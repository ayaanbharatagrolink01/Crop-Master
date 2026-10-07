"""Region context: turns the farmer's location into local facts for the agronomist.

Every source is optional; one that fails or is not configured is simply skipped.
  place       OpenStreetMap Nominatim reverse geocoding (free; max 1 request/second; cached)
  weather     Open-Meteo forecast API (free, no key)
  soil        ISRIC SoilGrids (free; a modelled estimate, not a soil test)
  mandi       data.gov.in Agmarknet daily prices (free key: DATA_GOV_API_KEY in .env)
  advisories  PDFs you add: district agromet advisories, ICAR pest alerts, KVK guides

Privacy: exact coordinates are never stored. Cache keys use coordinates rounded to 0.01 degree (about 1 km).
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import date, datetime

import httpx

from .config import ROOT, settings

log = logging.getLogger("agronomist")

DB_PATH = os.getenv("REGION_DB", str(ROOT / "data" / "region.db"))
UA = "CropMasterAgronomist/1.0 (" + (os.getenv("NOMINATIM_EMAIL") or "contact: set NOMINATIM_EMAIL in .env") + ")"
MANDI_RESOURCE = os.getenv("DATA_GOV_MANDI_RESOURCE", "9ef84268-d588-465a-a308-a864a43d0070")
PRICE_WORDS = re.compile(r"bhav|bhaav|mandi|price|rate|daam|keemat|kimat|bech|sell|भाव|मंडी|दाम|कीमत|बेच", re.I)
# Every source is on by default; switch one off with e.g. REGION_SOIL=off in .env.
def _on(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "on", "true", "yes")


SOW_WORDS = re.compile(r"\b(bo|bona|boye|bou|buvai|buwai|sow|sowing|grow|ugau|ugana|lagau|lagaye|lagana|"
                       r"kaun ?si fasal|kya fasal|which crop|what crop)\b|बो|बुवाई|उगा|फसल", re.I)


STOP = set("what which when where with that this have from your about crop crops plant plants field farm should "
           "could would there their they them give tell help need best good much many does than into also over "
           "under after before during more less very some kya hai kaise karen kare mere mera meri".split())


# ---------- storage: cache + regional documents ----------
class _Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires REAL);
            CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, title TEXT, source TEXT, state TEXT, district TEXT,
                crops TEXT, valid_until TEXT, added_at TEXT, pages INTEGER);
            CREATE TABLE IF NOT EXISTS chunks (doc_id TEXT, page INTEGER, text TEXT);
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);""")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(docs)")}
        if "doc_type" not in cols:
            self.db.execute("ALTER TABLE docs ADD COLUMN doc_type TEXT")
        self.db.commit()

    def cache_get(self, key: str):
        r = self.db.execute("SELECT value, expires FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r and r[1] > time.time() else None

    def cache_set(self, key: str, value, ttl: float):
        with self.lock:
            self.db.execute("REPLACE INTO cache VALUES (?,?,?)", (key, json.dumps(value), time.time() + ttl))
            self.db.commit()


store = _Store(DB_PATH)


def _coords(profile: dict) -> tuple[float, float] | None:
    try:
        lat, lon = float(profile.get("lat") or ""), float(profile.get("lon") or "")
    except ValueError:
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _k(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"


# ---------- place lookups: Nominatim first, Photon as backup ----------
# Both use OpenStreetMap data and are free. Nominatim sometimes refuses (403) apps or shared mobile IPs
# (e.g. carrier-grade NAT); then Photon answers instead.
EMAIL = os.getenv("NOMINATIM_EMAIL", "").strip()
INDIA_BBOX = "68.1,6.5,97.4,35.7"  # lon_min, lat_min, lon_max, lat_max
_geo_lock = asyncio.Lock()
_geo_last = 0.0
_nominatim_off_until = 0.0  # after a refusal, skip Nominatim for a while


def _clean(s: str) -> str:
    return re.sub(r"\s+(District|district|Tahsil|Tehsil|tahsil|tehsil)$", "", s or "").strip()


def _from_nominatim(a: dict, default_locality: str = "") -> dict:
    return {"locality": next((a[k] for k in ("village", "hamlet", "town", "city", "suburb", "neighbourhood",
                                             "municipality") if a.get(k)), default_locality),
            "subdistrict": _clean(a.get("county") or a.get("subdistrict") or ""),
            "district": _clean(a.get("state_district") or a.get("district") or ""),
            "state": a.get("state", "")}


def _from_photon(props: dict, default_locality: str = "") -> dict:
    is_place = props.get("osm_key") == "place" or props.get("type") in ("city", "town", "village", "locality")
    return {"locality": (props.get("name") if is_place else None) or props.get("city") or props.get("locality")
                        or props.get("name") or default_locality,
            "subdistrict": "",
            "district": _clean(props.get("county") or props.get("district") or ""),
            "state": props.get("state", "")}


async def _nominatim(path: str, params: dict):
    """Returns parsed JSON, or None if Nominatim refused or failed (so the caller tries Photon)."""
    global _geo_last, _nominatim_off_until
    if time.time() < _nominatim_off_until:
        return None
    if EMAIL:
        params = {**params, "email": EMAIL}
    async with _geo_lock:  # Nominatim's policy: at most one request per second
        wait = 1.1 - (time.time() - _geo_last)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(timeout=10, headers={"User-Agent": UA}) as http:
                r = await http.get(f"https://nominatim.openstreetmap.org/{path}", params=params)
        except httpx.HTTPError as e:
            log.warning("Nominatim unreachable (%s); using Photon", e)
            return None
        finally:
            _geo_last = time.time()
    if r.status_code in (403, 429):
        _nominatim_off_until = time.time() + 600
        log.warning("Nominatim refused the request (%s): %s ... Using Photon for the next 10 minutes. "
                    "Set NOMINATIM_EMAIL in .env if it is missing.", r.status_code,
                    re.sub(r"<[^>]+>|\s+", " ", r.text)[:160])
        return None
    if r.status_code != 200:
        log.warning("Nominatim error %s; using Photon", r.status_code)
        return None
    return r.json()


async def _photon(path: str, params: dict):
    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": UA}) as http:
        r = await http.get(f"https://photon.komoot.io/{path}", params={**params, "lang": "en"})
    r.raise_for_status()
    feats = [f for f in r.json().get("features", [])
             if (f.get("properties", {}).get("countrycode") or "IN").upper() == "IN"]
    return feats[0] if feats else None


async def place(lat: float, lon: float) -> dict | None:
    """Village/town, tehsil, district and state for a point."""
    key = "geo:" + _k(lat, lon)
    if (hit := store.cache_get(key)) is not None:
        return hit
    j = await _nominatim("reverse", {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14,
                                     "addressdetails": 1, "accept-language": "en"})
    if j and j.get("address"):
        out = _from_nominatim(j["address"])
    else:
        f = await _photon("reverse", {"lat": lat, "lon": lon})
        if not f:
            return None
        out = _from_photon(f.get("properties", {}))
        log.info("PLACE from Photon: %s", out)
    store.cache_set(key, out, 365 * 86400)
    return out


# Common parts of Indian place names; a farmer may type them joined ("ganjbasoda", "belkhedi")
NAME_PARTS = ("ganj", "pur", "pura", "nagar", "garh", "gaon", "khedi", "kheda", "khera", "wadi", "wada", "abad",
              "kot", "kalan", "khurd", "bazar", "mandi", "sagar", "tola", "sarai", "chauraha", "chouraha")


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _split_variants(text: str) -> list[str]:
    t = text.strip()
    if " " in t or len(t) < 7:
        return []
    low, out = t.lower(), []
    for part in NAME_PARTS:
        if low.startswith(part) and len(low) - len(part) >= 3:
            out.append(t[:len(part)] + " " + t[len(part):])
        if low.endswith(part) and len(low) - len(part) >= 3:
            out.append(t[:-len(part)] + " " + t[-len(part):])
    return list(dict.fromkeys(out))[:2]


def _lgd_guess(text: str, lgd: list, state: str, district: str) -> str | None:
    """'ganjbasoda' -> 'Basoda, Vidisha, Madhya Pradesh' when an official name is inside what was typed."""
    key, best = _letters(text), None
    for name, dist, st in lgd or []:
        k = _letters(name)
        if len(k) >= 5 and (k in key or key in k):
            score = (st.lower() == state.lower(), dist.lower() == district.lower(), len(k))
            if best is None or score > best[0]:
                best = (score, f"{name}, {dist}, {st}")
    return best[1] if best else None


async def _search_once(q: str, typed: str) -> dict | None:
    j = await _nominatim("search", {"q": q, "format": "jsonv2", "addressdetails": 1, "limit": 1,
                                    "countrycodes": "in", "accept-language": "en"})
    if j:
        return {"lat": float(j[0]["lat"]), "lon": float(j[0]["lon"]), **_from_nominatim(j[0].get("address", {}), typed)}
    try:  # Nominatim found nothing or refused: Photon is more forgiving with spelling
        f = await _photon("api/", {"q": q, "limit": 1, "bbox": INDIA_BBOX})
    except httpx.HTTPError as e:
        log.warning("Photon search failed: %s", e)
        return None
    if not f:
        return None
    lon, lat = f["geometry"]["coordinates"][:2]
    return {"lat": lat, "lon": lon, **_from_photon(f.get("properties", {}), typed)}


async def find_place(text: str, state: str = "", district: str = "", lgd: list | None = None) -> dict | None:
    """Coordinates and admin names for a place the farmer types (village, chauraha, town), within India."""
    text = text.strip()
    if len(text) < 3:
        return None
    hint = lambda name: ", ".join(x for x in (name, district.strip(), state.strip()) if x)
    key = "find2:" + hint(text).lower()  # "find2": ignores misses cached by the older, simpler search
    if (hit := store.cache_get(key)) is not None:
        return hit or None
    tries = [hint(text)] + [hint(v) for v in _split_variants(text)]
    guess = _lgd_guess(text, lgd, state, district)
    if guess:
        tries.append(guess)
    out = None
    for q in dict.fromkeys(tries):
        out = await _search_once(q, text)
        if out:
            if q != tries[0]:
                log.info("PLACE %r found as %r", text, q)
            break
    if not out:
        log.info("PLACE not found: %r (tried %s)", text, list(dict.fromkeys(tries)))
        store.cache_set(key, {}, 86400)  # remember a miss for a day
        return None
    out = {**out, "lat": round(out["lat"], 3), "lon": round(out["lon"], 3)}
    store.cache_set(key, out, 365 * 86400)
    store.cache_set("geo:" + _k(out["lat"], out["lon"]),  # so the answer step needs no second lookup
                    {k: out[k] for k in ("locality", "subdistrict", "district", "state")}, 365 * 86400)
    return out


# ---------- elevation (tells hills and cold deserts apart from the plains) ----------
async def elevation(lat: float, lon: float) -> float | None:
    key = "elev:" + _k(lat, lon)
    if (hit := store.cache_get(key)) is not None:
        return hit.get("m")
    async with httpx.AsyncClient(timeout=8, headers={"User-Agent": UA}) as http:
        r = await http.get("https://api.open-meteo.com/v1/elevation", params={"latitude": lat, "longitude": lon})
    r.raise_for_status()
    vals = r.json().get("elevation") or []
    m = round(float(vals[0])) if vals and vals[0] is not None else None
    store.cache_set(key, {"m": m}, 3650 * 86400)  # the ground doesn't move
    return m


def terrain(m: float | None) -> str:
    if m is None:
        return ""
    if m >= 2500:
        return f"elevation about {m} m: high-altitude cold region; plains seasons and crop lists do not apply"
    if m >= 1200:
        return f"elevation about {m} m: hill area; seasons and crops differ from the plains"
    if m >= 600:
        return f"elevation about {m} m: plateau"
    return f"elevation about {m} m: plains"


# ---------- weather ----------
async def weather(lat: float, lon: float) -> dict | None:
    key = "wx2:" + _k(lat, lon)
    if (hit := store.cache_get(key)) is not None:
        return hit
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": round(lat, 2), "longitude": round(lon, 2), "timezone": "Asia/Kolkata",
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "hourly": "relative_humidity_2m,soil_moisture_9_to_27cm",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,"
                     "wind_speed_10m_max,et0_fao_evapotranspiration",
            "past_days": 7, "forecast_days": 7})
    r.raise_for_status()
    j = r.json()
    rh = defaultdict(list)
    h = j.get("hourly", {})
    for t, v in zip(h.get("time", []), h.get("relative_humidity_2m", [])):
        if v is not None:
            rh[t[:10]].append(v)
    now = datetime.now().strftime("%Y-%m-%dT%H:00")
    moist = [v for t, v in zip(h.get("time", []), h.get("soil_moisture_9_to_27cm", [])) if t <= now and v is not None]
    d = j["daily"]
    today = date.today().isoformat()
    days = []
    for i, day in enumerate(d["time"]):
        days.append({"date": day, "tmax": d["temperature_2m_max"][i], "tmin": d["temperature_2m_min"][i],
                     "rain_mm": d["precipitation_sum"][i] or 0, "rain_prob": d["precipitation_probability_max"][i],
                     "wind_max": d["wind_speed_10m_max"][i],
                     "et0_mm": (d.get("et0_fao_evapotranspiration") or [None] * len(d["time"]))[i],
                     "rh_mean": round(sum(rh[day]) / len(rh[day])) if rh[day] else None})
    past = [x for x in days if x["date"] < today]
    fut = [x for x in days if x["date"] >= today]
    out = {"current": j.get("current", {}),
           "past_rain_mm": round(sum(x["rain_mm"] for x in past), 1),
           "past_et0_mm": round(sum(x["et0_mm"] or 0 for x in past), 1),
           "next_et0_mm": round(sum(x["et0_mm"] or 0 for x in fut), 1),
           "next_rain_mm": round(sum(x["rain_mm"] for x in fut), 1),
           "root_zone_moisture": round(moist[-1], 3) if moist else None,  # m3/m3 at 9-27 cm
           "forecast": fut}
    store.cache_set(key, out, 2 * 3600)
    return out


def weather_notes(w: dict) -> list[str]:
    """Rule-of-thumb flags the agronomist can use for spray, irrigation and disease timing."""
    f, notes = w["forecast"], []
    if not f:
        return notes
    soon = f[:2]
    if any((x["rain_prob"] or 0) >= 60 or x["rain_mm"] >= 2 for x in soon):
        notes.append("Rain likely today or tomorrow: avoid spraying and fertilizer top-dressing until it clears.")
    good = next((x["date"] for x in f if (x["rain_prob"] or 0) < 40 and x["rain_mm"] < 1
                 and (x["wind_max"] or 0) < 15), None)
    if good:
        notes.append(f"First suitable spray day (low rain chance, wind under 15 km/h): {good}.")
    humid = [x for x in f[:3] if (x["rh_mean"] or 0) >= 85]
    if len(humid) >= 2:
        notes.append("High humidity on most of the next 3 days: raised risk of fungal and bacterial diseases.")
    if any((x["tmax"] or 0) >= 40 for x in f[:5]):
        notes.append("Heat of 40 °C or more expected: irrigate in the evening; heat stress likely at flowering.")
    if any((x["tmin"] or 99) <= 4 for x in f[:5]):
        notes.append("Night temperature near 4 °C or below: frost risk for sensitive crops; light irrigation helps.")
    if w["past_rain_mm"] < 2 and sum(x["rain_mm"] for x in f[:5]) < 5:
        notes.append("Little rain in the last week and none expected soon: check soil moisture before skipping irrigation.")
    if w.get("past_et0_mm"):  # water balance: crop water use (reference ET0) against rain
        gap_past = w["past_et0_mm"] - w["past_rain_mm"]
        gap_next = w["next_et0_mm"] - w["next_rain_mm"]
        if gap_next >= 20:
            notes.append(f"Water: over the next 7 days a full crop will use about {w['next_et0_mm']:.0f} mm while only "
                         f"{w['next_rain_mm']:.0f} mm of rain is expected (gap ~{gap_next:.0f} mm): plan irrigation "
                         f"for standing crops.")
        elif gap_next <= -15:
            notes.append(f"Water: expected rain ({w['next_rain_mm']:.0f} mm) exceeds crop water use over the next "
                         f"7 days: hold irrigation and keep drainage open.")
        if gap_past >= 25:
            notes.append(f"Water: in the last 7 days crop water use was ~{w['past_et0_mm']:.0f} mm against "
                         f"{w['past_rain_mm']:.0f} mm of rain.")
    return notes


# ---------- soil ----------
def _texture(sand: float, silt: float, clay: float) -> str:
    if clay >= 40:
        return "clay"
    if clay >= 27:
        return "clay loam" if sand <= 45 else "sandy clay loam"
    if sand >= 70:
        return "sandy" if clay < 10 else "sandy loam"
    if silt >= 50:
        return "silt loam"
    return "loam" if sand < 52 else "sandy loam"


def _ph_class(ph: float) -> str:
    return ("strongly acidic" if ph < 5.5 else "slightly acidic" if ph < 6.5 else "neutral" if ph < 7.5
            else "alkaline" if ph < 8.5 else "strongly alkaline")


async def soil(lat: float, lon: float) -> dict | None:
    key = "soil2:" + _k(lat, lon)
    if (hit := store.cache_get(key)) is not None:
        return hit or None
    params = [("lon", round(lon, 3)), ("lat", round(lat, 3)), ("value", "mean")]
    params += [("property", p) for p in ("phh2o", "clay", "sand", "silt", "soc", "nitrogen", "cec")]
    params += [("depth", d) for d in ("0-5cm", "5-15cm")]
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.get("https://rest.isric.org/soilgrids/v2.0/properties/query", params=params)
    r.raise_for_status()
    vals = {}
    for layer in r.json().get("properties", {}).get("layers", []):
        dfac = (layer.get("unit_measure") or {}).get("d_factor") or 1
        nums = [x["values"]["mean"] / dfac for x in layer.get("depths", []) if x["values"].get("mean") is not None]
        if nums:
            vals[layer["name"]] = round(sum(nums) / len(nums), 1)
    out = {}
    if "phh2o" in vals:
        out["ph"], out["ph_class"] = vals["phh2o"], _ph_class(vals["phh2o"])
    if all(k in vals for k in ("sand", "silt", "clay")):
        out.update(sand_pct=vals["sand"], silt_pct=vals["silt"], clay_pct=vals["clay"],
                   texture=_texture(vals["sand"], vals["silt"], vals["clay"]))
    if "soc" in vals:
        out["organic_carbon_g_kg"] = vals["soc"]
    if "nitrogen" in vals:  # SoilGrids: cg/kg; after d_factor 100 -> g/kg
        out["total_nitrogen_g_kg"] = vals["nitrogen"]
    if "cec" in vals:  # mmol(c)/kg; after d_factor 10 -> cmol(c)/kg
        out["cec_cmol_kg"] = vals["cec"]
    store.cache_set(key, out, 365 * 86400)  # an empty result (water, city) is cached too
    return out or None


# ---------- mandi prices (data.gov.in / Agmarknet) ----------
async def mandi(state: str, district: str, crops: list[str]) -> list[dict]:
    api_key = os.getenv("DATA_GOV_API_KEY")
    if not api_key or not state:
        return []
    key = f"mandi:{state}:{district}".lower()
    rows = store.cache_get(key)
    if rows is None:
        async with httpx.AsyncClient(timeout=15) as http:
            base = {"api-key": api_key, "format": "json", "limit": 200, "filters[state.keyword]": state}
            rows = []
            if district:
                r = await http.get(f"https://api.data.gov.in/resource/{MANDI_RESOURCE}",
                                   params={**base, "filters[district]": district})
                r.raise_for_status()
                rows = r.json().get("records", [])
            if not rows:  # district names differ between sources: fall back to the whole state
                r = await http.get(f"https://api.data.gov.in/resource/{MANDI_RESOURCE}", params=base)
                r.raise_for_status()
                rows = r.json().get("records", [])
        store.cache_set(key, rows, 6 * 3600)
    want = [c for c in crops if c]
    if want:
        rows = [x for x in rows if any(_commodity_match(w, x.get("commodity") or "") for w in want)]
    return rows[:6]


# Agmarknet commodity names differ from everyday crop names
MANDI_ALIASES = {"gram": ["bengal gram", "chana"], "chana": ["bengal gram"], "chickpea": ["bengal gram"],
                 "pigeon pea": ["arhar", "tur"], "tur": ["arhar"], "arhar": ["arhar"], "moong": ["green gram"],
                 "mung": ["green gram"], "urad": ["black gram"], "rice": ["paddy", "rice"], "dhan": ["paddy"],
                 "mustard": ["mustard", "rapeseed"], "sarson": ["mustard"], "groundnut": ["groundnut"],
                 "maize": ["maize"], "makka": ["maize"], "gehu": ["wheat"], "soya": ["soyabean"]}


def _commodity_match(crop: str, commodity: str) -> bool:
    import difflib
    c, m = crop.lower().strip(), commodity.lower()
    if not c or not m:
        return False
    if c in MANDI_ALIASES:  # names that are ambiguous on their own ("gram" is also in "green gram")
        return any(a in m for a in MANDI_ALIASES[c]) or m.split("(")[0].strip() == c
    if c in m or m.split("(")[0].strip() in c:
        return True
    first = re.sub(r"[^a-z]", "", m.split("(")[0].split()[0]) if m.split() else ""
    return difflib.SequenceMatcher(None, re.sub(r"[^a-z]", "", c), first).ratio() >= 0.85  # soybean ~ soyabean


# ---------- regional documents (PDF advisories) ----------
# ---------- chunking: cleanup -> sections by heading -> recursive split with overlap ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))        # characters (~200-250 tokens)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
# try the biggest natural break first: paragraph, line, sentence (English and Hindi "।"), clause, word
SEPARATORS = ["\n\n", "\n", "। ", ". ", "? ", "; ", ", ", " ", ""]
_JUNK_LINE = re.compile(
    r"^\s*(?:"
    r"(?:page\s*)?[ivxlcdm\d]{1,5}\s*(?:\|\s*p\s*a\s*g\s*e)?"   # "12", "ii | P a g e", "Page 7"
    r"|.*\.{5,}\s*[ivxlcdm\d]{1,5}"                             # table-of-contents line "Intro ....... 12"
    r"|-+\s*\d+\s*-+"                                            # "- 12 -"
    r")\s*$", re.I)
_HEADING = re.compile(r"^\s*((?:\d{1,2}\.){0,3}\d{1,2}\.?\s+[A-Z\u0900-\u097F][^\n]{2,90}?|[A-Z][A-Z &/,()-]{5,80})\s*$")


def _is_heading(line: str) -> bool:
    """A section title, not a table row: short, mostly words, small chapter number, few other digits or commas."""
    s = line.strip()
    if not _HEADING.match(s) or len(s.split()) > 14 or s.endswith((",", ";")) or s.count(",") > 1:
        return False
    num = re.match(r"(\d{1,2})((?:\.\d{1,2}){0,3})(\.?)\s+(.*)", s)
    if num:
        rest = num.group(4)
        if not num.group(2) and (not num.group(3) or int(num.group(1)) > 20):
            return False  # a chapter is "3. Title"; a bare "4 CITY FOREST" is a table row
        if not num.group(2):  # "1. Overview of Ground Water" is a title; "1. For higher production adopt" a list item
            words = [w for w in num.group(4).split() if len(w) > 3]
            if words and sum(w[0].isupper() for w in words) < 0.5 * len(words):
                return False
        if len(re.findall(r"\d", rest)) > 2:
            return False  # "3.1 Station 17.409 77.880" is data, not a title
        return sum(ch.isalpha() for ch in rest) >= 0.7 * len(rest.replace(" ", ""))
    return not re.search(r"\d", s)  # unnumbered: an all-capitals title without figures


def _clean(text: str) -> str:
    lines = [l for l in text.splitlines() if not _JUNK_LINE.match(l)]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_BULLET = re.compile(r"^\s*(?:[-•●▪*➢✓]|\(?[a-z0-9]{1,3}[.)])\s")


def _reflow(body: str) -> str:
    """PDF text breaks every printed line; join lines that continue the same sentence so the splitter can
    cut at real sentence ends. Bullets, numbered items and lines after a full stop stay on their own line."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            out.append("")
        elif out and out[-1] and not re.search(r"[.!?:;।)]$", out[-1]) and not _BULLET.match(line):
            out[-1] += " " + line
        else:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def _recursive_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                     seps: list[str] = SEPARATORS) -> list[str]:
    """Split on the largest separator that keeps pieces under `size`, recursing to smaller ones as needed,
    then merge neighbouring pieces back up to `size` with `overlap` characters carried over."""
    sep, rest = "", []
    for i, s in enumerate(seps):
        if s == "" or s in text:
            sep, rest = s, seps[i + 1:]
            break
    pieces = text.split(sep) if sep else [text[i:i + size] for i in range(0, len(text), size)]
    out, small = [], []
    for piece in pieces:
        if len(piece) <= size:
            small.append(piece)
            continue
        if small:
            out += _merge(small, sep, size, overlap)
            small = []
        out += _recursive_split(piece, size, overlap, rest) if rest else [piece]
    if small:
        out += _merge(small, sep, size, overlap)
    return out


def _merge(pieces: list[str], sep: str, size: int, overlap: int) -> list[str]:
    out, cur, length = [], [], 0
    for p in pieces:
        add = len(p) + (len(sep) if cur else 0)
        if cur and length + add > size:
            out.append(sep.join(cur).strip())
            while cur and (length > overlap or length + add > size):  # keep a tail as overlap
                length -= len(cur[0]) + (len(sep) if len(cur) > 1 else 0)
                cur.pop(0)
        cur.append(p)
        length += len(p) + (len(sep) if len(cur) > 1 else 0)
    if cur:
        out.append(sep.join(cur).strip())
    return [c for c in out if c]


def _useful(chunk: str) -> bool:
    letters = sum(ch.isalpha() for ch in chunk)
    return len(chunk) >= 80 and letters / len(chunk) >= 0.5


def _chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """(page, chunk) pairs. Each chunk starts with the section heading it belongs to (carried across pages)."""
    out, trail = [], []  # heading path as (level, text); level 0 = unnumbered
    for page, raw in pages:
        sections, cur = [], []
        for line in _clean(raw).splitlines():
            if _is_heading(line):
                if cur:
                    sections.append((" > ".join(t for _, t in trail), "\n".join(cur)))
                    cur = []
                h = line.strip()
                num = re.match(r"(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s", h)
                if num:  # numbered heading: keep only its numbered parents (drops stray capitalised lines)
                    level = num.group(1).count(".") + 1
                    trail = [t for t in trail if t[0] and t[0] < level] + [(level, h)]
                else:    # unnumbered heading: deepest entry, replacing an earlier unnumbered one
                    trail = [t for t in trail if t[0]] + [(0, h)]
            else:
                cur.append(line)
        if cur:
            sections.append((" > ".join(t for _, t in trail), "\n".join(cur)))
        for head, body in sections:
            parts = head.split(" > ")
            while len(" > ".join(parts)) > 140 and len(parts) > 1:  # drop whole top levels, keep the specific ones
                parts.pop(0)
            head = ("… > " if len(parts) < len(head.split(" > ")) else "") + " > ".join(parts)
            room = CHUNK_SIZE - (len(head) + 4 if head else 0)
            for piece in _recursive_split(_reflow(body).strip(), size=max(room, 300)):
                if _useful(piece):
                    out.append((page, f"[{head}] {piece}" if head else piece))
    return out


def _chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:  # kept for older callers
    return [c for _, c in _chunk_pages([(0, text)])]


def _pdf_pages(pdf: bytes) -> list[tuple[int, str]]:
    """Text of each page. pypdfium2 first: pypdf garbles many government PDFs (spaces read as "D", "-" as "f")."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf)
        pages = []
        for i in range(len(doc)):
            t = doc[i].get_textpage().get_text_range()
            t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\ufffe", "-").replace("\u00ad", "-")
            pages.append((i + 1, t.strip()))
        return pages
    except Exception as e:
        log.info("pdfium text extraction failed (%s); using pypdf", e)
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf))
    return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]


OCR_PROMPT = ("Transcribe all text on this page of an Indian agriculture document exactly as written. Keep Hindi in "
              "Devanagari script. Write tables as rows with ' | ' between cells. Output only the page text.")


async def _ocr_pages(pdf: bytes, pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Read pages that have no text layer (scanned) with the vision model. Needs OCR_PDF=on and pypdfium2."""
    empty = [i for i, (_, t) in enumerate(pages) if len(t) < 40]
    if not empty:
        return pages
    if not _on("OCR_PDF", "off"):
        log.info("DOC %d of %d pages have no text (scanned). Set OCR_PDF=on in .env to read them with %s.",
                 len(empty), len(pages), settings.vision_model)
        return pages
    import base64
    import pypdfium2 as pdfium
    from openai import AsyncOpenAI
    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    doc = pdfium.PdfDocument(pdf)
    limit = int(os.getenv("OCR_MAX_PAGES", "40"))
    out, tokens = list(pages), [0, 0]
    done = failed = 0
    for i in empty[:limit]:
        try:  # one bad page must not stop the whole import
            img = doc[i].render(scale=150 / 72).to_pil().convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            resp = await oai.chat.completions.create(model=settings.vision_model, max_tokens=2000, temperature=0,
                                                     messages=[{"role": "user", "content": [
                                                         {"type": "text", "text": OCR_PROMPT},
                                                         {"type": "image_url", "image_url": {"url": url,
                                                                                             "detail": "high"}}]}])
        except ImportError as e:
            log.warning("OCR needs Pillow and pypdfium2 (pip install pillow pypdfium2): %s. Scanned pages skipped.", e)
            return out
        except Exception as e:
            failed += 1
            log.warning("OCR failed on page %d: %s", pages[i][0], e)
            continue
        out[i] = (pages[i][0], (resp.choices[0].message.content or "").strip())
        done += 1
        u = getattr(resp, "usage", None)
        tokens[0] += getattr(u, "prompt_tokens", 0) or 0
        tokens[1] += getattr(u, "completion_tokens", 0) or 0
    log.info("DOC OCR read %d scanned pages with %s (tokens in=%d out=%d)%s%s", done,
             settings.vision_model, tokens[0], tokens[1], f"; {failed} pages failed" if failed else "",
             f"; {len(empty) - limit} more skipped (OCR_MAX_PAGES)" if len(empty) > limit else "")
    return out


async def add_document(pdf: bytes, title: str, source: str = "", state: str = "", district: str = "",
                       crops: str = "", valid_until: str = "", doc_type: str = "") -> dict:
    """Store a PDF's text tagged by region, and index it in Qdrant when RAG is configured.
    Raises ValueError if no readable text was found."""
    from . import rag
    pages = await _ocr_pages(pdf, _pdf_pages(pdf))
    if sum(len(t) for _, t in pages) < 50:
        raise ValueError("No readable text in this PDF (scanned image). Set OCR_PDF=on in .env to read it.")
    chunks = _chunk_pages(pages)
    with store.lock:
        n = store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] + 1
        did = f"DOC{n:03d}"
        while store.db.execute("SELECT 1 FROM docs WHERE id=?", (did,)).fetchone():
            n += 1
            did = f"DOC{n:03d}"
        store.db.execute("INSERT INTO docs (id, title, source, state, district, crops, valid_until, added_at, pages, "
                         "doc_type) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (did, title, source, state.strip(), district.strip(), crops, valid_until or None,
                          datetime.now().isoformat(timespec="seconds"), len(pages), doc_type))
        store.db.executemany("INSERT INTO chunks VALUES (?,?,?)", [(did, p, c) for p, c in chunks])
        store.db.commit()
    doc = get_document(did)
    if rag.enabled():
        try:
            doc["indexed_chunks"] = await rag.index_document(doc, chunks)
        except Exception as e:  # the text is saved; `import_doc --reindex` can retry the index later
            log.warning("RAG indexing failed for %s: %s", did, e)
            doc["indexed_chunks"] = 0
    log.info("DOC added %s %r (%s / %s, %d pages, %d chunks)", did, title, state or "all states",
             district or "all districts", len(pages), len(chunks))
    return doc


async def add_document_from_url(url: str, **meta) -> dict:
    async with httpx.AsyncClient(timeout=60, follow_redirects=True, headers={"User-Agent": UA}) as http:
        r = await http.get(url)
    r.raise_for_status()
    if b"%PDF" not in r.content[:1024]:
        raise ValueError("That link did not return a PDF")
    return await add_document(r.content, source=url, **meta)


async def reindex_all() -> int:
    """Rebuild the Qdrant index from the texts saved in region.db."""
    from . import rag
    n = 0
    for d in list_documents():
        chunks = store.db.execute("SELECT page, text FROM chunks WHERE doc_id=?", (d["id"],)).fetchall()
        await rag.delete_document(d["id"])
        n += await rag.index_document(d, chunks)
    return n


def get_document(did: str) -> dict | None:
    r = store.db.execute("SELECT id, title, source, state, district, crops, valid_until, added_at, pages, doc_type "
                         "FROM docs WHERE id=?", (did.upper(),)).fetchone()
    if not r:
        return None
    keys = ("id", "title", "source", "state", "district", "crops", "valid_until", "added_at", "pages", "doc_type")
    return dict(zip(keys, r))


_UNITS = {"hafte": "weeks", "hafta": "weeks", "haftey": "weeks", "saptah": "weeks", "week": "weeks", "weeks": "weeks",
          "wk": "weeks", "din": "days", "dino": "days", "day": "days", "days": "days", "mahine": "months",
          "mahina": "months", "month": "months", "months": "months", "हफ्ते": "weeks", "सप्ताह": "weeks",
          "दिन": "days", "महीने": "months"}
_NUM_RE = re.compile(r"(\d{1,3})\s*(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\b", re.I)


def _number_phrases(question: str) -> list[str]:
    """'monsoon 2 hafte late' -> ['2 weeks']: documents are organised by these figures, and both keyword and
    vector search are weak at telling '2 weeks' from '6 weeks'."""
    return list(dict.fromkeys(f"{n} {_UNITS[u.lower()]}" for n, u in _NUM_RE.findall(question)))


# Farmers' words -> the words government plans use (for search, and to read the context around a figure)
TERM_MAP = {"late": "delay delayed", "der": "delay delayed", "deri": "delay", "jaldi": "early", "pehle": "early",
            "sukha": "drought dry spell", "sookha": "drought dry spell", "baarish": "rain rainfall",
            "barish": "rain rainfall", "barsat": "rain monsoon", "zyada": "excess heavy", "jyada": "excess heavy",
            "kam": "low deficit", "pala": "frost", "paala": "frost", "thand": "cold wave", "sheet": "cold wave",
            "loo": "heat wave", "garmi": "heat wave", "ole": "hailstorm", "ola": "hailstorm", "baadh": "flood",
            "badh": "flood", "jalbharav": "waterlogging", "paani": "water irrigation", "sinchai": "irrigation",
            "keet": "pest", "keede": "pest", "rog": "disease", "bimari": "disease", "mitti": "soil",
            "bona": "sowing", "buvai": "sowing", "bowai": "sowing", "beej": "seed variety", "kism": "variety"}


def _plan_terms(question: str) -> list[str]:
    words = re.findall(r"[a-z]+", question.lower())
    return list(dict.fromkeys(t for w in words for t in TERM_MAP.get(w, "").split()))


def _number_candidates(state: str, district: str, numbers: list[str], limit: int = 12) -> list[dict]:
    """Chunks in the farmer's region that contain the exact figure ("2 weeks"), however different the rest is."""
    if not numbers:
        return []
    st, di, today = (state or "").strip().lower(), (district or "").strip().lower(), date.today().isoformat()
    out = []
    for n, u in (x.split() for x in numbers):
        rows = store.db.execute(
            "SELECT d.id, d.title, d.state, d.district, c.page, c.text FROM chunks c JOIN docs d ON d.id = c.doc_id "
            "WHERE (d.valid_until IS NULL OR d.valid_until >= ?) AND c.text LIKE ?",
            (today, f"%{n} {u[:-1]}%")).fetchall()
        for did, title, dst, ddi, page, text in rows:
            dst_l, ddi_l = (dst or "").lower(), (ddi or "").lower()
            if (ddi_l and ddi_l != di) or (not ddi_l and dst_l and dst_l != st):
                continue
            out.append({"id": did, "title": title, "page": page, "text": text,
                        "scope": "district" if ddi_l else "state" if dst_l else "national"})
    return out[:limit]


def _rerank_numbers(docs: list[dict], numbers: list[str], terms: list[str] = ()) -> list[dict]:
    """Excerpts with the question's figure go first; among them, those where the question's words sit right next
    to the figure ("Delay by 2 weeks" for "2 hafte late", not "consecutive 2 weeks rainless")."""
    if not numbers:
        return docs
    pats = [re.compile(rf"\b{n}\s*{u[:-1]}s?\b", re.I) for n, u in (x.split() for x in numbers)]

    def score(d):
        t = d.get("text", "")
        best = 0
        for p in pats:
            for m in p.finditer(t):
                window = t[max(0, m.start() - 60):m.end() + 60].lower()
                best = max(best, 1 + sum(1 for w in terms if w in window))
        return -best
    return sorted(docs, key=score)


PAGE_CONTEXT_CHARS = int(os.getenv("PAGE_CONTEXT_CHARS", "2600"))  # ~650 tokens per expanded page
PAGES_EXPANDED = int(os.getenv("PAGES_EXPANDED", "2"))


def _page_text(doc_id: str, page: int) -> str:
    """The page's chunks joined back together (heading kept once, overlaps removed)."""
    rows = [t for (t,) in store.db.execute("SELECT text FROM chunks WHERE doc_id=? AND page=? ORDER BY rowid",
                                           (doc_id, page))]
    if not rows:
        return ""
    head = re.match(r"^\[([^\]]*)\]\s*", rows[0])
    text = ""
    for t in rows:
        t = re.sub(r"^\[[^\]]*\]\s*", "", t)
        for k in range(min(len(t), 220), 20, -1):  # drop the overlap carried from the previous chunk
            if text.endswith(t[:k]):
                t = t[k:]
                break
        text = (text + " " + t).strip()
    text = text[:PAGE_CONTEXT_CHARS] + (" …" if len(text) > PAGE_CONTEXT_CHARS else "")
    return (f"[{head.group(1)}] " if head else "") + text


def _expand_to_pages(docs: list[dict]) -> list[dict]:
    """Small-to-big: search matches small chunks; the agronomist reads the whole page around the best ones,
    because in contingency plans and advisories one page usually holds one complete situation."""
    out, seen = [], set()
    for d in docs:
        key = (d["id"], d.get("page"))
        if key in seen:
            continue
        seen.add(key)
        if len([x for x in out if x.get("expanded")]) < PAGES_EXPANDED and d.get("page"):
            full = _page_text(d["id"], d["page"])
            if full and len(full) > len(d["text"]):
                d = {**d, "text": full, "expanded": True}
        out.append(d)
    return out


def doc_id_by_source(source: str) -> str | None:
    row = store.db.execute("SELECT id FROM docs WHERE source=?", (source,)).fetchone()
    return row[0] if row else None


def has_source(source: str) -> bool:
    return store.db.execute("SELECT 1 FROM docs WHERE source=?", (source,)).fetchone() is not None


def list_documents() -> list[dict]:
    ids = [r[0] for r in store.db.execute("SELECT id FROM docs ORDER BY added_at DESC")]
    return [get_document(i) for i in ids]


async def delete_document(did: str) -> bool:
    from . import rag
    with store.lock:
        n = store.db.execute("DELETE FROM docs WHERE id=?", (did.upper(),)).rowcount
        store.db.execute("DELETE FROM chunks WHERE doc_id=?", (did.upper(),))
        store.db.commit()
    if n and rag.enabled():
        try:
            await rag.delete_document(did.upper())
        except Exception as e:
            log.warning("RAG delete failed for %s: %s", did, e)
    return n > 0


def search_documents(state: str, district: str, text: str, limit: int = 3) -> list[dict]:
    """Excerpts from documents for this district (or state-wide / national ones) that match the question."""
    st, di = (state or "").strip().lower(), (district or "").strip().lower()
    today = date.today().isoformat()
    rows = store.db.execute(
        "SELECT d.id, d.title, d.state, d.district, d.crops, d.added_at, c.page, c.text FROM chunks c "
        "JOIN docs d ON d.id = c.doc_id WHERE d.valid_until IS NULL OR d.valid_until >= ?", (today,)).fetchall()
    words = {w for w in re.findall(r"[a-z\u0900-\u097f]{4,}", text.lower()) if w not in STOP}
    scored = []
    for did, title, dst, ddi, crops, added, page, chunk in rows:
        dst_l, ddi_l = (dst or "").lower(), (ddi or "").lower()
        if ddi_l and ddi_l != di:
            continue  # another district's advisory
        if not ddi_l and dst_l and dst_l != st:
            continue  # another state's document
        hay = (chunk + " " + (crops or "") + " " + title).lower()
        s = sum(1 for w in words if w in hay)
        if ddi_l:
            s += 2 if s else 0
            if (datetime.now() - datetime.fromisoformat(added)).days <= 14:
                s += 1  # a recent advisory for this district is relevant even without matching words
        if s > 0:
            scored.append((s, {"id": did, "title": title, "page": page, "text": chunk}))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, r in scored:
        if (r["id"], r["page"]) in seen:
            continue
        seen.add((r["id"], r["page"]))
        out.append(r)
        if len(out) >= limit:
            break
    return out


# ---------- the block the agronomist sees ----------
async def build(profile: dict, question: str, crops: list[str]) -> tuple[str, list[dict], dict]:
    """Returns (context text, document refs, place). Never raises."""
    ll = _coords(profile)
    pl = wx = sl = None
    if ll:
        async def none():
            return None
        results = await asyncio.gather(place(*ll), weather(*ll) if _on("REGION_WEATHER") else none(),
                                       soil(*ll) if _on("REGION_SOIL") else none(),
                                       elevation(*ll) if os.getenv("REGION_ELEVATION", "on").lower() != "off" else none(),
                                       return_exceptions=True)
        for name, res in zip(("place", "weather", "soil", "elevation"), results):
            if isinstance(res, Exception):
                log.warning("REGION %s failed: %s", name, res)
        pl, wx, sl, elev = (None if isinstance(x, Exception) else x for x in results)
    else:
        elev = None
    state = profile.get("state") or (pl or {}).get("state", "")
    district = profile.get("district") or (pl or {}).get("district", "")

    prices = []
    if _on("REGION_MANDI") and PRICE_WORDS.search(question):  # only when switched on and the farmer asks
        try:
            prices = await mandi(state, district, crops)
        except Exception as e:
            log.warning("REGION mandi failed: %s", e)
    docs = []
    if _on("REGION_DOCS"):
        from . import rag
        numbers = _number_phrases(question)  # "2 hafte" -> "2 weeks", the plans' own wording
        terms = _plan_terms(question)         # "late" -> "delay delayed"
        query = f"{question} {' '.join(crops)} {' '.join(numbers)} {' '.join(terms)}".strip()
        if SOW_WORDS.search(question):  # pull the district profile / contingency-plan parts too
            query += " major crops cropping pattern sowing time rainfall soils irrigation"
        try:
            docs = await rag.search(state, district, query, limit=8) if rag.enabled() else \
                search_documents(state, district, query, limit=8)
        except Exception as e:
            log.warning("REGION documents failed (%s); using keyword search", e)
            try:
                docs = search_documents(state, district, query)
            except Exception:
                docs = []
        seen = {(d["id"], d.get("page")) for d in docs}
        docs += [d for d in _number_candidates(state, district, numbers) if (d["id"], d["page"]) not in seen]
        docs = _expand_to_pages(_rerank_numbers(docs, numbers, terms)[:4])

    lines = []
    if pl:
        bits = [pl["locality"], pl["subdistrict"] and f'{pl["subdistrict"]} tehsil',
                pl["district"] and f'{pl["district"]} district', pl["state"]]
        lines.append("Place (from the farmer's location): " + ", ".join(b for b in bits if b))
    if elev is not None:
        lines.append("Terrain: " + terrain(elev))
    if wx:
        c = wx.get("current", {})
        lines.append(f"Weather now: {c.get('temperature_2m')} °C, humidity {c.get('relative_humidity_2m')}%, "
                     f"wind {c.get('wind_speed_10m')} km/h. Rain in the last 7 days: {wx['past_rain_mm']} mm.")
        if wx.get("past_et0_mm") or wx.get("root_zone_moisture") is not None:
            m = wx.get("root_zone_moisture")
            lines.append(f"Water (Open-Meteo model): reference crop water use (ET0) last 7 days {wx.get('past_et0_mm')} mm, "
                         f"next 7 days {wx.get('next_et0_mm')} mm; expected rain next 7 days {wx.get('next_rain_mm')} mm"
                         + (f"; root-zone soil moisture (9-27 cm) {m} m3/m3 "
                            f"({'dry' if m < 0.15 else 'moist' if m < 0.32 else 'wet'}, model estimate)" if m is not None
                            else "") + ".")
        lines.append("Forecast: " + "; ".join(
            f"{x['date'][5:]} {x['tmin']:.0f}-{x['tmax']:.0f} °C, rain {x['rain_mm']:.0f} mm ({x['rain_prob']}%), "
            f"wind {x['wind_max']:.0f} km/h, RH {x['rh_mean']}%" for x in wx["forecast"]))
        notes = weather_notes(wx)
        if notes:
            lines.append("Weather notes: " + " ".join(notes))
    if sl:
        s = "Soil estimate (ISRIC SoilGrids model for 0-15 cm, not a lab test): "
        parts = []
        if "ph" in sl:
            parts.append(f"pH {sl['ph']} ({sl['ph_class']})")
        if "texture" in sl:
            parts.append(f"{sl['texture']} (clay {sl['clay_pct']:.0f}%, sand {sl['sand_pct']:.0f}%)")
        if "organic_carbon_g_kg" in sl:
            parts.append(f"organic carbon {sl['organic_carbon_g_kg']} g/kg")
        if "total_nitrogen_g_kg" in sl:
            parts.append(f"total nitrogen {sl['total_nitrogen_g_kg']} g/kg")
        if "cec_cmol_kg" in sl:
            parts.append(f"CEC {sl['cec_cmol_kg']} cmol/kg")
        lines.append(s + ", ".join(parts))
    if prices:
        lines.append("Mandi prices (Agmarknet via data.gov.in, Rs/quintal): " + "; ".join(
            f"{p.get('commodity')} ({p.get('variety')}) at {p.get('market')}: modal {p.get('modal_price')} "
            f"(min {p.get('min_price')}, max {p.get('max_price')}) on {p.get('arrival_date')}" for p in prices))
    if docs:
        lines.append("Regional advisories and guides:")
        lines += [f"[{d['id']}] {d['title']}, page {d['page']}"
                  + (f" ({d['scope']}-level)" if d.get("scope") else "") + f": \"{d['text']}\"" for d in docs]
    if not lines:
        return "", [], pl or {}
    log.info("REGION place=%s elevation=%s weather=%s soil=%s prices=%d docs=%s", bool(pl), elev, bool(wx),
             bool(sl), len(prices),
             [d["id"] for d in docs])
    refs = [{"id": d["id"], "title": d["title"], "flag": "Regional advisory"}
            for d in {d["id"]: d for d in docs}.values()]
    return "REGION CONTEXT\n" + "\n".join(lines), refs, pl or {}