"""Crop Master knowledge base: loads the Bharat Agrolink workbook, finds the rows
relevant to a farmer's question, and exposes lookup tools for the LLM."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DEVA = re.compile(r"[\u0900-\u097F]")
PUNCT = re.compile(r"[()\[\],.;:!?/\\\"'’‘“”\-–—+&|]")
REF_RE = re.compile(r"\b(CRP\d{3}|PST\d{3}|DIS\d{3}|CPM\d{3}|CDM\d{3}|CAT\d{2}|SEA\d{2}|LRN\d{3}|DOC\d{3})\b", re.I)
EMPTY = {"", "nan", "none", "—", "-"}
STOP = set(
    "what which when where with that this have from your about crop crops plant plants leaf leaves "
    "field farm should could would there their they them give tell help need best good much many does "
    "than into also over under after before during more less very some sowing spray".split()
)
INDIA_REGIONS = {
    "Jammu & Kashmir": ["jammu", "kashmir", "j&k", "srinagar"], "Ladakh": ["ladakh", "leh"],
    "Nagaland": ["nagaland", "kohima", "dimapur"], "Manipur": ["manipur", "imphal"],
    "Mizoram": ["mizoram", "aizawl"], "Meghalaya": ["meghalaya", "shillong"],
    "Tripura": ["tripura", "agartala"], "Arunachal Pradesh": ["arunachal", "itanagar"],
    "Sikkim": ["sikkim", "gangtok"], "Goa": ["goa"], "Delhi": ["delhi"],
    "Puducherry": ["puducherry", "pondicherry"], "Andaman & Nicobar": ["andaman", "nicobar"],
}
DROP = {"data_source", "source", "export_potential", "major_producing_districts", "local_names", "scientific_name"}
FULL_ROWS = 3


def _line(r):
    return "; ".join(f"{k}={v}" for k, v in r.items()
                     if k not in DROP and not (k.endswith("_hi") and "name" not in k))


def norm(s) -> str:
    s = PUNCT.sub(" ", str(s or "").lower())
    return " " + " ".join(s.split()) + " "


def base(s) -> str:
    return str(s or "").split("(")[0].strip()


def hit(qn: str, name: str) -> bool:
    if f" {name} " in qn:
        return True
    if DEVA.search(name):
        return name in qn
    return f" {name}s " in qn or f" {name}es " in qn


def _rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        d = {k: str(v).strip() for k, v in r.items() if isinstance(v, str) and str(v).strip().lower() not in EMPTY}
        if d:
            out.append(d)
    return out


def _names(rows: list[dict], fields: list[str]) -> list[tuple[dict, list[str]]]:
    out = []
    for r in rows:
        names: set[str] = set()
        for f in fields:
            if not r.get(f):
                continue
            vals = re.split(r"[,/]", r[f]) if f == "local_names" else [r[f], base(r[f])]
            for v in vals:
                n = norm(v).strip()
                if n and len(n) >= (2 if DEVA.search(n) else 3):
                    names.add(n)
        if r.get("scientific_name"):
            names.add(norm(" ".join(r["scientific_name"].split()[:2])).strip())
        out.append((r, sorted(names)))
    return out


class Knowledge:
    def __init__(self, path: str | Path):
        x = pd.read_excel(path, sheet_name=None, dtype=str)
        self.crops = _rows(x["Crop_Master"])
        self.pests = _rows(x["Pest_Master"])
        self.diseases = _rows(x["Disease_Master"])
        self.cpm = _rows(x["Crop_Pest_Map"])
        self.cdm = _rows(x["Crop_Disease_Map"])
        self.states = _rows(x["State_Crop_Map"])
        self.seasons = _rows(x["Season_Master"])
        self.cats = _rows(x["Category_Master"])
        lgd = x.get("LGD_Block_Master")
        self.districts: dict[str, list[str]] = (
            {s: sorted(g["district_name"].dropna().unique().tolist()) for s, g in lgd.groupby("state")}
            if lgd is not None else {}
        )

        # official tehsil / block / district names (LGD), used to correct typed place names
        self.lgd_places: list[tuple[str, str, str]] = []
        for sheet, col in (("LGD_SubDistrict_Master", "subdistrict_name_en"), ("LGD_Block_Master", "block_name_en")):
            df = x.get(sheet)
            if df is not None and col in df:
                for name, dist, st in df[[col, "district_name", "state"]].dropna().itertuples(index=False):
                    self.lgd_places.append((name.strip(), dist.strip(), st.strip()))
                    self.lgd_places.append((dist.strip(), dist.strip(), st.strip()))
        self.lgd_places = list(dict.fromkeys(self.lgd_places))

        self.by_id: dict[str, dict] = {}
        for rows, key in [(self.crops, "crop_id"), (self.pests, "pest_id"), (self.diseases, "disease_id"),
                          (self.cats, "cat_id"), (self.seasons, "season_id"), (self.cpm, "map_id"), (self.cdm, "map_id")]:
            for r in rows:
                if r.get(key):
                    self.by_id[r[key].upper()] = r

        self.crop_idx = _names(self.crops, ["crop_name_en", "crop_name_hi", "local_names"])
        self.pest_idx = _names(self.pests, ["pest_name_en", "pest_name_hi", "local_names"])
        self.dis_idx = _names(self.diseases, ["disease_name_en", "disease_name_hi"])


    # ---------- metadata for the UI ----------
    def meta(self) -> dict:
        verified = sum(1 for c in self.crops if c.get("accuracy_flag") == "Verified")
        return {
            "states": [{"state": s["state"], "code": s.get("state_code", "")} for s in self.states],
            "seasons": [{"en": s["season_en"], "hi": s.get("season_hi", "")} for s in self.seasons],
            "districts": self.districts,
            "crop_names": sorted({base(c["crop_name_en"]) for c in self.crops}),
            "stats": {"crops": len(self.crops), "pests": len(self.pests), "diseases": len(self.diseases),
                      "protocols": len(self.cpm) + len(self.cdm), "verified": verified},
        }

    def title(self, rid: str) -> str:
        r = self.by_id.get(rid.upper())
        if not r:
            return rid
        if r.get("map_id"):
            return f'{r.get("crop_name_en", "")} × {r.get("pest_name_en") or r.get("disease_name_en", "")}'
        return (r.get("crop_name_en") or r.get("pest_name_en") or r.get("disease_name_en")
                or r.get("season_en") or r.get("subcategory") or rid)

    def ref(self, rid: str) -> dict | None:
        r = self.by_id.get(rid.upper())
        if not r:
            return None
        return {"id": rid.upper(), "title": self.title(rid), "flag": r.get("accuracy_flag", "")}

    def record(self, rid: str) -> dict | None:
        r = self.by_id.get(rid.upper())
        if not r:
            return None
        rec = dict(r)
        if rec.get("season_id") in self.by_id and rid.upper().startswith("CRP"):
            rec["season_id"] = f'{self.by_id[rec["season_id"]]["season_en"]} ({rec["season_id"]})'
        if rec.get("cat_id") in self.by_id and rid.upper().startswith("CRP"):
            c = self.by_id[rec["cat_id"]]
            rec["cat_id"] = f'{c.get("category_en")} / {c.get("subcategory")} ({rec["cat_id"]})'
        return {"id": rid.upper(), "title": self.title(rid), "flag": r.get("accuracy_flag", ""), "record": rec}

    # ---------- retrieval ----------
    def retrieve(self, question: str, profile: dict) -> dict:
        qn = norm(f'{question} {profile.get("crop") or ""}')
        q_only = norm(question)

        crops = [r for r, names in self.crop_idx if any(hit(qn, n) for n in names)]
        bases = sorted({base(c["crop_name_en"]).lower() for c in crops})

        def crop_key(c):
            generic = 1 if "(" in c["crop_name_en"] else 0
            asked = 0 if hit(q_only, norm(base(c["crop_name_en"])).strip()) else 1
            unverified = 0 if c.get("accuracy_flag") == "Verified" else 1
            return (generic, asked, unverified)

        crops.sort(key=crop_key)
        variants = [f'{c["crop_id"]} {c["crop_name_en"]}' for c in crops[8:38]]
        crops = crops[:8]
        crop_ids = {c["crop_id"] for c in crops}

        tokens = [t for t in q_only.split() if len(t) >= 4 and t not in STOP]

        def related(r):
            hay = (r.get("primary_crops", "") + " " + r.get("secondary_crops", "")).lower()
            return any(b in hay for b in bases)

        def pick(idx, name_field, cap):
            scored = []
            for r, names in idx:
                direct = any(hit(q_only, n) for n in names)
                rel = related(r)
                if bases and not (direct or rel):
                    continue
                s = 10 if direct else 0
                nm = (r.get(name_field, "") + " " + r.get("local_names", "")).lower()
                s += sum(3 for t in tokens if t in nm)
                symp = " ".join(r.get(k, "") for k in ("symptom_early", "symptom_late", "identification_signs",
                                                       "damage_description_en")).lower()
                s += sum(1 for t in tokens if t in symp)
                if rel:
                    s += 4
                if s >= 3:
                    scored.append((s, r))
            scored.sort(key=lambda x: -x[0])
            return [r for _, r in scored[:cap]]

        pests = pick(self.pest_idx, "pest_name_en", 5)
        diseases = pick(self.dis_idx, "disease_name_en", 5)
        p_ids = {p["pest_id"] for p in pests}
        d_ids = {d["disease_id"] for d in diseases}

        def in_base(name):
            return any(b in (name or "").lower() for b in bases)

        cpm = [m for m in self.cpm if m.get("crop_id") in crop_ids or m.get("pest_id") in p_ids or in_base(m.get("crop_name_en"))][:10]
        cdm = [m for m in self.cdm if m.get("crop_id") in crop_ids or m.get("disease_id") in d_ids or in_base(m.get("crop_name_en"))][:10]
        for m in cpm:
            pid = m.get("pest_id")
            if pid and pid not in p_ids and pid in self.by_id and len(pests) < 8:
                pests.append(self.by_id[pid]); p_ids.add(pid)
        for m in cdm:
            did = m.get("disease_id")
            if did and did not in d_ids and did in self.by_id and len(diseases) < 8:
                diseases.append(self.by_id[did]); d_ids.add(did)

        q_raw = f" {question} "
        states = [s for s in self.states if s["state"] == profile.get("state")
                  or f' {s["state"].lower()} ' in qn
                  or (s.get("state_code") and re.search(rf'\b{s["state_code"]}\b', q_raw))]
        seasons = [s for s in self.seasons if s["season_en"] == profile.get("season")
                   or hit(qn, s["season_en"].lower()) or (s.get("season_hi") and s["season_hi"] in qn)]
        cats = [self.by_id[c] for c in dict.fromkeys(c.get("cat_id") for c in crops) if c in self.by_id]
        q_low = question.lower()
        known = {s["state"] for s in self.states}
        uncovered = [r for r, keys in INDIA_REGIONS.items()
                     if r not in known and any(k in q_low for k in keys)]

        return dict(crops=crops, variants=variants, pests=pests, diseases=diseases,
                    cpm=cpm, cdm=cdm, states=states, seasons=seasons, cats=cats, uncovered=uncovered)

    @staticmethod
    def context_block(ctx: dict) -> str:
        parts = []
        if ctx.get("uncovered"):
            parts.append("REGIONS WITH NO DATABASE ROWS: " + ", ".join(ctx["uncovered"])
                         + ". Answer for these from general agronomy knowledge and label it as such.")
        for label, key in [("CROPS", "crops"), ("CROP-PEST PROTOCOLS", "cpm"), ("CROP-DISEASE PROTOCOLS", "cdm"),
                           ("STATE", "states"), ("SEASON", "seasons")]:
            if ctx.get(key):
                parts.append(label + ":\n" + "\n".join(_line(r) for r in ctx[key]))
        for label, key, idf, namef in [("PESTS", "pests", "pest_id", "pest_name_en"),
                                       ("DISEASES", "diseases", "disease_id", "disease_name_en")]:
            rows = ctx.get(key) or []
            if rows:
                txt = label + ":\n" + "\n".join(_line(r) for r in rows[:FULL_ROWS])
                if rows[FULL_ROWS:]:
                    txt += "\nAlso related (use get_records for details): " + ", ".join(
                        f"{r[idf]} {r[namef]}" for r in rows[FULL_ROWS:])
                parts.append(txt)
        if ctx.get("variants"):
            parts.append("Other variety rows (get_records for details): " + "; ".join(ctx["variants"]))
        return "\n\n".join(parts) or "No rows matched directly. Use the search tools if needed."

    def context_ids(self, ctx: dict) -> list[str]:
        ids = [c["crop_id"] for c in ctx["crops"]] + [m["map_id"] for m in ctx["cpm"]] + [m["map_id"] for m in ctx["cdm"]]
        ids += [p["pest_id"] for p in ctx["pests"]] + [d["disease_id"] for d in ctx["diseases"]]
        return ids

    # ---------- tools callable by the LLM ----------
    def _crop_row(self, c):
        return {"id": c["crop_id"], "name": c["crop_name_en"], "hi": c.get("crop_name_hi"),
                "season": self.by_id.get(c.get("season_id", ""), {}).get("season_en", c.get("season_id")),
                "days": c.get("duration_days"), "states": c.get("primary_states"), "soil": c.get("suitable_soil"),
                "water_mm": c.get("water_need_mm"), "yield_qtl_ha": c.get("avg_yield_qtl_ha"),
                "msp": c.get("msp_2024_rs_qtl"), "tier": c.get("market_value_tier"), "flag": c.get("accuracy_flag")}

    def search_crops(self, text="", state_code="", season="", soil="", max_water_mm=0, limit=15):
        text, state_code, soil = str(text or "").lower().strip(), str(state_code or "").upper().strip(), str(soil or "").lower()
        limit = max(1, min(int(limit or 15), 25))
        season_id = next((s["season_id"] for s in self.seasons if s["season_en"].lower() == str(season or "").lower()), None)
        try:
            max_water = float(max_water_mm or 0)
        except (TypeError, ValueError):
            max_water = 0
        out = []
        for c in self.crops:
            if text:
                cat = self.by_id.get(c.get("cat_id", ""), {})
                hay = " ".join([c.get(k, "") for k in ("crop_name_en", "crop_name_hi", "local_names", "subcategory",
                                                       "suitable_soil", "notes", "scientific_name")] + [cat.get("category_en", "")]).lower()
                if not all(w in hay for w in text.split()):
                    continue
            if state_code and state_code not in [s.strip() for s in c.get("primary_states", "").split(",")]:
                continue
            if season_id and c.get("season_id") != season_id:
                continue
            if soil and soil not in c.get("suitable_soil", "").lower():
                continue
            if max_water:
                m = re.match(r"\s*(\d+)", c.get("water_need_mm", ""))
                if not m or float(m.group(1)) > max_water:
                    continue
            out.append(c)
        return {"total_matches": len(out), "rows": [self._crop_row(c) for c in out[:limit]]}

    def get_records(self, ids, used: set):
        ids = ids if isinstance(ids, list) else [ids]
        res = {}
        for i in [str(x).upper() for x in ids][:6]:
            r = self.by_id.get(i)
            if not r:
                res[i] = "not found"
                continue
            used.add(i)
            res[i] = {k: v for k, v in r.items() if k not in ("data_source", "source")}
            if i.startswith("CRP"):
                res[i + "_pest_map"] = [m for m in self.cpm if m.get("crop_id") == i]
                res[i + "_disease_map"] = [m for m in self.cdm if m.get("crop_id") == i]
        return res

    def search_pests_diseases(self, text=""):
        ws = [w for w in str(text or "").lower().split() if len(w) >= 3]

        def score(hay):
            return sum(1 for w in ws if w in hay)

        p = sorted(((score(" ".join(r.get(k, "") for k in ("pest_name_en", "pest_name_hi", "local_names", "primary_crops",
                    "secondary_crops", "damage_description_en", "identification_signs", "damage_type")).lower()), r)
                    for r in self.pests), key=lambda x: -x[0])
        d = sorted(((score(" ".join(r.get(k, "") for k in ("disease_name_en", "disease_name_hi", "causal_organism",
                    "primary_crops", "secondary_crops", "symptom_early", "symptom_late", "affected_part")).lower()), r)
                    for r in self.diseases), key=lambda x: -x[0])
        return {
            "pests": [{"id": r["pest_id"], "name": r["pest_name_en"], "crops": r.get("primary_crops"),
                       "signs": r.get("identification_signs") or r.get("damage_description_en"), "flag": r.get("accuracy_flag")}
                      for s, r in p[:6] if s],
            "diseases": [{"id": r["disease_id"], "name": r["disease_name_en"], "crops": r.get("primary_crops"),
                          "early": r.get("symptom_early"), "late": r.get("symptom_late"), "flag": r.get("accuracy_flag")}
                         for s, r in d[:6] if s],
        }

    def run_tool(self, name: str, args: dict, used: set):
        if name == "search_crops":
            return self.search_crops(**{k: args.get(k) for k in ("text", "state_code", "season", "soil", "max_water_mm", "limit") if k in args})
        if name == "get_records":
            return self.get_records(args.get("ids", []), used)
        if name == "search_pests_diseases":
            return self.search_pests_diseases(args.get("text", ""))
        return {"error": f"unknown tool {name}"}

    def find_named(self, name: str, crop: str = "") -> str | None:
        """Return the PST/DIS id for a free-text pest/disease name, or None.
        When a crop is given, only a row for that crop counts, so another crop's row is never used in its place."""
        key = re.sub(r"[^a-z]", "", (name or "").lower())
        if len(key) < 5:
            return None
        rows = self.diseases + self.pests
        crop_words = self._crop_words()
        # pass 1: common names (ignoring parts that are just a crop name, e.g. "Chilli")
        hits = []
        for r in rows:
            nm = r.get("disease_name_en") or r.get("pest_name_en", "")
            for part in re.split(r"[/(),]", nm):
                k = re.sub(r"[^a-z]", "", part.lower())
                if len(k) >= 5 and k not in crop_words and (k in key or key in k):
                    hits.append(r)
                    break
        # pass 2: scientific names ("Genus species"); the genus alone only if just one row has it
        if not hits:
            for r in rows:
                sci = self._binomial(r)
                if sci and (sci[0] + sci[1] in key or (sci[0] in key and self._genus_count().get(sci[0]) == 1)):
                    hits.append(r)
        if crop.strip():
            hits = [r for r in hits if self._row_has_crop(r, crop)]
        if hits:
            return hits[0].get("disease_id") or hits[0].get("pest_id")
        return None

    def _crop_words(self) -> set[str]:
        if not hasattr(self, "_cw"):
            self._cw = {re.sub(r"[^a-z]", "", base(c["crop_name_en"]).lower()) for c in self.crops}
        return self._cw

    def _row_has_crop(self, r: dict, crop: str) -> bool:
        """True if the pest/disease row covers this crop (matching names like soya/soybean, rice/paddy)."""
        qn = norm(crop)
        bases = {base(c["crop_name_en"]).lower() for c, names in self.crop_idx if any(hit(qn, n) for n in names)}
        bases |= {crop.strip().lower()}
        hay = " ".join([r.get("primary_crops", ""), r.get("secondary_crops", ""),
                        r.get("disease_name_en") or r.get("pest_name_en", "")]).lower()
        return any(len(b) >= 3 and b in hay for b in bases)

    @staticmethod
    def _binomial(r: dict) -> tuple[str, str] | None:
        """Latin name from causal_organism / scientific_name, or None if the field is not a Latin binomial."""
        m = re.match(r"\s*([A-Z][a-z]{4,})\s+([a-z]{3,})\b", r.get("causal_organism") or r.get("scientific_name") or "")
        return (m.group(1).lower(), m.group(2)) if m else None

    def _genus_count(self) -> dict:
        if not hasattr(self, "_genera"):
            self._genera: dict[str, int] = {}
            for r in self.diseases + self.pests:
                b = self._binomial(r)
                if b:
                    self._genera[b[0]] = self._genera.get(b[0], 0) + 1
        return self._genera

    def crop_reference(self, text: str, limit: int = 15) -> str:
        """Known pests/diseases of the crop named in `text`, with typical signs (a reference for the vision step)."""
        qn = norm(text)
        bases = {base(r["crop_name_en"]).lower() for r, names in self.crop_idx if any(hit(qn, n) for n in names)}
        if not bases:
            return ""

        def for_crop(r):
            hay = (r.get("primary_crops", "") + " " + r.get("secondary_crops", "")).lower()
            return any(b in hay for b in bases)

        lines = [f'{r["disease_name_en"]} (disease): {r.get("symptom_early", "")} {r.get("symptom_late", "")}'.strip()
                 for r in self.diseases if for_crop(r)]
        lines += [f'{r["pest_name_en"]} (pest): {r.get("identification_signs") or r.get("damage_description_en", "")}'.strip()
                  for r in self.pests if for_crop(r)]
        return "\n".join(lines[:limit])

    def parse_refs(self, text: str) -> list[str]:
        m = re.search(r"REFS:\s*([^\n]*)\s*$", text, re.I)
        return list(dict.fromkeys(x.upper() for x in REF_RE.findall(m.group(1)))) if m else []


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_crops",
        "description": "Search the 500-row Crop_Master. Filters are optional and combine with AND. Returns compact rows "
                       "(id, name, season, duration, states, soil, water, yield, MSP, market tier, accuracy flag). "
                       "Use for recommendations such as which crops suit a state in a season.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Words to match in crop name, Hindi/local name, category, soil or notes"},
            "state_code": {"type": "string", "description": "Two-letter state code like MP, MH, RJ"},
            "season": {"type": "string", "enum": ["Kharif", "Rabi", "Zaid", "Perennial", "Kharif+Rabi"]},
            "soil": {"type": "string"},
            "max_water_mm": {"type": "number"},
            "limit": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "get_records",
        "description": "Fetch full database rows by id: CRP### (crop), PST### (pest), DIS### (disease), CAT##, SEA##, "
                       "CPM###/CDM### (protocols). Crop ids also return their pest and disease protocol rows. Max 6 ids.",
        "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                       "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "search_pests_diseases",
        "description": "Find pests and diseases by name, symptom words or crop. Returns id, name, crops, signs/symptoms "
                       "and accuracy flag. Use when diagnosing symptoms.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]