"""Web research for topics missing from the Crop Master database.

Step 1: a search API (Tavily, Brave, Serper or Google Programmable Search) finds pages.
        This uses no OpenAI tokens.
Step 2: one small gpt-4o-mini call turns the search results into a structured record.
The record is saved in the learned store, so the same topic is never searched twice.

Choose the provider in .env:
  SEARCH_PROVIDER=tavily | brave | serper | google
  SEARCH_API_KEY=...            (the provider's key)
  GOOGLE_CSE_ID=...             (only for google)
"""
from __future__ import annotations

import json
import logging
import os

import httpx

from .config import settings
from .learned import store

log = logging.getLogger("agronomist")

# Searched first, and ranked first in results.
TRUSTED_DOMAINS = [
    "icar.org.in", "icar.gov.in", "agritech.tnau.ac.in", "tnau.ac.in", "ppqs.gov.in", "vikaspedia.in",
    "iari.res.in", "rkmp.co.in", "agricoop.nic.in", "agriwelfare.gov.in", "irri.org",
    "plantwiseplusknowledgebank.org", "cabidigitallibrary.org",
]
MAX_RESULTS = 6
MAX_CHARS_PER_RESULT = 1200   # keeps the extraction call small (about 2,000 tokens in total)

EXTRACT_PROMPT = """Convert the search results into JSON about the crop problem named. Use only facts stated in the results; use null for anything missing. Never invent a dose.
Return:
{"kind": "disease|pest|disorder|practice",
 "name": "standard English name",
 "aliases": ["other English names", "Hindi name in Devanagari if given"],
 "crop": "main crop, or comma-separated crops",
 "scientific_name": null, "vector": null,
 "symptoms_early": null, "symptoms_late": null, "look_alikes": null,
 "favourable_conditions": null, "etl": null,
 "cultural_control": null, "bio_control": null,
 "chemical_control": [{"active_ingredient": "", "formulation": "", "dose": "", "method": "", "timing": "", "phi_days": null, "source_url": ""}],
 "notes": null,
 "confidence": "high|medium|low"}
Include a chemical only if a result states its dose; set source_url to that result's URL. Prefer facts from Indian official sources (ICAR, state agricultural universities, TNAU, Directorate of Plant Protection).
When a result covers several diseases or pests, use only the part that is explicitly about the named topic; never combine facts from sections about other problems.
The record must include the defining symptom of the named problem; if the results do not describe it, set confidence to "low"."""


# ---------- search providers: each returns [{"url", "title", "content"}] ----------
async def _tavily(http: httpx.AsyncClient, key: str, query: str, domains: list[str] | None):
    body = {"api_key": key, "query": query, "search_depth": "basic", "max_results": MAX_RESULTS}
    if domains:
        body["include_domains"] = domains
    r = await http.post("https://api.tavily.com/search", json=body, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return [{"url": x.get("url", ""), "title": x.get("title", ""), "content": x.get("content", "")}
            for x in r.json().get("results", [])]


async def _brave(http: httpx.AsyncClient, key: str, query: str, domains: list[str] | None):
    r = await http.get("https://api.search.brave.com/res/v1/web/search",
                       params={"q": query, "count": MAX_RESULTS + 4, "country": "IN"},
                       headers={"X-Subscription-Token": key, "Accept": "application/json"})
    r.raise_for_status()
    return [{"url": x.get("url", ""), "title": x.get("title", ""),
             "content": " ".join([x.get("description", "")] + (x.get("extra_snippets") or []))}
            for x in r.json().get("web", {}).get("results", [])]


async def _serper(http: httpx.AsyncClient, key: str, query: str, domains: list[str] | None):
    r = await http.post("https://google.serper.dev/search", json={"q": query, "gl": "in", "num": MAX_RESULTS + 4},
                        headers={"X-API-KEY": key})
    r.raise_for_status()
    return [{"url": x.get("link", ""), "title": x.get("title", ""), "content": x.get("snippet", "")}
            for x in r.json().get("organic", [])]


async def _google(http: httpx.AsyncClient, key: str, query: str, domains: list[str] | None):
    r = await http.get("https://www.googleapis.com/customsearch/v1",
                       params={"key": key, "cx": os.getenv("GOOGLE_CSE_ID", ""), "q": query, "num": 10, "gl": "in"})
    r.raise_for_status()
    return [{"url": x.get("link", ""), "title": x.get("title", ""), "content": x.get("snippet", "")}
            for x in r.json().get("items", [])]


PROVIDERS = {"tavily": _tavily, "brave": _brave, "serper": _serper, "google": _google}


def _trusted(url: str) -> bool:
    return any(d in url for d in TRUSTED_DOMAINS)


async def search_web(query: str) -> list[dict]:
    """Search with the configured API. Trusted sources first; falls back to an open search if none found."""
    name = os.getenv("SEARCH_PROVIDER", "tavily").lower()
    key = os.getenv("SEARCH_API_KEY", "")
    fn = PROVIDERS.get(name)
    if not fn or not key:
        log.warning("Web search not configured: set SEARCH_PROVIDER (%s) and SEARCH_API_KEY in .env",
                    "/".join(PROVIDERS))
        return []
    async with httpx.AsyncClient(timeout=20) as http:
        results = await fn(http, key, query, TRUSTED_DOMAINS)
        if name == "tavily" and not results:  # nothing on trusted sites: one open search
            results = await fn(http, key, query, None)
    seen, out = set(), []
    for r in sorted(results, key=lambda x: not _trusted(x["url"])):  # trusted sources first
        if r["url"] and r["url"] not in seen and r["content"].strip():
            seen.add(r["url"])
            out.append(r)
    return out[:MAX_RESULTS]


async def research(client, topic: str, crop: str = "") -> dict | None:
    """Return a learned record for topic/crop, searching the web only if it is not stored yet."""
    cached = store.find(topic, crop)
    if cached:
        store.hit(cached["id"])
        log.info("LEARNED hit %s (%s)", cached["id"], cached["name"])
        return cached

    crop_in_topic = crop and crop.lower() in topic.lower()
    query = " ".join(x for x in (topic, "" if crop_in_topic else crop, "symptoms management India") if x)
    results = await search_web(query)
    trusted = sum(_trusted(r["url"]) for r in results)
    log.info("SEARCH %s query=%r results=%d trusted=%d", os.getenv("SEARCH_PROVIDER", "tavily"),
             query, len(results), trusted)
    if not results:
        return None  # nothing found; don't store

    notes = "\n\n".join(f"[{i + 1}] {r['title']}\nURL: {r['url']}\n{r['content'][:MAX_CHARS_PER_RESULT]}"
                        for i, r in enumerate(results))
    crop_part = f" in {crop}" if crop else ""
    ext = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "system", "content": EXTRACT_PROMPT},
                  {"role": "user", "content": f"Topic: {topic}{crop_part}\n\nSearch results:\n{notes}"}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=900,
    )
    u = getattr(ext, "usage", None)
    log.info("EXTRACT %s in=%s out=%s", settings.openai_model,
             getattr(u, "prompt_tokens", "?"), getattr(u, "completion_tokens", "?"))
    try:
        data = json.loads(ext.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return None

    if data.get("confidence") == "low":
        log.info("EXTRACT low confidence for %r; not saved", topic)
        return None
    urls = {r["url"] for r in results}
    # keep a chemical only if it has a dose and its source is one of the pages actually found
    data["chemical_control"] = [c for c in (data.get("chemical_control") or [])
                                if isinstance(c, dict) and c.get("dose") and c.get("source_url") in urls]
    name = data.pop("name", None) or topic
    kind = data.pop("kind", None) or "disease"
    aliases = [a for a in (data.pop("aliases", None) or []) if a] + ([topic] if topic.lower() != name.lower() else [])
    rec_crop = data.pop("crop", None) or crop
    sources = [{"url": r["url"], "title": r["title"]} for r in results]
    return store.add(kind, name, rec_crop, aliases, data, sources)