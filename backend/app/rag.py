"""PDF knowledge for the agronomist (RAG): text chunks of government PDFs, embedded with OpenAI and stored in
Qdrant, tagged by state, district, crops and expiry date.

SQLite (region.db) stays the record of what was imported (DOC ids, titles, chunk text); Qdrant is the search
index. If Qdrant is lost or the embedding model changes, rebuild it with:  python -m app.import_doc --reindex

.env:
  QDRANT_URL=http://localhost:6333        (a Qdrant server; or use QDRANT_PATH instead)
  QDRANT_PATH=../data/qdrant_local        (local mode: no server needed, data in this folder; only one
                                           program can use it at a time, so stop the server while importing)
  QDRANT_API_KEY=...                      (only for Qdrant Cloud or a secured server)
  QDRANT_PREFIX=qdrant                    (only if your server puts Qdrant under a path, e.g. https://host/qdrant)
  QDRANT_COLLECTION=crop_master_docs
  EMBED_MODEL=text-embedding-3-small      (1536 dims; text-embedding-3-large = 3072)
  RAG_MIN_SCORE=0.30                      (similarity below this is ignored)
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import date

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

from .config import settings

log = logging.getLogger("agronomist")

COLLECTION = os.getenv("QDRANT_COLLECTION", "crop_master_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DIM = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}.get(
    EMBED_MODEL, int(os.getenv("EMBED_DIM", "1536")))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))
ALL = "_all"  # payload value meaning "every state" / "every district"

_qdrant: AsyncQdrantClient | None = None
_oai: AsyncOpenAI | None = None
_ready = False


def enabled() -> bool:
    return bool(os.getenv("QDRANT_URL") or os.getenv("QDRANT_PATH"))


def describe() -> str:
    if os.getenv("QDRANT_PATH") and not os.getenv("QDRANT_URL"):
        return f"local folder {os.getenv('QDRANT_PATH')}"
    return os.getenv("QDRANT_URL", "")


def qdrant() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        url = os.getenv("QDRANT_URL", "")
        path = os.getenv("QDRANT_PATH", "")
        if path and not url:  # local mode: no server, data kept in this folder (one process at a time)
            _qdrant = AsyncQdrantClient(path=path)
            return _qdrant
        _qdrant = (AsyncQdrantClient(location=":memory:") if url == ":memory:"
                   else AsyncQdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30,
                                          prefix=os.getenv("QDRANT_PREFIX", "").strip("/") or None))
    return _qdrant


def oai() -> AsyncOpenAI:
    global _oai
    if _oai is None:
        _oai = AsyncOpenAI(api_key=settings.openai_api_key)
    return _oai


def _tag(s: str) -> str:
    return (s or "").strip().lower() or ALL


async def ensure_collection() -> None:
    global _ready
    if _ready:
        return
    c = qdrant()
    if not await c.collection_exists(COLLECTION):
        await c.create_collection(COLLECTION, vectors_config=models.VectorParams(size=DIM,
                                                                                 distance=models.Distance.COSINE))
        for field in ("doc_id", "state", "district", "crops", "doc_type"):
            await c.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
        await c.create_payload_index(COLLECTION, "valid_until", models.PayloadSchemaType.INTEGER)
        log.info("RAG created Qdrant collection %s (%d dims)", COLLECTION, DIM)
    _ready = True


async def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """Embeddings in batches; returns (vectors, tokens used)."""
    vectors, tokens = [], 0
    for i in range(0, len(texts), 96):
        r = await oai().embeddings.create(model=EMBED_MODEL, input=texts[i:i + 96])
        vectors += [d.embedding for d in r.data]
        tokens += getattr(getattr(r, "usage", None), "total_tokens", 0) or 0
    return vectors, tokens


async def index_document(doc: dict, chunks: list[tuple[int, str]]) -> int:
    """Embed and store a document's chunks. doc: id, title, state, district, crops, valid_until, doc_type."""
    if not chunks:
        return 0
    await ensure_collection()
    vectors, tokens = await embed([f"{doc['title']}\n{text}" for _, text in chunks])
    valid = int(str(doc.get("valid_until") or "9999-12-31").replace("-", "")[:8])
    crops = [c.strip().lower() for c in (doc.get("crops") or "").split(",") if c.strip()]
    points = [models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['id']}:{i}")), vector=v,
        payload={"doc_id": doc["id"], "title": doc["title"], "state": _tag(doc.get("state")),
                 "district": _tag(doc.get("district")), "crops": crops, "doc_type": doc.get("doc_type") or "",
                 "valid_until": valid, "page": page, "text": text})
        for i, ((page, text), v) in enumerate(zip(chunks, vectors))]
    for i in range(0, len(points), 128):
        await qdrant().upsert(COLLECTION, points=points[i:i + 128])
    log.info("RAG indexed %s: %d chunks, %d embedding tokens", doc["id"], len(points), tokens)
    return len(points)


async def delete_document(doc_id: str) -> None:
    await ensure_collection()
    await qdrant().delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))


def _region_filter(state: str, district: str) -> models.Filter:
    """This district's documents, this state's state-wide ones, and national ones; never expired ones."""
    st, di = _tag(state), _tag(district)
    eq = lambda k, v: models.FieldCondition(key=k, match=models.MatchValue(value=v))
    should = [eq("state", ALL)]
    if st != ALL:
        should.append(models.Filter(must=[eq("state", st), eq("district", ALL)]))
        if di != ALL:
            should.append(models.Filter(must=[eq("state", st), eq("district", di)]))
    today = int(date.today().strftime("%Y%m%d"))
    return models.Filter(must=[models.FieldCondition(key="valid_until", range=models.Range(gte=today))],
                         should=should)


async def search(state: str, district: str, text: str, limit: int = 4) -> list[dict]:
    """Best-matching excerpts for the farmer's region: at most 2 per document, one per page."""
    await ensure_collection()
    (vec,), _ = await embed([text[:2000]])
    res = await qdrant().query_points(COLLECTION, query=vec, query_filter=_region_filter(state, district),
                                      limit=limit * 4, with_payload=True)
    bonus = {"district": 0.10, "state": 0.05}  # local sources first when relevance is close
    scope = lambda pl: "district" if pl.get("district") != ALL else "state" if pl.get("state") != ALL else "national"
    points = sorted(res.points, key=lambda p: -(p.score + bonus.get(scope(p.payload or {}), 0)))
    out, per_doc, pages = [], {}, set()
    for p in points:
        pl = p.payload or {}
        if p.score < MIN_SCORE or (pl.get("doc_id"), pl.get("page")) in pages or \
                per_doc.get(pl.get("doc_id"), 0) >= max(2, limit // 2):
            continue
        pages.add((pl.get("doc_id"), pl.get("page")))
        per_doc[pl.get("doc_id")] = per_doc.get(pl.get("doc_id"), 0) + 1
        out.append({"id": pl.get("doc_id"), "title": pl.get("title"), "page": pl.get("page"),
                    "text": pl.get("text"), "score": round(p.score, 3), "scope": scope(pl)})
        if len(out) >= limit:
            break
    rank = {"district": 0, "state": 1, "national": 2}
    return sorted(out, key=lambda d: rank[d["scope"]])  # most local first