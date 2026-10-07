"""FastAPI app: JSON API for the Crop Master agronomist + serves the frontend."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

import csv
import io

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent import answer_stream, summarise_unsummarised
from .config import settings
from .knowledge import Knowledge
from .learned import store
from . import region
from .public_api import router as public_router
from .auth import get_current_user, request_otp, verify_otp
from .farmers import (
    create_conversation, get_conversation, get_messages,
    get_profile_for_user, init_db, list_conversations, save_message,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
kb: Knowledge | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb
    init_db()
    kb = Knowledge(settings.data_path)
    app.state.kb = kb  # shared with the /v1 public API
    logging.getLogger("agronomist").info("Loaded %d crops, %d pests, %d diseases from %s",
                                         len(kb.crops), len(kb.pests), len(kb.diseases), settings.data_path)
    yield


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"  # so Windows PowerShell shows Hindi correctly


app = FastAPI(title="Crop Master Agronomist API", version="1.0.0", lifespan=lifespan, default_response_class=UTF8JSONResponse)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["*"])


class Profile(BaseModel):
    state: str = ""
    district: str = ""
    crop: str = ""
    season: str = ""
    area: str = ""
    water: str = ""
    soil: str = ""
    lang: str = "auto"
    lat: str = ""   # from the browser's GPS or a typed place; never stored on the server
    lon: str = ""
    place: str = ""  # e.g. "Belkhedi, Narsinghpur, Madhya Pradesh"
    location_source: str = ""  # gps | typed | district

    @field_validator("*")
    @classmethod
    def trim(cls, v: str) -> str:
        return (v or "").strip()[:80]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    profile: Profile = Profile()
    history: list[Turn] = Field(default_factory=list, max_length=40)
    image: str | None = Field(default=None, description="Optional data URL (image/jpeg|png|webp)")
    images: list[str] = Field(default_factory=list, max_length=3, description="Up to 3 data URLs of the same problem")
    photo_repeat: bool = False
    conversation_id: int | None = None

    @field_validator("images")
    @classmethod
    def check_images(cls, v: list[str]) -> list[str]:
        return [cls.check_image(x) for x in v if x]

    @field_validator("image")
    @classmethod
    def check_image(cls, v: str | None) -> str | None:
        if not v:
            return None
        if not v.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")):
            raise ValueError("image must be a JPEG, PNG or WebP data URL")
        if len(v) * 3 / 4 > settings.max_image_mb * 1024 * 1024:
            raise ValueError(f"image larger than {settings.max_image_mb} MB")
        return v


@app.get("/api/health")
def health():
    return {"ok": True, "model": settings.openai_model, "key_configured": bool(settings.openai_api_key)}


@app.get("/api/meta")
def meta():
    return kb.meta()


@app.get("/api/record/{rid}")
def record(rid: str):
    if rid.upper().startswith("DOC"):
        d = region.get_document(rid)
        if not d:
            raise HTTPException(404, f"No record {rid}")
        return {"id": d["id"], "title": d["title"], "flag": "Regional advisory",
                "record": {k: v for k, v in d.items() if v and k not in ("id", "title")}}
    if rid.upper().startswith("LRN"):
        rec = store.get(rid)
        if not rec:
            raise HTTPException(404, f"No record {rid}")
        flat = {"name": rec["name"], "kind": rec["kind"], "crop": rec["crop"], "status": rec["status"]}
        for k, v in rec["data"].items():
            if v:
                flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        flat["sources"] = "\n".join(s.get("url", "") for s in rec["sources"])
        flat["researched_on"] = rec["created_at"]
        return {"id": rec["id"], "title": store.ref(rec)["title"], "flag": store.flag(rec), "record": flat}
    r = kb.record(rid)
    if not r:
        raise HTTPException(404, f"No record {rid}")
    return r


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=200)):
    """Plain database lookup without the LLM — useful for testing retrieval."""
    ctx = kb.retrieve(q, {})
    return {"refs": [kb.ref(i) for i in kb.context_ids(ctx)]}


@app.post("/api/auth/request-otp")
async def auth_request_otp(body: dict):
    return request_otp(body.get("mobile", ""))


@app.post("/api/auth/verify-otp")
async def auth_verify_otp(body: dict):
    result = verify_otp(body.get("mobile", ""), body.get("otp", ""))
    user = result["user"]
    try:
        await summarise_unsummarised(user["id"])
    except Exception:
        logging.getLogger("agronomist").exception("case-note summarisation failed on login")
    return {
        "token": result["token"],
        "user_id": user["id"],
        "mobile_last4": user["mobile_last4"],
        "is_existing_customer": bool(user["is_existing_customer"]),
        "profile": get_profile_for_user(user),
    }


@app.get("/api/auth/me")
async def auth_me(user=Depends(get_current_user(optional=False))):
    return {
        "user_id": user["id"],
        "is_existing_customer": bool(user["is_existing_customer"]),
        "profile": get_profile_for_user(user),
    }


@app.post("/api/auth/logout")
async def auth_logout():
    return {"ok": True}


@app.get("/api/conversations")
async def conversations(user=Depends(get_current_user(optional=False))):
    return list_conversations(user["id"])


@app.get("/api/conversations/{conversation_id}")
async def conversation(conversation_id: int, user=Depends(get_current_user(optional=False))):
    c = get_conversation(user["id"], conversation_id)
    if not c:
        raise HTTPException(404, "Conversation not found")
    return {"conversation": c, "messages": get_messages(user["id"], conversation_id) or []}


@app.post("/api/conversations")
async def new_conversation(user=Depends(get_current_user(optional=False))):
    cid = create_conversation(user["id"])
    return {"id": cid}


def merge_farmer_profile(customer: dict, incoming: dict) -> dict:
    merged = dict(customer or {})
    if incoming.get("lat") and incoming.get("lon"):
        merged.update({k: incoming[k] for k in ("lat", "lon", "place") if incoming.get(k)})
    elif incoming.get("place"):
        merged["place"] = incoming["place"]
    for k in ("state", "district", "crop", "season", "area", "water", "soil", "lang"):
        if incoming.get(k):
            merged[k] = incoming[k]
    return merged


@app.post("/api/chat")
async def chat(req: ChatRequest, user=Depends(get_current_user(optional=True))):
    if not user:
        history = [t.model_dump() for t in req.history]
        return StreamingResponse(
            answer_stream(kb, req.question.strip(), req.profile.model_dump(), history, req.images or req.image, req.photo_repeat),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    cid = req.conversation_id
    if cid is None:
        cid = create_conversation(user["id"])
    elif not get_conversation(user["id"], cid):
        raise HTTPException(404, "Conversation not found")

    customer = get_profile_for_user(user)
    merged = merge_farmer_profile(customer, req.profile.model_dump())
    rows = get_messages(user["id"], cid, limit=settings.max_history_turns * 2) or []
    history = [{"role": r["role"], "content": r["content"]} for r in rows]
    save_message(user["id"], cid, "user", req.question.strip(), refs=[], photo_note=None)

    return StreamingResponse(
        answer_stream(
            kb, req.question.strip(), merged, history,
            req.images or req.image, req.photo_repeat,
            farmer_record=customer, user_id=user["id"], conversation_id=cid,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- review of learned (web-researched) rows ----------
def _admin(key: str | None):
    if not settings.admin_key or key != settings.admin_key:
        raise HTTPException(401, "Set ADMIN_KEY in .env and send it in the X-Admin-Key header")


@app.get("/api/learned")
def learned_list(status: str | None = Query(None, pattern="^(unverified|verified|rejected)$"),
                 x_admin_key: str | None = Header(None)):
    _admin(x_admin_key)
    return store.list(status)


@app.post("/api/learned/{rid}/{action}")
def learned_review(rid: str, action: str, x_admin_key: str | None = Header(None)):
    _admin(x_admin_key)
    if action not in ("verify", "reject", "unverify"):
        raise HTTPException(400, "action must be verify, reject or unverify")
    rec = store.set_status(rid, {"verify": "verified", "reject": "rejected", "unverify": "unverified"}[action])
    if not rec:
        raise HTTPException(404, f"No record {rid}")
    return rec


@app.get("/api/learned/export.csv", response_class=PlainTextResponse)
def learned_export(status: str = "verified", x_admin_key: str | None = Header(None)):
    """Rows ready to paste into the master workbook."""
    _admin(x_admin_key)
    rows = store.list(status)
    keys = ["id", "kind", "name", "crop", "status"] + sorted({k for r in rows for k in r["data"]}) + ["sources"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        line = {k: r.get(k, "") for k in ("id", "kind", "name", "crop", "status")}
        for k, v in r["data"].items():
            line[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else (v or "")
        line["sources"] = " | ".join(s.get("url", "") for s in r["sources"])
        w.writerow(line)
    return buf.getvalue()


# Public API for other apps (/v1/...)
app.include_router(public_router)


# ---------- region: place lookup for the sidebar, and regional documents ----------
@app.get("/api/region")
async def region_lookup(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Village/district/state for a GPS point, so the page can fill the farm profile."""
    try:
        return {"place": await region.place(lat, lon)}
    except Exception as e:
        raise HTTPException(502, f"Location lookup failed: {e}")


@app.get("/api/region/search")
async def region_search(request: Request, q: str = Query(min_length=3, max_length=120), state: str = "",
                        district: str = ""):
    """Find a village/town the farmer types (no GPS needed)."""
    lgd = request.app.state.kb.lgd_places
    try:
        p = await region.find_place(q, state, district, lgd)
        if not p and (state or district):  # the saved district may be wrong; try the name alone
            p = await region.find_place(q, lgd=lgd)
    except Exception as e:
        raise HTTPException(502, f"Place search failed: {e}")
    if not p:
        raise HTTPException(404, "Place not found; try adding the tehsil or district")
    return {"place": p}


class DocFromUrl(BaseModel):
    url: str = Field(description="Link to a PDF (agromet advisory, ICAR pest alert, KVK guide)")
    title: str
    state: str = ""
    district: str = Field("", description="Leave empty for a state-wide (or, with no state, national) document")
    crops: str = ""
    valid_until: str = Field("", description="YYYY-MM-DD; leave empty if it does not expire")
    doc_type: str = Field("", description="advisory, contingency_plan, pest_alert, guide, groundwater, soil_report...")


@app.post("/api/region-docs/upload")
async def region_doc_upload(file: UploadFile = File(..., description="PDF"), title: str = Form(...),
                            state: str = Form(""), district: str = Form(""), crops: str = Form(""),
                            valid_until: str = Form(""), doc_type: str = Form(""),
                            x_admin_key: str | None = Header(None)):
    """Upload a PDF (agromet advisory, contingency plan, ICAR guide, pest alert...) for the agronomist to use."""
    _admin(x_admin_key)
    data = await file.read()
    if b"%PDF" not in data[:1024]:  # some PDFs (like CGWB reports) start with a blank line
        raise HTTPException(422, "That file is not a PDF")
    try:
        return await region.add_document(data, title=title, source=file.filename or "", state=state,
                                         district=district, crops=crops, valid_until=valid_until, doc_type=doc_type)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/region-docs/status")
async def region_docs_status(x_admin_key: str | None = Header(None)):
    """Is the PDF knowledge base (Qdrant) connected, and how much is in it?"""
    _admin(x_admin_key)
    from . import rag
    out = {"documents": len(region.list_documents()), "rag_enabled": rag.enabled(), "collection": rag.COLLECTION,
           "embed_model": rag.EMBED_MODEL}
    if rag.enabled():
        try:
            await rag.ensure_collection()
            out["indexed_chunks"] = (await rag.qdrant().count(rag.COLLECTION)).count
        except Exception as e:
            out["qdrant_error"] = str(e)
    return out


@app.post("/api/region-docs/from-url")
async def region_doc_from_url(body: DocFromUrl, x_admin_key: str | None = Header(None)):
    _admin(x_admin_key)
    try:
        return await region.add_document_from_url(body.url, title=body.title, state=body.state,
                                                  district=body.district, crops=body.crops,
                                                  valid_until=body.valid_until, doc_type=body.doc_type)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/region-docs")
def region_docs(x_admin_key: str | None = Header(None)):
    _admin(x_admin_key)
    return region.list_documents()


@app.delete("/api/region-docs/{did}")
async def region_doc_delete(did: str, x_admin_key: str | None = Header(None)):
    _admin(x_admin_key)
    if not await region.delete_document(did):
        raise HTTPException(404, f"No document {did}")
    return {"deleted": did.upper()}


# Frontend (mounted last so /api routes win)
if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")