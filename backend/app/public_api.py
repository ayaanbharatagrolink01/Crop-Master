"""Public API (v1) for other apps that use the Crop Master agronomist.

Auth:   every request sends its key in the X-API-Key header.
        Keys are set in .env as CLIENT_API_KEYS=name:key,name2:key2
Limits: RATE_LIMIT_PER_MIN chat requests per minute per client (default 20).
Memory: send conversation_id to continue a conversation; the server keeps its history and last photo.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .agent import answer_stream
from .config import ROOT, settings
from .learned import store

log = logging.getLogger("agronomist")
router = APIRouter(prefix="/v1", tags=["Agronomist API v1"])

MAX_TURNS_KEPT = 20      # turns stored per conversation
PHOTO_FOLLOW_UPS = 3     # follow-up questions that re-check the last photo
REFS_RE = re.compile(r"\n?\s*REFS:[^\n]*\s*$", re.I)


# ---------- auth and rate limit ----------
def _load_clients() -> dict[str, str]:
    """CLIENT_API_KEYS=sandeep:key1,mobileapp:key2  ->  {key: name}"""
    out = {}
    for item in os.getenv("CLIENT_API_KEYS", "").split(","):
        if ":" in item:
            name, key = item.split(":", 1)
            if name.strip() and key.strip():
                out[key.strip()] = name.strip()
    return out


CLIENTS = _load_clients()
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def api_client(x_api_key: str | None = Header(None, description="Your client API key")) -> str:
    for key, name in CLIENTS.items():
        if x_api_key and secrets.compare_digest(x_api_key, key):
            return name
    raise HTTPException(401, "Missing or invalid X-API-Key header")


def rate_limited_client(client: str = Depends(api_client)) -> str:
    now = time.time()
    with _hits_lock:
        q = _hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, f"Rate limit reached: {RATE_LIMIT} chat requests per minute")
        q.append(now)
    return client


# ---------- conversation store ----------
def _load_photos(v: str | None) -> list[str]:
    """last_photo is stored as a JSON list; older rows hold a single data URL."""
    if not v:
        return []
    if v.startswith("["):
        return json.loads(v)
    return [v]


class Conversations:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, client TEXT, history TEXT, last_photo TEXT,
            photo_uses INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)""")
        self.db.commit()

    def get(self, cid: str, client: str) -> dict | None:
        r = self.db.execute("SELECT history, last_photo, photo_uses, created_at, updated_at FROM conversations "
                            "WHERE id=? AND client=?", (cid, client)).fetchone()
        if not r:
            return None
        return {"id": cid, "history": json.loads(r[0] or "[]"), "last_photo": _load_photos(r[1]), "photo_uses": r[2],
                "created_at": r[3], "updated_at": r[4]}

    def create(self, client: str) -> dict:
        cid = "conv_" + secrets.token_hex(8)
        now = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            self.db.execute("INSERT INTO conversations (id, client, history, created_at, updated_at) "
                            "VALUES (?,?,?,?,?)", (cid, client, "[]", now, now))
            self.db.commit()
        return self.get(cid, client)

    def save(self, conv: dict):
        with self.lock:
            self.db.execute("UPDATE conversations SET history=?, last_photo=?, photo_uses=?, updated_at=? WHERE id=?",
                            (json.dumps(conv["history"][-MAX_TURNS_KEPT:], ensure_ascii=False),
                             json.dumps(conv["last_photo"]) if conv.get("last_photo") else None,
                             conv["photo_uses"], datetime.now().isoformat(timespec="seconds"), conv["id"]))
            self.db.commit()

    def delete(self, cid: str, client: str) -> bool:
        with self.lock:
            n = self.db.execute("DELETE FROM conversations WHERE id=? AND client=?", (cid, client)).rowcount
            self.db.commit()
        return n > 0


conversations = Conversations(os.getenv("CONVERSATIONS_DB", ROOT / "data" / "conversations.db"))


# ---------- request and response models ----------
class Profile(BaseModel):
    state: str = Field("", examples=["Madhya Pradesh"])
    district: str = Field("", examples=["Indore"])
    crop: str = Field("", examples=["Soybean"])
    season: str = Field("", examples=["Kharif"])
    area: str = Field("", description="Acres", examples=["5"])
    water: str = Field("", examples=["Rainfed"])
    soil: str = Field("", examples=["Black"])
    lang: str = Field("auto", description="auto (match the question), English, Hindi or Hinglish")
    lat: str = Field("", description="Farmer's latitude from GPS, e.g. \"22.98\" (optional; enables local weather, "
                                     "soil and advisories)", examples=["22.98"])
    lon: str = Field("", description="Farmer's longitude from GPS", examples=["79.21"])
    place: str = Field("", description="Village/town name, e.g. \"Belkhedi, Vidisha, Madhya Pradesh\". Use "
                                       "GET /v1/places?q= to turn a typed village into lat/lon", examples=["Belkhedi"])

    @field_validator("*")
    @classmethod
    def trim(cls, v: str) -> str:
        return (v or "").strip()[:80]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["Dhan ki baali me hare-peele gole kya hai?"])
    conversation_id: str | None = Field(None, description="Omit to start a new conversation; send the returned id "
                                                          "to continue it")
    profile: Profile = Profile()
    image: str | None = Field(None, description="Optional crop photo: a data URL (data:image/jpeg;base64,...) or plain "
                                                "base64 of a JPEG, PNG or WebP. Resize to about 1024px first.")
    images: list[str] = Field(default_factory=list, max_length=3,
                              description="Up to 3 photos of the same problem (same formats as image); use this "
                                          "instead of image to send several angles, e.g. a leaf and the stem")
    history: list[Turn] | None = Field(None, max_length=40, description="Only for clients that keep history "
                                                                        "themselves; ignored when conversation_id is "
                                                                        "sent")

    @field_validator("images")
    @classmethod
    def check_images(cls, v: list[str]) -> list[str]:
        return [cls.check_image(x) for x in v if x]

    @field_validator("image")
    @classmethod
    def check_image(cls, v: str | None) -> str | None:
        if not v:
            return None
        if v.startswith("data:"):
            if not v.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")):
                raise ValueError("image must be JPEG, PNG or WebP")
            b64 = v.split(",", 1)[1]
        else:
            b64 = v
        try:
            raw = base64.b64decode(b64[:64] + "=" * (-len(b64[:64]) % 4))
        except (binascii.Error, ValueError):
            raise ValueError("image is not valid base64")
        if len(b64) * 3 / 4 > settings.max_image_mb * 1024 * 1024:
            raise ValueError(f"image larger than {settings.max_image_mb} MB")
        if v.startswith("data:"):
            return v
        mime = ("image/png" if raw.startswith(b"\x89PNG") else "image/webp" if raw[8:12] == b"WEBP"
                else "image/jpeg" if raw.startswith(b"\xff\xd8") else None)
        if not mime:
            raise ValueError("image must be JPEG, PNG or WebP")
        return f"data:{mime};base64,{b64}"


class Ref(BaseModel):
    id: str = Field(examples=["DIS019"])
    title: str = Field(examples=["Soybean Charcoal Rot"])
    flag: str = Field("", description="Verified, Indicative (seed data awaiting review) or Unverified (web)")


class Usage(BaseModel):
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    vision_input_tokens: int = 0
    vision_output_tokens: int = 0
    cost_usd: float = 0
    cost_inr: float = 0


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str = Field(description="The agronomist's reply (Markdown)")
    refs: list[Ref] = Field(description="Database records the answer relied on")
    used_web_research: bool = Field(description="True if any record came from web research (LRN ids)")
    photo_note: str | None = Field(None, description="What the photo step saw, when an image was sent")
    usage: Usage | None = None


# ---------- helpers ----------
def _prepare(req: ChatRequest, client: str):
    """Load or create the conversation and decide which history and photo to send to the agent."""
    if req.conversation_id:
        conv = conversations.get(req.conversation_id, client)
        if not conv:
            raise HTTPException(404, "Unknown conversation_id")
        history = conv["history"]
    else:
        conv = conversations.create(client)
        history = [t.model_dump() for t in (req.history or [])]

    image, photo_repeat = req.images or ([req.image] if req.image else []), False
    if image:
        conv["last_photo"], conv["photo_uses"] = image, 0
    elif conv.get("last_photo") and conv.get("photo_uses", 0) < PHOTO_FOLLOW_UPS:
        image, photo_repeat = conv["last_photo"], True
        conv["photo_uses"] += 1
    return conv, history, image, photo_repeat


def _finish(conv: dict, history: list, question: str, answer: str, photo_note: str | None, had_photo: bool):
    user = question + (f"\n[Photo showed: {photo_note}]" if photo_note else "\n[sent a crop photo]" if had_photo else "")
    conv["history"] = history + [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    conversations.save(conv)


# ---------- endpoints ----------
@router.get("/health", summary="Service status (no key needed)")
def health():
    return {"ok": True, "model": settings.openai_model, "vision_model": settings.vision_model}


@router.post("/chat", response_model=ChatResponse, summary="Ask the agronomist (JSON in, JSON out)")
async def chat(req: ChatRequest, request: Request, client: str = Depends(rate_limited_client)):
    conv, history, image, photo_repeat = _prepare(req, client)
    text, refs, note, usage, error = "", [], None, None, None
    async for chunk in answer_stream(request.app.state.kb, req.question.strip(), req.profile.model_dump(),
                                     history, image, photo_repeat):
        ev = json.loads(chunk[len("data: "):])
        if ev["type"] == "delta":
            text += ev["text"]
        elif ev["type"] == "reset":
            text = ""
        elif ev["type"] == "done":
            refs, note, usage = ev.get("refs", []), ev.get("photo_note"), ev.get("usage")
        elif ev["type"] == "error":
            error = ev
    if error:
        raise HTTPException(429 if error.get("code") == "rate_limited" else 502, error.get("text", "AI service error"))
    answer = REFS_RE.sub("", text).strip()
    _finish(conv, history, req.question.strip(), answer, note, bool(req.image or req.images))
    log.info("API client=%s conv=%s refs=%s cost_inr=%s", client, conv["id"], [r["id"] for r in refs],
             (usage or {}).get("cost_inr"))
    return ChatResponse(conversation_id=conv["id"], answer=answer, refs=refs,
                        used_web_research=any(r["id"].startswith("LRN") for r in refs),
                        photo_note=note, usage=usage)


@router.post("/chat/stream", summary="Ask the agronomist with a streamed reply (Server-Sent Events)")
async def chat_stream(req: ChatRequest, request: Request, client: str = Depends(rate_limited_client)):
    conv, history, image, photo_repeat = _prepare(req, client)

    async def events():
        yield f"data: {json.dumps({'type': 'conversation', 'conversation_id': conv['id']})}\n\n"
        text, note = "", None
        async for chunk in answer_stream(request.app.state.kb, req.question.strip(), req.profile.model_dump(),
                                         history, image, photo_repeat):
            ev = json.loads(chunk[len("data: "):])
            if ev["type"] == "delta":
                text += ev["text"]
            elif ev["type"] == "reset":
                text = ""
            elif ev["type"] == "done":
                note = ev.get("photo_note")
            yield chunk
        if text:
            _finish(conv, history, req.question.strip(), REFS_RE.sub("", text).strip(), note, bool(req.image or req.images))

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/conversations/{conversation_id}", summary="Read a conversation's history")
def get_conversation(conversation_id: str, client: str = Depends(api_client)):
    conv = conversations.get(conversation_id, client)
    if not conv:
        raise HTTPException(404, "Unknown conversation_id")
    return {"conversation_id": conv["id"], "history": conv["history"], "has_photo": bool(conv["last_photo"]),
            "created_at": conv["created_at"], "updated_at": conv["updated_at"]}


@router.delete("/conversations/{conversation_id}", summary="Delete a conversation")
def delete_conversation(conversation_id: str, client: str = Depends(api_client)):
    if not conversations.delete(conversation_id, client):
        raise HTTPException(404, "Unknown conversation_id")
    return {"deleted": conversation_id}


@router.get("/records/{record_id}", summary="Full database record (CRP, PST, DIS, CPM, CDM, LRN ids)")
def record(record_id: str, request: Request, client: str = Depends(api_client)):
    if record_id.upper().startswith("LRN"):
        rec = store.get(record_id)
        if not rec or rec["status"] == "rejected":
            raise HTTPException(404, f"No record {record_id}")
        return {"id": rec["id"], "title": store.ref(rec)["title"], "flag": store.flag(rec),
                "record": {"name": rec["name"], "kind": rec["kind"], "crop": rec["crop"], **rec["data"]},
                "sources": rec["sources"]}
    r = request.app.state.kb.record(record_id)
    if not r:
        raise HTTPException(404, f"No record {record_id}")
    return r


@router.get("/search", summary="Search the crop database without the AI (free, instant)")
def search(request: Request, q: str = Query(min_length=1, max_length=200), client: str = Depends(api_client)):
    kb = request.app.state.kb
    ctx = kb.retrieve(q, {})
    return {"query": q, "results": [kb.ref(i) for i in kb.context_ids(ctx)]}


@router.get("/meta", summary="States, seasons, districts and crop names (for building forms)")
def meta(request: Request, client: str = Depends(api_client)):
    return request.app.state.kb.meta()


@router.get("/places", summary="Find a village or town (returns lat/lon to send in profile)")
async def find_place(request: Request, q: str = Query(min_length=3, max_length=120), state: str = "",
                     district: str = "", client: str = Depends(api_client)):
    from . import region
    try:
        lgd = request.app.state.kb.lgd_places
        p = await region.find_place(q, state, district, lgd) or \
            (await region.find_place(q, lgd=lgd) if (state or district) else None)
    except Exception as e:
        raise HTTPException(502, f"Place search failed: {e}")
    if not p:
        raise HTTPException(404, "Place not found; try adding the tehsil or district")
    return {"place": p}