"""SQLite persistence for authenticated farmer profiles and conversation memory."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

DB_PATH = settings.farmers_db


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 mobile_hash TEXT UNIQUE NOT NULL,
 mobile_last4 TEXT NOT NULL,
 created_at TEXT NOT NULL,
 last_login TEXT,
 is_existing_customer INTEGER NOT NULL DEFAULT 0,
 consent_personalise INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS customer_profiles (
 mobile_hash TEXT PRIMARY KEY,
 lat REAL,
 lng REAL,
 pincode TEXT,
 city TEXT,
 district TEXT,
 state TEXT,
 crops TEXT,
 soil TEXT,
 water TEXT,
 season TEXT,
 recent_products TEXT,
 last_order_date TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 user_id INTEGER NOT NULL,
 started_at TEXT NOT NULL,
 last_message_at TEXT NOT NULL,
 summarised INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 conversation_id INTEGER NOT NULL,
 role TEXT NOT NULL,
 content TEXT NOT NULL,
 refs TEXT,
 photo_note TEXT,
 created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_notes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 user_id INTEGER NOT NULL,
 conversation_id INTEGER NOT NULL,
 date TEXT NOT NULL,
 crop TEXT,
 problem TEXT,
 diagnosis TEXT,
 advice TEXT,
 refs TEXT,
 follow_up_after TEXT
);

CREATE TABLE IF NOT EXISTS otp_challenges (
 mobile_hash TEXT PRIMARY KEY,
 nonce TEXT NOT NULL,
 otp_hash TEXT,
 expires_at INTEGER NOT NULL,
 wrong_attempts INTEGER NOT NULL DEFAULT 0,
 request_count INTEGER NOT NULL DEFAULT 0,
 request_window_start INTEGER NOT NULL,
 locked_until INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conversations_user
ON conversations(user_id, last_message_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
ON messages(conversation_id, id);

CREATE INDEX IF NOT EXISTS idx_case_notes_user
ON case_notes(user_id, date DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    con.row_factory = sqlite3.Row

    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db():
        pass


def _json(value, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def get_user_by_id(user_id: int):
    with db() as c:
        r = c.execute(
            "SELECT * FROM users WHERE id=?",
            (user_id,),
        ).fetchone()

        if not r:
            return None

        d = dict(r)
        d["consent_personalise"] = bool(
            d["consent_personalise"]
        )

        return d


def get_user_by_mobile_hash(mh: str):
    with db() as c:
        r = c.execute(
            "SELECT * FROM users WHERE mobile_hash=?",
            (mh,),
        ).fetchone()

        return dict(r) if r else None


def create_user(
    mh,
    last4,
    is_existing_customer=False,
):
    with db() as c:
        c.execute(
            """
            INSERT OR IGNORE INTO users(
                mobile_hash,
                mobile_last4,
                created_at,
                is_existing_customer
            )
            VALUES(?,?,?,?)
            """,
            (
                mh,
                last4,
                _now(),
                int(is_existing_customer),
            ),
        )

        r = c.execute(
            "SELECT * FROM users WHERE mobile_hash=?",
            (mh,),
        ).fetchone()

        return dict(r)


def record_login(user_id):
    with db() as c:
        c.execute(
            "UPDATE users SET last_login=? WHERE id=?",
            (_now(), user_id),
        )


def get_profile_by_hash(mh):
    with db() as c:
        r = c.execute(
            """
            SELECT *
            FROM customer_profiles
            WHERE mobile_hash=?
            """,
            (mh,),
        ).fetchone()

        if not r:
            return {}

        d = dict(r)
        d["recent_products"] = _json(
            d.get("recent_products"),
            [],
        )

        return d


def get_profile_for_user(user):
    return get_profile_by_hash(
        user["mobile_hash"]
    )


def upsert_profile(mh, profile):
    with db() as c:
        c.execute(
            """
            INSERT INTO customer_profiles(
                mobile_hash,
                lat,
                lng,
                pincode,
                city,
                district,
                state,
                crops,
                soil,
                water,
                season,
                recent_products,
                last_order_date
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)

            ON CONFLICT(mobile_hash)
            DO UPDATE SET
                lat=excluded.lat,
                lng=excluded.lng,
                pincode=excluded.pincode,
                city=excluded.city,
                district=excluded.district,
                state=excluded.state,
                crops=excluded.crops,
                soil=excluded.soil,
                water=excluded.water,
                season=excluded.season,
                recent_products=excluded.recent_products,
                last_order_date=excluded.last_order_date
            """,
            (
                mh,
                profile.get("lat"),
                profile.get("lng"),
                profile.get("pincode"),
                profile.get("city"),
                profile.get("district"),
                profile.get("state"),
                profile.get("crops"),
                profile.get("soil"),
                profile.get("water"),
                profile.get("season"),
                json.dumps(
                    profile.get(
                        "recent_products",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                profile.get(
                    "last_order_date"
                ),
            ),
        )


def upsert_otp_challenge(
    mh,
    max_requests,
    ttl,
    otp_hash=None,
):
    now = int(time.time())

    with db() as c:
        r = c.execute(
            "SELECT * FROM otp_challenges WHERE mobile_hash=?",
            (mh,),
        ).fetchone()

        if (
            not r
            or now - r["request_window_start"] >= 3600
        ):
            nonce = hashlib.sha256(
                f"{mh}:{now}".encode()
            ).hexdigest()

            c.execute(
                """
                INSERT OR REPLACE INTO otp_challenges(
                    mobile_hash,
                    nonce,
                    otp_hash,
                    expires_at,
                    wrong_attempts,
                    request_count,
                    request_window_start,
                    locked_until
                )
                VALUES(?,?,?,?,?,?,?,0)
                """,
                (
                    mh,
                    nonce,
                    None,
                    now + ttl,
                    0,
                    0,
                    now,
                ),
            )

            r = c.execute(
                "SELECT * FROM otp_challenges WHERE mobile_hash=?",
                (mh,),
            ).fetchone()

        if (
            r["request_count"] >= max_requests
            and otp_hash is None
        ):
            return {
                "allowed": False,
                "nonce": r["nonce"],
            }

        if otp_hash is not None:
            c.execute(
                """
                UPDATE otp_challenges
                SET
                    otp_hash=?,
                    expires_at=?,
                    wrong_attempts=0,
                    locked_until=0,
                    request_count=request_count+1
                WHERE mobile_hash=?
                """,
                (
                    otp_hash,
                    now + ttl,
                    mh,
                ),
            )

        else:
            return {
                "allowed": True,
                "nonce": r["nonce"],
            }

        return {
            "allowed": True,
            "nonce": r["nonce"],
        }


def verify_otp_challenge(
    mh,
    otp,
    pepper,
    max_wrong,
):
    now = int(time.time())

    with db() as c:
        r = c.execute(
            "SELECT * FROM otp_challenges WHERE mobile_hash=?",
            (mh,),
        ).fetchone()

        if not r or not r["otp_hash"]:
            return {
                "ok": False,
                "status": "invalid",
            }

        if (
            r["locked_until"] > now
            or r["wrong_attempts"] >= max_wrong
        ):
            return {
                "ok": False,
                "status": "locked",
            }

        if r["expires_at"] < now:
            return {
                "ok": False,
                "status": "expired",
            }

        expected = hashlib.sha256(
            (
                otp
                + pepper
                + r["nonce"]
            ).encode()
        ).hexdigest()

        if not hmac.compare_digest(
            expected,
            r["otp_hash"],
        ):
            wrong = r["wrong_attempts"] + 1

            c.execute(
                """
                UPDATE otp_challenges
                SET wrong_attempts=?,
                    locked_until=?
                WHERE mobile_hash=?
                """,
                (
                    wrong,
                    now + 3600
                    if wrong >= max_wrong
                    else 0,
                    mh,
                ),
            )

            return {
                "ok": False,
                "status": (
                    "locked"
                    if wrong >= max_wrong
                    else "invalid"
                ),
            }

        c.execute(
            """
            UPDATE otp_challenges
            SET otp_hash=NULL,
                wrong_attempts=0
            WHERE mobile_hash=?
            """,
            (mh,),
        )

        exists = c.execute(
            """
            SELECT 1
            FROM customer_profiles
            WHERE mobile_hash=?
            """,
            (mh,),
        ).fetchone() is not None

        return {
            "ok": True,
            "is_existing_customer": exists,
        }


def create_conversation(user_id):
    now = _now()

    with db() as c:
        r = c.execute(
            """
            INSERT INTO conversations(
                user_id,
                started_at,
                last_message_at
            )
            VALUES(?,?,?)
            RETURNING id
            """,
            (
                user_id,
                now,
                now,
            ),
        ).fetchone()

        return r[0]


def touch_conversation(
    conversation_id,
    user_id,
):
    with db() as c:
        c.execute(
            """
            UPDATE conversations
            SET last_message_at=?
            WHERE id=? AND user_id=?
            """,
            (
                _now(),
                conversation_id,
                user_id,
            ),
        )


def list_conversations(user_id):
    with db() as c:
        rows = c.execute(
            """
            SELECT *
            FROM conversations
            WHERE user_id=?
            ORDER BY last_message_at DESC
            """,
            (user_id,),
        ).fetchall()

        out = []

        for r in rows:
            title = c.execute(
                """
                SELECT content
                FROM messages
                WHERE conversation_id=?
                  AND role='user'
                ORDER BY id
                LIMIT 1
                """,
                (r["id"],),
            ).fetchone()

            out.append(
                {
                    **dict(r),
                    "summarised": bool(
                        r["summarised"]
                    ),
                    "title": (
                        title[0][:70]
                        if title
                        else "New conversation"
                    ),
                }
            )

        return out


def get_conversation(
    user_id,
    conversation_id,
):
    with db() as c:
        r = c.execute(
            """
            SELECT *
            FROM conversations
            WHERE id=? AND user_id=?
            """,
            (
                conversation_id,
                user_id,
            ),
        ).fetchone()

        return dict(r) if r else None


def get_messages(
    user_id,
    conversation_id,
    limit=20,
):
    if not get_conversation(
        user_id,
        conversation_id,
    ):
        return None

    with db() as c:
        rows = c.execute(
            """
            SELECT *
            FROM messages
            WHERE conversation_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                conversation_id,
                limit,
            ),
        ).fetchall()

        return [
            dict(
                r,
                refs=_json(
                    r["refs"],
                    [],
                ),
            )
            for r in reversed(rows)
        ]


def save_message(
    user_id,
    conversation_id,
    role,
    content,
    refs=None,
    photo_note=None,
):
    if not get_conversation(
        user_id,
        conversation_id,
    ):
        raise PermissionError(
            "conversation does not belong to user"
        )

    with db() as c:
        c.execute(
            """
            INSERT INTO messages(
                conversation_id,
                role,
                content,
                refs,
                photo_note,
                created_at
            )
            VALUES(?,?,?,?,?,?)
            """,
            (
                conversation_id,
                role,
                content,
                json.dumps(
                    refs or [],
                    ensure_ascii=False,
                ),
                photo_note,
                _now(),
            ),
        )

        c.execute(
            """
            UPDATE conversations
            SET last_message_at=?
            WHERE id=?
            """,
            (
                _now(),
                conversation_id,
            ),
        )


def update_latest_user_photo_note(
    user_id,
    conversation_id,
    photo_note,
):
    if not get_conversation(
        user_id,
        conversation_id,
    ):
        raise PermissionError(
            "conversation does not belong to user"
        )

    with db() as c:
        c.execute(
            """
            UPDATE messages
            SET photo_note=?
            WHERE id=(
                SELECT id
                FROM messages
                WHERE conversation_id=?
                  AND role='user'
                ORDER BY id DESC
                LIMIT 1
            )
            """,
            (
                photo_note,
                conversation_id,
            ),
        )


def unsummarised_conversations(user_id):
    with db() as c:
        return [
            dict(r)
            for r in c.execute(
                """
                SELECT *
                FROM conversations
                WHERE user_id=?
                  AND summarised=0
                ORDER BY last_message_at
                """,
                (user_id,),
            ).fetchall()
        ]


def mark_summarised(
    conversation_id,
    user_id,
):
    with db() as c:
        c.execute(
            """
            UPDATE conversations
            SET summarised=1
            WHERE id=? AND user_id=?
            """,
            (
                conversation_id,
                user_id,
            ),
        )


def save_case_notes(
    user_id,
    conversation_id,
    notes,
):
    with db() as c:
        for n in notes[:3]:
            c.execute(
                """
                INSERT INTO case_notes(
                    user_id,
                    conversation_id,
                    date,
                    crop,
                    problem,
                    diagnosis,
                    advice,
                    refs,
                    follow_up_after
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    user_id,
                    conversation_id,
                    n.get("date")
                    or datetime.now(
                        timezone.utc
                    ).date().isoformat(),
                    n.get("crop", ""),
                    n.get("problem", ""),
                    n.get("diagnosis", ""),
                    str(
                        n.get(
                            "advice",
                            "",
                        )
                    )[:40],
                    json.dumps(
                        n.get("refs", []),
                        ensure_ascii=False,
                    ),
                    n.get("follow_up_after"),
                ),
            )


def latest_case_notes(
    user_id,
    limit=3,
):
    with db() as c:
        rows = c.execute(
            """
            SELECT *
            FROM case_notes
            WHERE user_id=?
            ORDER BY date DESC, id DESC
            LIMIT ?
            """,
            (
                user_id,
                limit,
            ),
        ).fetchall()

        return [
            dict(
                r,
                refs=_json(
                    r["refs"],
                    [],
                ),
            )
            for r in rows
        ]
