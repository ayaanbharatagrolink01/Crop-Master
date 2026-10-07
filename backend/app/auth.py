"""Phone OTP authentication and JWT sessions for Crop Master."""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .farmers import (
    create_user,
    get_user_by_id,
    get_user_by_mobile_hash,
    record_login,
    upsert_otp_challenge,
    verify_otp_challenge,
)

bearer = HTTPBearer(auto_error=False)

OTP_TTL = 300
OTP_MAX_WRONG = 5
OTP_MAX_REQUESTS_HOUR = 5


def normalize_mobile(value: str) -> str:
    s = re.sub(r"[\s-]", "", str(value or ""))

    if s.startswith("+91"):
        s = s[3:]
    elif s.startswith("0091"):
        s = s[4:]
    elif s.startswith("91") and len(s) in (11, 12):
        s = s[2:]

    if s.startswith("0") and len(s) == 11:
        s = s[1:]

    if not re.fullmatch(r"[6-9]\d{9}", s):
        raise ValueError(
            "mobile must be a valid 10-digit Indian number starting 6-9"
        )

    return s


def mobile_hash(mobile: str) -> str:
    return hashlib.sha256(
        (
            normalize_mobile(mobile)
            + settings.mobile_pepper
        ).encode()
    ).hexdigest()


def _jwt_secret(website: bool = False) -> str:
    secret = (
        settings.website_jwt_secret
        if website
        else settings.jwt_secret
    )

    if not secret:
        raise HTTPException(
            500,
            "JWT secret is not configured",
        )

    return secret


def issue_session(user_id: int) -> str:
    now = datetime.now(timezone.utc)

    claims = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(days=30),
        "typ": "crop_master",
    }

    return jwt.encode(
        claims,
        _jwt_secret(),
        algorithm="HS256",
    )


def request_otp(mobile: str) -> dict[str, Any]:
    normalized = normalize_mobile(mobile)
    mh = mobile_hash(normalized)

    challenge = upsert_otp_challenge(
        mh,
        OTP_MAX_REQUESTS_HOUR,
        OTP_TTL,
    )

    if not challenge["allowed"]:
        raise HTTPException(
            429,
            "Too many OTP requests. Try again later.",
        )

    otp = f"{secrets.randbelow(1_000_000):06d}"

    otp_hash = hashlib.sha256(
        (
            otp
            + settings.mobile_pepper
            + challenge["nonce"]
        ).encode()
    ).hexdigest()

    if settings.otp_provider == "console":
        print(
            f"[Crop Master OTP] {normalized[-4:]} -> {otp}",
            flush=True,
        )

    elif settings.otp_provider == "msg91":
        raise HTTPException(
            501,
            "MSG91 OTP provider is not wired yet",
        )

    upsert_otp_challenge(
        mh,
        OTP_MAX_REQUESTS_HOUR,
        OTP_TTL,
        otp_hash=otp_hash,
    )

    return {
        "ok": True,
        "expires_in": OTP_TTL,
    }


def verify_otp(mobile: str, otp: str) -> dict[str, Any]:
    normalized = normalize_mobile(mobile)
    mh = mobile_hash(normalized)

    challenge = verify_otp_challenge(
        mh,
        str(otp),
        settings.mobile_pepper,
        OTP_MAX_WRONG,
    )

    if not challenge["ok"]:
        status = challenge.get(
            "status",
            "invalid",
        )

        if status == "locked":
            raise HTTPException(
                429,
                "Too many wrong OTP attempts. "
                "Request a new OTP later.",
            )

        if status == "expired":
            raise HTTPException(
                400,
                "OTP expired. Request a new OTP.",
            )

        raise HTTPException(
            400,
            "Invalid OTP.",
        )

    user = get_user_by_mobile_hash(mh)

    if not user:
        user = create_user(
            mh,
            normalized[-4:],
            is_existing_customer=bool(
                challenge.get(
                    "is_existing_customer"
                )
            ),
        )

    record_login(user["id"])

    return {
        "token": issue_session(user["id"]),
        "user": get_user_by_id(user["id"]),
    }


def _decode(
    token: str,
    secret: str,
    website: bool = False,
) -> dict[str, Any] | None:
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
        )
    except jwt.PyJWTError:
        return None


def get_current_user(optional: bool = False):
    """
    FastAPI dependency factory.

    The second website JWT type can later be accepted here using
    WEBSITE_JWT_SECRET and either mobile or user_id.
    """

    async def dependency(
        credentials: HTTPAuthorizationCredentials | None = Depends(
            bearer
        ),
    ):
        if not credentials:
            if optional:
                return None

            raise HTTPException(
                401,
                "Authentication required",
            )

        token = credentials.credentials

        claims = _decode(
            token,
            _jwt_secret(),
            False,
        )

        if claims:
            uid = (
                claims.get("sub")
                or claims.get("user_id")
            )

            if uid is not None:
                user = get_user_by_id(int(uid))

                if user:
                    return user

        if settings.website_jwt_secret:
            claims = _decode(
                token,
                settings.website_jwt_secret,
                True,
            )

            if claims:
                mobile = claims.get("mobile")
                uid = (
                    claims.get("user_id")
                    or claims.get("sub")
                )

                user = None

                if mobile:
                    try:
                        user = get_user_by_mobile_hash(
                            mobile_hash(str(mobile))
                        )
                    except ValueError:
                        user = None

                if not user and uid is not None:
                    try:
                        user = get_user_by_id(
                            int(uid)
                        )
                    except (TypeError, ValueError):
                        user = None

                if user:
                    return user

        raise HTTPException(
            401,
            "Invalid or expired session",
        )

    return dependency
