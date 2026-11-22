"""Auth utilities: anonymous device tokens and optional signed JWT sessions.
Device tokens are UUID-based, stored as user.id, sent via X-Device-Token header.
No passwords are collected; this is an academic prototype recommender.
"""
import datetime
import hashlib
import hmac
import json
import logging
import os
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.db import ALLOWED_PROFILE_KEYS, User, create_tables

logger = logging.getLogger("lic_recommender.auth")

TOKEN_SECRET = os.environ.get("TOKEN_SECRET", "change-me-before-deploy")
TOKEN_EXPIRY_DAYS = int(os.environ.get("TOKEN_EXPIRY_DAYS", "365"))


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# ----- Token signing (simple HMAC-SHA256, no external library needed) -----

def _sign(payload: dict) -> str:
    """HMAC-SHA256 sign a JSON payload; returns `{b64_payload}.{b64_sig}`."""
    import base64
    raw = json.dumps(payload, sort_keys=True).encode()
    b64_payload = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    sig = hmac.new(TOKEN_SECRET.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
    return f"{b64_payload}.{sig}"


def _verify(token: str) -> Optional[dict]:
    """Verify token signature; return payload dict or None on failure."""
    import base64
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        b64_payload, sig = parts
        expected = hmac.new(TOKEN_SECRET.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        padding = "=" * (4 - len(b64_payload) % 4)
        raw = base64.urlsafe_b64decode(b64_payload + padding)
        payload = json.loads(raw)
        # Expiry check
        exp = payload.get("exp")
        if exp and _now_utc().timestamp() > exp:
            return None
        return payload
    except Exception:
        return None


# ----- Device token CRUD --------------------------------------------------

def issue_device_token(db: Session) -> tuple[str, str]:
    """Create a new anonymous user record and return (user_id, signed_token)."""
    create_tables()
    user_id = str(uuid.uuid4())
    user = User(id=user_id, created_at=_now_utc())
    db.add(user)
    db.commit()

    exp = (_now_utc() + datetime.timedelta(days=TOKEN_EXPIRY_DAYS)).timestamp()
    payload = {"uid": user_id, "exp": exp}
    token = _sign(payload)
    logger.info("Issued new device token for user_id=%s", user_id)
    return user_id, token


def resolve_token(token: str, db: Session) -> Optional[User]:
    """Validate a device token and return the User record or None."""
    payload = _verify(token)
    if not payload:
        return None
    user_id = payload.get("uid")
    if not user_id:
        return None
    user = db.query(User).filter(User.id == user_id, User.deleted_at.is_(None)).first()
    return user


# ----- Profile attribute helpers ------------------------------------------

DISALLOWED_PROFILE_KEYS = frozenset({
    "date_of_birth", "child_dob", "credit_card", "aadhaar_number", "pan_card", "bank_account", "password"
})


def is_valid_profile_key(key: str) -> bool:
    """Check if key is permitted (standard allowed key or clean manual attribute slug)."""
    if not key or not isinstance(key, str):
        return False
    clean_k = key.lower().strip()
    if clean_k in DISALLOWED_PROFILE_KEYS:
        return False
    if clean_k in ALLOWED_PROFILE_KEYS:
        return True
    import re
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_]{1,63}$", clean_k))


def get_confirmed_profile(user: User) -> dict:
    """Return a dict of field -> {value, source, confirmed_at} for a user's profile."""
    profile = {}
    for attr in user.profile_attrs:
        if is_valid_profile_key(attr.key):
            profile[attr.key] = {
                "value": _cast_value(attr.key, attr.value),
                "source": attr.source,
                "stated_at": attr.stated_at.isoformat() + "Z",
                "confirmed_at": attr.confirmed_at.isoformat() + "Z",
            }
    return profile


def _cast_value(key: str, raw: str):
    """Cast stored string value back to native type."""
    int_fields = {"customer_age", "child_age", "preferred_term", "preferred_sum_assured", "annual_income"}
    if key in int_fields:
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None
    return raw


def upsert_profile_attribute(
    user: User,
    key: str,
    value,
    source: str,
    db: Session
) -> bool:
    """Upsert a confirmed profile attribute. Returns False on invalid key/source."""
    clean_k = key.lower().strip()
    if not is_valid_profile_key(clean_k):
        logger.warning("Attempted to store disallowed profile key: %s", key)
        return False
    if source not in {"user_stated", "form", "llm_extracted_confirmed"}:
        logger.warning("Attempted to store disallowed source: %s", source)
        return False

    now = _now_utc()
    existing = next((a for a in user.profile_attrs if a.key == clean_k), None)
    if existing:
        existing.value = str(value)
        existing.source = source
        existing.confirmed_at = now
    else:
        from app.db import ProfileAttribute
        attr = ProfileAttribute(
            user_id=user.id,
            key=clean_k,
            value=str(value),
            source=source,
            stated_at=now,
            confirmed_at=now,
        )
        db.add(attr)
    db.commit()
    return True


def delete_profile_attribute(user: User, key: str, db: Session) -> bool:
    """Remove a single attribute; return True if found and deleted."""
    existing = next((a for a in user.profile_attrs if a.key == key), None)
    if not existing:
        return False
    db.delete(existing)
    db.commit()
    return True


def hard_delete_user(user: User, db: Session):
    """Hard delete all user data: attributes, queries, consents, and user record."""
    db.delete(user)
    db.commit()


# ----- Consent helpers ----------------------------------------------------

def get_consent_status(user: User) -> dict:
    """Return current consent flags from the User table (single source of truth)."""
    return {
        "consent_profile": bool(user.consent_profile),
        "consent_history": bool(user.consent_history),
    }


def update_consent(user: User, purpose: str, granted: bool, db: Session) -> bool:
    """Grant or revoke a consent purpose; writes to users table and consent audit log."""
    from app.db import ALLOWED_CONSENT_PURPOSES, Consent
    if purpose not in ALLOWED_CONSENT_PURPOSES:
        return False

    now = _now_utc()

    # Update single-source-of-truth flags on User record
    if purpose == "profile_memory":
        user.consent_profile = granted
        if not granted:
            # Revoke: delete all stored profile attributes immediately
            for attr in list(user.profile_attrs):
                db.delete(attr)
    elif purpose == "query_history":
        user.consent_history = granted
        if not granted:
            # Revoke immediately: delete all stored query history
            for q in list(user.queries):
                db.delete(q)

    # Write audit record
    log = Consent(
        user_id=user.id,
        purpose=purpose,
        granted_at=now if granted else None,
        revoked_at=now if not granted else None,
    )
    db.add(log)
    db.commit()
    logger.info("User %s consent '%s' set to %s", user.id, purpose, granted)
    return True
