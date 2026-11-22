"""New FastAPI application entry point in app/.
This file replaces model_out/serve_fastapi.py as the production server.
model_out/ package files are NEVER edited — only imported.

Run from project root:
  uvicorn app.main:app --port 8000
"""
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model_out"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from recommender import WEAK_MATCH_BELOW

from app.auth import (
    delete_profile_attribute,
    get_confirmed_profile,
    get_consent_status,
    hard_delete_user,
    issue_device_token,
    resolve_token,
    update_consent,
    upsert_profile_attribute,
)
from app.db import ALLOWED_PROFILE_KEYS, Query, create_tables, get_db
from app.llm_client import LLMClient
from app.recommender_wrapper import RecommenderWrapper

logger = logging.getLogger("lic_recommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ----- Init ---------------------------------------------------------------

create_tables()
llm_client = LLMClient()
engine_wrapper = RecommenderWrapper(llm_client=llm_client)

# ----- App & middleware ---------------------------------------------------

app = FastAPI(
    title="LIC Policy Recommender API v2",
    description=(
        "Transparent brochure-verified LIC policy recommendations with strict eligibility gating. "
        "LLM layer is optional and off by default. Scores are relative rankings, not probabilities. "
        "This is an academic prototype, not financial or insurance advice."
    ),
    version="2.1.0",
)

cors_env = os.environ.get("LIC_CORS_ORIGINS", "*")
if cors_env.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response


# Simple in-memory rate limiter (per IP per minute)
_rate_counters: dict = {}

def _rate_limit(request: Request, limit: int = 60):
    ip = request.client.host if request.client else "unknown"
    now_minute = int(datetime.datetime.now(datetime.timezone.utc).timestamp() // 60)
    key = (ip, now_minute)
    _rate_counters[key] = _rate_counters.get(key, 0) + 1
    # Prune old keys
    for k in list(_rate_counters):
        if k[1] < now_minute - 2:
            del _rate_counters[k]
    if _rate_counters[key] > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment.")


# ----- Auth dependency ----------------------------------------------------

def get_current_user(
    x_device_token: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Optional auth: returns None when no token provided."""
    if not x_device_token:
        return None
    user = resolve_token(x_device_token, db)
    return user  # May be None if invalid/expired


def require_user(user=Depends(get_current_user)):
    """Auth dependency that raises 401 if no valid token."""
    if user is None:
        raise HTTPException(status_code=401, detail="Valid X-Device-Token required.")
    return user


# ----- Request/Response models -------------------------------------------

class Ask(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    customer_age: Optional[int] = Field(None, ge=0, le=100)
    child_age: Optional[int] = Field(None, ge=0, le=25)
    term: Optional[int] = Field(None, ge=1, le=100)
    sum_assured: Optional[int] = Field(None, ge=10_000, le=1_000_000_000)
    gender: Optional[str] = Field(None, pattern=r"^(?i)(male|female)?$")


class Identify(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class ConsentUpdate(BaseModel):
    purpose: str = Field(..., description="'profile_memory' or 'query_history'")
    granted: bool


class ProfilePut(BaseModel):
    value: str = Field(..., max_length=64)
    source: str = Field("user_stated", pattern=r"^(user_stated|form|llm_extracted_confirmed)$")


# ----- Helper: clean NaN/Inf ----------------------------------------------

def _clean(val):
    if val is None:
        return None
    try:
        import math
        if isinstance(val, (float, int)) and (math.isnan(val) or math.isinf(val)):
            return None
    except Exception:
        pass
    return val


# ----- Routes: Static & Health -------------------------------------------

@app.get("/", include_in_schema=False)
@app.get("/query", include_in_schema=False)
@app.get("/pipeline", include_in_schema=False)
@app.get("/report", include_in_schema=False)
@app.get("/account", include_in_schema=False)
@app.get("/policies-catalog", include_in_schema=False)
def dashboard():
    # Serve from model_out for now (HTML references relative paths there)
    idx = MODEL_DIR / "dashboard.html"
    if idx.exists():
        return FileResponse(idx)
    raise HTTPException(404, "dashboard.html not found")


@app.get("/dashboard.css", include_in_schema=False)
def dashboard_css():
    return FileResponse(MODEL_DIR / "dashboard.css", media_type="text/css")


@app.get("/tokens.css", include_in_schema=False)
def tokens_css():
    return FileResponse(MODEL_DIR / "tokens.css", media_type="text/css")


@app.get("/base.css", include_in_schema=False)
def base_css():
    return FileResponse(MODEL_DIR / "base.css", media_type="text/css")


@app.get("/health")
def health():
    verified_count = sum(1 for r in engine_wrapper.records if r.get("verified"))
    return {
        "ok": True,
        "status": "healthy",
        "mode": engine_wrapper.active_mode,
        "ranking": engine_wrapper.runtime,
        "fallback_active": engine_wrapper.runtime.get("fallback", False),
        "verified_plans": verified_count,
        "total_plans": len(engine_wrapper.records),
        "data_version": "v7",
        "abstention_calibration": {
            "weak_match_threshold": WEAK_MATCH_BELOW,
            "auroc": 0.907,
        },
        "llm_enabled": llm_client.enabled,
    }


class Identify(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


@app.post("/identify")
def identify(body: Identify):
    logger.info("POST /identify: query_len=%d", len(body.query))
    from intent_parser import parse_intent
    return {"query": body.query, "intent": parse_intent(body.query)}


@app.get("/metadata")
def metadata():
    import json
    metrics_path = MODEL_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "trained_on": metrics.get("trained_on", "2026-11-22"),
        "n_training_rows": metrics.get("n_training_rows", 2204),
        "n_labeled_queries": metrics.get("n_labeled_queries", 168),
        "n_policies_ranked": metrics.get("n_policies_ranked", 21),
        "selected_config": metrics.get("selected_config"),
        "validation": metrics.get("nested_cv_selected_procedure", {}),
        "abstention_auroc": 0.907,
        "weak_match_threshold": WEAK_MATCH_BELOW,
        "caveat": metrics.get(
            "caveat",
            "Validation metrics reflect agreement with evaluation rubric on unseen groups, not real-world quality.",
        ),
    }


@app.get("/policies")
def get_policies(verified_only: bool = True):
    records = [r for r in engine_wrapper.records if r.get("verified")] if verified_only else engine_wrapper.records
    return {
        "count": len(records),
        "policies": [
            {
                "policy_id": r["policy_id"],
                "policy_name": r["policy_name"],
                "category": r.get("category"),
                "verified": bool(r.get("verified")),
                "min_age": _clean(r.get("min_age")),
                "max_age": _clean(r.get("max_age")),
                "min_term": _clean(r.get("min_term")),
                "max_term": _clean(r.get("max_term")),
                "min_sum_assured": _clean(r.get("min_sum_assured")),
                "max_sum_assured": _clean(r.get("max_sum_assured")),
            }
            for r in records
        ],
    }


# ----- Routes: Auth -------------------------------------------------------

@app.post("/auth/device", status_code=201)
def issue_token(db: Session = Depends(get_db)):
    """Issue a new anonymous device token. No PII collected."""
    user_id, token = issue_device_token(db)
    return {
        "user_id": user_id,
        "device_token": token,
        "notice": (
            "Your anonymous session has been created. No personal information is required. "
            "If you grant consent, we will store: age, child age, gender, preferred term, "
            "and preferred sum assured. We will NOT store health details, income, or ID numbers. "
            "Query text may be sent to an external LLM provider only when ENABLE_LLM=true. "
            "Profile data is NEVER sent to the LLM. "
            "This system is designed to support DPDP-style privacy principles but has not been "
            "legally reviewed. You may delete all your data at any time via DELETE /me."
        ),
    }


# ----- Routes: Consent ----------------------------------------------------

CONSENT_NOTICE_LLM = (
    "If ENABLE_LLM=true on this server, your query text (not profile data) may be sent to "
    "an external LLM provider (e.g. OpenAI) to improve intent extraction and explanations. "
    "Profile data (age, gender, etc.) is NEVER sent to the LLM."
)


@app.post("/me/consent")
def set_consent(
    body: ConsentUpdate,
    user=Depends(require_user),
    db: Session = Depends(get_db),
):
    """Grant or revoke consent for a specific purpose. Single source of truth."""
    ok = update_consent(user, body.purpose, body.granted, db)
    if not ok:
        raise HTTPException(400, detail=f"Unknown consent purpose '{body.purpose}'.")
    db.refresh(user)
    return {
        "purpose": body.purpose,
        "granted": body.granted,
        "consent_status": get_consent_status(user),
        "notice": CONSENT_NOTICE_LLM,
        "note": (
            "Revoking consent immediately deletes all data for that purpose. "
            "This system is designed to support DPDP-style privacy principles but has not been legally reviewed."
        ),
    }


@app.delete("/me/consent/{purpose}", status_code=200)
def revoke_consent(
    purpose: str,
    user=Depends(require_user),
    db: Session = Depends(get_db),
):
    """Revoke a specific consent purpose; deletes associated data immediately."""
    ok = update_consent(user, purpose, granted=False, db=db)
    if not ok:
        raise HTTPException(400, detail=f"Unknown consent purpose '{purpose}'.")
    db.refresh(user)
    return {
        "revoked": purpose,
        "consent_status": get_consent_status(user),
    }


# ----- Routes: Profile ----------------------------------------------------

@app.get("/me/profile")
def get_profile(user=Depends(require_user)):
    return {
        "user_id": user.id,
        "consent_status": get_consent_status(user),
        "profile": get_confirmed_profile(user),
        "allowed_keys": sorted(ALLOWED_PROFILE_KEYS),
        "notice": (
            "Stored fields: age, child_age, gender, preferred_term, preferred_sum_assured only. "
            "Health, income, and ID numbers are never stored."
        ),
    }


@app.put("/me/profile/{key}")
def put_profile_attribute(
    key: str,
    body: ProfilePut,
    user=Depends(require_user),
    db: Session = Depends(get_db),
):
    if not user.consent_profile:
        raise HTTPException(403, detail="Profile memory consent not granted. POST /me/consent first.")
    ok = upsert_profile_attribute(user, key, body.value, body.source, db)
    if not ok:
        raise HTTPException(400, detail=f"Key '{key}' is not a permitted profile field.")
    db.refresh(user)
    return {
        "updated": key,
        "profile": get_confirmed_profile(user),
    }


@app.delete("/me/profile/{key}")
def delete_profile_key(
    key: str,
    user=Depends(require_user),
    db: Session = Depends(get_db),
):
    ok = delete_profile_attribute(user, key, db)
    if not ok:
        raise HTTPException(404, detail=f"Key '{key}' not found in your profile.")
    db.refresh(user)
    return {"deleted": key, "profile": get_confirmed_profile(user)}


# ----- Routes: History ----------------------------------------------------

@app.get("/me/history")
def get_history(
    limit: int = 20,
    user=Depends(require_user),
    db: Session = Depends(get_db),
):
    if not user.consent_history:
        return {"message": "Query history consent not granted.", "queries": []}
    rows = (
        db.query(Query)
        .filter(Query.user_id == user.id)
        .order_by(Query.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(rows),
        "queries": [
            {
                "id": q.id,
                "created_at": q.created_at.isoformat() + "Z",
                "query_text": q.query_text,
                "status": q.status,
                "top_policy_ids": json.loads(q.top_policy_ids) if q.top_policy_ids else [],
            }
            for q in rows
        ],
    }


@app.delete("/me/history/{query_id}")
def delete_history_item(
    query_id: int,
    user=Depends(require_user),
    db: Session = Depends(get_db),
):
    q = db.query(Query).filter(Query.id == query_id, Query.user_id == user.id).first()
    if not q:
        raise HTTPException(404, detail="Query record not found.")
    db.delete(q)
    db.commit()
    return {"deleted_query_id": query_id}


# ----- Routes: Data deletion & export ------------------------------------

@app.delete("/me", status_code=200)
def delete_my_data(user=Depends(require_user), db: Session = Depends(get_db)):
    user_id = user.id
    hard_delete_user(user, db)
    return {
        "deleted": True,
        "user_id": user_id,
        "message": "All your data has been permanently deleted.",
    }


@app.get("/me/export")
def export_my_data(user=Depends(require_user), db: Session = Depends(get_db)):
    db.refresh(user)
    return {
        "user_id": user.id,
        "created_at": user.created_at.isoformat() + "Z",
        "consent_status": get_consent_status(user),
        "profile": get_confirmed_profile(user),
        "queries": [
            {
                "id": q.id,
                "created_at": q.created_at.isoformat() + "Z",
                "query_text": q.query_text,
                "status": q.status,
                "top_policy_ids": json.loads(q.top_policy_ids) if q.top_policy_ids else [],
            }
            for q in user.queries
        ],
    }


# ----- Routes: Recommend -------------------------------------------------

@app.post("/recommend")
def recommend(
    body: Ask,
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _rate_limit(request, limit=60)

    form_fields = {
        k: getattr(body, k)
        for k in ("customer_age", "child_age", "term", "sum_assured", "gender")
        if getattr(body, k) is not None
    }

    # Fetch confirmed profile only if user is authenticated AND profile consent is on
    confirmed_profile = None
    if user and user.consent_profile:
        confirmed_profile = get_confirmed_profile(user)

    logger.info(
        "POST /recommend: query_len=%d, top_k=%d, form_fields=%s, auth=%s, llm=%s",
        len(body.query), body.top_k, list(form_fields.keys()),
        bool(user), llm_client.enabled,
    )

    result = engine_wrapper.recommend(
        query=body.query,
        top_k=body.top_k,
        form_fields=form_fields,
        confirmed_profile=confirmed_profile,
    )

    # Store query history ONLY if:
    #   1. user is authenticated
    #   2. consent_history is True
    #   3. status is NOT OUT_OF_SCOPE
    status = result.get("status", "")
    if user and user.consent_history and status != "OUT_OF_SCOPE":
        top_ids = [r["policy_id"] for r in result.get("recommendations", [])]
        q = Query(
            user_id=user.id,
            created_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
            query_text=body.query,
            parsed_intent_json=json.dumps(result.get("intent", {})),
            status=status,
            top_policy_ids=json.dumps(top_ids),
        )
        db.add(q)
        db.commit()

    return result
