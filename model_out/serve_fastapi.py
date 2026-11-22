"""FastAPI serving backend for the LIC policy recommender.
Run: uvicorn serve_fastapi:app --port 8000
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from intent_parser import parse_intent
from recommender import Recommender, WEAK_MATCH_BELOW

# Setup privacy-preserving logging (omits raw user query text by default)
logger = logging.getLogger("lic_recommender")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MODEL_DIR = Path(__file__).resolve().parent
engine = Recommender(MODEL_DIR)

app = FastAPI(
    title="LIC Policy Recommender API",
    description="Transparent brochure-verified LIC policy recommendations with strict eligibility gating.",
    version="2.0.0"
)

# Configurable CORS support
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


class Ask(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User search query in free text (max 500 chars)")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")
    customer_age: Optional[int] = Field(None, ge=0, le=100, description="Customer entry age in years (0-100)")
    child_age: Optional[int] = Field(None, ge=0, le=25, description="Child age in years for child plans (0-25)")
    term: Optional[int] = Field(None, ge=1, le=100, description="Policy term in years (1-100)")
    sum_assured: Optional[int] = Field(None, ge=10_000, le=1_000_000_000, description="Sum assured in rupees")
    gender: Optional[str] = Field(None, pattern=r"^(?i)(male|female)?$", description="Policyholder gender")


class Identify(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


@app.get("/", include_in_schema=False)
def dashboard():
    index_file = MODEL_DIR / "dashboard.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(index_file)


@app.get("/dashboard.css", include_in_schema=False)
def dashboard_styles():
    return FileResponse(MODEL_DIR / "dashboard.css", media_type="text/css")


@app.get("/tokens.css", include_in_schema=False)
def theme_tokens():
    return FileResponse(MODEL_DIR / "tokens.css", media_type="text/css")


@app.get("/base.css", include_in_schema=False)
def theme_base():
    return FileResponse(MODEL_DIR / "base.css", media_type="text/css")


@app.get("/health")
def health():
    verified_count = sum(1 for r in engine.records if r.get("verified"))
    return {
        "ok": True,
        "status": "healthy",
        "mode": engine.active_mode,
        "ranking": engine.runtime,
        "fallback_active": engine.runtime.get("fallback", False),
        "verified_plans": verified_count,
        "total_plans": len(engine.records),
        "trained_on": engine.meta.get("trained_on", "2026-11-22"),
        "data_version": "v7",
        "abstention_calibration": {
            "weak_match_threshold": WEAK_MATCH_BELOW,
            "auroc": 0.907,
            "match_recall": 1.0,
            "abstention_precision": 1.0,
            "abstention_recall": 0.286
        }
    }


@app.get("/metadata")
def metadata():
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    return {
        "trained_on": metrics.get("trained_on", "2026-11-22"),
        "n_training_rows": metrics.get("n_training_rows", 2204),
        "n_labeled_queries": metrics.get("n_labeled_queries", 168),
        "n_policies_ranked": metrics.get("n_policies_ranked", 21),
        "selected_config": metrics.get("selected_config"),
        "validation": metrics.get("nested_cv_selected_procedure", {}),
        "abstention_auroc": 0.907,
        "weak_match_threshold": WEAK_MATCH_BELOW,
        "selection_note": metrics.get("hybrid_stages_4_to_7", {}).get("selection_note", "gbr_only mode selected"),
        "caveat": metrics.get("caveat", "Validation metrics describe agreement with the grading scheme on unseen persona groups, not real-world quality."),
    }


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


@app.get("/policies")
def get_policies(verified_only: bool = True):
    records = [r for r in engine.records if r.get("verified")] if verified_only else engine.records
    facts = []
    for r in records:
        facts.append({
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
            "policy_url": r.get("policy_url") if isinstance(r.get("policy_url"), str) else None,
            "brochure_url": r.get("brochure_url") if isinstance(r.get("brochure_url"), str) else None,
        })
    return {"count": len(facts), "policies": facts}


@app.post("/identify")
def identify(body: Identify):
    # Log request length without storing query text
    logger.info("POST /identify: query_length=%d", len(body.query))
    return {
        "query": body.query,
        "intent": parse_intent(body.query)
    }


@app.post("/recommend")
def recommend(body: Ask):
    overrides = {
        k: getattr(body, k)
        for k in ("customer_age", "child_age", "term", "sum_assured", "gender")
        if getattr(body, k) is not None
    }
    # Log privacy-safe metadata
    logger.info(
        "POST /recommend: query_length=%d, top_k=%d, overridden_fields=%s",
        len(body.query), body.top_k, list(overrides.keys())
    )

    result = engine.recommend(body.query, top_k=body.top_k, overrides=overrides)
    # Echo parsed intent vs overrides for transparency and correction
    result["understood_details"] = {
        "parsed_from_text": parse_intent(body.query),
        "structured_overrides": overrides,
        "effective_intent": result["intent"]
    }
    return result
