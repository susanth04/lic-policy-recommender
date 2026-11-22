"""API contract tests for serve_fastapi.py."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_out"))

from serve_fastapi import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["verified_plans"] == 21
    assert data["total_plans"] == 47
    assert data["data_version"] == "v7"
    assert data["mode"] == "gbr_only"
    assert "abstention_calibration" in data
    assert data["abstention_calibration"]["weak_match_threshold"] == 1.2
    assert data["abstention_calibration"]["auroc"] == 0.907


def test_policies_endpoint():
    res = client.get("/policies?verified_only=true")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 21
    ids = {p["policy_id"] for p in data["policies"]}
    assert "LIC_734" in ids  # Jeevan Tarun
    assert "LIC_771" in ids  # Jeevan Utsav
    assert "LIC_876" in ids  # Digi Term


def test_out_of_scope_status():
    """Pension/annuity queries return OUT_OF_SCOPE with plain message and 0 recs."""
    res = client.post("/recommend", json={"query": "I'm 58 and want a pension plan with guaranteed income"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OUT_OF_SCOPE"
    assert len(data["recommendations"]) == 0
    assert "Pension / Annuity" in data["message"]


def test_needs_info_child_age():
    """Children's education query without child_age asks for child_age."""
    res = client.post("/recommend", json={"query": "parent saving for daughter education plan"})
    assert res.status_code == 200
    data = res.json()
    # Should include a clarification for child_age
    fields = [c["field"] for c in data.get("clarifications", [])]
    assert "child_age" in fields
    q_map = {c["field"]: c.get("question") for c in data.get("clarifications", [])}
    assert "How old is your child?" in q_map.get("child_age", "")


def test_needs_info_unspecified():
    """Vague query returns NEEDS_INFO."""
    res = client.post("/recommend", json={"query": "I want some kind of insurance"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NEEDS_INFO"
    assert len(data["clarifications"]) > 0


def test_no_eligible_policies():
    """Age 95 query correctly yields zero eligible policies under the gate."""
    res = client.post("/recommend", json={"query": "I am 95 years old looking for a low premium policy with 5 lakh sum assured for 10 years"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NO_ELIGIBLE_POLICIES"
    assert data["eligible_count"] == 0
    assert len(data["recommendations"]) == 0


def test_ok_recommendations_and_gate_integrity():
    """Standard term plan query returns OK and all recommendations are verified gate-eligible."""
    res = client.post("/recommend", json={
        "query": "I'm 30 and want a pure term plan with 25 lakh cover for 20 years",
        "top_k": 5
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert data["eligible_count"] >= 1
    assert len(data["recommendations"]) > 0

    # Ensure every single recommended policy is gate-eligible
    for r in data["recommendations"]:
        facts = r["explanation"]["gate_facts"]
        assert any("eligibility gate result: eligible" in f for f in facts)


def test_input_validation_422():
    """Pydantic rejects out-of-range ages, negative terms, or excessive query length."""
    # Age > 100
    res_age = client.post("/recommend", json={"query": "term plan", "customer_age": 150})
    assert res_age.status_code == 422

    # Negative term
    res_term = client.post("/recommend", json={"query": "term plan", "term": -5})
    assert res_term.status_code == 422

    # Query too long (>500 chars)
    res_long = client.post("/recommend", json={"query": "a" * 501})
    assert res_long.status_code == 422
