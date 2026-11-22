"""Regression probe tests verifying expected behaviors for key benchmark queries."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_out"))

from serve_fastapi import app, engine
from eligibility_gate import evaluate_eligibility

client = TestClient(app)


def test_probe_1_pure_term_plan_30_25lakh_20yr():
    """Probe 1: pure term plan age 30 25 lakh 20 years."""
    res = client.post("/recommend", json={"query": "I am 30 and want a pure term plan with 25 lakh cover for 20 years"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert data["eligible_count"] >= 1
    # Check top rec category
    top = data["recommendations"][0]
    assert top["category"] == "Term Assurance"
    # Verify gate eligibility for each recommendation
    for rec in data["recommendations"]:
        cat_rec = engine.by_id[rec["policy_id"]]
        gate_res = evaluate_eligibility(cat_rec, data["intent"])
        assert gate_res["status"] == "eligible"


def test_probe_2_female_savings_plan_35():
    """Probe 2: female savings plan age 35."""
    res = client.post("/recommend", json={"query": "I am a 35 year old woman looking for a savings plan"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OK"
    assert data["eligible_count"] >= 1
    # At least one female or general savings plan
    pids = [r["policy_id"] for r in data["recommendations"]]
    # Should recommend Endowment or Money Back plans
    for rec in data["recommendations"]:
        assert rec["category"] in ("Endowment", "Money Back", "Whole Life")


def test_probe_3_parent_daughter_education():
    """Probe 3: parent saving for daughter's education.
    Without child_age: asks for child_age.
    With child_age=6: child plans appear (Jeevan Tarun LIC_734 or Children Money Back LIC_732).
    """
    # Part A: Without child age
    res_a = client.post("/recommend", json={"query": "I am saving for my daughter's education"})
    assert res_a.status_code == 200
    data_a = res_a.json()
    clarify_fields = [c["field"] for c in data_a.get("clarifications", [])]
    assert "child_age" in clarify_fields

    # Part B: With child_age provided
    res_b = client.post("/recommend", json={
        "query": "I am saving for my 6-year-old daughter's college education with 10 lakh cover",
        "child_age": 6
    })
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert data_b["status"] == "OK"
    pids_b = [r["policy_id"] for r in data_b["recommendations"]]
    # Child plan(s) must appear in recommendations
    assert any(pid in pids_b for pid in ("LIC_734", "LIC_732")), f"Expected child plans, got: {pids_b}"


def test_probe_4_vague_insurance_query():
    """Probe 4: 'I want some kind of insurance' -> status NEEDS_INFO."""
    res = client.post("/recommend", json={"query": "I want some kind of insurance"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NEEDS_INFO"


def test_probe_5_age_95_boundary():
    """Probe 5: age 95 -> status NO_ELIGIBLE_POLICIES."""
    res = client.post("/recommend", json={"query": "I am 95 years old looking for insurance"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NO_ELIGIBLE_POLICIES"
    assert len(data["recommendations"]) == 0


def test_probe_6_pension_query_out_of_scope():
    """Probe 6: pension query -> status OUT_OF_SCOPE."""
    res = client.post("/recommend", json={"query": "I want a pension scheme for regular monthly pension after retirement"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OUT_OF_SCOPE"
    assert len(data["recommendations"]) == 0
    assert "Pension / Annuity" in data["message"]


def test_probe_7_contradictory_query_abstention():
    """Probe 7: 'zero risk very high returns guaranteed' -> should not produce a confident recommendation."""
    res = client.post("/recommend", json={"query": "zero risk very high returns guaranteed market growth"})
    assert res.status_code == 200
    data = res.json()
    # It must flag contradiction or weak_match (abstention)
    assert data["weak_match"] is True or len(data["intent"].get("contradictions", [])) > 0
