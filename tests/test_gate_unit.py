"""Unit tests for brochure-backed eligibility gate rules."""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_out"))

from eligibility_gate import evaluate_eligibility

@pytest.fixture(scope="module")
def catalogue():
    df = pd.read_csv(ROOT / "model_out" / "catalogue.csv")
    return {r["policy_id"]: r for r in df.to_dict("records")}


def test_child_plans_child_age_rules(catalogue):
    """Jeevan Tarun (LIC_734) and Children's Money Back (LIC_732) evaluate on child_age (0-12)."""
    tarun = catalogue["LIC_734"]
    children_mb = catalogue["LIC_732"]

    # 1. child_age = 6 -> eligible
    intent_valid = {"child_age": 6, "customer_age": 35, "sum_assured": 200_000}
    res_tarun = evaluate_eligibility(tarun, intent_valid)
    res_cmb = evaluate_eligibility(children_mb, intent_valid)
    assert res_tarun["status"] == "eligible", f"Expected eligible, got: {res_tarun}"
    assert res_cmb["status"] == "eligible", f"Expected eligible, got: {res_cmb}"

    # 2. child_age missing -> unknown (child age is required)
    intent_missing = {"customer_age": 35, "sum_assured": 200_000}
    res_missing_tarun = evaluate_eligibility(tarun, intent_missing)
    assert res_missing_tarun["status"] == "unknown"
    assert "child age is required" in res_missing_tarun["reasons"]

    # 3. child_age = 13 -> ineligible (max child entry age is 12)
    intent_ineligible = {"child_age": 13, "customer_age": 40, "sum_assured": 200_000}
    res_ineligible = evaluate_eligibility(tarun, intent_ineligible)
    assert res_ineligible["status"] == "ineligible"
    assert "age exceeds the maximum entry age" in res_ineligible["reasons"]


def test_bima_shree_max_entry_age_by_term(catalogue):
    """LIC's Bima Shree (LIC_748) has term-dependent max entry age."""
    bima_shree = catalogue["LIC_748"]

    # Term 14: max entry age is 55
    res_55_t14 = evaluate_eligibility(bima_shree, {"customer_age": 55, "term": 14, "sum_assured": 1_000_000})
    assert res_55_t14["status"] == "eligible"

    res_56_t14 = evaluate_eligibility(bima_shree, {"customer_age": 56, "term": 14, "sum_assured": 1_000_000})
    assert res_56_t14["status"] == "ineligible"
    assert "age exceeds the maximum entry age" in res_56_t14["reasons"]

    # Term 28: max entry age is 41
    res_41_t28 = evaluate_eligibility(bima_shree, {"customer_age": 41, "term": 28, "sum_assured": 1_000_000})
    assert res_41_t28["status"] == "eligible"

    res_42_t28 = evaluate_eligibility(bima_shree, {"customer_age": 42, "term": 28, "sum_assured": 1_000_000})
    assert res_42_t28["status"] == "ineligible"
    assert "age exceeds the maximum entry age" in res_42_t28["reasons"]

    # Disallowed term (e.g. 15) -> ineligible
    res_bad_term = evaluate_eligibility(bima_shree, {"customer_age": 30, "term": 15, "sum_assured": 1_000_000})
    assert res_bad_term["status"] == "ineligible"
    assert "requested term is not an allowed term" in res_bad_term["reasons"]


def test_jeevan_utsav_term_unknown(catalogue):
    """Jeevan Utsav (LIC_771) is a Whole Life plan with unverified fixed term limits; stating a term yields unknown."""
    utsav = catalogue["LIC_771"]

    # Without term: eligible
    res_no_term = evaluate_eligibility(utsav, {"customer_age": 30, "sum_assured": 500_000})
    assert res_no_term["status"] == "eligible"

    # With term specified: unknown (term_verified=False)
    res_with_term = evaluate_eligibility(utsav, {"customer_age": 30, "term": 20, "sum_assured": 500_000})
    assert res_with_term["status"] == "unknown"
    assert "policy-term limits are not brochure-verified" in res_with_term["reasons"]


def test_money_back_25_age_term_limits(catalogue):
    """New Money Back Plan - 25 Years (LIC_721) requires exactly term 25 and entry age 13-45."""
    mb25 = catalogue["LIC_721"]

    # Exactly term 25 and valid age: eligible (min sum assured is 200,000)
    res_valid = evaluate_eligibility(mb25, {"customer_age": 30, "term": 25, "sum_assured": 200_000})
    assert res_valid["status"] == "eligible"

    # Term 20 -> ineligible
    res_wrong_term = evaluate_eligibility(mb25, {"customer_age": 30, "term": 20, "sum_assured": 100_000})
    assert res_wrong_term["status"] == "ineligible"
    assert "requested term is below the minimum" in res_wrong_term["reasons"]

    # Age 12 -> ineligible
    res_young = evaluate_eligibility(mb25, {"customer_age": 12, "term": 25, "sum_assured": 100_000})
    assert res_young["status"] == "ineligible"
    assert "age is below the minimum entry age" in res_young["reasons"]

    # Age 46 -> ineligible
    res_old = evaluate_eligibility(mb25, {"customer_age": 46, "term": 25, "sum_assured": 100_000})
    assert res_old["status"] == "ineligible"
    assert "age exceeds the maximum entry age" in res_old["reasons"]
