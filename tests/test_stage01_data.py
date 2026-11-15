import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_export_preserves_claims_without_verifying_them():
    policies = pd.read_csv(ROOT / "data" / "policies.csv")

    assert len(policies) == 47
    assert not policies["verified"].astype(bool).any()
    assert policies.loc[policies["source_note"].isna(), "claimed_verification"].notna().all()
    additions = policies.loc[policies["source_note"].fillna("").ne("")]
    assert len(additions) == 27
    assert additions[["min_age", "max_age", "min_term", "max_term"]].isna().all().all()


def test_added_plan_identifiers_and_notes_are_preserved():
    policies = pd.read_csv(ROOT / "data" / "policies.csv").set_index("policy_id")

    assert policies.loc["LIC_770", "plan_number"] == "770"
    assert policies.loc["LIC_770", "uin"] == "512N397V01"
    assert policies.loc["LIC_770", "source_note"] == "Press release; on sale from 7 Sep 2026"
    assert policies.loc["LIC_SMART_PENSION", "uin"] == "512N386V01"
    assert pd.isna(policies.loc["LIC_SMART_PENSION", "plan_number"])


def test_queries_are_synthetic_and_not_prelabelled():
    queries = pd.read_csv(ROOT / "artifacts" / "queries.csv")

    assert len(queries) == 18
    assert queries["provenance"].eq("SYNTHETIC").all()
    assert queries["group_id"].is_unique
    assert "expected_goals" not in queries.columns


def test_splits_are_group_disjoint_and_template_is_ungraded():
    splits = json.loads((ROOT / "artifacts" / "splits.json").read_text(encoding="utf-8"))
    status = json.loads((ROOT / "results" / "data_status.json").read_text(encoding="utf-8"))
    groups = splits["group_split"]["groups"]

    assert not (set(groups["train"]) & set(groups["validation"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["validation"]) & set(groups["test"]))
    assert status["template_rows"] == 360
    assert status["template_graded_rows"] == 0
    assert status["template_annotators_with_grades"] == []
