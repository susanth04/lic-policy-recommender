import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_catalogue_uses_plan_ids_and_merges_only_checked_eligibility():
    policies = pd.read_csv(ROOT / "data" / "policies.csv")

    assert len(policies) == 47
    assert policies["policy_id"].is_unique
    assert policies.apply(
        lambda row: row["policy_id"] == f"LIC_{row['plan_number']}", axis=1
    ).all()
    assert policies["verified"].sum() == 16
    assert not any(column.startswith("eligibility_") for column in policies.columns)
    verified_rows = policies.loc[policies["verified"]]
    assert verified_rows["verification_status"].str.startswith("Brochure eligibility verified: ").all()
    assert verified_rows["description"].fillna("").str.strip().ne("").all()
    assert verified_rows["description_source_page_section"].fillna("").str.strip().ne("").all()
    assert {
        "description",
        "description_source_page_section",
    }.issubset(policies.columns)
    for column in (
        "financial_goal",
        "risk_tolerance_inferred",
        "premium_level_inferred",
    ):
        assert policies[column].fillna("").astype(str).str.strip().eq("").all()
    assert policies.loc[~policies["verified"], "verified_scope"].fillna("").eq("").all()
    assert policies.loc[policies["verified"], "verified_scope"].fillna("").ne("").all()
    assert policies["category_verified"].eq(False).all()
    assert "LIC_906" not in set(policies["policy_id"])
    assert "LIC_PMJJBY" not in set(policies["policy_id"])
    assert "LIC_854" not in set(policies["policy_id"])
    unresolved = policies.loc[policies["status"].eq("unresolved")]
    assert set(unresolved["plan_number"]) == {867, 873}
    assert not unresolved["recommendation_pool"].astype(bool).any()
    assert not policies.loc[policies["status"].eq("withdrawn"), "recommendation_pool"].astype(bool).any()
    assert {
        "min_entry_age_days",
        "max_maturity_age",
        "min_sum_assured",
        "max_sum_assured",
        "rules_json",
        "policy_url",
        "brochure_url",
    }.issubset(policies.columns)
    eligibility = pd.read_csv(ROOT / "data" / "source" / "eligibility_from_brochures.csv")
    assert len(eligibility) == 16
    assert eligibility["description"].str.strip().ne("").all()
    assert eligibility["description_source_page_section"].str.strip().ne("").all()
    descriptions = eligibility.set_index("plan_no")
    for row in verified_rows.itertuples(index=False):
        source = descriptions.loc[row.plan_number]
        assert row.description == source["description"]
        assert row.description_source_page_section == source["description_source_page_section"]
    assert eligibility["policy_url"].str.startswith("https://licindia.in/").all()
    assert eligibility["brochure_url"].str.startswith("https://licindia.in/").all()
    assert set(eligibility.loc[~eligibility["eligibility_verified"], "plan_no"]) == set()
    unmatched = policies.loc[~policies["plan_number"].isin(eligibility["plan_no"])]
    assert unmatched[["min_age", "max_age", "min_term", "max_term"]].isna().all().all()


def test_added_plan_identifiers_and_notes_are_preserved():
    policies = pd.read_csv(ROOT / "data" / "policies.csv").set_index("policy_id")

    assert policies.loc["LIC_770", "status"] == "active"
    assert policies.loc["LIC_879", "plan_number"] == 879
    assert policies.loc["LIC_736", "supersedes"] == 936
    assert policies.loc["LIC_736", "withdrawn_versions"] == "836/512N304V01;936/512N304V02"
    assert "854/512N333V01" in policies.loc["LIC_954", "withdrawn_versions"]
    assert policies.loc["LIC_954", "supersedes"] == 854
    assert policies.loc["LIC_939", "status"] == "withdrawn"
    assert policies.loc["LIC_873", "status"] == "unresolved"
    assert policies.loc["LIC_867", "status"] == "unresolved"
    assert policies.loc["LIC_875", "verified"] == True
    assert policies.loc["LIC_875", "min_term"] == 15
    assert policies.loc["LIC_736", "verified"] == True
    superseding_plans = {
        936: 736,
        933: 733,
        914: 714,
        915: 715,
        854: 954,
        948: 748,
        932: 732,
        934: 734,
        945: 745,
        871: 771,
        858: 758,
        852: 752,
        849: 749,
        951: 751,
    }
    for old_number, current_number in superseding_plans.items():
        assert policies.loc[f"LIC_{current_number}", "supersedes"] == old_number


def test_queries_are_synthetic_and_not_prelabelled():
    queries = pd.read_csv(ROOT / "artifacts" / "queries.csv")

    assert len(queries) == 18
    assert queries["provenance"].eq("SYNTHETIC").all()
    assert queries.groupby("persona")["group_id"].nunique().eq(1).all()
    assert queries["group_id"].nunique() < len(queries)
    assert "expected_goals" not in queries.columns


def test_splits_are_group_disjoint_and_template_is_ungraded():
    splits = json.loads((ROOT / "artifacts" / "splits.json").read_text(encoding="utf-8"))
    status = json.loads((ROOT / "results" / "data_status.json").read_text(encoding="utf-8"))
    groups = splits["group_split"]["groups"]

    assert not (set(groups["train"]) & set(groups["validation"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["validation"]) & set(groups["test"]))
    assert status["template_rows"] == 324
    assert status["template_graded_rows"] == 0
    assert status["template_annotators_with_grades"] == []
    assert status["human_graded_rows"] == 0
    assert status["human_label_source"] is None
    assert status["excluded_human_graded_rows"] == 0


def test_synthetic_smoke_labels_are_marked_for_smoke_training_and_evaluation():
    labels = pd.read_csv(ROOT / "labels" / "relevance_labels.csv")
    status = json.loads((ROOT / "results" / "data_status.json").read_text(encoding="utf-8"))

    assert status["catalogue_source"] == "data/policies.csv"
    assert labels.columns.tolist() == ["query_id", "group_id", "policy_id", "grade", "annotator"]
    assert len(labels) == 324
    assert labels["policy_id"].str.fullmatch(r"LIC_\d+").all()
    assert "LIC_954" in set(labels["policy_id"])
    assert "LIC_854" not in set(labels["policy_id"])
    assert labels["grade"].between(0, 3).all()
    assert labels["annotator"].eq("SYNTHETIC_SMOKE_TEST").all()
    assert status["synthetic_smoke_label_rows"] == 324
    assert status["human_graded_rows"] == 0
    assert status["synthetic_labels_used_for_training"] is True
    assert status["synthetic_labels_used_for_evaluation"] is True
    assert status["evidentiary_status"] == "SYNTHETIC_SMOKE_TEST_NOT_EVIDENCE_OF_REAL_RELEVANCE_OR_KG_BENEFIT"
    synthetic_artifact = pd.read_parquet(ROOT / "artifacts" / "synthetic_smoke_labels.parquet")
    assert synthetic_artifact["provenance"].eq("SYNTHETIC").all()
    assert synthetic_artifact["annotator"].eq("SYNTHETIC_SMOKE_TEST").all()
    withdrawn_ids = set(
        pd.read_csv(ROOT / "data" / "policies.csv").loc[
            lambda frame: frame["status"].eq("withdrawn"), "policy_id"
        ]
    )
    assert labels.loc[labels["policy_id"].isin(withdrawn_ids), "grade"].eq(0).all()


def test_human_reviewed_labels_are_separate_from_synthetic_training():
    queries = pd.read_csv(ROOT / "artifacts" / "human_eval_queries.csv")
    labels = pd.read_parquet(ROOT / "artifacts" / "human_eval_labels.parquet")
    status = json.loads(
        (ROOT / "results" / "human_evaluation_data_status.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(queries) == 60
    assert len(labels) == 549
    assert labels["grade"].between(0, 3).all()
    assert labels["annotator_type"].eq("human").all()
    assert labels["provenance"].eq("HUMAN_REVIEWED_BY_USER").all()
    assert labels["annotator"].eq("HUMAN_UNSPECIFIED").all()
    assert status["graded_query_count"] == 55
    assert status["training_labels"] is False
