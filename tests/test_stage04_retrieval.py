import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retrieval_state():
    notebook = nbformat.read(ROOT / "04_retrieval_fusion_rerank.ipynb", as_version=4)
    state = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(ROOT / "04_retrieval_fusion_rerank.ipynb"), "exec"), state)
    return state


def test_synthetic_ranking_is_eligible_monotone_and_validation_tuned(retrieval_state):
    results = json.loads((ROOT / "results" / "retrieval_smoke.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT / "results" / "weights_smoke.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "results" / "synthetic_smoke_evaluation.json").read_text(encoding="utf-8"))

    assert results["eligibility_violations"] == 0
    assert results["eligibility_violation_rate"] == 0.0
    assert results["provenance"] == "SYNTHETIC"
    assert all(
        recommendation["eligibility"] == "eligible"
        and recommendation["score_kind"] == "ranking_score_not_probability"
        for row in results["queries"]
        for recommendation in row["recommendations"]
    )
    assert weights["status"] == "COMPLETED_SYNTHETIC_SMOKE_TEST"
    assert 0 <= weights["selected_alpha"] <= 1
    assert 0 <= weights["selected_beta"] <= 1
    assert len(weights["sweep"]) == 441
    assert weights["human_labels_used"] is False
    for model_sweep in weights["models"].values():
        assert model_sweep["selection_method"] in {
            "SYNTHETIC_VALIDATION_SWEEP",
            "UNTUNED_DOCUMENTED_DEFAULT",
        }
        for row in model_sweep["sweep"]:
            if (
                row["beta"] == 0
                and row["scored_validation_list_count"] > 0
                and row["all_equal_final_score_lists"]
                == row["scored_validation_list_count"]
            ):
                assert row["informative"] is False
                assert row["selection_note"] == "NON_INFORMATIVE_ALL_SCORES_TIED"
    assert evaluation["provenance"] == "SYNTHETIC"
    assert evaluation["test_metric_status"] == "NOT_EVALUABLE_NO_ELIGIBLE_RELEVANT_JUDGMENTS"
    assert evaluation["test_results"]["fine_tuned"]["eligible_relevant_judgments"] == 0
    assert set(evaluation["test_customer_profiles"]).isdisjoint(
        evaluation["train_customer_profiles"]
    )
    assert set(evaluation["test_customer_profiles"]).isdisjoint(
        evaluation["validation_customer_profiles"]
    )
    assert set(evaluation["test_results"]) == {"baseline", "fine_tuned"}


def test_offline_fallbacks_and_repeatable_ranking(retrieval_state):
    assert retrieval_state["minmax"]([0.0, 0.0]) == pytest.approx([0.5, 0.5])
    fallback = retrieval_state["backend_plan"](False, False, False, False)
    assert fallback["vector"] == "numpy_exact"
    assert fallback["encoder"] == "tfidf_fallback"
    assert fallback["cross_encoder"] == "dense_ranking_score_fallback"
    assert fallback["network"] == "disabled_local_only"

    query = retrieval_state["queries_df"].iloc[0]
    intents = retrieval_state["intents"]
    first = retrieval_state["rank_query"](query.raw_query, intents[query.query_id]["intent"])
    second = retrieval_state["rank_query"](query.raw_query, intents[query.query_id]["intent"])
    first_ids = [record["policy_id"] for record in first["recommendations"]]
    second_ids = [record["policy_id"] for record in second["recommendations"]]

    assert first_ids == second_ids
    scores = [record["final_ranking_score"] for record in first["recommendations"]]
    assert all(left >= right for left, right in zip(scores, scores[1:]))
    assert all(score > 0.0 for score in scores)


def test_score_ties_use_dense_then_bm25_then_cross_encoder(retrieval_state):
    order = retrieval_state["rank_order"](
        ["LIC_2", "LIC_1"],
        [0.5, 0.5],
        [0.2, 0.9],
        [0.9, 0.1],
        [0.9, 0.1],
    )
    assert order == [1, 0]

    order = retrieval_state["rank_order"](
        ["LIC_2", "LIC_1"],
        [0.5, 0.5],
        [0.9, 0.9],
        [0.1, 0.1],
        [0.2, 0.8],
    )
    assert order == [1, 0]


def test_tied_rank_metrics_and_collapse_guard(retrieval_state):
    rank_list = lambda scores: {
        "recommendations": [
            {"final_ranking_score": score} for score in scores
        ]
    }

    metrics = retrieval_state["summarize_rank_ties"](
        [rank_list([0.5, 0.5]), rank_list([0.9, 0.4])]
    )
    assert metrics["tied_rank_fraction"] == pytest.approx(0.5)
    assert metrics["all_equal_list_fraction"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="20%"):
        retrieval_state["assert_tie_collapse_limit"](
            [rank_list([1, 1]), rank_list([1, 1]), rank_list([1, 2]), rank_list([1, 2]), rank_list([1, 2])]
        )


def test_ranker_baselines_are_label_free(retrieval_state):
    query = retrieval_state["queries_df"].iloc[0]
    intent = retrieval_state["intents"][query.query_id]["intent"]
    id_order = retrieval_state["rank_query"](
        query.raw_query, intent, query_id=query.query_id, ranker="id_order"
    )
    bm25_only = retrieval_state["rank_query"](
        query.raw_query, intent, query_id=query.query_id, ranker="bm25_only"
    )

    assert [row["policy_id"] for row in id_order["recommendations"]] == sorted(
        row["policy_id"] for row in id_order["recommendations"]
    )
    assert len({row["final_ranking_score"] for row in id_order["recommendations"]}) == len(
        id_order["recommendations"]
    )
    bm25_scores = [row["final_ranking_score"] for row in bm25_only["recommendations"]]
    assert all(left >= right for left, right in zip(bm25_scores, bm25_scores[1:]))


def test_missing_brochure_kg_fields_are_excluded_from_fusion(retrieval_state):
    policy = retrieval_state["policy_by_id"]["LIC_714"]
    intent = {
        "all_goals": ["Life Cover"],
        "risk_tolerance": "Low",
        "premium_level": "Low",
    }

    assert retrieval_state["kg_features"](policy, intent) == {}
    assert retrieval_state["kg_preference"](
        policy, intent, {"goal": 1 / 3, "risk": 1 / 3, "premium": 1 / 3}
    ) is None

    query = retrieval_state["queries_df"].iloc[0]
    query_intent = retrieval_state["intents"][query.query_id]["intent"]
    without_kg = retrieval_state["rank_query"](
        query.raw_query, query_intent, beta=0, query_id=query.query_id
    )
    with_kg_weight = retrieval_state["rank_query"](
        query.raw_query, query_intent, beta=1, query_id=query.query_id
    )
    assert [r["policy_id"] for r in without_kg["recommendations"]] == [
        r["policy_id"] for r in with_kg_weight["recommendations"]
    ]


def test_all_eligible_policies_are_ranked_before_display_truncation(retrieval_state):
    intent = {"customer_age": 40, "term": None, "all_goals": []}
    result = retrieval_state["rank_query"]("life cover", intent)
    eligible_count = sum(
        status == "eligible" for status in result["eligibility"].values()
    )

    assert result["eligible_candidate_count"] == eligible_count
    assert len(result["ranked_policy_ids"]) == eligible_count
    assert eligible_count <= 16
    assert len(result["recommendations"]) == min(
        eligible_count, result["display_cap"]
    )
    assert result["display_omitted_count"] == max(
        0, eligible_count - result["display_cap"]
    )
    assert eligible_count > result["display_cap"]


def test_corrected_intent_audit_rechecks_every_returned_policy():
    audit = json.loads(
        (ROOT / "results" / "eligibility_corrected_intent_audit.json").read_text(
            encoding="utf-8"
        )
    )

    assert audit["status"] == "PASSED"
    assert audit["case_count"] == 5
    assert audit["ineligible_returned_policy_count"] == 0
    assert len(audit["cases"]) == 5
    assert all(
        case["returned_policy_count"] > 0
        and not case["ineligible_returned_policy_ids"]
        for case in audit["cases"]
    )


def test_corrected_catalogue_hard_gates_apply_before_ranking(retrieval_state):
    gate = retrieval_state["eligibility_status"]
    policies = retrieval_state["policy_by_id"]

    assert gate(policies["LIC_911"], {"customer_age": 40, "term": 20}) == "ineligible"
    assert gate(policies["LIC_873"], {"customer_age": 40, "term": 20}) == "unknown"
    assert gate(policies["LIC_887"], {"customer_age": 65, "term": 35}) == "eligible"
    assert gate(policies["LIC_879"], {"customer_age": 95, "term": None}) == "unknown"
    age_95 = retrieval_state["rank_query"](
        "age 95", {"customer_age": 95, "term": None, "all_goals": []}
    )
    assert age_95["status"] == "NO_ELIGIBLE_POLICIES"
    assert age_95["recommendations"] == []


def test_human_evaluation_is_separate_and_eligibility_filtered():
    evaluation = json.loads(
        (ROOT / "results" / "human_evaluation.json").read_text(encoding="utf-8")
    )
    candidates = pd.read_csv(ROOT / "results" / "human_evaluation_candidates.csv")

    assert evaluation["provenance"] == "HUMAN_REVIEWED_BY_USER"
    assert evaluation["query_provenance"] == ["UNSPECIFIED"]
    assert evaluation["training_labels_used"] is False
    assert evaluation["human_labels_used_for_tuning"] is False
    assert evaluation["eligibility_violations"] == 0
    assert set(evaluation["non_evaluable_query_ids"]).issuperset(
        evaluation["ungraded_query_ids"]
    )
    assert candidates["eligibility"].eq("eligible").all()
    assert {"human_grade", "policy_id", "query_id"}.issubset(candidates.columns)
    assert set(evaluation["ranker_results"]) == {"pipeline", "id_order", "bm25_only"}
    assert set(evaluation["baselines"]) == {"id_order", "bm25_only"}
    assert all(
        0 <= result["tied_rank_fraction"] <= 1
        for result in evaluation["ranker_results"].values()
    )
