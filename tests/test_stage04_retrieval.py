import json
from pathlib import Path

import nbformat
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


def test_smoke_ranking_is_eligible_monotone_and_not_tuned(retrieval_state):
    results = json.loads((ROOT / "results" / "retrieval_smoke.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT / "results" / "weights_smoke.json").read_text(encoding="utf-8"))

    assert results["eligibility_violations"] == 0
    assert results["eligibility_violation_rate"] == 0.0
    assert all(
        recommendation["eligibility"] == "eligible"
        and recommendation["score_kind"] == "ranking_score_not_probability"
        for row in results["queries"]
        for recommendation in row["recommendations"]
    )
    assert weights["selected_alpha"] is None
    assert weights["selected_beta"] is None
    assert weights["selected_kg_preference_weights"] is None
    assert weights["status"] == "SKIPPED_NO_HUMAN_GRADED_LABELS"


def test_offline_fallbacks_and_repeatable_ranking(retrieval_state):
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
