from pathlib import Path

import nbformat
import pytest
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def notebook_state():
    notebook = nbformat.read(ROOT / "02_intent_and_kg.ipynb", as_version=4)
    state = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(ROOT / "02_intent_and_kg.ipynb"), "exec"), state)
    return state


def test_entry_age_and_term_boundaries_are_hard_gates(notebook_state):
    policy = notebook_state["policies_df"].iloc[0].to_dict()
    gate = notebook_state["eligibility"]

    assert gate(policy, {"customer_age": int(policy["min_age"]), "term": None})["status"] == "eligible"
    assert gate(policy, {"customer_age": int(policy["max_age"]), "term": None})["status"] == "eligible"
    outside_age = int(policy["min_age"]) - 1
    assert gate(policy, {"customer_age": outside_age, "term": None})["status"] == "ineligible"
    assert gate(policy, {"customer_age": int(policy["min_age"]), "term": int(policy["min_term"])})["status"] == "eligible"
    assert gate(policy, {"customer_age": int(policy["min_age"]), "term": int(policy["max_term"])})["status"] == "eligible"
    assert gate(policy, {"customer_age": int(policy["min_age"]), "term": int(policy["max_term"]) + 1})["status"] == "ineligible"


def test_added_policies_with_unknown_limits_are_not_eligible(notebook_state):
    gate = notebook_state["eligibility"]
    added = notebook_state["policies_df"].loc[
        notebook_state["policies_df"]["source_note"].fillna("").ne("")
    ]

    assert len(added) == 27
    assert all(
        gate(row, {"customer_age": 40, "term": None})["status"] == "unknown"
        for row in added.to_dict(orient="records")
    )
    assert all(
        gate(row, {"customer_age": 40, "term": 20})["status"] == "unknown"
        for row in added.to_dict(orient="records")
    )


def test_unknown_age_and_age_95_never_produce_recommendations(notebook_state):
    result = notebook_state["recommend"]

    assert result("Looking for life cover")["status"] == "NO_ELIGIBLE_POLICIES"
    assert all(state["status"] != "eligible" for state in result("Looking for life cover")["eligibility"].values())
    old = result("I am 95 years old and want a policy")
    assert old["status"] == "NO_ELIGIBLE_POLICIES"
    assert old["recommendations"] == []


def test_age_disambiguation_and_contradiction_parser(notebook_state):
    parse = notebook_state["parse_intent"]

    assert (parse("I am 36 with a 3 year old son")["customer_age"], parse("I am 36 with a 3 year old son")["child_age"]) == (36, 3)
    assert parse("5 Lakhs")["customer_age"] is None
    assert parse("I need a 30 years term")["customer_age"] is None
    assert parse("I want high market returns with zero risk")["contradictions"]
    assert parse("")["confidence"] == 0.0


def test_graphml_keeps_every_policy_unverified():
    graph = nx.read_graphml(ROOT / "artifacts" / "kg.graphml")
    policy_nodes = [attrs for _, attrs in graph.nodes(data=True) if attrs.get("type") == "Policy"]

    assert len(policy_nodes) == 47
    assert all(attrs.get("verified") in (False, "False") for attrs in policy_nodes)
