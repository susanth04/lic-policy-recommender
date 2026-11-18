from pathlib import Path

import nbformat
import pandas as pd
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
    policy = {
        "status": "active",
        "recommendation_pool": True,
        "age_verified": True,
        "term_verified": True,
        "verified": True,
        "sum_assured_verified": True,
        "min_age": 18,
        "max_age": 65,
        "min_age_days": None,
        "min_term": 5,
        "max_term": 40,
        "max_maturity_age": 100,
        "source_note": "",
        "eligibility_conditions": "",
    }
    gate = notebook_state["eligibility"]

    assert gate(policy, {"customer_age": 18, "term": None})["status"] == "eligible"
    assert gate(policy, {"customer_age": 65, "term": None})["status"] == "eligible"
    outside_age = 17
    assert gate(policy, {"customer_age": outside_age, "term": None})["status"] == "ineligible"
    assert gate(policy, {"customer_age": 18, "term": 5})["status"] == "eligible"
    assert gate(policy, {"customer_age": 18, "term": 40})["status"] == "eligible"
    assert gate(policy, {"customer_age": 18, "term": 41})["status"] == "ineligible"


def test_blank_rules_json_nan_and_optional_gender(notebook_state):
    gate = notebook_state["eligibility"]
    policy = {
        "status": "active",
        "recommendation_pool": True,
        "verified": True,
        "age_verified": True,
        "term_verified": True,
        "sum_assured_verified": True,
        "min_age": 18,
        "max_age": 65,
        "min_term": 5,
        "max_term": 40,
        "rules_json": float("nan"),
    }
    assert gate(policy, {"customer_age": 30, "term": 20})["status"] == "eligible"
    assert gate({**policy, "rules_json": ""}, {"customer_age": 30, "term": 20})["status"] == "eligible"
    assert gate({**policy, "eligible_gender": float("nan")}, {"customer_age": 30, "term": 20})["status"] == "eligible"
    assert gate({**policy, "eligible_gender": "female"}, {"customer_age": 30, "term": 20})["status"] == "unknown"
    assert gate(
        {**policy, "eligible_gender": "female"},
        {"customer_age": 30, "term": 20, "gender": "female"},
    )["status"] == "eligible"


def test_corrected_catalogue_status_and_conditional_limits_gate_recommendations(notebook_state):
    gate = notebook_state["eligibility"]
    policies = notebook_state["policies_df"].set_index("policy_id")
    policy = lambda policy_id: policies.loc[policy_id].to_dict()

    assert gate(policy("LIC_770"), {"customer_age": 40, "term": None})["status"] == "eligible"
    assert gate(policy("LIC_888"), {"customer_age": 40, "term": 20})["status"] == "unknown"
    assert gate(policy("LIC_907"), {"customer_age": 40, "term": 15})["status"] == "unknown"
    assert gate(policy("LIC_887"), {"customer_age": 65, "term": 35})["status"] == "eligible"
    assert gate(policy("LIC_911"), {"customer_age": 40, "term": 20})["status"] == "ineligible"
    assert gate(policy("LIC_873"), {"customer_age": 40, "term": 20})["status"] == "unknown"
    assert gate(policy("LIC_867"), {"customer_age": 40, "term": 20})["status"] == "unknown"
    assert gate(
        policy("LIC_770"),
        {"customer_age": 11, "term": 17, "premium_paying_term": 7},
    )["status"] == "eligible"
    assert gate(policy("LIC_912"), {"customer_age": 40, "term": None})["status"] == "eligible"
    assert gate(policy("LIC_770"), {"customer_age": 40, "term": None})["status"] == "eligible"
    assert gate(policy("LIC_770"), {"customer_age": 61, "term": None})["status"] == "ineligible"
    assert gate(policy("LIC_912"), {"customer_age": 40, "term": None, "premium_paying_term": 12})["status"] == "eligible"
    assert gate(
        policy("LIC_881"),
        {"customer_age": 40, "term": 25, "sum_assured": 200_000, "gender": "female"},
    )["status"] == "eligible"


def test_brochure_eligibility_rules_and_sum_assured_limits(notebook_state):
    gate = notebook_state["eligibility"]
    policies = notebook_state["policies_df"].set_index("policy_id")
    policy = lambda policy_id: policies.loc[policy_id].to_dict()

    assert gate(
        policy("LIC_887"),
        {"customer_age": 40, "term": 20, "sum_assured": 5_000_000},
    )["status"] == "ineligible"
    assert gate(
        policy("LIC_736"),
        {"customer_age": 59, "term": 16, "sum_assured": 500_000},
    )["status"] == "eligible"
    assert gate(
        policy("LIC_736"),
        {"customer_age": 40, "term": 20, "sum_assured": 500_000},
    )["status"] == "ineligible"
    assert gate(
        policy("LIC_875"),
        {"customer_age": 45, "term": 40},
    )["status"] == "ineligible"
    assert gate(policy("LIC_875"), {"customer_age": 46, "term": None})["status"] == "ineligible"
    assert gate(
        policy("LIC_736"),
        {"customer_age": 59, "term": 21, "sum_assured": 500_000},
    )["status"] == "ineligible"


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
    amount = parse("I need ₹50 lakh cover for a 20-year term at age 40")
    assert amount["sum_assured"] == 5_000_000
    assert amount["term"] == 20
    assert amount["customer_age"] == 40
    assert parse("2 crore sum assured")["sum_assured"] == 20_000_000
    assert parse("I want high market returns with zero risk")["contradictions"]
    assert parse("PPT 10 with a 20-year term")["premium_paying_term"] == 10
    assert parse("I am female")["gender"] == "female"
    assert parse("")["confidence"] == 0.0


def test_requested_amount_and_term_intents(notebook_state):
    parse = notebook_state["parse_intent"]
    cases = (
        (
            "I'm 30 and have a small budget, want a basic policy with 5 lakh coverage for 10 year term",
            500_000,
            10,
        ),
        (
            "Daily wage worker aged 35, want small premium policy with 5 lakh sum assured, 15 year term",
            500_000,
            15,
        ),
        (
            "Age 26, first time buyer, need low premium cover of 10 lakh for 20 years term",
            1_000_000,
            20,
        ),
        (
            "Looking for 50 lakh term cover at age 40 for 20 years",
            5_000_000,
            20,
        ),
    )
    for query, expected_sum_assured, expected_term in cases:
        intent = parse(query)
        assert intent["sum_assured"] == expected_sum_assured
        assert intent["term"] == expected_term

    premium_query = "I'm 37, want to put 1 lakh yearly in an index linked plan"
    premium_intent = parse(premium_query)
    assert premium_intent["sum_assured"] is None
    assert premium_intent["budget"] == {"amount": "1", "unit": "lakh"}
    assert parse("premium of 5000")["budget"] == {"amount": "5000", "unit": "rupees"}


def test_spouse_pronouns_infer_female_when_gender_is_omitted(notebook_state):
    parse = notebook_state["parse_intent"]

    assert parse("I need a policy for my wife")["gender"] == "female"
    assert parse("She needs life cover")["gender"] == "female"
    assert parse("A woman needs life cover")["gender"] == "female"


def test_graphml_preserves_corrected_verification_scope_and_status():
    graph = nx.read_graphml(ROOT / "artifacts" / "kg.graphml")
    policy_nodes = [attrs for _, attrs in graph.nodes(data=True) if attrs.get("type") == "Policy"]

    assert len(policy_nodes) == 47
    assert len(policy_nodes) == len(pd.read_csv(ROOT / "data" / "policies.csv"))
    assert graph.nodes["LIC_859"]["identity_verified"] in (True, "True")
    assert graph.nodes["LIC_859"]["verified"] in (True, "True")
    assert graph.nodes["LIC_911"]["status"] == "withdrawn"
