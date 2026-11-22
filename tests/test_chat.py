"""Unit and integration tests for the Policy Chat AI feature."""
import pytest
from fastapi.testclient import TestClient

from app.llm_client import FakeLLMClient, LLMTimeoutError
from app.main import app
from app.policy_chat import (
    build_policy_knowledge_summary,
    chat_about_policies,
    detect_cited_policies,
    get_chat_system_prompt,
    load_verified_policies,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_load_verified_policies():
    """Verify that catalogue.csv loads all 21 verified candidate plans."""
    plans = load_verified_policies()
    assert len(plans) == 21
    for p in plans:
        assert p["policy_id"].startswith("LIC_")
        assert len(p["policy_name"]) > 0
        assert p["category"] in ("Endowment", "Whole Life", "Money Back", "Term Assurance")


def test_build_policy_knowledge_summary():
    """Ensure knowledge summary contains core verified plans."""
    summary = build_policy_knowledge_summary()
    assert "Tech Term" in summary or "954" in summary
    assert "Jeevan Labh" in summary or "736" in summary
    assert "Jeevan Utsav" in summary or "771" in summary


def test_get_chat_system_prompt():
    """System prompt must enforce grounding and disclaimers."""
    prompt = get_chat_system_prompt()
    assert "STRICT RULES" in prompt
    assert "OUT-OF-SCOPE" in prompt
    assert "NO GUARANTEE CLAIMS" in prompt


def test_detect_cited_policies():
    """Ensure mentions of plan names and numbers are extracted into citations."""
    sample_text = (
        "Based on verified brochures, LIC's Tech Term (Plan 954) offers pure term cover. "
        "Alternatively, LIC's Jeevan Labh Plan 736 is an endowment option."
    )
    cited = detect_cited_policies(sample_text)
    assert len(cited) >= 2
    plan_numbers = {c["plan_number"] for c in cited}
    assert "954" in plan_numbers
    assert "736" in plan_numbers


def test_chat_about_policies_with_fake_llm():
    """Test policy chat using FakeLLMClient."""
    fake_client = FakeLLMClient(enabled=True)
    res = chat_about_policies(
        llm_client=fake_client,
        message="What is the maximum entry age for Tech Term?",
        history=[]
    )
    assert res["ok"] is True
    assert len(res["reply"]) > 0
    assert "disclaimer" in res
    assert isinstance(res["cited_policies"], list)


def test_chat_fallback_on_llm_timeout():
    """When LLM times out, chat_about_policies falls back to catalogue keyword extraction."""
    def timeout_handler(sys, msgs):
        raise LLMTimeoutError("Simulation timeout")

    fake_client = FakeLLMClient(custom_chat_handler=timeout_handler, enabled=True)
    res = chat_about_policies(
        llm_client=fake_client,
        message="Tell me about Jeevan Labh Plan 736",
        history=[]
    )
    assert res["ok"] is True
    # Should contain brochure details from fallback
    assert "Jeevan Labh" in res["reply"] or "736" in res["reply"]


def test_chat_api_endpoint(client):
    """Test POST /chat API endpoint."""
    resp = client.post("/chat", json={
        "message": "Can a 30-year-old buy Jeevan Umang?",
        "history": []
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "reply" in data
    assert "cited_policies" in data
    assert "disclaimer" in data


def test_chat_api_empty_message_validation(client):
    """Empty messages should be rejected by Pydantic validation."""
    resp = client.post("/chat", json={
        "message": "",
        "history": []
    })
    assert resp.status_code == 422
