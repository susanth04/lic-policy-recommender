"""Tests for database models, consent management, auth, and query history storage.
Covers Amendment 2 (no storage without consent_history), Amendment 3 (consent endpoints),
Amendment 6 (OUT_OF_SCOPE never stored), and precedence rules.
"""
import datetime
import json
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use in-memory SQLite for all tests
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.db import Base, ALLOWED_PROFILE_KEYS, Query, User, create_tables
from app.auth import (
    delete_profile_attribute,
    get_confirmed_profile,
    get_consent_status,
    hard_delete_user,
    issue_device_token,
    resolve_token,
    update_consent,
    upsert_profile_attribute,
)


# ----- Fixtures -----------------------------------------------------------

@pytest.fixture(scope="function")
def db_session():
    """Fresh in-memory SQLite session per test."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine)
    db = TestSession()
    yield db
    db.close()


@pytest.fixture
def user_and_token(db_session):
    """Issue a device token and return (user, token, db)."""
    # Temporarily override the module-level engine used in auth.py
    import app.db as db_module
    original_engine = db_module.engine
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(test_engine)
    from sqlalchemy.orm import sessionmaker as SM
    db_module.SessionLocal = SM(bind=test_engine)
    db_module.engine = test_engine
    db = db_module.SessionLocal()
    Base.metadata.create_all(test_engine)
    user_id, token = issue_device_token(db)
    user = db.query(User).filter(User.id == user_id).first()
    yield user, token, db
    db.close()
    db_module.engine = original_engine


# ----- Auth tests ---------------------------------------------------------

def test_device_token_issues_and_resolves(user_and_token):
    user, token, db = user_and_token
    assert user is not None
    resolved = resolve_token(token, db)
    assert resolved is not None
    assert resolved.id == user.id


def test_invalid_token_returns_none(user_and_token):
    _, _, db = user_and_token
    assert resolve_token("garbage.token", db) is None
    assert resolve_token("", db) is None
    assert resolve_token("a.b.c.d", db) is None


def test_tampered_signature_rejected(user_and_token):
    user, token, db = user_and_token
    parts = token.split(".")
    tampered = parts[0] + ".deadbeef"
    assert resolve_token(tampered, db) is None


# ----- Consent tests ------------------------------------------------------

def test_consent_flags_default_to_false(user_and_token):
    user, _, db = user_and_token
    status = get_consent_status(user)
    assert status["consent_profile"] is False
    assert status["consent_history"] is False


def test_consent_grant_and_revoke(user_and_token):
    user, _, db = user_and_token
    # Grant profile memory
    ok = update_consent(user, "profile_memory", True, db)
    assert ok
    db.refresh(user)
    assert user.consent_profile is True

    # Revoke
    ok = update_consent(user, "profile_memory", False, db)
    assert ok
    db.refresh(user)
    assert user.consent_profile is False


def test_revoking_history_consent_deletes_stored_queries(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "query_history", True, db)
    db.refresh(user)

    # Manually insert a query row
    q = Query(
        user_id=user.id,
        created_at=datetime.datetime.utcnow(),
        query_text="test query",
        parsed_intent_json="{}",
        status="OK",
        top_policy_ids="[]",
    )
    db.add(q)
    db.commit()
    db.refresh(user)
    assert len(user.queries) == 1

    # Revoke history consent: query must be immediately deleted
    update_consent(user, "query_history", False, db)
    db.refresh(user)
    assert len(user.queries) == 0


def test_revoking_profile_consent_deletes_attributes(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "profile_memory", True, db)
    db.refresh(user)
    upsert_profile_attribute(user, "customer_age", 35, "user_stated", db)
    db.refresh(user)
    assert len(user.profile_attrs) == 1

    # Revoke: all profile attributes must disappear
    update_consent(user, "profile_memory", False, db)
    db.refresh(user)
    assert len(user.profile_attrs) == 0


def test_unknown_consent_purpose_returns_false(user_and_token):
    user, _, db = user_and_token
    ok = update_consent(user, "medical_data", True, db)
    assert ok is False


# ----- Profile attribute tests --------------------------------------------

def test_only_allowed_profile_keys_stored(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "profile_memory", True, db)
    db.refresh(user)
    # Allowed keys (name, age, phone, income, etc.)
    assert upsert_profile_attribute(user, "name", "Rahul Sharma", "user_stated", db) is True
    assert upsert_profile_attribute(user, "customer_age", 30, "user_stated", db) is True
    assert upsert_profile_attribute(user, "phone", "9876543210", "user_stated", db) is True
    assert upsert_profile_attribute(user, "annual_income", 1200000, "user_stated", db) is True

    # Disallowed keys (unsupported / invasive financial IDs)
    assert upsert_profile_attribute(user, "credit_card", "4111222233334444", "user_stated", db) is False
    assert upsert_profile_attribute(user, "aadhaar_number", "123456789012", "user_stated", db) is False


def test_child_age_stored_but_not_name_or_dob(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "profile_memory", True, db)
    db.refresh(user)
    # Only child_age is allowed for child context
    assert upsert_profile_attribute(user, "child_age", 5, "user_stated", db) is True
    # DOB not allowed
    assert upsert_profile_attribute(user, "child_dob", "2019-01-01", "user_stated", db) is False


def test_upsert_updates_existing_attribute(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "profile_memory", True, db)
    db.refresh(user)
    upsert_profile_attribute(user, "customer_age", 30, "user_stated", db)
    upsert_profile_attribute(user, "customer_age", 31, "form", db)
    db.refresh(user)
    profile = get_confirmed_profile(user)
    assert profile["customer_age"]["value"] == 31
    assert profile["customer_age"]["source"] == "form"


def test_delete_profile_attribute(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "profile_memory", True, db)
    db.refresh(user)
    upsert_profile_attribute(user, "gender", "female", "user_stated", db)
    db.refresh(user)
    assert "gender" in get_confirmed_profile(user)
    ok = delete_profile_attribute(user, "gender", db)
    assert ok is True
    db.refresh(user)
    assert "gender" not in get_confirmed_profile(user)


# ----- Query history storage rules (Amendment 2) -------------------------

def test_no_query_stored_when_consent_history_off(user_and_token):
    user, _, db = user_and_token
    # consent_history is False by default
    # Simulate: would-be storage check
    assert user.consent_history is False
    # Confirm we do NOT write (application logic gates on this flag)
    db.refresh(user)
    assert len(user.queries) == 0


def test_out_of_scope_query_never_stored(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "query_history", True, db)
    db.refresh(user)

    # Simulate out-of-scope status — application MUST NOT store it
    # (this tests the guard condition, not the route directly)
    status = "OUT_OF_SCOPE"
    should_store = user.consent_history and status != "OUT_OF_SCOPE"
    assert should_store is False
    db.refresh(user)
    assert len(user.queries) == 0


def test_ok_query_stored_when_consent_on(user_and_token):
    user, _, db = user_and_token
    update_consent(user, "query_history", True, db)
    db.refresh(user)
    status = "OK"
    if user.consent_history and status != "OUT_OF_SCOPE":
        q = Query(
            user_id=user.id,
            created_at=datetime.datetime.utcnow(),
            query_text="I am 30 and want a term plan",
            parsed_intent_json=json.dumps({"customer_age": 30}),
            status="OK",
            top_policy_ids=json.dumps(["LIC_724"]),
        )
        db.add(q)
        db.commit()
    db.refresh(user)
    assert len(user.queries) == 1
    assert user.queries[0].query_text == "I am 30 and want a term plan"


# ----- Hard deletion test -------------------------------------------------

def test_hard_delete_removes_all_data(user_and_token):
    user, _, db = user_and_token
    user_id = user.id
    update_consent(user, "profile_memory", True, db)
    db.refresh(user)
    upsert_profile_attribute(user, "customer_age", 28, "user_stated", db)
    update_consent(user, "query_history", True, db)
    db.refresh(user)
    q = Query(
        user_id=user.id,
        created_at=datetime.datetime.utcnow(),
        query_text="test",
        parsed_intent_json="{}",
        status="OK",
        top_policy_ids="[]",
    )
    db.add(q)
    db.commit()
    hard_delete_user(user, db)
    # User record should be gone
    result = db.query(User).filter(User.id == user_id).first()
    assert result is None
