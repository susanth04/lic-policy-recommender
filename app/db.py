"""SQLAlchemy models for user profiles, query history, and consent management.
Single source of truth: SQLite for dev, Postgres-compatible by swapping DATABASE_URL.
"""
import datetime
import os
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ----- Engine & session factory ------------------------------------------

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "lic_recommender.db"
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH}"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


# ----- ORM models ---------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)          # UUID as string
    created_at = Column(DateTime, nullable=False,
                        default=lambda: datetime.datetime.now(datetime.timezone.utc))
    # Granular consent flags (single source of truth — updated via /me/consent)
    consent_profile = Column(Boolean, nullable=False, default=False)
    consent_history = Column(Boolean, nullable=False, default=False)
    deleted_at = Column(DateTime, nullable=True)       # soft-delete marker

    profile_attrs = relationship("ProfileAttribute", back_populates="user",
                                 cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="user",
                           cascade="all, delete-orphan")
    consents = relationship("Consent", back_populates="user",
                            cascade="all, delete-orphan")


ALLOWED_PROFILE_KEYS = frozenset({
    "name",                    # user's full name
    "customer_age",            # integer age in years
    "child_age",               # integer age in years (child plans only)
    "gender",                  # "male" / "female" / "other"
    "email",                   # contact email
    "phone",                   # contact phone number
    "annual_income",           # annual income in INR (integer)
    "occupation",              # "salaried", "business", "self_employed", "student", "homemaker"
    "marital_status",          # "single", "married"
    "city",                    # city / location
    "preferred_term",          # integer years
    "preferred_sum_assured",   # integer rupees
})

ALLOWED_SOURCES = frozenset({"user_stated", "form", "llm_extracted_confirmed"})


class ProfileAttribute(Base):
    """Stores only confirmed, minimized profile attributes for a user."""
    __tablename__ = "profile_attributes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key = Column(String(32), nullable=False)       # must be in ALLOWED_PROFILE_KEYS
    value = Column(String(128), nullable=False)    # stored as string; cast on read
    source = Column(String(32), nullable=False)    # must be in ALLOWED_SOURCES
    stated_at = Column(DateTime, nullable=False,
                       default=lambda: datetime.datetime.now(datetime.timezone.utc))
    confirmed_at = Column(DateTime, nullable=False,
                          default=lambda: datetime.datetime.now(datetime.timezone.utc))

    user = relationship("User", back_populates="profile_attrs")

    __table_args__ = (
        Index("ix_profile_user_key", "user_id", "key"),
    )


class Query(Base):
    """Records a completed /recommend request (only stored when consent_history is on)."""
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, nullable=False,
                        default=lambda: datetime.datetime.now(datetime.timezone.utc))
    # Both columns NULL when status is OUT_OF_SCOPE (never stored)
    # or when consent_history is off at call time
    query_text = Column(Text, nullable=True)       # only if consent_history=True
    parsed_intent_json = Column(Text, nullable=True)
    status = Column(String(32), nullable=False)
    top_policy_ids = Column(Text, nullable=True)   # JSON array string

    user = relationship("User", back_populates="queries")

    __table_args__ = (
        Index("ix_queries_user_created", "user_id", "created_at"),
    )


ALLOWED_CONSENT_PURPOSES = frozenset({"profile_memory", "query_history"})


class Consent(Base):
    """Audit log of consent grants and revocations — single source of truth."""
    __tablename__ = "consents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    purpose = Column(String(32), nullable=False)   # must be in ALLOWED_CONSENT_PURPOSES
    granted_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="consents")

    __table_args__ = (
        Index("ix_consents_user_purpose", "user_id", "purpose"),
    )


def create_tables():
    """Idempotent schema creation — acts as a lightweight migration."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a SQLAlchemy session and closes it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
