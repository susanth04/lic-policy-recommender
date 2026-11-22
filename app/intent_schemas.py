"""Pydantic schemas for LLM intent extraction with strict bounds validation.
Rejects any unsupported keys (extra='forbid').
"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ExtractedIntent(BaseModel):
    """Extracted intent parameters validated strictly against brochure bounds."""
    model_config = ConfigDict(extra="forbid")

    customer_age: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Customer entry age in years (0-100)"
    )
    child_age: Optional[int] = Field(
        None,
        ge=0,
        le=25,
        description="Child age in years for child plans (0-25)"
    )
    term: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Policy term in years (1-100)"
    )
    sum_assured: Optional[int] = Field(
        None,
        ge=10_000,
        le=1_000_000_000,
        description="Requested sum assured in rupees (10,000 to 1,000,000,000)"
    )
    gender: Optional[str] = Field(
        None,
        pattern=r"^(?i)(male|female)?$",
        description="Policyholder gender ('male', 'female', or None)"
    )
    goal: Optional[str] = Field(
        None,
        max_length=100,
        description="User financial or protection goal"
    )
    risk_note: Optional[str] = Field(
        None,
        max_length=100,
        description="User risk preference or tolerance"
    )
    out_of_scope_flag: Optional[bool] = Field(
        False,
        description="Flag indicating if the query appears to be for an out-of-scope product"
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Model extraction confidence score (0.0 to 1.0)"
    )
