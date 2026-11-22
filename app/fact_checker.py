"""Fact verification filter for LLM-generated explanations.
Constrains explanations strictly to brochure facts and rejects hallucinations.
"""
import re
from typing import Iterable, Set


PROHIBITED_CLAIMS_PATTERN = re.compile(
    r"\b(guaranteed\s+(?:high\s+)?returns?|market[- ]linked|100%\s+safe|medical\s+cover(?:age)?|health\s+benefits?|pension\s+income|annuity\s+for\s+life)\b",
    re.IGNORECASE
)


def extract_numbers(text: str) -> Set[int]:
    """Extract all standalone integers from text."""
    # Matches digits, stripping currency symbols and commas
    cleaned = re.sub(r"[,\u20b9$]", "", text)
    nums = set()
    for token in re.findall(r"\b\d+\b", cleaned):
        try:
            nums.add(int(token))
        except ValueError:
            pass
    return nums


def verify_explanation(
    llm_text: str,
    target_plan_name: str,
    target_plan_id: str,
    gate_facts_list: list[str],
    all_other_plan_names: Iterable[str]
) -> bool:
    """Verify that an LLM-generated explanation adheres strictly to supplied facts.
    Returns True if valid, False if hallucination or unsupported claim is detected.
    """
    if not llm_text or not isinstance(llm_text, str):
        return False

    # 1. Prohibited claims check (financial advice, guarantees, medical, pension)
    if PROHIBITED_CLAIMS_PATTERN.search(llm_text):
        return False

    # 2. Foreign plan names check: must not mention any other LIC policy name
    norm_text = llm_text.lower()
    for foreign_name in all_other_plan_names:
        if not foreign_name:
            continue
        fn_lower = foreign_name.lower().strip()
        # Avoid false positives on common words like "jeevan" if the foreign name is longer
        if len(fn_lower) > 5 and fn_lower != target_plan_name.lower().strip():
            # Check for full name occurrence
            if re.search(rf"\b{re.escape(fn_lower)}\b", norm_text):
                return False

    # 3. Numeric hallucination check:
    # All numbers in the explanation must originate from:
    # - gate facts (e.g. age bounds, term bounds, sum assured)
    # - target plan name / ID (e.g., "721", "25" in "Money Back 25")
    # - harmless formatting digits (1, 2, 3)
    facts_joined = " ".join(gate_facts_list) + f" {target_plan_name} {target_plan_id}"
    allowed_numbers = extract_numbers(facts_joined)
    # Allow small list counters 1, 2, 3
    allowed_numbers.update({1, 2, 3, 4, 5})

    explanation_numbers = extract_numbers(llm_text)
    for num in explanation_numbers:
        if num not in allowed_numbers:
            # Hallucinated number detected (e.g. fabricated interest rate, unstated premium or term)
            return False

    return True
