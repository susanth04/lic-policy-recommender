
import re

GOAL_KEYWORDS = {
    "Life Cover": ("term", "life cover", "protection", "death benefit", "sum assured"),
    "Child Education": ("child", "daughter", "son", "education", "school", "college", "marriage"),
    "Retirement": ("retirement", "pension", "annuity", "retire", "old age"),
    "Savings": ("savings", "endowment", "guaranteed", "money back", "bachat", "maturity"),
    "Wealth Creation": ("ulip", "wealth", "market", "equity", "high returns", "growth"),
    "Health Protection": ("cancer", "health", "hospital", "mediclaim", "illness", "medical", "icu"),
    "Income": ("regular income", "monthly income", "annual income", "survival benefit", "cash flow"),
}

OUT_OF_SCOPE_RULES = {
    "Pension / Annuity": [
        r"\bpension(?:s|\s+plan|\s+scheme)?\b",
        r"\bannuit(?:y|ies)\b",
        r"\bretire(?:ment|\s+plan|\s+fund|\s+income)?\b",
        r"\bpost[- ]retirement\b",
        r"\bold[- ]age\s+(?:pension|income|benefit)\b",
    ],
    "Health / Medical": [
        r"\bhealth\s+(?:plan|insurance|cover)\b",
        r"\bmediclaim\b",
        r"\bhospital(?:ization)?\b",
        r"\bcancer(?:\s+cover|\s+insurance)?\b",
        r"\bcritical\s+illness\b",
        r"\bmedical\s+(?:cover|insurance|claim)\b",
    ],
    "Unit Linked (ULIP)": [
        r"\bulip(?:s)?\b",
        r"\bunit[- ]linked\b",
        r"\bmarket[- ]linked\b",
        r"\bequity\s+fund\b",
        r"\bnav\s+growth\b",
    ],
    "Micro Insurance": [
        r"\bmicro[- ]insurance\b",
        r"\bmicro[- ]term\b",
        r"\bmicro[- ]plan\b",
    ],
}


def parse_intent(text):
    text = text if isinstance(text, str) else ""
    normalized = re.sub(r"[-‐‑–—]+", " ", text.lower())
    # spoken amounts: "half a crore", "quarter crore", "one crore", "two lakh"
    normalized = re.sub(r"\b(?:half a crore|half crore|half a cr)\b", "50 lakh", normalized)
    normalized = re.sub(r"\b(?:a quarter crore|quarter crore|quarter of a crore)\b", "25 lakh", normalized)
    _words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    normalized = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(crores?|lakhs?|lacs?)\b",
                        lambda m: f"{_words[m.group(1)]} {m.group(2)}", normalized)
    child_matches = []
    for pattern in (
        r"(?P<age>\d{1,2})\s*(?:year|yr)s?\s*old\s*(?:daughter|son|child|kid|baby)\b",
        r"(?:daughter|son|child|kid)\s*(?:is|aged|of)\s*(?P<age>\d{1,2})\b",
    ):
        child_matches.extend(re.finditer(pattern, normalized))
    child_age = next((int(match.group("age")) for match in child_matches if int(match.group("age")) <= 25), None)

    customer_age = None
    customer_pattern = (
        r"\b(?:(?:i am|i'm|my age is|age|aged|at age|senior citizen aged|turning|turned)"
        r"\s*(?:a\s*)?(?P<explicit>\d{1,3})\b|"
        r"(?P<year_old>\d{1,3})\s*[- ]?\s*(?:years?|yrs?)\s*[- ]old\b)"
    )
    for match in re.finditer(customer_pattern, normalized):
        age_group = "explicit" if match.group("explicit") else "year_old"
        start, end = match.span(age_group)
        if any(start < child.end("age") and end > child.start("age") for child in child_matches):
            continue
        age = int(match.group(age_group))
        if 16 <= age <= 105:
            customer_age = age
            break

    if customer_age is None:      # "At 24, I ..." / "to save at 58"; never "retire at 60" or "returns at 12%"
        for match in re.finditer(r"\bat\s+(\d{2})(?!\d)(?!\s*(?:%|percent|per cent|lakh|lac|crore|cr\b|k\b))", normalized):
            before = normalized[max(0, match.start() - 24):match.start()]
            age = int(match.group(1))
            if 16 <= age <= 105 and not re.search(r"retir|pension|till|until|after|by|from|stop|matur|annuity", before):
                customer_age = age
                break

    term_match = re.search(
        r"\b(\d{1,2})\s*[- ]?\s*(?:years?|yrs?)\s*(?:policy\s*)?term\b|"
        r"\bterm\s*(?:of\s*)?(\d{1,2})\s*(?:years?|yrs?)?\b",
        normalized,
    )
    term = int(next(group for group in term_match.groups() if group is not None)) if term_match else None
    if term is None:
        for_match = re.search(r"\bfor\s*(?:a\s*)?(\d{1,2})\s*[- ]?\s*(?:years?|yrs?)\b(?!\s*old)", normalized)
        term = int(for_match.group(1)) if for_match else None
    ppt_match = re.search(
        r"\b(?:ppt|premium[- ]paying term|premium payment term|paying term)\s*(?:of\s*)?(\d{1,2})\b",
        normalized,
    )
    premium_paying_term = int(ppt_match.group(1)) if ppt_match else None
    amount_match = re.search(
        r"(?<!\w)(?:(?:sum\s+assured|sum\s+insured|cover(?:age)?)\s*(?:of|is|for)?\s*)?"
        r"(?:₹|rs\.?\s*)?(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
        r"(?P<unit>crores?|cr|lakhs?|lacs?|l|k)\b",
        normalized,
    )
    sum_assured = None
    if amount_match:
        preceding_text = normalized[max(0, amount_match.start() - 14):amount_match.start()]
        if not re.search(r"\b(?:premium of|budget of|budget|under|below|up to)\s*(?:rs\.?|₹)?\s*$", preceding_text):
            amount = float(amount_match.group("amount").replace(",", ""))
            unit = amount_match.group("unit")
            multiplier = 10_000_000 if unit.startswith(("crore", "cr")) else 100_000 if unit.startswith(("lakh", "lac", "l")) else 1_000
            sum_assured = int(amount * multiplier)

    gender_match = re.search(r"\b(male|female|man|woman)\b", normalized)
    gender_map = {"man": "male", "woman": "female"}
    gender = gender_map.get(gender_match.group(1), gender_match.group(1)) if gender_match else None

    budget_match = re.search(
        r"\b(?:under|below|up to|budget(?: of)?|premium of)?\s*(?:₹|rs\.?\s*)?"
        r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|rupees?|rs\.?|k)\b",
        normalized,
    )
    budget = None if not budget_match or sum_assured is not None else {
        "amount": budget_match.group(1).replace(",", ""),
        "unit": budget_match.group(2).lower(),
    }

    goals = [goal for goal, keywords in GOAL_KEYWORDS.items() if any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords)]
    risk = next((value for value, keywords in (("High", ("high risk", "equity", "aggressive", "market linked", "ulip")), ("Moderate", ("moderate risk", "balanced", "medium risk")), ("Low", ("safe", "low risk", "risk free", "guaranteed", "no risk", "zero risk", "secure"))) if any(keyword in normalized for keyword in keywords)), None)
    premium = next((value for value, keywords in (("Low", ("cheap", "low premium", "affordable", "budget", "small premium", "micro")), ("Moderate", ("moderate premium", "reasonable premium", "regular premium")), ("High", ("single premium", "lump sum", "high premium", "hni"))) if any(keyword in normalized for keyword in keywords)), None)
    contradictions = []
    if any(term in normalized for term in ("high returns", "market returns", "wealth")) and any(term in normalized for term in ("zero risk", "no risk", "risk free", "risk-free")):
        contradictions.append("High market returns conflict with a zero-risk requirement.")

    out_of_scope_categories = []
    for category, patterns in OUT_OF_SCOPE_RULES.items():
        if any(re.search(pat, normalized) for pat in patterns):
            out_of_scope_categories.append(category)

    confidence = sum(value is not None for value in (customer_age, goals[0] if goals else None, risk, premium)) / 4
    return {
        "customer_age": customer_age,
        "customer_age_days": None,
        "child_age": child_age,
        "term": term,
        "premium_paying_term": premium_paying_term,
        "sum_assured": sum_assured,
        "gender": gender,
        "budget": budget,
        "all_goals": goals,
        "primary_goal": goals[0] if goals else None,
        "risk_tolerance": risk,
        "premium_level": premium,
        "contradictions": contradictions,
        "is_out_of_scope": len(out_of_scope_categories) > 0,
        "out_of_scope_categories": out_of_scope_categories,
        "is_ambiguous": not any((customer_age is not None, term is not None, sum_assured is not None, goals, risk, premium, budget)),
        "confidence": round(confidence, 2),
    }


def merge_intent(parsed, overrides=None):
    """Apply structured fields from a UI form (age, term, sum_assured, gender, ...) on top of the parsed text."""
    merged = dict(parsed)
    for key, value in (overrides or {}).items():
        if value is not None and value != "":
            merged[key] = value
    return merged
