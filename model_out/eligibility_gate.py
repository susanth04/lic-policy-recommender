
import json
import math


def _number(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _is_true(value):
    return str(value).strip().lower() in {"true", "1"}


def _rules(policy):
    value = policy.get("rules_json")
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if isinstance(value, str) and not value.strip():
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("rules_json must contain a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("rules_json must contain a JSON object")
    return value


def evaluate_eligibility(policy, intent):
    product_status = str(policy.get("status", "")).strip().lower()
    pool_eligible = _is_true(policy.get("recommendation_pool"))
    if product_status == "withdrawn":
        return {"status": "ineligible", "reasons": ["policy is withdrawn"]}
    if product_status != "active" or not pool_eligible:
        return {"status": "unknown", "reasons": ["policy status is unresolved or outside the candidate pool"]}

    age = _number(intent.get("customer_age"))
    term = _number(intent.get("term"))
    ppt = _number(intent.get("premium_paying_term"))
    sum_assured = _number(intent.get("sum_assured"))
    rules = _rules(policy)
    by_ppt = rules.get("by_ppt") or {}
    if by_ppt:
        if age is None:
            return {"status": "unknown", "reasons": ["customer age is required"]}
        option_keys = [str(int(ppt))] if ppt is not None else sorted(
            (str(key) for key in by_ppt),
            key=lambda key: _number(key) if _number(key) is not None else float("inf"),
        )
        option_results = []
        for option_key in option_keys:
            option_rule = by_ppt.get(option_key)
            if not isinstance(option_rule, dict):
                option_results.append({
                    "status": "ineligible" if _is_true(policy.get("term_verified")) else "unknown",
                    "reasons": ["requested premium-paying term is not allowed"],
                })
            else:
                option_results.append(_evaluate_scenario(policy, intent, rules, {}, option_rule))
        if any(result["status"] == "eligible" for result in option_results):
            return {"status": "eligible", "reasons": []}
        if all(result["status"] == "ineligible" for result in option_results):
            reasons = list(dict.fromkeys(reason for result in option_results for reason in result["reasons"]))
            return {"status": "ineligible", "reasons": reasons}
        reasons = list(dict.fromkeys(reason for result in option_results for reason in result["reasons"]))
        return {"status": "unknown", "reasons": reasons or ["no premium-paying-term option can be decided"]}

    return _evaluate_scenario(policy, intent, rules, _term_rule(rules, term), {})


def _term_rule(rules, term):
    by_term = rules.get("by_term") or {}
    if term is None or not isinstance(by_term, dict):
        return {}
    rule = by_term.get(str(int(term)))
    return rule if isinstance(rule, dict) else {}


def _evaluate_scenario(policy, intent, rules, term_rule, ppt_rule):
    # v7: plans whose rules_json sets "age_subject": "child" (children's plans) are priced on the CHILD's entry age.
    subject_child = str(rules.get("age_subject", "")).strip().lower() == "child"
    age = _number(intent.get("child_age")) if subject_child else _number(intent.get("customer_age"))
    term = _number(intent.get("term"))
    sum_assured = _number(intent.get("sum_assured"))
    age_verified = _is_true(policy.get("age_verified"))
    term_verified = _is_true(policy.get("term_verified"))
    sum_assured_verified = _is_true(policy.get("sum_assured_verified"))
    failures = []
    missing = []

    by_term = rules.get("by_term") or {}
    if term is not None and by_term:
        if not isinstance(by_term, dict):
            (failures if term_verified else missing).append("term-specific rules are invalid")
        elif str(int(term)) not in by_term:
            (failures if term_verified else missing).append("requested term has no term-specific rule")

    def rule_number(key, *aliases):
        for rule in (term_rule, ppt_rule):
            for name in (key, *aliases):
                number = _number(rule.get(name))
                if number is not None:
                    return number
        return None

    min_age = rule_number("min_entry_age", "min_entry_age_years")
    if min_age is None:
        min_age = _number(policy.get("min_age"))
    max_age = rule_number("max_entry_age")
    if max_age is None:
        max_age = _number(policy.get("max_age"))

    if age is None:
        missing.append("child age is required" if subject_child else "customer age is required")
    else:
        if min_age is not None and age < min_age:
            (failures if age_verified else missing).append("age is below the minimum entry age")
        if max_age is not None and age > max_age:
            (failures if age_verified else missing).append("age exceeds the maximum entry age")
        if min_age is None or max_age is None:
            missing.append("entry-age limits are incomplete")
        min_age_days = rule_number("min_entry_age_days")
        if min_age_days is None:
            min_age_days = _number(policy.get("min_entry_age_days", policy.get("min_age_days")))
        if age == 0 and min_age_days is not None and min_age_days > 0:
            age_days = _number(intent.get("customer_age_days"))
            if age_days is None:
                missing.append("age in days is required")
            elif age_days < min_age_days:
                failures.append("age is below the minimum entry age in days")
        if not age_verified:
            missing.append("entry-age limits are not brochure-verified")

    if term is not None:
        min_term = rule_number("min_term")
        if min_term is None:
            min_term = _number(policy.get("min_term"))
        max_term = rule_number("max_term")
        if max_term is None:
            max_term = _number(policy.get("max_term"))
        if min_term is not None and term < min_term:
            (failures if term_verified else missing).append("requested term is below the minimum")
        if max_term is not None and term > max_term:
            (failures if term_verified else missing).append("requested term exceeds the maximum")
        if min_term is None or max_term is None:
            missing.append("policy-term limits are incomplete")

        allowed_terms = rules.get("allowed_terms")
        if allowed_terms is not None and term not in {_number(value) for value in allowed_terms}:
            (failures if term_verified else missing).append("requested term is not an allowed term")

        max_maturity_age = _number(policy.get("max_maturity_age"))
        if age is not None and max_maturity_age is not None and age + term > max_maturity_age:
            failures.append("age plus requested term exceeds maximum maturity age")

        if not term_verified:
            missing.append("policy-term limits are not brochure-verified")

    if sum_assured is not None:
        min_sum_assured = _number(policy.get("min_sum_assured"))
        max_sum_assured = _number(policy.get("max_sum_assured"))
        unlimited_max = _is_true(policy.get("max_sum_assured_unlimited"))
        if min_sum_assured is not None and sum_assured < min_sum_assured:
            (failures if sum_assured_verified else missing).append("requested sum assured is below the minimum")
        elif min_sum_assured is None:
            missing.append("minimum sum assured is missing")
        if max_sum_assured is not None and sum_assured > max_sum_assured:
            (failures if sum_assured_verified else missing).append("requested sum assured exceeds the maximum")
        elif max_sum_assured is None and not unlimited_max:
            missing.append("maximum sum assured is missing")
        if not sum_assured_verified:
            missing.append("sum-assured limits are not brochure-verified")

    eligible_gender = policy.get("eligible_gender")
    if eligible_gender is not None and not (
        isinstance(eligible_gender, float) and math.isnan(eligible_gender)
    ) and str(eligible_gender).strip().lower() not in {"", "nan", "none"}:
        requested_gender = intent.get("gender")
        if requested_gender is None or not str(requested_gender).strip():
            missing.append("gender is required")
        else:
            allowed = {item.strip().lower() for item in str(eligible_gender).split("|")}
            if str(requested_gender).strip().lower() not in allowed:
                failures.append("gender is not eligible")

    if not _is_true(policy.get("verified")):
        missing.append("brochure eligibility is not verified")

    if failures:
        return {"status": "ineligible", "reasons": list(dict.fromkeys(failures))}
    if missing:
        return {"status": "unknown", "reasons": list(dict.fromkeys(missing))}
    return {"status": "eligible", "reasons": []}
