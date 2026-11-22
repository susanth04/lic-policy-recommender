"""Recommender wrapper implementing Part A (LLM layer) and Part B (precedence & memory).
Does NOT edit or mutate generated package files in model_out/.
"""
import copy
import datetime
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Add model_out to sys.path so we can import directly without modifying package files
MODEL_DIR = Path(__file__).resolve().parent.parent / "model_out"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from intent_parser import parse_intent, merge_intent, OUT_OF_SCOPE_RULES
from recommender import Recommender, WEAK_MATCH_BELOW, CLARIFY_FIELDS

from app.fact_checker import verify_explanation
from app.intent_schemas import ExtractedIntent
from app.llm_client import LLMClient, LLMError, LLMTimeoutError
from app.prompt_safety import wrap_user_query

logger = logging.getLogger("lic_recommender.wrapper")

CORE_FIELDS = ("customer_age", "child_age", "term", "sum_assured", "gender")
STALE_DAYS_DEFAULT = 180


class RecommenderWrapper:
    """Wraps the pristine v7 recommender engine with LLM assistance and profile memory."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        stale_days: int = STALE_DAYS_DEFAULT
    ):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.engine = Recommender(self.model_dir)
        self.llm_client = llm_client or LLMClient()
        self.stale_days = stale_days

        # Cache known policy names for hallucination checks
        self.all_policy_names = {r["policy_name"] for r in self.engine.records}

    @property
    def records(self):
        return self.engine.records

    @property
    def active_mode(self):
        return self.engine.active_mode

    @property
    def runtime(self):
        return self.engine.runtime

    @property
    def meta(self):
        return self.engine.meta

    def parse_query_with_llm(
        self,
        query: str
    ) -> Tuple[Dict[str, Any], Optional[ExtractedIntent], List[Dict[str, Any]], Optional[str], bool]:
        """Extract intent from query text, combining rule parser with optional LLM parser.
        Returns:
            (rule_intent, llm_intent, disagreements, out_of_scope_note, llm_fallback_used)
        """
        rule_intent = parse_intent(query)

        if not self.llm_client.enabled:
            return rule_intent, None, [], None, False

        safe_prompt = wrap_user_query(query)
        system_prompt = (
            "You are a strict data extractor for life insurance intents. "
            "Extract customer_age (int 0-100), child_age (int 0-25), term (int 1-100), "
            "sum_assured (int 10000-1000000000), gender ('male'/'female'), goal, risk_note, "
            "and out_of_scope_flag (true if asking for pension, health, ULIP, or micro-insurance). "
            "Return JSON matching the schema strictly. Never guess or hallucinate."
        )

        llm_fallback = False
        llm_intent: Optional[ExtractedIntent] = None
        disagreements: List[Dict[str, Any]] = []
        out_of_scope_prompt: Optional[str] = None

        try:
            raw_json = self.llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=safe_prompt,
                timeout_budget=3.0
            )
            # Validate with Pydantic (extra keys forbidden, bounds enforced)
            llm_intent = ExtractedIntent.model_validate(raw_json)

            # Check confidence
            if llm_intent.confidence < 0.70:
                logger.info("LLM confidence low (%.2f); falling back to rule parser", llm_intent.confidence)
                llm_fallback = True

        except (LLMError, Exception) as exc:
            logger.warning("LLM extraction failed or timed out (%s); falling back to rule parser", exc)
            llm_fallback = True
            llm_intent = None

        # Compare rule parser vs LLM parser
        if llm_intent and not llm_fallback:
            for field in CORE_FIELDS:
                r_val = rule_intent.get(field)
                l_val = getattr(llm_intent, field)
                if r_val is not None and l_val is not None and r_val != l_val:
                    disagreements.append({
                        "field": field,
                        "rule_extracted": r_val,
                        "llm_extracted": l_val,
                        "clarification_prompt": (
                            f"We detected multiple possible values for {field.replace('_', ' ')}: "
                            f"'{r_val}' and '{l_val}'. Which one would you like to use?"
                        )
                    })

            # LLM out-of-scope flag check
            # Rule check stays authoritative; if LLM flagged out-of-scope but rule didn't:
            if llm_intent.out_of_scope_flag and not rule_intent.get("is_out_of_scope"):
                out_of_scope_prompt = (
                    "Your query looks like it might refer to pension/annuity, health coverage, or ULIP plans. "
                    "This system only recommends brochure-verified Endowment, Term, Money Back, and Whole Life plans. "
                    "Please confirm if you are looking for life coverage or another category."
                )

        return rule_intent, llm_intent, disagreements, out_of_scope_prompt, llm_fallback

    def resolve_attributes(
        self,
        form_fields: Optional[Dict[str, Any]],
        parsed_query_values: Dict[str, Any],
        confirmed_profile: Optional[Dict[str, Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply strict precedence:
        explicit form fields > values stated in current query > confirmed profile.
        Profile ONLY fills fields the user did not provide now.
        """
        form = {k: v for k, v in (form_fields or {}).items() if v is not None and v != ""}
        effective_overrides: Dict[str, Any] = {}
        profile_applied: List[Dict[str, Any]] = []
        needs_confirmation: List[Dict[str, Any]] = []
        now = datetime.datetime.now(datetime.timezone.utc)

        for field in CORE_FIELDS:
            # 1. Explicit form field wins first
            if field in form:
                effective_overrides[field] = form[field]
                continue

            # 2. Value in current query wins second
            if parsed_query_values.get(field) is not None:
                effective_overrides[field] = parsed_query_values[field]
                continue

            # 3. Confirmed profile attribute only fills unprovided fields
            if confirmed_profile and field in confirmed_profile:
                prof_item = confirmed_profile[field]
                val = prof_item.get("value")
                confirmed_at_str = prof_item.get("confirmed_at")

                # Staleness check for age-like attributes
                is_stale = False
                days_old = 0
                if confirmed_at_str and field in ("customer_age", "child_age"):
                    try:
                        confirmed_dt = datetime.datetime.fromisoformat(confirmed_at_str.replace("Z", "+00:00"))
                        days_old = (now - confirmed_dt).days
                        if days_old > self.stale_days:
                            is_stale = True
                    except Exception:
                        pass

                if is_stale:
                    # Do not silently apply stale age attributes to the gate!
                    needs_confirmation.append({
                        "field": field,
                        "stored_value": val,
                        "days_old": days_old,
                        "prompt": f"Last time you noted your {field.replace('_', ' ')} as {val} ({days_old} days ago). Is this still accurate?"
                    })
                elif val is not None and val != "":
                    effective_overrides[field] = val
                    profile_applied.append({
                        "field": field,
                        "value": val,
                        "source": "confirmed_profile"
                    })

        return effective_overrides, profile_applied, needs_confirmation

    def generate_friendly_clarifications(
        self,
        clarifications: List[Dict[str, Any]],
        user_query: str
    ) -> List[Dict[str, Any]]:
        """Use LLM to generate friendly phrasing for NEEDS_INFO items computed by the gate."""
        if not self.llm_client.enabled or not clarifications:
            return clarifications

        friendly_list = []
        for item in clarifications:
            field = item.get("field")
            system_prompt = (
                "You are an empathetic LIC life insurance assistant. "
                "The user is looking for a policy, but the eligibility gate needs one more detail. "
                f"Generate a friendly, concise 1-sentence question asking the user for their '{field}'. "
                "Return JSON with key 'question'."
            )
            safe_user = wrap_user_query(f"Field required: {field}. Original query: {user_query}")
            try:
                resp = self.llm_client.complete_json(system_prompt, safe_user, timeout_budget=1.5)
                q_text = resp.get("question")
                if q_text and isinstance(q_text, str) and len(q_text) < 200:
                    friendly_list.append({**item, "question": q_text})
                    continue
            except Exception:
                pass
            friendly_list.append(item)

        return friendly_list

    def rewrite_explanations_fact_constrained(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rewrite explanation text into plain language using ONLY supplied brochure facts.
        Drops model_score from the LLM prompt. Post-checks against hallucinations.
        """
        if not self.llm_client.enabled:
            return recommendations

        for rec in recommendations:
            orig_exp = rec.get("explanation", {})
            gate_facts = orig_exp.get("gate_facts", [])
            kg_links = orig_exp.get("kg_links", [])
            kg_conflicts = orig_exp.get("kg_conflicts", [])
            plan_name = rec.get("policy_name")
            plan_id = rec.get("policy_id")

            # Notice: model_score is intentionally dropped from the LLM payload (Amendment 6)
            payload = {
                "plan_name": plan_name,
                "category": rec.get("category"),
                "gate_facts": gate_facts,
                "matched_links": kg_links,
                "unmatched_links": kg_conflicts
            }

            system_prompt = (
                "You are an insurance explainer summarizing why this LIC policy is eligible. "
                "Rules: "
                "1. Use ONLY the supplied gate_facts, category, and matched_links. "
                "2. NEVER invent numbers, interest rates, returns, or bonuses not in gate_facts. "
                "3. NEVER mention other policy names. "
                "4. Keep it to 2-3 plain-language sentences. "
                "Return JSON with key 'plain_explanation'."
            )

            try:
                resp = self.llm_client.complete_json(
                    system_prompt,
                    wrap_user_query(json.dumps(payload)),
                    timeout_budget=2.0
                )
                plain_text = resp.get("plain_explanation")
                if plain_text and verify_explanation(
                    plain_text,
                    plan_name,
                    plan_id,
                    gate_facts,
                    self.all_policy_names
                ):
                    rec["explanation"]["text"] = plain_text
                    rec["explanation"]["llm_verified"] = True
                else:
                    logger.info("LLM explanation rejected by fact checker; kept template for %s", plan_name)
                    rec["explanation"]["llm_verified"] = False
            except Exception as exc:
                logger.debug("LLM explanation generation skipped for %s: %s", plan_name, exc)
                rec["explanation"]["llm_verified"] = False

        return recommendations

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        form_fields: Optional[Dict[str, Any]] = None,
        confirmed_profile: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Full recommendation pipeline honoring precedence, memory, and LLM constraints."""
        # 1. Parse query with rule parser + optional LLM parser
        rule_intent, llm_intent, disagreements, oos_prompt, llm_fallback = self.parse_query_with_llm(query)

        # Merge parsed query values (LLM parsed values take precedence over rule parsed values if valid and agreed)
        parsed_query_values = dict(rule_intent)
        if llm_intent and not llm_fallback:
            for field in CORE_FIELDS:
                l_val = getattr(llm_intent, field)
                # If there is a disagreement on this field, do NOT overwrite; let user resolve
                field_has_disagreement = any(d["field"] == field for d in disagreements)
                if l_val is not None and not field_has_disagreement:
                    parsed_query_values[field] = l_val

        # 2. Resolve attributes with strict precedence:
        # explicit form fields > current query > confirmed profile
        effective_overrides, profile_applied, needs_confirmation = self.resolve_attributes(
            form_fields=form_fields,
            parsed_query_values=parsed_query_values,
            confirmed_profile=confirmed_profile
        )

        # 3. Call underlying pristine recommender
        result = self.engine.recommend(query, top_k=top_k, overrides=effective_overrides)

        # 4. Handle NEEDS_INFO friendly phrasing
        if result.get("status") == "NEEDS_INFO" and result.get("clarifications"):
            result["clarifications"] = self.generate_friendly_clarifications(
                result["clarifications"],
                query
            )

        # 5. Handle OK status explanations rewriting
        if result.get("status") == "OK" and result.get("recommendations"):
            result["recommendations"] = self.rewrite_explanations_fact_constrained(
                result["recommendations"]
            )

        # 6. Add metadata, profile memory audit, and confirmations
        result["profile_applied"] = profile_applied
        result["needs_confirmation"] = needs_confirmation
        result["disagreements"] = disagreements
        if oos_prompt:
            result["out_of_scope_confirmation"] = oos_prompt

        result["precedence_audit"] = {
            "order": "explicit_form > current_query > confirmed_profile",
            "effective_overrides": effective_overrides,
            "llm_assisted": self.llm_client.enabled and not llm_fallback
        }

        return result
