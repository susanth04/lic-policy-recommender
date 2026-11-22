"""Policy Chatbot module grounded in the 21 brochure-verified LIC policies.
Uses Gemini via LLMClient, with safety sanitization, citation extraction,
and strict anti-hallucination fact checking.
"""
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.fact_checker import PROHIBITED_CLAIMS_PATTERN
from app.llm_client import LLMClient, LLMError
from app.prompt_safety import sanitize_user_prompt

logger = logging.getLogger("lic_recommender.chat")

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = ROOT / "model_out" / "catalogue.csv"

_VERIFIED_PLANS_CACHE: Optional[List[Dict[str, Any]]] = None
_SYSTEM_PROMPT_CACHE: Optional[str] = None


def load_verified_policies() -> List[Dict[str, Any]]:
    """Load and cache the 21 verified policies from catalogue.csv."""
    global _VERIFIED_PLANS_CACHE
    if _VERIFIED_PLANS_CACHE is not None:
        return _VERIFIED_PLANS_CACHE

    plans = []
    if not CATALOGUE_PATH.exists():
        logger.warning("Catalogue path not found: %s", CATALOGUE_PATH)
        return plans

    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("verified", "").strip().lower() == "true":
                plans.append({
                    "policy_id": row.get("policy_id", "").strip(),
                    "policy_name": row.get("policy_name", "").strip(),
                    "plan_number": row.get("plan_number", "").strip(),
                    "category": row.get("category", "").strip(),
                    "min_age": float(row["min_age"]) if row.get("min_age") else None,
                    "max_age": float(row["max_age"]) if row.get("max_age") else None,
                    "min_term": float(row["min_term"]) if row.get("min_term") else None,
                    "max_term": float(row["max_term"]) if row.get("max_term") else None,
                    "min_sum_assured": float(row["min_sum_assured"]) if row.get("min_sum_assured") else None,
                    "max_sum_assured": float(row["max_sum_assured"]) if row.get("max_sum_assured") else None,
                    "description": row.get("description", "").strip(),
                    "brochure_url": row.get("brochure_url", "").strip(),
                    "notes": row.get("brochure_notes", "").strip(),
                })

    _VERIFIED_PLANS_CACHE = plans
    return plans


def build_policy_knowledge_summary() -> str:
    """Build a concise text knowledge base of all verified policies for the system prompt."""
    plans = load_verified_policies()
    lines = []
    for p in plans:
        age_str = f"{p['min_age']}-{p['max_age']} yrs" if p['min_age'] is not None and p['max_age'] is not None else "brochure specified"
        term_str = f"{p['min_term']}-{p['max_term']} yrs" if p['min_term'] is not None and p['max_term'] is not None else "brochure specified"
        sa_str = f"Rs {int(p['min_sum_assured']):,}" if p['min_sum_assured'] else "brochure specified"
        lines.append(
            f"• {p['policy_name']} (Plan {p['plan_number']}, {p['category']}): "
            f"Entry age: {age_str}; Term: {term_str}; Min Sum Assured: {sa_str}. "
            f"Description: {p['description']} Note: {p['notes'][:120]}"
        )
    return "\n".join(lines)


def get_chat_system_prompt() -> str:
    """Return the system prompt grounded with brochure knowledge."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE

    knowledge = build_policy_knowledge_summary()
    prompt = (
        "You are the LIC Policy Guide, an AI assistant for transparent, brochure-verified LIC life insurance exploration.\n"
        "Your task is to answer user questions accurately, concisely, and helpfully using ONLY the 21 official verified LIC policies listed below.\n\n"
        "=== STRICT RULES ===\n"
        "1. GROUNDING: Answer facts (ages, terms, sum assured, categories, benefits) strictly from the verified policies knowledge base below. "
        "Do not invent or assume policy numbers or eligibility criteria.\n"
        "2. OUT-OF-SCOPE: If the user asks about health insurance, pension/annuity plans (like Jeevan Shanti), ULIPs (like SIIP/Nivesh Plus), "
        "or micro-insurance, politely inform them that this recommender specifically covers brochure-verified Life/Term/Endowment/Money-Back plans.\n"
        "3. NO GUARANTEE CLAIMS: Never claim guaranteed investment returns, dividends, or market-beating returns. Non-linked policies participate in LIC profits/bonuses unless explicitly guaranteed (e.g. New Bima Jyoti guaranteed additions).\n"
        "4. CITE PLANS: When recommending or explaining a policy, always mention the policy name and its Plan Number (e.g., 'LIC's Tech Term (Plan 954)').\n"
        "5. TONE: Be helpful, structured (use bullet points when comparing), objective, and concise. Keep answers within 2-4 short paragraphs or bullet lists.\n"
        "6. DISCLAIMER: Remind the user that this is an academic prototype for transparent exploration, not certified financial advice.\n\n"
        "=== VERIFIED LIC POLICIES KNOWLEDGE BASE ===\n"
        f"{knowledge}\n"
    )
    _SYSTEM_PROMPT_CACHE = prompt
    return prompt


def detect_cited_policies(text: str) -> List[Dict[str, Any]]:
    """Scan the response text to find which of the 21 verified policies were mentioned."""
    plans = load_verified_policies()
    cited = []
    text_lower = text.lower()

    for p in plans:
        name_lower = p["policy_name"].lower()
        plan_num = p["plan_number"]
        # Match by plan number (e.g. "954", "plan 954", "736")
        matched = False
        if plan_num and (f"plan {plan_num}" in text_lower or f"plan no. {plan_num}" in text_lower or f"plan no.{plan_num}" in text_lower or f"({plan_num})" in text_lower or f" {plan_num} " in text_lower):
            matched = True
        elif name_lower.replace("lic's ", "").replace("lic’s ", "") in text_lower:
            matched = True

        if matched and p not in cited:
            cited.append({
                "policy_id": p["policy_id"],
                "policy_name": p["policy_name"],
                "plan_number": p["plan_number"],
                "category": p["category"],
                "min_age": p["min_age"],
                "max_age": p["max_age"],
                "min_sum_assured": p["min_sum_assured"],
                "brochure_url": p["brochure_url"] or f"https://licindia.in",
            })

    return cited


def chat_about_policies(
    llm_client: LLMClient,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    policy_id_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a conversational interaction regarding the verified policies."""
    # 1. Sanitize user input
    clean_message = sanitize_user_prompt(message, max_length=1000)
    if not clean_message:
        return {
            "ok": True,
            "reply": "Please describe what policy, feature, or eligibility criteria you would like to ask about.",
            "cited_policies": [],
            "disclaimer": "Academic decision support system. Verify with official brochures."
        }

    # 2. Check if a specific policy focus is requested
    system_prompt = get_chat_system_prompt()
    if policy_id_filter:
        plans = load_verified_policies()
        target = next((p for p in plans if p["policy_id"] == policy_id_filter or p["plan_number"] == policy_id_filter), None)
        if target:
            system_prompt += f"\nNote: The user is specifically asking about {target['policy_name']} (Plan {target['plan_number']}). Prioritize details about this policy.\n"

    # 3. Format message history (limit to last 6 turns to keep context fast and focused)
    formatted_messages = []
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if content and role in ("user", "assistant", "model"):
                formatted_messages.append({"role": "user" if role == "user" else "assistant", "content": content})

    formatted_messages.append({"role": "user", "content": clean_message})

    # 4. Generate response via LLMClient
    disclaimer = "Academic prototype for transparent policy exploration, not financial or insurance advice. Always verify with official LIC brochures."

    try:
        reply_text = llm_client.chat(
            messages=formatted_messages,
            system_prompt=system_prompt,
            timeout_budget=12.0
        )
    except LLMError as err:
        logger.warning("LLM chat fallback: %s", err)
        # Fallback offline rule-based search
        plans = load_verified_policies()
        matched_plans = []
        msg_words = clean_message.lower().split()
        for p in plans:
            score = sum(1 for w in msg_words if len(w) > 3 and w in p["policy_name"].lower() or w in p["category"].lower() or w == p["plan_number"])
            if score > 0:
                matched_plans.append((score, p))
        matched_plans.sort(key=lambda x: x[0], reverse=True)

        if matched_plans:
            top_p = matched_plans[0][1]
            reply_text = (
                f"Regarding your query about **{top_p['policy_name']} (Plan {top_p['plan_number']})**:\n\n"
                f"• **Category**: {top_p['category']}\n"
                f"• **Entry Age Range**: {top_p['min_age']} to {top_p['max_age']} years\n"
                f"• **Policy Term**: {top_p['min_term']} to {top_p['max_term']} years\n"
                f"• **Minimum Sum Assured**: ₹{int(top_p['min_sum_assured']):,}\n"
                f"• **Overview**: {top_p['description']}\n\n"
                f"*(Note: LLM is currently running in fallback mode; brochure figures shown above are verified directly from our catalogue.)*"
            )
        else:
            reply_text = (
                "We cover 21 brochure-verified LIC policies across Endowment, Whole Life, Money Back, and Term Assurance. "
                "You can ask about entry age, minimum sum assured, policy term, or compare specific plans like Tech Term (954), "
                "Jeevan Labh (736), or Jeevan Umang."
            )

    # 5. Fact check check for prohibited claims
    if PROHIBITED_CLAIMS_PATTERN.search(reply_text):
        reply_text += "\n\n*(Notice: LIC returns are subject to declared performance and not guaranteed unless specified as guaranteed additions.)*"

    # 6. Detect cited policies
    cited = detect_cited_policies(reply_text)

    return {
        "ok": True,
        "reply": reply_text,
        "cited_policies": cited,
        "disclaimer": disclaimer,
    }
