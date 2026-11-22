
"""Knowledge-graph matching. Attributes are DERIVED (category + description lexicon), not read from the empty catalogue columns."""
import re

import networkx as nx
import numpy as np

from intent_parser import GOAL_KEYWORDS

CATEGORY_GOALS = {
    "Term Assurance": ("Life Cover",),
    "Endowment": ("Savings", "Life Cover"),
    "Money Back": ("Savings", "Income"),
    "Whole Life": ("Life Cover", "Savings"),
    "Pension": ("Retirement", "Income"),
    "Unit Linked": ("Wealth Creation", "Life Cover"),
    "Health Insurance": ("Health Protection",),
    "Micro Insurance": ("Life Cover", "Savings"),
}
EXTRA_GOAL_LEXICON = {
    "Life Cover": ("pure risk", "credit life", "term plan", "life cover"),
    "Savings": ("maturity sum", "money back", "guaranteed addition", "savings plan"),
    "Retirement": ("deferred annuity", "immediate annuity"),
    "Child Education": ("children",),
    "Wealth Creation": ("unit linked", "market linked"),
    "Income": ("guaranteed income", "income benefit", "income benefits"),
    "Health Protection": ("critical illness",),
}
LEVELS = ["Low", "Moderate", "High"]
CATEGORY_PREMIUM = {"Term Assurance": "Low", "Micro Insurance": "Low"}
WEIGHTS = {"goal": 0.45, "category": 0.20, "risk": 0.20, "premium": 0.15}
KG_FEATURE_NAMES = ["kg_score", "kg_goal", "kg_category", "kg_risk", "kg_premium"]


def _s(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[-_/]", " ", str(text).lower())).strip()


def _strip_negated(text):
    """Drop phrases like 'no maturity benefit' / 'no survival or maturity benefit' so they do not create goal links."""
    return re.sub(r"\b(?:no|without|not)\b(?:\s+\w+){0,4}?\s+benefits?", " ", text)


def _has(text, kw):
    return re.search(rf"\b{re.escape(_norm(kw))}\b", text) is not None


def derive_profile(rec):
    cat = _s(rec.get("category"))
    text = _strip_negated(_norm(_s(rec.get("policy_name")) + " . " + _s(rec.get("description"))))
    goals = {}
    for i, g in enumerate(CATEGORY_GOALS.get(cat, ())):
        goals[g] = {"weight": 1.0 if i == 0 else 0.7, "source": f"category {cat}"}
    for g, kws in GOAL_KEYWORDS.items():
        words = tuple(kws) + EXTRA_GOAL_LEXICON.get(g, ())
        hit = next((k for k in words if _has(text, k)), None)
        if hit and goals.get(g, {}).get("weight", 0.0) < 0.6:
            goals[g] = {"weight": 0.6, "source": f"description keyword '{hit}'"}
    linked = any(_has(text, k) for k in ("unit linked", "market linked", "ulip"))
    if cat == "Unit Linked" or linked:
        risk = ("High", "unit/market-linked plan")
    else:
        risk = ("Low", "non-linked plan (no market-linked investment risk)")
    if _has(text, "single premium"):
        prem = ("High", "text says single premium")
    elif cat in CATEGORY_PREMIUM:
        prem = (CATEGORY_PREMIUM[cat], f"category {cat}")
    else:
        prem = ("Moderate", "default for regular/limited-premium savings plans")
    return {"category": cat, "goals": goals, "risk": risk, "premium": prem}


def build_kg(records):
    G = nx.DiGraph()
    for g in GOAL_KEYWORDS:
        G.add_node(f"goal:{g}", kind="Goal", label=g)
    for lv in LEVELS:
        G.add_node(f"risk:{lv}", kind="Risk", label=lv)
        G.add_node(f"premium:{lv}", kind="Premium", label=lv)
    for cat, goals in CATEGORY_GOALS.items():
        G.add_node(f"category:{cat}", kind="Category", label=cat)
        for i, g in enumerate(goals):
            G.add_edge(f"category:{cat}", f"goal:{g}", relation="SERVES_GOAL", weight=1.0 if i == 0 else 0.7, derived=True, source="category prior")
    for rec in records:
        pid = rec["policy_id"]
        prof = derive_profile(rec)
        node = f"policy:{pid}"
        G.add_node(node, kind="Policy", label=_s(rec.get("policy_name")), policy_id=pid, category=prof["category"], verified=bool(rec.get("verified")))
        cat_node = f"category:{prof['category']}"
        if not G.has_node(cat_node):
            G.add_node(cat_node, kind="Category", label=prof["category"])
        G.add_edge(node, cat_node, relation="IN_CATEGORY", weight=1.0, derived=False, source="catalogue category (not brochure-verified)")
        for g, d in prof["goals"].items():
            G.add_edge(node, f"goal:{g}", relation="HAS_GOAL", weight=d["weight"], derived=True, source=d["source"])
        G.add_edge(node, f"risk:{prof['risk'][0]}", relation="HAS_RISK", weight=1.0, derived=True, source=prof["risk"][1])
        G.add_edge(node, f"premium:{prof['premium'][0]}", relation="HAS_PREMIUM_LEVEL", weight=1.0, derived=True, source=prof["premium"][1])
    return G


def _ordinal(a, b):
    return {0: 1.0, 1: 0.5}.get(abs(LEVELS.index(a) - LEVELS.index(b)), 0.0)


def kg_match(G, pid, intent):
    node = f"policy:{pid}"
    goals, cat, risk, prem = {}, None, None, None
    for _, tgt, d in G.out_edges(node, data=True):
        label = G.nodes[tgt]["label"]
        rel = d["relation"]
        if rel == "HAS_GOAL":
            goals[label] = d
        elif rel == "IN_CATEGORY":
            cat = label
        elif rel == "HAS_RISK":
            risk = label
        elif rel == "HAS_PREMIUM_LEVEL":
            prem = label
    comps, links, conflicts = {}, [], []
    intent_goals = list(intent.get("all_goals") or [])
    primary = intent.get("primary_goal")
    if intent_goals:
        wt = {g: (1.0 if g == primary else 0.5) for g in intent_goals}
        comps["goal"] = sum(wt[g] * goals[g]["weight"] for g in intent_goals if g in goals) / sum(wt.values())
        for g in intent_goals:
            if g in goals:
                links.append(f"goal '{g}' (from {goals[g]['source']})")
            else:
                conflicts.append(f"no graph link to requested goal '{g}'")
    if primary and cat:
        edge = G.get_edge_data(f"category:{cat}", f"goal:{primary}")
        comps["category"] = float(edge["weight"]) if edge else 0.0
        if edge:
            links.append(f"category '{cat}' serves goal '{primary}'")
        else:
            conflicts.append(f"category '{cat}' does not serve goal '{primary}'")
    want_risk = intent.get("risk_tolerance")
    if want_risk in LEVELS and risk:
        comps["risk"] = _ordinal(want_risk, risk)
        (links if comps["risk"] >= 0.5 else conflicts).append(f"risk: you asked for {want_risk}, plan is {risk}" if comps["risk"] < 1 else f"risk level {risk} matches")
    want_prem = intent.get("premium_level")
    if want_prem in LEVELS and prem:
        comps["premium"] = _ordinal(want_prem, prem)
        (links if comps["premium"] >= 0.5 else conflicts).append(f"premium: you asked for {want_prem}, plan is {prem}" if comps["premium"] < 1 else f"premium level {prem} matches")
    wsum = sum(WEIGHTS[k] for k in comps)
    score = sum(WEIGHTS[k] * v for k, v in comps.items()) / wsum if comps else 0.5
    return {"policy_id": pid, "score": float(score), "components": comps, "matched_links": links, "conflicts": conflicts,
            "signals_used": sorted(comps)}


def kg_match_many(G, policy_ids, intent):
    return [kg_match(G, p, intent) for p in policy_ids]


def kg_feature_matrix(results):
    rows = [[r["score"]] + [r["components"].get(k, 0.5) for k in ("goal", "category", "risk", "premium")] for r in results]
    return np.array(rows, dtype=float).reshape(len(results), len(KG_FEATURE_NAMES))


def export_graphml(G, path):
    nx.write_graphml(G, str(path))
