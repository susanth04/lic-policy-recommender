
"""Inference entry point v2: query -> parse -> eligibility gate (unchanged) -> rank eligible plans -> template explanations."""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import fusion
import kg_matching
import semantic_scoring
from eligibility_gate import evaluate_eligibility
from intent_parser import merge_intent, parse_intent

DISCLAIMER = (
    "Educational decision support, not financial advice. Check the official LIC brochure before buying."
)
WEAK_MATCH_BELOW = 1.2   # Calibrated on nested out-of-fold GBR predictions (abstention AUROC = 0.907)
CLARIFY_FIELDS = {
    "customer age is required": "customer_age",
    "child age is required": "child_age",
    "gender is required": "gender",
    "age in days is required": "customer_age_days",
}
CLARIFY_QUESTIONS = {
    "customer_age": "What is the policyholder's age?",
    "child_age": "How old is your child?",
    "gender": "What is the policyholder's gender (male/female)?",
    "customer_age_days": "What is the infant's age in days?",
}
NOTE = (
    "predicted_relevance is a 0-3 ranking score on the human-grade scale, not a probability. "
    "fused_score, kg_score, semantic_score and cross_encoder_score are relative scores, also not probabilities. "
    "weak_match=true means even the top eligible plan is a poor fit for the stated need."
)
MODE_TEXT = {
    "gbr_only": "the gradient-boosted ranker trained on human grades",
    "gbr_features": "the gradient-boosted ranker with knowledge-graph / semantic / cross-encoder scores as extra features",
    "fusion": "a weighted fusion of the gradient-boosted ranker, knowledge-graph and semantic scores",
}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def gate_facts(r, intent):
    """Catalogue facts behind an 'eligible' answer. Descriptive only: the gate itself is eligibility_gate.evaluate_eligibility."""
    facts = ["eligibility gate result: eligible", "plan is active and in the recommendation pool"]

    def ver(flag):
        return " (brochure-verified)" if bool(r.get(flag)) else " (not brochure-verified)"

    # Identify whether plan prices on child age or customer age
    rules = {}
    try:
        if r.get("rules_json") and isinstance(r.get("rules_json"), str):
            rules = json.loads(r.get("rules_json"))
        elif isinstance(r.get("rules_json"), dict):
            rules = r.get("rules_json")
    except Exception:
        pass
    subject_child = str(rules.get("age_subject", "")).strip().lower() == "child"

    lo, hi = _num(r.get("min_age")), _num(r.get("max_age"))
    if subject_child:
        child_age = _num(intent.get("child_age"))
        if child_age is not None and lo is not None and hi is not None:
            facts.append(f"child age {child_age:.0f} is inside the catalogue entry-age range {lo:.0f}-{hi:.0f}" + ver("age_verified"))
        elif child_age is not None:
            facts.append(f"child age {child_age:.0f} evaluated" + ver("age_verified"))
    else:
        cust_age = _num(intent.get("customer_age"))
        if cust_age is not None and lo is not None and hi is not None:
            facts.append(f"age {cust_age:.0f} is inside the catalogue entry-age range {lo:.0f}-{hi:.0f}" + ver("age_verified"))

    term, tlo, thi = _num(intent.get("term")), _num(r.get("min_term")), _num(r.get("max_term"))
    if term is not None and tlo is not None and thi is not None:
        facts.append(f"term {term:.0f} years is inside the catalogue range {tlo:.0f}-{thi:.0f} years" + ver("term_verified"))
    elif str(r.get("policy_id")) == "LIC_771":
        if term is not None:
            facts.append("term specified, but Jeevan Utsav is a Whole Life plan with no verified fixed policy-term limits (catalog unverified)")
        else:
            facts.append("Whole Life plan: coverage continues for life (up to age 100); no fixed policy term is required")
    elif term is not None and (tlo is None or thi is None):
        facts.append("policy term not verified in brochure")

    sa, slo, shi = _num(intent.get("sum_assured")), _num(r.get("min_sum_assured")), _num(r.get("max_sum_assured"))
    if sa is not None and slo is not None:
        if bool(r.get("max_sum_assured_unlimited")) or shi is None:
            facts.append(f"sum assured Rs {sa:,.0f} meets the minimum Rs {slo:,.0f} (no upper limit stored)" + ver("sum_assured_verified"))
        else:
            facts.append(f"sum assured Rs {sa:,.0f} is inside Rs {slo:,.0f}-{shi:,.0f}" + ver("sum_assured_verified"))

    gender = str(r.get("eligible_gender") or "").strip().lower()
    if gender in ("female", "male") and intent.get("gender"):
        facts.append(f"gender matches the plan's {gender}-only rule")

    missing = []
    if subject_child:
        if intent.get("child_age") is None:
            missing.append("child age")
    else:
        if intent.get("customer_age") is None:
            missing.append("age")
    if term is None and str(r.get("policy_id")) != "LIC_771":
        missing.append("term")
    if sa is None:
        missing.append("sum assured")
    if missing:
        facts.append("not provided, so not checked against the plan: " + ", ".join(missing))
    return facts


class Recommender:
    def __init__(self, model_dir="."):
        d = Path(model_dir)
        self.dir = d
        v1 = joblib.load(d / "lic_ranker.joblib")
        self.featurizer, self.base_model, self.meta = v1["featurizer"], v1["model"], v1["meta"]
        self.catalogue = pd.read_csv(d / "catalogue.csv")
        self.records = self.catalogue.to_dict("records")
        self.by_id = {r["policy_id"]: r for r in self.records}
        self.kg = kg_matching.build_kg(self.records)
        cfg_path = d / "hybrid_config.json"
        self.cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"mode": "gbr_only", "needs": {}}
        self.variant = self.sem = self.ce = None
        self.active_mode = "gbr_only"
        self.runtime = {"requested_mode": self.cfg.get("mode", "gbr_only"), "active_mode": "gbr_only", "fallback": False,
                        "fallback_reason": None, "semantic_loaded": False, "cross_encoder_loaded": False}
        self._init_hybrid()

    def _resolve(self, spec):
        local = self.dir / spec["local_dir"] if spec.get("local_dir") else None
        return str(local) if local is not None and local.exists() else spec["name"]

    def _init_hybrid(self):
        mode = self.cfg.get("mode", "gbr_only")
        if mode == "gbr_only":
            return
        needs = self.cfg.get("needs", {})
        try:
            if os.environ.get("LIC_FORCE_FALLBACK"):
                raise RuntimeError("LIC_FORCE_FALLBACK is set (fallback test)")
            variant = joblib.load(self.dir / "lic_hybrid.joblib") if mode == "gbr_features" else None
            sem = ce = None
            if needs.get("semantic"):
                spec = self.cfg["semantic_model"]
                enc = semantic_scoring.load_encoder(self._resolve(spec))
                sem = semantic_scoring.SemanticScorer.from_catalogue(enc, spec["name"], self.records, self.dir / self.cfg["policy_embeddings"])
            if needs.get("cross_encoder"):
                ce = semantic_scoring.load_cross_encoder(self._resolve(self.cfg["cross_encoder_model"]))
        except Exception as e:
            self.runtime.update(fallback=True, fallback_reason=f"{type(e).__name__}: {e}"[:300])
            return
        self.variant, self.sem, self.ce = variant, sem, ce
        self.active_mode = mode
        self.runtime.update(active_mode=mode, semantic_loaded=sem is not None, cross_encoder_loaded=ce is not None)

    # ---- scoring --------------------------------------------------------------------------
    def _score(self, query, intent, eligible):
        feats = self.featurizer.transform(query, intent, eligible)
        kg_res = kg_matching.kg_match_many(self.kg, eligible, intent)
        sem = np.asarray(self.sem.scores(query, eligible), dtype=float) if self.sem is not None else None
        ce = None
        if self.ce is not None:
            pairs = [(query, semantic_scoring.policy_text(self.by_id[p])) for p in eligible]
            ce = np.asarray(self.ce.predict(pairs, show_progress_bar=False), dtype=float).ravel()
        if self.active_mode == "gbr_features":
            blk, _ = fusion.build_extra(self.cfg.get("feature_extras", []), kg_mat=kg_matching.kg_feature_matrix(kg_res), sem=sem, ce=ce)
            raw = self.variant["model"].predict(np.hstack([feats, blk]))
        else:
            raw = self.base_model.predict(feats)
        return {"pred": np.clip(raw, 0, 3), "kg": np.array([r["score"] for r in kg_res]), "sem": sem, "ce": ce, "kg_res": kg_res}

    def _rank(self, sc, eligible):
        fused = None
        if self.active_mode == "fusion":
            comps = {"gbr": sc["pred"], "kg": sc["kg"]}
            if sc["sem"] is not None:
                comps["sem"] = sc["sem"]
            fused = fusion.fuse(comps, self.cfg["fusion"]["weights"])
            order = fusion.rank_order(fused, eligible)
            rr = self.cfg.get("rerank", {})
            if rr.get("enabled") and sc["ce"] is not None:
                order = fusion.rerank_topk(fused, sc["ce"], eligible, rr["k"], rr["weight"])
        else:
            order = sorted(range(len(eligible)), key=lambda i: (-sc["pred"][i], eligible[i]))
        return order, fused

    # ---- explanation (fixed templates, no generated text) ------------------------------------
    def _explain(self, r, intent, kg_r, rec):
        facts = gate_facts(r, intent)
        links, conflicts = kg_r["matched_links"], kg_r["conflicts"]
        parts = [f"Eligibility: {'; '.join(facts)}."]
        parts.append("Knowledge-graph links matched: " + ("; ".join(links) if links else "none (no goal, risk or premium preference was stated)") + ".")
        if conflicts:
            parts.append("Not matched: " + "; ".join(conflicts) + ".")
        s = [f"model relevance {rec['predicted_relevance']:.2f} on a 0-3 scale", f"KG suitability {rec['kg_score']:.2f}"]
        if rec["semantic_score"] is not None:
            s.append(f"semantic similarity {rec['semantic_score']:.2f}")
        if rec["cross_encoder_score"] is not None:
            s.append(f"cross-encoder score {rec['cross_encoder_score']:.2f}")
        parts.append("Scores (relative, not probabilities): " + ", ".join(s) + ".")
        parts.append("Ranked by " + MODE_TEXT[self.active_mode] + ".")
        return {"gate_facts": facts, "kg_links": links, "kg_conflicts": conflicts, "text": " ".join(parts)}

    def recommend(self, query, top_k=5, overrides=None):
        intent = merge_intent(parse_intent(query), overrides)

        if intent.get("is_out_of_scope"):
            categories = ", ".join(intent.get("out_of_scope_categories", []))
            return {
                "status": "OUT_OF_SCOPE",
                "query": query,
                "intent": intent,
                "recommendations": [],
                "weak_match": True,
                "eligible_count": 0,
                "clarifications": [],
                "message": (
                    f"The requested category ({categories}) is outside the scope of this recommender. "
                    "This system only recommends brochure-verified LIC Endowment, Term Assurance, "
                    "Money Back, and Whole Life plans. Pension/annuity, health insurance, ULIP (market-linked), "
                    "and micro-insurance plans are not supported."
                ),
                "ranking_meta": {
                    **self.runtime,
                    "ranking_description": MODE_TEXT[self.active_mode],
                    "gate": "unchanged: only gate-eligible plans are ranked",
                    "abstention_calibration": {
                        "threshold": WEAK_MATCH_BELOW,
                        "auroc": 0.907,
                        "match_recall": 1.0,
                        "abstention_precision": 1.0,
                    },
                },
                "note": NOTE,
                "disclaimer": DISCLAIMER,
            }

        states = {r["policy_id"]: evaluate_eligibility(r, intent) for r in self.records}
        eligible = sorted(pid for pid, s in states.items() if s["status"] == "eligible")

        has_child_context = (
            intent.get("child_age") is not None
            or "Child Education" in (intent.get("all_goals") or [])
            or any(re.search(rf"\b{w}\b", query.lower()) for w in (
                "child", "children", "daughter", "son", "kid", "baby", "education", "school", "college", "marriage"
            ))
        )

        clarify = {}
        for r in self.records:
            state = states[r["policy_id"]]
            if state["status"] != "unknown" or not r.get("verified"):
                continue
            for reason in state["reasons"]:
                field = CLARIFY_FIELDS.get(reason)
                if field == "child_age" and not has_child_context:
                    continue
                if field:
                    clarify.setdefault(field, {
                        "field": field,
                        "reason": reason,
                        "question": CLARIFY_QUESTIONS.get(field, reason),
                        "policy_ids": []
                    })["policy_ids"].append(r["policy_id"])

        recs = []
        if eligible:
            sc = self._score(query, intent, eligible)
            order, fused = self._rank(sc, eligible)
            for rank, i in enumerate(order[:top_k], start=1):
                r = self.by_id[eligible[i]]
                rec = {
                    "rank": rank,
                    "policy_id": r["policy_id"],
                    "policy_name": r["policy_name"],
                    "category": r["category"],
                    "predicted_relevance": round(float(sc["pred"][i]), 3),
                    "fused_score": round(float(fused[i]), 3) if fused is not None else None,
                    "kg_score": round(float(sc["kg"][i]), 3),
                    "semantic_score": round(float(sc["sem"][i]), 3) if sc["sem"] is not None else None,
                    "cross_encoder_score": round(float(sc["ce"][i]), 3) if sc["ce"] is not None else None,
                    "policy_url": r.get("policy_url") or None,
                    "brochure_url": r.get("brochure_url") or None,
                }
                rec["explanation"] = self._explain(r, intent, sc["kg_res"][i], rec)
                recs.append(rec)
        weak_match = bool(recs) and recs[0]["predicted_relevance"] < WEAK_MATCH_BELOW
        status = "OK" if recs else ("NEEDS_INFO" if clarify else "NO_ELIGIBLE_POLICIES")
        return {
            "status": status,
            "query": query,
            "intent": intent,
            "recommendations": recs,
            "weak_match": weak_match,
            "eligible_count": len(eligible),
            "clarifications": list(clarify.values()),
            "ranking_meta": {
                **self.runtime,
                "ranking_description": MODE_TEXT[self.active_mode],
                "gate": "unchanged: only gate-eligible plans are ranked",
                "abstention_calibration": {
                    "threshold": WEAK_MATCH_BELOW,
                    "auroc": 0.907,
                    "match_recall": 1.0,
                    "abstention_precision": 1.0,
                },
            },
            "note": NOTE,
            "disclaimer": DISCLAIMER,
        }
