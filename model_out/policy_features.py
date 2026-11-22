
"""Query x policy feature builder shared by training and serving."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from intent_parser import GOAL_KEYWORDS

GOALS = list(GOAL_KEYWORDS)


def goal_vector(text):
    t = str(text).lower()
    return np.array(
        [float(any(re.search(rf"\b{re.escape(k)}\b", t) for k in kws)) for kws in GOAL_KEYWORDS.values()]
    )


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


class PolicyFeaturizer:
    """Fitted on the policy catalogue only (no labels), so it cannot leak label information."""

    def fit(self, catalogue):
        cat = catalogue.reset_index(drop=True)
        self.policy_ids = cat["policy_id"].tolist()
        self.index = {p: i for i, p in enumerate(self.policy_ids)}
        text = (
            cat["policy_name"].fillna("").astype(str) + ". "
            + cat["category"].fillna("").astype(str) + ". "
            + cat["description"].fillna("").astype(str)
        ).str.lower()
        self.categories = sorted(cat["category"].dropna().unique().tolist())
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(text)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(text)
        self.Xw = self.word_vec.transform(text)
        self.Xc = self.char_vec.transform(text)
        self.pgoal = np.vstack([goal_vector(t) for t in text])
        self.pcat = np.array([[float(c == k) for k in self.categories] for c in cat["category"]])
        self.min_age = cat["min_age"].map(_num).to_numpy(float)
        self.max_age = cat["max_age"].map(_num).to_numpy(float)
        self.min_sa = cat["min_sum_assured"].map(_num).to_numpy(float)
        self.min_term = cat["min_term"].map(_num).to_numpy(float)
        return self

    @property
    def feature_names(self):
        return (
            [f"q_goal:{g}" for g in GOALS]
            + [f"p_goal:{g}" for g in GOALS]
            + [f"match:{g}" for g in GOALS]
            + ["goal_overlap", "goal_any_match"]
            + [f"cat:{c}" for c in self.categories]
            + ["tfidf_word_sim", "tfidf_char_sim", "missing_age", "missing_sum_assured", "missing_term",
               "age_above_min", "age_below_max", "log10_sa_over_min", "term_above_min"]
        )

    def transform(self, raw_query, intent, policy_ids):
        idx = np.array([self.index[p] for p in policy_ids])
        n = len(idx)
        q = str(raw_query).lower()
        qv = goal_vector(q)
        qg = np.tile(qv, (n, 1))
        pg = self.pgoal[idx]
        overlap = (qg * pg).sum(axis=1)
        sw = (self.word_vec.transform([q]) @ self.Xw[idx].T).toarray()[0]
        sc = (self.char_vec.transform([q]) @ self.Xc[idx].T).toarray()[0]
        age, sa, term = intent.get("customer_age"), intent.get("sum_assured"), intent.get("term")
        zeros = np.zeros(n)
        age_above_min = np.nan_to_num((age - self.min_age[idx]) / 50.0) if age is not None else zeros
        age_below_max = np.nan_to_num((self.max_age[idx] - age) / 50.0) if age is not None else zeros
        if sa:
            ms = np.where(np.isnan(self.min_sa[idx]) | (self.min_sa[idx] <= 0), 1.0, self.min_sa[idx])
            log_sa = np.log10(sa / ms)
        else:
            log_sa = zeros
        term_above_min = np.nan_to_num((term - self.min_term[idx]) / 40.0) if term is not None else zeros
        cols = [
            qg, pg, qg * pg, overlap[:, None], (overlap > 0).astype(float)[:, None], self.pcat[idx],
            sw[:, None], sc[:, None],
            np.full((n, 1), float(age is None)), np.full((n, 1), float(not sa)), np.full((n, 1), float(term is None)),
            age_above_min[:, None], age_below_max[:, None], log_sa[:, None], term_above_min[:, None],
        ]
        return np.hstack(cols)
