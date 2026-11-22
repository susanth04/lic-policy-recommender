
"""Score fusion helpers shared by the notebook and the serving code."""
import numpy as np

EXTRA_ORDER = ["kg", "sem", "ce"]
EXTRA_WIDTH = {"kg": 5, "sem": 1, "ce": 1}


def minmax(values):
    """Min-max to [0, 1]; a constant (or empty) vector maps to 0.5."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.full(v.shape, 0.5)
    return (v - lo) / (hi - lo)


def rank_order(scores, ids):
    """Indices sorted by score desc; ties (to 12 decimals) broken by policy id."""
    return sorted(range(len(ids)), key=lambda i: (-round(float(scores[i]), 12), str(ids[i])))


def fuse(components, weights):
    """components: name -> raw score vector; weights: name -> weight. Weights are renormalised over the names given."""
    names = [n for n in weights if n in components]
    total = sum(weights[n] for n in names)
    n_items = len(next(iter(components.values()))) if components else 0
    if not names or total <= 0:
        return np.full(n_items, 0.5)
    return sum(weights[n] * minmax(components[n]) for n in names) / total


def rerank_topk(fused, ce, ids, k, w):
    """Re-order only the fused top-k by w*norm(CE) + (1-w)*norm(fused) (normalised inside the top-k); the rest keep fused order."""
    order = rank_order(fused, ids)
    top, rest = order[:k], order[k:]
    if len(top) < 2 or w <= 0:
        return list(top) + list(rest)
    f_n = minmax(np.asarray(fused, dtype=float)[top])
    c_n = minmax(np.asarray(ce, dtype=float)[top])
    blend = w * c_n + (1 - w) * f_n
    new = sorted(range(len(top)), key=lambda j: (-round(float(blend[j]), 12), str(ids[top[j]])))
    return [top[j] for j in new] + list(rest)


def simplex_grid(n, parts=10):
    def rec(k, left):
        if k == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for r in rec(k - 1, left - i):
                yield (i,) + r
    return [tuple(v / parts for v in c) for c in rec(n, parts)]


def build_extra(extras, kg_mat=None, sem=None, ce=None):
    """Column block of extra GBR features in the fixed order kg, sem, ce."""
    cols, names = [], []
    for e in EXTRA_ORDER:
        if e not in extras:
            continue
        if e == "kg":
            cols.append(np.asarray(kg_mat, dtype=float))
            names += ["kg_score", "kg_goal", "kg_category", "kg_risk", "kg_premium"]
        elif e == "sem":
            cols.append(np.asarray(sem, dtype=float).reshape(-1, 1))
            names.append("semantic_cosine")
        elif e == "ce":
            cols.append(np.asarray(ce, dtype=float).reshape(-1, 1))
            names.append("cross_encoder_logit")
    if not cols:
        return None, []
    return np.hstack(cols), names
