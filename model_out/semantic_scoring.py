
"""Semantic scoring with cached policy embeddings. Raises SemanticUnavailable when the encoder cannot be loaded."""
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticUnavailable(RuntimeError):
    pass


def _s(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def policy_text(rec):
    return ". ".join(p for p in (_s(rec.get("policy_name")), _s(rec.get("category")), _s(rec.get("description"))) if p)


def load_encoder(name_or_path, device=None):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(str(name_or_path), device=device)
    except Exception as e:
        raise SemanticUnavailable(f"could not load sentence encoder {name_or_path}: {e!r}"[:300]) from e


def load_cross_encoder(name_or_path, device=None):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(str(name_or_path), device=device)
    except Exception as e:
        raise SemanticUnavailable(f"could not load cross-encoder {name_or_path}: {e!r}"[:300]) from e


def digest(model_id, texts):
    h = hashlib.sha256(model_id.encode())
    for t in texts:
        h.update(b"\x00" + t.encode())
    return h.hexdigest()


class SemanticScorer:
    def __init__(self, encoder, ids, emb):
        self.encoder, self.ids, self.emb = encoder, list(ids), np.asarray(emb, dtype=np.float32)
        self.index = {p: i for i, p in enumerate(self.ids)}

    @classmethod
    def from_catalogue(cls, encoder, model_id, records, cache_path):
        ids = [r["policy_id"] for r in records]
        texts = [policy_text(r) for r in records]
        dg = digest(model_id, texts)
        cache_path = Path(cache_path)
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            if str(z["digest"]) == dg and list(z["ids"]) == ids:
                return cls(encoder, ids, z["emb"])
        emb = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        np.savez(cache_path, ids=np.array(ids), emb=emb.astype(np.float32), digest=np.array(dg), model=np.array(model_id))
        return cls(encoder, ids, emb)

    def embed_queries(self, queries):
        return self.encoder.encode(list(queries), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def vec(self, pid):
        return self.emb[self.index[pid]]

    def scores(self, query, policy_ids):
        q = self.embed_queries([query])[0]
        return np.array([float(q @ self.emb[self.index[p]]) for p in policy_ids])
