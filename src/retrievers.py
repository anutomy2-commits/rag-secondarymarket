"""
Retrievers behind one interface so the eval is apples-to-apples.

BaselineRetriever  : naive chunks + dense cosine top-k. No structure, no hybrid.
HybridRetriever    : structured chunks + dense + BM25 (reciprocal-rank fusion),
                     with a configurable dense/sparse weight (default 0.7/0.3,
                     matching the guide's spec) instead of treating both
                     signals as equal.
Reranker           : cross-encoder second pass. Takes a wider candidate pool
                     (e.g. top-20 from HybridRetriever) and rescores each one
                     directly against the query for actual relevance -- RRF/
                     BM25/cosine are all proxies for relevance; a cross-encoder
                     reads query+chunk together and scores relevance directly,
                     which is why it typically outperforms the fusion ranking
                     alone. Keeps only the top-k after rescoring.
RerankedRetriever  : wraps any base retriever + a Reranker into one .search()
                     call, so it drops into the eval harness the same way.

All expose .search(query, k) -> list[chunk dicts] with a 'score'.
Embeddings + reranker: local (no API key). Reranker model is small
(cross-encoder/ms-marco-MiniLM-L-6-v2, ~80MB) and CPU-friendly.
"""
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

_MODEL = None
def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

_RERANKER = None
def _reranker_model():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER

def _embed(texts):
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)

def _tok(s):
    return [w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split() if w]


class BaselineRetriever:
    """Naive: fixed-size chunks, single dense embedding, cosine top-k."""
    name = "baseline_naive"
    def __init__(self, chunks):
        self.chunks = chunks
        self.emb = _embed([c["text"] for c in chunks])
    def search(self, query, k=5):
        q = _embed([query])[0]
        sims = self.emb @ q
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            c = dict(self.chunks[int(i)]); c["score"] = float(sims[int(i)])
            out.append(c)
        return out


class HybridRetriever:
    """
    Structured chunks + dense + BM25, fused by WEIGHTED reciprocal rank
    fusion. dense_weight/sparse_weight let you tune how much each signal
    counts -- e.g. 0.7/0.3 leans toward semantic matches but still lets
    exact keyword hits (a specific law number, a fee figure) pull weight.
    Defaults to 0.5/0.5 (unweighted, same behavior as before) unless set.
    """
    name = "hybrid_structured"
    def __init__(self, chunks, rrf_k=60, dtype_boost=None,
                 dense_weight=0.5, sparse_weight=0.5):
        self.chunks = chunks
        self.emb = _embed([c["text"] for c in chunks])
        self.bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
        self.rrf_k = rrf_k
        self.dtype_boost = dtype_boost or {}
        # normalize weights so they always sum to 1, regardless of what's passed in
        total = dense_weight + sparse_weight
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total

    def search(self, query, k=5, pool=30):
        # dense ranking
        q = _embed([query])[0]
        dense = np.argsort(-(self.emb @ q))[:pool]
        # sparse ranking
        sparse = np.argsort(-self.bm25.get_scores(_tok(query)))[:pool]
        # weighted reciprocal rank fusion
        score = {}
        for rank, i in enumerate(dense):
            score[int(i)] = score.get(int(i), 0) + self.dense_weight * (1.0/(self.rrf_k+rank))
        for rank, i in enumerate(sparse):
            score[int(i)] = score.get(int(i), 0) + self.sparse_weight * (1.0/(self.rrf_k+rank))
        # optional doc_type boost (light prior toward likely-relevant types)
        for i in list(score):
            b = self.dtype_boost.get(self.chunks[i]["doc_type"], 0.0)
            score[i] += b
        ranked = sorted(score.items(), key=lambda x: -x[1])[:k]
        out = []
        for i, s in ranked:
            c = dict(self.chunks[i]); c["score"] = float(s)
            out.append(c)
        return out


class Reranker:
    """
    Cross-encoder second pass. Unlike dense/BM25 (which score the QUERY and
    the CHUNK separately, then compare), a cross-encoder reads them TOGETHER
    in one forward pass and outputs a direct relevance score -- slower per
    item, so it's only run on a small candidate pool, not the whole corpus.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
        model = _reranker_model()
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
        order = np.argsort(-scores)[:top_k]
        out = []
        for i in order:
            i = int(i)
            c = dict(candidates[i])
            c["rerank_score"] = float(scores[i])
            c["prerank_score"] = c.get("score")  # keep the original fusion score for comparison
            out.append(c)
        return out


class RerankedRetriever:
    """
    Wraps a base retriever + Reranker into one .search() call:
    base retriever pulls a WIDE pool (e.g. top-20), reranker narrows it
    to the final top-k using direct query-chunk relevance scoring.
    Drops into the eval harness exactly like the other retrievers.
    """
    def __init__(self, base_retriever, reranker=None, pool_size=20):
        self.base = base_retriever
        self.reranker = reranker or Reranker()
        self.pool_size = pool_size
        self.name = f"{base_retriever.name}_reranked"

    def search(self, query, k=5):
        pool = self.base.search(query, k=self.pool_size)
        return self.reranker.rerank(query, pool, top_k=k)