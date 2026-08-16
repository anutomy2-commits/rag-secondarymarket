"""
PRODUCTION retriever — same search logic as retrievers.py, but hardened
for real use instead of a learning script. What changed, and why:

1. PERSISTENT STORAGE (ChromaDB) instead of recomputing embeddings every run.
   The learning version re-embeds all 760 chunks EVERY time you run it —
   ~20-30 seconds wasted, every single time, forever. Production systems
   embed ONCE, save to disk, and just load the saved index on every
   later run (milliseconds instead of seconds).

2. LOGGING instead of nothing. If retrieval breaks in production, you
   need to know WHERE and WHY without adding print statements and
   re-running. Every stage logs what it did.

3. ERROR HANDLING. The learning version assumes everything works. This
   version assumes things WILL occasionally fail (a malformed chunk, an
   embedding model hiccup, a missing index) and fails gracefully with a
   clear message instead of crashing the whole process.

4. CONFIG instead of hardcoded values. Model name, k, thresholds are
   set in one place (RetrieverConfig) instead of scattered as magic
   numbers, so you can tune them without hunting through the code.

5. INCREMENTAL UPDATES. If you add one new PDF, the learning version
   forces you to re-embed EVERYTHING. This version can add just the
   new chunks to the existing index.

6. ACCESS CONTROL. search() accepts a role/policy and scopes retrieval to
   the doc_types that role may see -- enforced on BOTH the dense (Chroma
   `where`) and sparse (BM25) paths, plus a final hard re-check, so a user
   never retrieves, generates from, or cites a document they aren't
   entitled to. See access_control.py.

Same public interface as before: .search(query, k) -> list of chunk dicts.
Anything that used the old BaselineRetriever/HybridRetriever can switch
to this with minimal changes.
"""

import os, logging, time, pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from access_control import AccessPolicy, resolve_policy, filter_chunks_by_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("retriever")


@dataclass
class RetrieverConfig:
    persist_dir: str = "./data/chroma_store"   # where the index lives on disk
    collection_name: str = "offplan_resale_corpus"
    embedding_model: str = "all-MiniLM-L6-v2"
    rrf_k: int = 60                            # reciprocal-rank-fusion constant
    default_k: int = 5
    dtype_boost: Optional[dict] = None
    dense_weight: float = 0.7                   # match the tuned hybrid (0.7 dense / 0.3 sparse)
    sparse_weight: float = 0.3


class _EmbeddingModel:
    """Loads the embedding model once, reused everywhere. Loading it is
    the slow part (a few seconds) -- we don't want to pay that cost twice."""
    _instance = None

    @classmethod
    def get(cls, name):
        if cls._instance is None:
            log.info(f"Loading embedding model '{name}' (first call only)...")
            t = time.time()
            cls._instance = SentenceTransformer(name)
            log.info(f"  loaded in {time.time()-t:.1f}s")
        return cls._instance


def _tok(s):
    return [w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split() if w]


class ProductionRetriever:
    """
    Hybrid (dense + BM25) retriever backed by a persistent Chroma collection.

    First run:  build_index(chunks)  -> embeds everything once, saves to disk.
    Later runs: just instantiate + .search(...) -> loads existing index, no
                re-embedding, starts in milliseconds.
    """
    name = "production_hybrid"

    def __init__(self, config: RetrieverConfig = None):
        self.cfg = config or RetrieverConfig()
        self.model = _EmbeddingModel.get(self.cfg.embedding_model)

        try:
            self.client = chromadb.PersistentClient(
                path=self.cfg.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:
            log.error(f"Could not open Chroma store at {self.cfg.persist_dir}: {e}")
            raise

        self.collection = self.client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_chunks = None  # cache: id -> chunk dict, for BM25 lookups
        self._bm25_path = os.path.join(self.cfg.persist_dir, "bm25_index.pkl")
        self._fingerprint_path = os.path.join(self.cfg.persist_dir, "index_fingerprint.json")
        self._load_bm25_if_exists()

    def save_fingerprint(self, fingerprint):
        """Record how this index was built, so a later startup can detect
        whether the corpus/model/chunking has changed since."""
        import json as _json
        os.makedirs(self.cfg .persist_dir, exist_ok=True)
        with open(self._fingerprint_path, "w") as f:
            _json.dump(fingerprint, f, indent=2)

    def load_fingerprint(self):
        """Return the fingerprint saved when the index was last built, or None."""
        import json as _json
        if os.path.exists(self._fingerprint_path):
            try:
                with open(self._fingerprint_path) as f:
                    return _json.load(f)
            except Exception:
                return None
        return None

    def is_index_valid(self, current_fingerprint):
        """
        Decide whether the persisted index can be trusted for the CURRENT
        corpus state. Returns (valid: bool, reason: str). The reason string
        makes the rebuild decision auditable in logs -- you can see exactly
        WHY a rebuild was (or wasn't) triggered.
        """
        try:
            count = self.collection.count()
        except Exception as e:
            return False, f"could not read index ({e})"
        if count == 0:
            return False, "index is empty"
        saved = self.load_fingerprint()
        if saved is None:
            return False, "no fingerprint on disk (index provenance unknown)"
        if saved.get("hash") != current_fingerprint.get("hash"):
            # spell out what changed, for the log
            changed = []
            for key in ("embedding_model", "chunking_version", "n_source_files"):
                if saved.get(key) != current_fingerprint.get(key):
                    changed.append(f"{key}: {saved.get(key)} -> {current_fingerprint.get(key)}")
            detail = "; ".join(changed) if changed else "source files changed (size/mtime)"
            return False, f"fingerprint mismatch ({detail})"
        return True, "fingerprint matches current corpus"

    def _load_bm25_if_exists(self):
        """BM25 has no native persistence (unlike Chroma's dense index), so
        we pickle it ourselves. Without this, every fresh process would lose
        keyword search until build_index() was called again."""
        if os.path.exists(self._bm25_path):
            try:
                with open(self._bm25_path, "rb") as f:
                    self._bm25, self._bm25_ids, self._bm25_chunks = pickle.load(f)
                log.info(f"Loaded BM25 index from disk ({len(self._bm25_ids)} chunks)")
            except Exception as e:
                log.warning(f"Could not load cached BM25 index, will rebuild on next "
                            f"build_index() call: {e}")

    def _save_bm25(self):
        os.makedirs(self.cfg.persist_dir, exist_ok=True)
        with open(self._bm25_path, "wb") as f:
            pickle.dump((self._bm25, self._bm25_ids, self._bm25_chunks), f)

    # ------------------------------------------------------------------
    def build_index(self, chunks, batch_size=64, fingerprint=None):
        """
        Embed and store all chunks. Safe to call again later with NEW
        chunks only (e.g. after adding a PDF) -- Chroma upserts by id,
        so re-adding the same chunk_id just overwrites, no duplicates.
        """
        if not chunks:
            log.warning("build_index called with 0 chunks -- nothing to do.")
            return

        log.info(f"Indexing {len(chunks)} chunks in batches of {batch_size}...")
        t0 = time.time()
        n_ok, n_failed = 0, 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                texts = [c["text"] for c in batch]
                embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
                self.collection.upsert(
                    ids=[c["id"] for c in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[{"source": c["source"], "doc_type": c["doc_type"],
                                "page": c["page"], "strategy": c["strategy"],
                                "char_count": c.get("char_count", len(c["text"])),
                                "chunk_index": c.get("chunk_index", -1)}
                               for c in batch],
                )
                n_ok += len(batch)
            except Exception as e:
                # one bad batch shouldn't kill the whole indexing run
                log.error(f"Batch {i}-{i+len(batch)} failed to index: {e}")
                n_failed += len(batch)

        log.info(f"Indexed {n_ok} chunks ({n_failed} failed) in {time.time()-t0:.1f}s "
                  f"-> saved to {self.cfg.persist_dir}")

        # BM25 needs the full corpus in memory for keyword search --
        # rebuild it whenever the index changes
        self._rebuild_bm25(chunks)

        # record how this index was built, so a later startup can detect
        # whether the corpus/model/chunking changed since
        if fingerprint is not None:
            self.save_fingerprint(fingerprint)
            log.info(f"Saved index fingerprint: {fingerprint.get('hash')}")

    def _rebuild_bm25(self, chunks):
        self._bm25_chunks = {c["id"]: c for c in chunks}
        self._bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
        self._bm25_ids = [c["id"] for c in chunks]
        self._save_bm25()
        log.info(f"BM25 index saved to {self._bm25_path}")

    # ------------------------------------------------------------------
    def search(self, query, k=None, pool=30, policy=None, role=None):
        """
        Retrieve top-k chunks for `query`, scoped to the caller's access
        policy. Pass either a resolved AccessPolicy (`policy`) or a role
        string (`role`); if neither is given, the least-privilege default
        role applies (see access_control.resolve_policy).

        Access control is enforced on BOTH retrieval paths:
          - dense (Chroma): via the `where` doc_type filter, applied by the
            store so out-of-scope chunks are never even returned.
          - sparse (BM25): filtered explicitly here, because BM25 is an
            in-memory index with no `where` support -- a keyword hit would
            otherwise bypass the filter.
          - final pass: filter_chunks_by_policy re-checks everything returned,
            guaranteeing the invariant regardless of which path produced a
            chunk. Belt and suspenders on purpose.
        """
        k = k or self.cfg.default_k
        if not query or not query.strip():
            log.warning("Empty query passed to search() -- returning no results.")
            return []

        # Resolve entitlements. Fail closed: unknown role raises in
        # resolve_policy; missing role -> least-privilege default.
        if policy is None:
            policy = resolve_policy(role)
        where = policy.where_clause()   # None == unrestricted

        try:
            qvec = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        except Exception as e:
            log.error(f"Embedding the query failed: {e}")
            return []

        # dense search via Chroma (persistent, disk-backed), scoped by `where`
        try:
            if where is not None:
                dense = self.collection.query(query_embeddings=[qvec], n_results=pool, where=where)
            else:
                dense = self.collection.query(query_embeddings=[qvec], n_results=pool)
        except Exception as e:
            log.error(f"Chroma query failed: {e} -- falling back to BM25-only if available")
            dense = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        dense_ids = dense["ids"][0]

        # sparse (BM25) search, if we have an in-memory index for it.
        # BM25 has no `where` support, so filter its candidates against the
        # policy explicitly -- otherwise keyword hits bypass access control.
        sparse_ids = []
        if self._bm25 is not None:
            try:
                scores = self._bm25.get_scores(_tok(query))
                order = np.argsort(-scores)
                for i in order:
                    cid = self._bm25_ids[i]
                    if policy.unrestricted:
                        sparse_ids.append(cid)
                    else:
                        c = self._bm25_chunks.get(cid) if self._bm25_chunks else None
                        if c and policy.can_see(c.get("doc_type", "")):
                            sparse_ids.append(cid)
                    if len(sparse_ids) >= pool:
                        break
            except Exception as e:
                log.warning(f"BM25 search failed, continuing dense-only: {e}")

        # weighted reciprocal rank fusion (matches the tuned hybrid)
        fused = {}
        for rank, cid in enumerate(dense_ids):
            fused[cid] = fused.get(cid, 0) + self.cfg.dense_weight * (1.0 / (self.cfg.rrf_k + rank))
        for rank, cid in enumerate(sparse_ids):
            fused[cid] = fused.get(cid, 0) + self.cfg.sparse_weight * (1.0 / (self.cfg.rrf_k + rank))

        if not fused:
            log.warning(f"No results for query {query!r} under role {policy.role!r}")
            return []

        # rank ALL fused ids; we trim to k AFTER access-filtering, so a
        # restricted role still gets up to k visible results from the pool
        ranked_ids = sorted(fused, key=lambda x: -fused[x])

        # assemble full chunk dicts for the results.
        # dense results carry their own metadata/documents; BM25-only ids
        # (fused in but outside the dense pool) must resolve from the cached
        # bm25 chunk map. Resolve from whichever source has the id.
        meta_by_id = {cid: m for cid, m in zip(dense["ids"][0], dense["metadatas"][0])}
        doc_by_id = {cid: d for cid, d in zip(dense["ids"][0], dense["documents"][0])}

        out, unresolved = [], []
        for cid in ranked_ids:
            if cid in meta_by_id:
                m, txt = meta_by_id[cid], doc_by_id[cid]
            elif self._bm25_chunks and cid in self._bm25_chunks:
                c = self._bm25_chunks[cid]
                m = {"source": c["source"], "doc_type": c["doc_type"], "page": c["page"]}
                txt = c["text"]
            else:
                # id fused from a ranking but resolvable in neither store --
                # log it rather than silently returning fewer than k results
                unresolved.append(cid)
                continue
            out.append({"id": cid, "text": txt, "source": m["source"],
                        "doc_type": m["doc_type"], "page": m["page"],
                        "score": float(fused[cid])})

        if unresolved:
            log.warning(f"{len(unresolved)} ranked id(s) could not be resolved to a "
                        f"chunk (not in dense results or BM25 cache); "
                        f"Unresolved: {unresolved[:5]}")

        # FINAL hard guarantee: every returned chunk must be policy-visible,
        # regardless of which path produced it. Then trim to k.
        out = filter_chunks_by_policy(out, policy)
        return out[:k]

    def stats(self):
        """Quick health check -- how big is the index right now."""
        try:
            count = self.collection.count()
            return {"indexed_chunks": count, "persist_dir": self.cfg.persist_dir,
                    "bm25_ready": self._bm25 is not None}
        except Exception as e:
            return {"error": str(e)}