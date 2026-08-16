"""
FastAPI service exposing the RAG pipeline over HTTP.

Design notes:
  - The retriever is built ONCE at startup and held in memory (loading
    embeddings per-request would add ~40s to every call). This is the
    standard "load once, serve many" production pattern.
  - /ask is fast by default (retrieve + generate + confidence). Faithfulness
    verification (LLM-as-judge on every claim) is OPTIONAL via verify=true,
    because judging every live request is slow and costs money -- in
    production you verify offline/selectively, not on every query.
  - ACCESS CONTROL: each /ask carries a `role` that scopes which doc_types
    are retrievable. See the SECURITY NOTE on the endpoint -- in the demo the
    role comes from the request; in production it MUST come from the
    authenticated user, not the client.

  NOTE ON WORKERS: SessionMemoryStore is in-process memory. Run a SINGLE
  worker, or move sessions to shared storage (e.g. Redis) -- otherwise a
  user's follow-up can land on a worker that never saw the first turn, and
  conversation memory (which the follow-up query rewriter depends on) breaks.

Run it:
    pip install fastapi uvicorn
    uvicorn src.api:app --reload
    # then open http://localhost:8000/docs  (interactive API page)

Needs ANTHROPIC_API_KEY (in .env) for /ask. /health and /documents work
without it.
"""
import os, sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ingest import structured_chunks_deduped, corpus_fingerprint, CORPUS
from retrievers_production import ProductionRetriever, RetrieverConfig
from generate_llm import generate_answer_llm, validate_citation_numbers
from verify import verify_citations, confidence_score
from memory import SessionMemoryStore, contextualize_query

# ---- app state: built once at startup, reused across requests ----
STATE = {"retriever": None, "n_chunks": 0, "sessions": SessionMemoryStore()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts
    print("Starting up: loading ChromaDB-backed retriever...")
    cfg = RetrieverConfig(persist_dir="./data/chroma_store",
                          dense_weight=0.7, sparse_weight=0.3)
    retriever = ProductionRetriever(cfg)

    # Decide whether the persisted index is still valid for the CURRENT
    # corpus, embedding model, and chunking version -- rebuild only if not.
    fp = corpus_fingerprint(embedding_model=cfg.embedding_model)
    valid, reason = retriever.is_index_valid(fp)
    if valid:
        print(f"  index valid ({reason}) -> loading from disk, no rebuild")
    else:
        print(f"  rebuilding index -> reason: {reason}")
        chunks = structured_chunks_deduped()
        retriever.build_index(chunks, fingerprint=fp)

    stats = retriever.stats()
    STATE["retriever"] = retriever
    STATE["n_chunks"] = stats.get("indexed_chunks", 0)
    print(f"Ready: {STATE['n_chunks']} chunks, bm25_ready={stats.get('bm25_ready')}")
    yield


app = FastAPI(
    title="Dubai Off-Plan Resale RAG",
    description="Hybrid-search RAG (ChromaDB) with grounded, cited answers over a Dubai real-estate corpus.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- request/response schemas (FastAPI validates + documents these) ----
class AskRequest(BaseModel):
    question: str = Field(..., description="The question to answer from the corpus.")
    k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve.")
    verify: bool = Field(False, description="Run LLM-as-judge faithfulness verification (slower, costs API calls).")
    use_memory: bool = Field(True, description="Use conversation history to resolve follow-up questions.")
    session_id: str = Field("default", description="Conversation/session id -- isolates each user's memory.")
    role: str = Field("public", description="Caller's access role; scopes which doc_types are retrievable. "
                                            "DEMO ONLY -- see SECURITY NOTE on /v1/ask.")


class Citation(BaseModel):
    block: int
    source: str
    page: int


class AskResponse(BaseModel):
    question: str
    rewritten_query: str | None = None
    answer: str
    citations: list[Citation]
    confidence: dict
    faithfulness: float | None = None
    retrieved_sources: list[str]
    role: str | None = None   # which entitlement the answer was produced under (auditability)


# ---- endpoints ----
@app.get("/health")
def health():
    """Liveness check -- is the server up and the retriever loaded?"""
    return {"status": "ok", "retriever_ready": STATE["retriever"] is not None,
            "n_chunks": STATE["n_chunks"]}


@app.get("/")
def home():
    """Serve the web UI."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/v1/documents")
def documents():
    """List the source documents in the corpus and their declared types."""
    docs = [{"source": name, "doc_type": dtype} for name, dtype in CORPUS.items()]
    return {"count": len(docs), "documents": docs}


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Answer a question from the corpus with grounded, cited output.

    SECURITY NOTE (access control): retrieval is scoped to `req.role` -- a
    user only retrieves, and can only be answered from, doc_types their role
    permits (enforced in ProductionRetriever.search on both the dense and
    BM25 paths). In THIS demo the role is taken from the request body, which
    is fine to demonstrate the mechanism. In PRODUCTION you MUST derive the
    role from the AUTHENTICATED user (SSO / Entra ID / IAM claims) and IGNORE
    any client-supplied role -- otherwise a caller could self-grant access by
    setting role="analyst". The enforcement code does not change; only the
    source of the role does. That is the single line to harden before this
    touches real data.
    """
    if STATE["retriever"] is None:
        raise HTTPException(status_code=503, detail="Retriever not ready yet.")

    memory = STATE["sessions"].get(req.session_id)

    # 1. contextualize: rewrite a follow-up into a standalone query using
    #    history, so retrieval works even when the question says "it"/"that".
    search_query = req.question
    rewritten = None
    if req.use_memory:
        try:
            search_query, changed = contextualize_query(req.question, memory)
            rewritten = search_query if changed else None
        except RuntimeError:
            search_query = req.question   # no key etc -> fall back to raw question

    # 2. retrieve on the (possibly rewritten) standalone query, scoped to role
    retrieved = STATE["retriever"].search(search_query, k=req.k, role=req.role)

    # 3. generate, passing recent history so the answer can refer back naturally
    history = memory.history_text() if req.use_memory else ""
    try:
        ans = generate_answer_llm(req.question, retrieved, history=history)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    faithfulness = None
    verification = None
    if req.verify:
        verification = verify_citations(ans, retrieved)
        faithfulness = verification.get("faithfulness_rate")

    conf = confidence_score(ans, retrieved, verification)

    # 4. remember this turn for future follow-ups
    if req.use_memory:
        memory.add(req.question, ans["answer"])

    return AskResponse(
        question=req.question,
        rewritten_query=rewritten,
        answer=ans["answer"],
        citations=[Citation(block=c["block"], source=c["source"], page=c["page"])
                   for c in ans["citations"]],
        confidence=conf,
        faithfulness=faithfulness,
        retrieved_sources=[c["source"] for c in retrieved],
        role=req.role,
    )


class ResetRequest(BaseModel):
    session_id: str = Field("default", description="Which conversation to clear.")


@app.post("/v1/reset")
def reset(req: ResetRequest):
    """Clear one session's conversation memory (start a fresh conversation)."""
    STATE["sessions"].reset(req.session_id)
    return {"status": "conversation cleared", "session_id": req.session_id}