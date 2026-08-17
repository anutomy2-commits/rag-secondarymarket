"""
Gradio UI for the RAG pipeline -- a free-tier-deployable front door for
Hugging Face Spaces (Docker Spaces require a paid tier; Gradio Spaces are
free). Same pipeline, same src/ modules as src/api.py's /v1/ask -- this
file only swaps the transport (HTTP JSON -> chat UI). Nothing under src/
was changed; see api.py's lifespan() and ask() for the reference this
mirrors step for step.

Run it locally (from the repo root, so ./data/chroma_store resolves):
    pip install -r requirements.txt
    python src/app.py
    # opens http://127.0.0.1:7860

Needs ANTHROPIC_API_KEY (in .env locally, or a Space Secret on Spaces).
"""
import uuid

import gradio as gr

from ingest import structured_chunks_deduped, corpus_fingerprint
from retrievers_production import ProductionRetriever, RetrieverConfig
from generate_llm import generate_answer_llm
from verify import verify_citations, confidence_score
from memory import SessionMemoryStore, contextualize_query

ROLES = ["public", "agent", "legal", "analyst"]

# ---- app state: built ONCE at import time, mirrors api.py's lifespan() ----
STATE = {"retriever": None, "n_chunks": 0, "sessions": SessionMemoryStore()}


def _build_retriever():
    print("Starting up: loading ChromaDB-backed retriever...")
    cfg = RetrieverConfig(persist_dir="./data/chroma_store",
                          dense_weight=0.7, sparse_weight=0.3)
    retriever = ProductionRetriever(cfg)

    # same rebuild-only-if-stale decision as api.py's lifespan
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


_build_retriever()


# ---- formatting helpers for the sources/confidence side panels ----
def _format_sources(retrieved):
    if not retrieved:
        return ("_No sources retrieved -- either nothing matched, or this "
                "role has no visibility into this topic._")
    return "\n".join(
        f"- **{c['source']}** (p.{c['page']}, `{c['doc_type']}`) — score {c['score']:.4f}"
        for c in retrieved
    )


def _format_confidence(conf, faithfulness, role, rewritten):
    lines = [f"**Role used:** `{role}`"]
    if rewritten:
        lines.append(f"**Rewritten query:** {rewritten}")
    lines.append(f"**Composite confidence:** {conf['composite']}")
    lines.append(f"- retrieval confidence: {conf['retrieval_confidence']}")
    lines.append(f"- citation coverage: {conf['citation_coverage']}")
    lines.append(f"- completeness: {conf['completeness']}")
    if faithfulness is not None:
        lines.append(f"**Faithfulness (verified):** {faithfulness:.2f}")
    return "\n".join(lines)


# ---- per-message pipeline, mirrors api.py's ask() step for step ----
def respond(message, chat_history, role, verify, session_id):
    if not message or not message.strip():
        return chat_history, gr.update(), gr.update(), ""

    if STATE["retriever"] is None:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Retriever not ready yet -- try again in a moment."},
        ]
        return chat_history, gr.update(), gr.update(), ""

    memory = STATE["sessions"].get(session_id)

    # 1. contextualize: rewrite a follow-up into a standalone query using
    #    history, so retrieval works even when the question says "it"/"that".
    search_query = message
    rewritten = None
    try:
        search_query, changed = contextualize_query(message, memory)
        rewritten = search_query if changed else None
    except RuntimeError:
        search_query = message   # no key etc -> fall back to raw question

    # 2. retrieve on the (possibly rewritten) standalone query, scoped to role
    retrieved = STATE["retriever"].search(search_query, k=5, role=role)

    # 3. generate, passing recent history so the answer can refer back naturally
    history_text = memory.history_text()
    try:
        ans = generate_answer_llm(message, retrieved, history=history_text)
    except RuntimeError as e:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Generation failed: {e}"},
        ]
        return chat_history, _format_sources(retrieved), gr.update(), ""

    # 4. optional faithfulness verification, then confidence
    verification = None
    faithfulness = None
    if verify:
        verification = verify_citations(ans, retrieved)
        faithfulness = verification.get("faithfulness_rate")
    conf = confidence_score(ans, retrieved, verification)

    # 5. remember this turn for future follow-ups
    memory.add(message, ans["answer"])

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ans["answer"]},
    ]
    return (chat_history,
            _format_sources(retrieved),
            _format_confidence(conf, faithfulness, role, rewritten),
            "")


def clear_conversation(session_id):
    STATE["sessions"].reset(session_id)
    return [], "", ""


def _new_session_id():
    return str(uuid.uuid4())


with gr.Blocks(title="Dubai Off-Plan Resale RAG") as demo:
    session_id = gr.State()
    demo.load(_new_session_id, None, session_id)   # one isolated session per browser tab

    gr.Markdown(
        "# Dubai Off-Plan Resale RAG\n"
        "Grounded, cited answers over DLD laws/regulations and PropertyFinder "
        "listings. Switch **Role** and re-ask the same question to see access "
        "control change which sources come back."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, label="Conversation")
            question = gr.Textbox(placeholder="Ask about off-plan resale rules, fees, listings...",
                                  label="Question", lines=2)
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear conversation")
        with gr.Column(scale=1):
            role = gr.Dropdown(ROLES, value="public", label="Role (access control)")
            verify = gr.Checkbox(value=False, label="Verify citations (slower, extra LLM calls)")
            gr.Markdown("### Retrieved sources")
            sources_out = gr.Markdown()
            gr.Markdown("### Confidence")
            confidence_out = gr.Markdown()

    ask_btn.click(respond, [question, chatbot, role, verify, session_id],
                  [chatbot, sources_out, confidence_out, question])
    question.submit(respond, [question, chatbot, role, verify, session_id],
                    [chatbot, sources_out, confidence_out, question])
    clear_btn.click(clear_conversation, [session_id],
                    [chatbot, sources_out, confidence_out])


if __name__ == "__main__":
    demo.launch()
