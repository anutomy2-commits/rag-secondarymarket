"""
Gradio UI for the RAG pipeline -- the free-tier-deployable front door for
Hugging Face Spaces. Same pipeline, same src/ modules as src/api.py's
/v1/ask -- this file only swaps the transport (HTTP JSON -> chat UI).
Nothing under src/ was changed; see api.py's lifespan() and ask() for the
reference this mirrors step for step.

Run it locally (from the repo root, so ./data/chroma_store resolves):
    pip install -r requirements.txt
    python src/app.py
    # opens http://127.0.0.1:7860

Needs ANTHROPIC_API_KEY (in .env locally, or a Space Secret on Spaces).
"""
import uuid

# Free-tier Gradio Spaces run on ZeroGPU hardware, which refuses to start
# unless `spaces` is imported before anything that touches CUDA AND at least
# one @spaces.GPU function exists. This app is CPU-only (embeddings/BM25
# locally, generation via the Anthropic API), so the requirement is satisfied
# with a placeholder below that is never called -- no GPU is ever allocated.
# Import is optional so local runs don't need the Spaces-only package.
try:
    import spaces
except ImportError:
    spaces = None

import gradio as gr

from ingest import structured_chunks_deduped, corpus_fingerprint
from retrievers_production import ProductionRetriever, RetrieverConfig
from generate_llm import generate_answer_llm
from verify import verify_citations, confidence_score
from memory import SessionMemoryStore, contextualize_query

ROLES = ["public", "agent", "legal", "analyst"]


if spaces is not None:
    @spaces.GPU
    def _zerogpu_placeholder():
        """Never called -- exists only so ZeroGPU's startup check passes."""
        return None

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


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_700",
    block_border_color_dark="*neutral_700",
    block_label_background_fill="*neutral_800",
    block_title_text_weight="600",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_500, *secondary_400)",
    button_primary_text_color="white",
)

CUSTOM_CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#hero {
    padding: 26px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1e293b 0%, #1e1b4b 55%, #172554 100%);
    border: 1px solid rgba(99,102,241,0.35);
    margin-bottom: 20px;
    box-shadow: 0 8px 24px -8px rgba(30,27,75,0.6);
}
#hero h1 {
    margin: 0 0 6px 0;
    font-size: 1.6rem;
    background: linear-gradient(90deg, #a5b4fc, #93c5fd);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
#hero p {margin: 0; opacity: 0.85; line-height: 1.5; font-size: 0.94rem;}

/* ---- composer (question + ask + examples) ---- */
#composer {
    border-radius: 16px;
    padding: 16px 18px 12px 18px;
    background: rgba(30,41,59,0.55);
    border: 1px solid rgba(148,163,184,0.15);
    margin-bottom: 18px;
}
#composer_row {align-items: flex-end; gap: 10px;}
#question_box textarea {
    background: rgba(15,23,42,0.65) !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    border-radius: 12px !important;
    font-size: 0.98rem !important;
    padding: 12px 14px !important;
}
#question_box textarea:focus {
    border-color: rgba(129,140,248,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
#ask_btn {
    font-weight: 600;
    border-radius: 12px !important;
    min-width: 96px;
    height: 46px;
}
#composer_footer {
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;
    flex-wrap: wrap;
    gap: 8px;
}
#examples_row {background: transparent !important; border: none !important; padding: 0 !important; flex: 1;}
#examples_row .label {display: none !important;}
#examples_row .gallery {
    display: flex !important;
    flex-wrap: wrap;
    gap: 6px;
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
#examples_row .gallery-item {
    border: 1px solid rgba(148,163,184,0.25) !important;
    border-radius: 999px !important;
    background: rgba(15,23,42,0.55) !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease;
    width: auto !important;
}
#examples_row .gallery-item:hover {
    border-color: rgba(129,140,248,0.6) !important;
    background: rgba(79,70,229,0.15) !important;
    transform: translateY(-1px);
}
#examples_row .gallery-item div {
    font-size: 0.8rem !important;
    color: #cbd5e1 !important;
    white-space: nowrap !important;
    --local-text-width: auto !important;
}
#clear_btn {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 8px !important;
}
#clear_btn:hover {color: #e2e8f0 !important;}

/* ---- chat panel ---- */
#chatbot_card {
    border-radius: 16px !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
}

/* ---- sidebar ---- */
#sidebar {
    border-radius: 16px;
    padding: 18px;
    background: rgba(30,41,59,0.55);
    border: 1px solid rgba(148,163,184,0.15);
    height: 100%;
}
#sidebar h3 {
    margin-top: 14px;
    margin-bottom: 6px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #93c5fd;
    border-bottom: 1px solid rgba(148,163,184,0.2);
    padding-bottom: 4px;
}
#sidebar h3:first-child {margin-top: 0;}
#sources_box, #confidence_box {
    min-height: 40px;
    font-size: 0.9rem;
    line-height: 1.5;
}
#role_row {gap: 8px;}
footer {display: none !important;}
"""

with gr.Blocks(title="Dubai Off-Plan Resale RAG") as demo:
    session_id = gr.State()
    demo.load(_new_session_id, None, session_id)   # one isolated session per browser tab

    with gr.Column(elem_id="hero"):
        gr.Markdown(
            "# 🏗️ Dubai Off-Plan Resale RAG\n"
            "Grounded, cited answers over DLD laws/regulations and PropertyFinder "
            "listings. Switch **Role** and re-ask the same question to see access "
            "control change which sources come back."
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Column(elem_id="composer"):
                with gr.Row(elem_id="composer_row"):
                    question = gr.Textbox(placeholder="Ask about off-plan resale rules, fees, listings...",
                                          show_label=False, lines=1, max_lines=6,
                                          elem_id="question_box", scale=6, container=False)
                    ask_btn = gr.Button("Ask", variant="primary", elem_id="ask_btn", scale=1)
                with gr.Row(elem_id="composer_footer"):
                    gr.Examples(
                        examples=[
                            "What fees apply when reselling an off-plan unit before handover?",
                            "Can I assign my SPA before the project is complete?",
                            "What documents does DLD require for an off-plan resale (Oqood transfer)?",
                        ],
                        inputs=question,
                        label="",
                        elem_id="examples_row",
                    )
                    clear_btn = gr.Button("Clear conversation", elem_id="clear_btn", size="sm")
            chatbot = gr.Chatbot(
                height=440, label="Conversation", elem_id="chatbot_card",
                avatar_images=(None, "🏗️"), buttons=["copy"],
                placeholder="### 💬 Ask about DLD regulations, resale fees, or listings\n"
                            "Answers are grounded in the retrieved documents and cited by source.",
            )
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("### Access control")
            with gr.Row(elem_id="role_row"):
                role = gr.Dropdown(ROLES, value="public", label="Role", scale=2)
            verify = gr.Checkbox(value=False, label="Verify citations (slower, extra LLM calls)")
            gr.Markdown("### Retrieved sources")
            sources_out = gr.Markdown(elem_id="sources_box")
            gr.Markdown("### Confidence")
            confidence_out = gr.Markdown(elem_id="confidence_box")

    ask_btn.click(respond, [question, chatbot, role, verify, session_id],
                  [chatbot, sources_out, confidence_out, question])
    question.submit(respond, [question, chatbot, role, verify, session_id],
                    [chatbot, sources_out, confidence_out, question])
    clear_btn.click(clear_conversation, [session_id],
                    [chatbot, sources_out, confidence_out])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
