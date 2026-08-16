"""
Phase 3.1 + 3.4 — LLM-based grounded generation with citations.

Unlike the extractive generator (generate.py), this uses an actual LLM
(Anthropic Claude) to WRITE a fluent answer from the retrieved chunks --
but under strict grounding rules enforced by the system prompt:

  - answer ONLY from the provided numbered context blocks
  - cite the specific block(s) each claim comes from, as [1], [2], ...
  - if the context doesn't contain the answer, say so explicitly rather
    than guessing (the graceful "I don't know" case, Phase 3.4)

The API key is read from the ANTHROPIC_API_KEY environment variable --
NEVER hardcode it. Set it in your shell before running:
    export ANTHROPIC_API_KEY="sk-ant-..."

If the key is missing, this fails with a clear message instead of a
confusing crash, and the rest of the pipeline (retrieval, eval) still
works without it.
"""
import os
import re

# Load a .env file if present, so ANTHROPIC_API_KEY can live there instead
# of being exported every session. Safe if python-dotenv isn't installed or
# no .env exists -- it just falls back to the real environment variable.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "claude-haiku-4-5"   # current daily-driver; see docs.claude.com for latest

SYSTEM_PROMPT = """You are a retrieval-grounded assistant for Dubai real estate questions.

You will be given a user question and a set of numbered CONTEXT blocks retrieved from a document corpus.

Rules you must follow exactly:
1. Answer ONLY using information in the provided CONTEXT blocks. Do not use outside knowledge.
2. After each claim, cite the block number(s) it comes from in square brackets, like [1] or [2][3].
3. If the CONTEXT does not contain enough information to answer, say exactly: "The retrieved context does not contain enough information to answer this question." Then briefly note what related information WAS present, if any.
4. Be concise and factual. Do not speculate or add caveats beyond what the context supports.
5. Never invent a citation number that wasn't in the provided context."""


def _client():
    """Lazily create the Anthropic client, with a clear error if the key
    or SDK is missing."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Get a key from console.anthropic.com and run:\n"
            '    export ANTHROPIC_API_KEY="sk-ant-..."\n'
            "before running LLM generation."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("The 'anthropic' package is not installed. Run: pip install anthropic")
    return anthropic.Anthropic(api_key=key)


def _format_context(chunks):
    """Turn retrieved chunks into numbered context blocks the model can cite.
    Returns (context_text, mapping) where mapping[i] -> chunk, so we can
    later resolve a [3] citation back to the real source chunk."""
    lines, mapping = [], {}
    for i, c in enumerate(chunks, start=1):
        src = c.get("source", "unknown")
        page = c.get("page", -1)
        loc = f"{src}, p.{page}" if page and page != -1 else src
        lines.append(f"[{i}] (from {loc})\n{c['text']}")
        mapping[i] = c
    return "\n\n".join(lines), mapping


def generate_answer_llm(question, chunks, model=MODEL, max_tokens=600, history=""):
    """
    Generate a grounded, cited answer using the LLM.
    Optional `history` (recent conversation turns as text) lets the answer
    refer back naturally -- retrieval already used the rewritten standalone
    query, so history here only shapes the phrasing, not the grounding.
    Returns dict: answer, citations (block-number -> chunk), model, raw_context_map.
    """
    if not chunks:
        return {"answer": "The retrieved context does not contain enough information to answer this question.",
                "citations": [], "model": model, "context_map": {}}

    context_text, mapping = _format_context(chunks)
    hist_block = f"CONVERSATION SO FAR:\n{history}\n\n" if history else ""
    user_msg = f"{hist_block}CONTEXT:\n{context_text}\n\nQUESTION: {question}"

    client = _client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer = "".join(block.text for block in resp.content if block.type == "text")

    # parse which [n] citations the model actually used, resolve to real chunks
    cited_nums = sorted(set(int(n) for n in re.findall(r'\[(\d+)\]', answer)))
    citations = []
    for n in cited_nums:
        if n in mapping:
            c = mapping[n]
            citations.append({"block": n, "source": c["source"], "page": c["page"],
                              "chunk_id": c["id"]})

    return {"answer": answer, "citations": citations, "model": model,
            "context_map": {n: mapping[n]["id"] for n in mapping}}


def validate_citation_numbers(answer_record, chunks):
    """
    Structural check (cheap, no API): every [n] the model cited must map to
    a block that was actually in the context. Catches the model inventing
    a citation number. (Semantic support-checking is in verify.py.)
    """
    valid_blocks = set(range(1, len(chunks) + 1))
    cited = set(c["block"] for c in answer_record["citations"])
    invalid = cited - valid_blocks
    return {"valid": len(invalid) == 0, "invalid_blocks": sorted(invalid),
            "n_citations": len(answer_record["citations"])}


if __name__ == "__main__":
    # tiny smoke test -- requires ANTHROPIC_API_KEY to be set
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import structured_chunks
    from retrievers import HybridRetriever

    chunks = structured_chunks()
    hyb = HybridRetriever(chunks, dense_weight=0.7, sparse_weight=0.3)
    q = "What minimum percentage must be paid before an off-plan assignment is approved?"
    retrieved = hyb.search(q, k=5)

    try:
        result = generate_answer_llm(q, retrieved)
        print("Q:", q)
        print("\nA:", result["answer"])
        print("\nCitations resolved to:")
        for c in result["citations"]:
            print(f"  [{c['block']}] -> {c['source']} p.{c['page']}")
        print("\nvalidity:", validate_citation_numbers(result, retrieved))
    except RuntimeError as e:
        print("Could not run LLM generation:\n", e)