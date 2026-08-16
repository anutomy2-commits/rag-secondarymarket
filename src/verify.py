"""
Phase 3.2 + 3.3 — Citation verification (LLM-as-judge) + confidence scoring.

verify_citations(): for each claim+citation in an answer, ask an LLM judge
  whether the cited chunk ACTUALLY supports the claim. This is the semantic
  check that structural validation (does [n] exist?) cannot do. Flags any
  claim whose citation doesn't hold up -- the faithfulness layer.

confidence_score(): combine three signals into one composite number:
  - retrieval confidence: how relevant were the top retrieved chunks?
  - citation coverage:     what fraction of the answer's claims are cited
                           AND verified as supported?
  - completeness:          did the answer actually attempt the question, or
                           fall back to "not enough information"?

Key read from ANTHROPIC_API_KEY (same as generate_llm.py).
"""
import os
import re
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JUDGE_MODEL = "claude-haiku-4-5"

# Single source of truth for detecting a graceful non-answer. Both
# _split_claims and confidence_score used to independently substring-match
# ONE exact phrase -- any drift in the generator's fallback wording
# ("insufficient information", "not enough context") silently scored the
# answer as complete. Centralizing means one place to maintain, and it
# matches several phrasings, not just the literal prompt string.
_NON_ANSWER_MARKERS = [
    "does not contain enough information",
    "not enough information",
    "insufficient information",
    "not enough context",
    "cannot answer",
    "unable to answer",
]

def is_non_answer(answer):
    a = (answer or "").lower()
    return any(m in a for m in _NON_ANSWER_MARKERS)


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set -- see generate_llm.py for setup.")
    import anthropic
    return anthropic.Anthropic(api_key=key)


def _split_claims(answer):
    """Break an answer into sentences. Returns (cited, uncited) where:
      cited   = [(claim_text, [block_nums]), ...]  -- sentences WITH [n] markers
      uncited = [claim_text, ...]                  -- factual-looking sentences
                                                      with NO citation
    We track uncited sentences so citation COVERAGE can count them against
    the answer (an answer that cites one sentence and leaves five factual
    claims uncited should not score full coverage)."""

    #answer = "The DLD transfer fee is 4% of the property value [1]. Buyers usually split it with the seller [2][3]. Payment is due at the time of transfer. OK."

    if is_non_answer(answer):
        return [], []
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    cited, uncited = [], []
    for s in sentences:
        nums = [int(n) for n in re.findall(r'\[(\d+)\]', s)]
        text = re.sub(r'\s*\[\d+\]', '', s).strip()
        if not text or len(text) < 15:
            continue  # skip fragments / non-sentences
        if nums:
            cited.append((text, nums))
        else:
            uncited.append(text)
    return cited, uncited



JUDGE_SYSTEM = """You are a strict citation verifier. You are given a CLAIM and a SOURCE passage.
Decide whether the SOURCE actually supports the CLAIM.

Respond with ONLY a JSON object, no other text:
{"supported": true or false, "reason": "<one short sentence>"}

"supported" is true ONLY if the SOURCE contains information that directly backs the CLAIM.
If the SOURCE is unrelated, contradicts the claim, or is only loosely topical, "supported" is false."""


def verify_citations(answer_record, retrieved_chunks, model=JUDGE_MODEL):
    """
    For each cited claim, ask the judge if the cited chunk supports it.
    Returns per-claim verdicts + a summary faithfulness rate + the count of
    uncited factual sentences (so coverage can penalize them).
    You're verifying the LLM's answer against the chunks the LLM was actually shown, not against a fresh retrieval.
    """
    # SAFETY: the generator was shown these chunks as 1-indexed blocks and
    # cited [n] against THAT order. We must verify against the same list in
    # the same order, or [n] resolves to the wrong chunk. If the caller
    # passed a context_map from generation, assert it still matches.

    ctx_map = answer_record.get("context_map")
    if ctx_map:
        current = {i: c["id"] for i, c in enumerate(retrieved_chunks, start=1)}
        if current != ctx_map:
            raise ValueError(
                "Block-numbering mismatch: the chunks passed to verify_citations "
                "are not the same (or same order) as those the generator cited "
                "against. Citations would resolve to the wrong sources. Pass the "
                "exact retrieved_chunks used for generation."
            )

    block_to_chunk = {i: c for i, c in enumerate(retrieved_chunks, start=1)}
    cited, uncited = _split_claims(answer_record["answer"])

    if not cited and not uncited:
        return {"verdicts": [], "faithfulness_rate": None, "n_uncited": 0,
                "note": "No verifiable claims (answer may be a graceful non-answer)."}

    client = _client()
    verdicts = []
    for claim_text, block_nums in cited:
        claim_supported = False
        reasons = []
        for n in block_nums:
            chunk = block_to_chunk.get(n)
            if not chunk:
                reasons.append(f"[{n}] not in context")
                continue
            msg = f"CLAIM: {claim_text}\n\nSOURCE:\n{chunk['text']}"
            resp = client.messages.create(
                model=model, max_tokens=120, system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": msg}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            # non-greedy + anchored: tolerate a '}' inside the reason string
            m = re.search(r'\{.*?"supported".*?\}', raw, re.S)
            try:
                verdict = json.loads(m.group(0)) if m else json.loads(raw)
            except Exception:
                verdict = {"supported": False, "reason": "could not parse judge output"}
            if verdict.get("supported"):
                claim_supported = True
            reasons.append(f"[{n}] {verdict.get('reason','')}")
        verdicts.append({"claim": claim_text, "blocks": block_nums,
                         "supported": claim_supported, "detail": "; ".join(reasons)})

    n_supported = sum(1 for v in verdicts if v["supported"])
    return {"verdicts": verdicts,
            "faithfulness_rate": n_supported / len(verdicts) if verdicts else None,
            "n_claims": len(verdicts), "n_supported": n_supported,
            "n_uncited": len(uncited),          # factual sentences with NO citation
            "uncited_claims": uncited}


def confidence_score(answer_record, retrieved_chunks, verification=None):
    """
    Composite confidence in [0,1] from three sub-scores. Cheap -- no API call
    (reuses verification if you pass it in).
    """
    # 1. retrieval confidence: how good are the retrieved chunks in ABSOLUTE
    #    terms -- NOT where the mean sits in this batch's own spread (that
    #    older approach returned 1.0 for uniformly-terrible scores and
    #    buried a strong top hit behind a weak tail).
    #
    #    Different retrievers emit different score scales (cosine ~0-1, RRF
    #    fusion ~0.01-0.02, cross-encoder unbounded), so we calibrate per
    #    scorer type against a reference range, then weight the top hit most
    #    (for "did retrieval surface something relevant", the #1 result
    #    matters more than the tail).
    scores = [c.get("rerank_score", c.get("score", 0)) for c in retrieved_chunks[:5]]
    if scores:
        # detect scale and map the TOP score into [0,1] against a sane range
        top = scores[0]
        if any("rerank_score" in c for c in retrieved_chunks[:5]):
            # cross-encoder logits: ~<0 irrelevant, ~>5 strongly relevant
            top_norm = max(0.0, min(1.0, (top + 2.0) / 10.0))
        elif top <= 0.05:
            # RRF fusion scores are tiny; ~0.03+ is a strong single-list hit, Hybrid Retriever
            top_norm = max(0.0, min(1.0, top / 0.03))
        else:
            # cosine-like, already ~0-1
            top_norm = max(0.0, min(1.0, top))
        # blend: 70% the best hit, 30% the average of the rest, so a strong
        # #1 isn't dragged down by a weak tail, but breadth still counts
        rest = scores[1:] or [top]
        rest_mean = sum(rest) / len(rest)
        if any("rerank_score" in c for c in retrieved_chunks[:5]):
            rest_norm = max(0.0, min(1.0, (rest_mean + 2.0) / 10.0))
        elif top <= 0.05:
            rest_norm = max(0.0, min(1.0, rest_mean / 0.03))
        else:
            rest_norm = max(0.0, min(1.0, rest_mean))
        retrieval_conf = 0.7 * top_norm + 0.3 * rest_norm
    else:
        retrieval_conf = 0.0

    # 2. citation COVERAGE: true coverage = supported cited claims / ALL
    #    factual claims (cited + uncited). The old version reused
    #    faithfulness_rate, which only looked at CITED claims -- so an answer
    #    that cited one sentence and left five factual claims uncited scored
    #    1.0. Now uncited factual sentences count against coverage.
    if verification and verification.get("faithfulness_rate") is not None:
        n_supported = verification.get("n_supported", 0)
        n_cited = verification.get("n_claims", 0)
        n_uncited = verification.get("n_uncited", 0)
        total_claims = n_cited + n_uncited
        citation_cov = (n_supported / total_claims) if total_claims else 0.0
    else:
        citation_cov = 0.0

    # 3. completeness: did it actually answer, or fall back? (shared helper,
    #    not a brittle one-phrase substring match)
    completeness = 0.0 if is_non_answer(answer_record["answer"]) else 1.0

    composite = 0.4 * retrieval_conf + 0.4 * citation_cov + 0.2 * completeness
    return {"composite": round(composite, 3),
            "retrieval_confidence": round(retrieval_conf, 3),
            "citation_coverage": round(citation_cov, 3),
            "completeness": completeness}

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import structured_chunks
    from retrievers import HybridRetriever
    from generate_llm import generate_answer_llm

    chunks = structured_chunks()
    hyb = HybridRetriever(chunks, dense_weight=0.7, sparse_weight=0.3)
    q = "Which Dubai law regulates the interim property register for off-plan sales?"
    retrieved = hyb.search(q, k=5)

    # resolve [n] -> chunk, same 1-indexing the generator/verifier use
    block_to_chunk = {i: c for i, c in enumerate(retrieved, start=1)}

    def _src(c):
        src = c.get("source", "unknown")
        page = c.get("page", -1)
        return f"{src} p.{page}" if page and page != -1 else src

    try:
        ans = generate_answer_llm(q, retrieved)
        print("Q:", q, "\n")          

        print("A:", ans["answer"], "\n")

        ver = verify_citations(ans, retrieved)
        print("Faithfulness rate:", ver.get("faithfulness_rate"))
        for v in ver.get("verdicts", []):
            flag = "OK" if v["supported"] else "UNSUPPORTED"
            print(f"  [{flag}] {v['claim'][:70]}")
            for n in v["blocks"]:
                c = block_to_chunk.get(n)
                if c:
                    print(f"        [{n}] -> {_src(c)}")
                    print(f"            {c['text'][:160].strip()}...")
                else:
                    print(f"        [{n}] -> (not in retrieved context)")

        conf = confidence_score(ans, retrieved, ver)
        print("\nConfidence:", conf)
    except RuntimeError as e:
        print("Could not run:", e)