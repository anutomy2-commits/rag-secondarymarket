"""
Phase 3/4 runner — the full pipeline, end to end, across eval questions.

For each question:
    retrieve (hybrid) -> generate cited answer (LLM) -> verify each citation
    (LLM judge) -> confidence score
and saves everything to data/processed/pipeline_results.json plus prints a
summary table (faithfulness, coverage, confidence).

IMPORTANT: the SAME retrieved list is passed to both generation and
verification, in the same order, so the block numbers [1][2] line up and
verify.py's context-map assertion passes.

Usage:
    python3 src/run_pipeline.py            # cheap test: first 3 questions
    python3 src/run_pipeline.py 20         # full run: all 20 questions
    python3 src/run_pipeline.py 5          # any N

Needs ANTHROPIC_API_KEY set (in .env or exported).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))

from ingest import structured_chunks_deduped
from retrievers import HybridRetriever
from generate_llm import generate_answer_llm, validate_citation_numbers
from verify import verify_citations, confidence_score

HERE = os.path.dirname(__file__)
EVAL = json.load(open(os.path.join(HERE, "..", "data", "raw", "eval_set.json")))["questions"]
OUT = os.path.join(HERE, "..", "data", "processed", "pipeline_results.json")


def run(n_questions=3, k=5):
    print(f"Loading corpus and building hybrid retriever...")
    chunks = structured_chunks_deduped()   # deduplicated: pipeline uses the clean chunk set
    boost = {"reference":0.002,"regulation":0.001,"market_data":0.001,"listings":0.002,"listings_data":0.0}
    retriever = HybridRetriever(chunks, dtype_boost=boost, dense_weight=0.7, sparse_weight=0.3)

    questions = EVAL[:n_questions]
    print(f"Running full pipeline on {len(questions)} question(s)...\n")

    results = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"[{i}/{len(questions)}] {question}")

        # 1. retrieve -- this exact list feeds BOTH generation and verification
        retrieved = retriever.search(question, k=k)

        # 2. generate cited answer
        try:
            ans = generate_answer_llm(question, retrieved)
        except RuntimeError as e:
            print(f"    generation failed: {e}")
            return
        struct_valid = validate_citation_numbers(ans, retrieved)

        # 3. verify citations (LLM judge) -- same retrieved list
        verification = verify_citations(ans, retrieved)

        # 4. confidence
        conf = confidence_score(ans, retrieved, verification)

        print(f"    answer: {ans['answer'][:90].strip()}...")
        fr = verification.get("faithfulness_rate")
        print(f"    faithfulness: {fr if fr is not None else 'n/a'}  "
              f"coverage: {conf['citation_coverage']}  confidence: {conf['composite']}")
        print()

        results.append({
            "id": q["id"], "question": question,
            "gold_source": q.get("gold_source"),
            "answer": ans["answer"],
            "citations": ans["citations"],
            "structural_validity": struct_valid,
            "verification": verification,
            "confidence": conf,
            "retrieved_sources": [c["source"] for c in retrieved],
        })

    # summary
    print("=" * 60)
    print(f"{'Q':5} {'faithful':>9} {'coverage':>9} {'confidence':>11} {'answered':>9}")
    for r in results:
        fr = r["verification"].get("faithfulness_rate")
        fr_s = f"{fr:.2f}" if fr is not None else "n/a"
        answered = "no" if r["confidence"]["completeness"] == 0 else "yes"
        print(f"{r['id']:5} {fr_s:>9} {r['confidence']['citation_coverage']:>9} "
              f"{r['confidence']['composite']:>11} {answered:>9}")

    # aggregate (over questions that actually produced verifiable claims)
    rated = [r for r in results if r["verification"].get("faithfulness_rate") is not None]
    if rated:
        avg_faith = sum(r["verification"]["faithfulness_rate"] for r in rated)/len(rated)
        avg_conf = sum(r["confidence"]["composite"] for r in results)/len(results)
        print("-" * 60)
        print(f"mean faithfulness (rated Qs): {avg_faith:.3f}   mean confidence (all Qs): {avg_conf:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {os.path.relpath(OUT, os.path.join(HERE,'..'))}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run(n_questions=n)