"""
Retrieval eval: does the right SOURCE document surface in top-k?
Gold label = gold_source per question. Metrics: Hit@k, Recall@k (source-level),
MRR. Runs baseline vs hybrid on the same eval set and prints a comparison.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from ingest import naive_chunks, structured_chunks, structured_chunks_deduped
from retrievers import BaselineRetriever, HybridRetriever, RerankedRetriever

EVAL = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "eval_set.json")))["questions"]

def source_hit(results, q):
    gold = set(q.get("gold_source_alt") or [q["gold_source"]])
    return [r["source"] in gold for r in results]

def evaluate(retriever, k=5):
    hits_at = {1:0, 3:0, 5:0}
    rr_sum = 0.0
    per_q = []
    for q in EVAL:
        res = retriever.search(q["question"], k=k)
        hitmask = source_hit(res, q)
        # reciprocal rank of first correct source
        rank = next((i+1 for i, h in enumerate(hitmask) if h), None)
        rr = 1.0/rank if rank else 0.0
        rr_sum += rr
        for kk in hits_at:
            if any(hitmask[:kk]):
                hits_at[kk] += 1
        per_q.append({"id": q["id"], "rank": rank, "top_source": res[0]["source"]})
    n = len(EVAL)
    return {
        "hit@1": hits_at[1]/n,
        "hit@3": hits_at[3]/n,
        "hit@5": hits_at[5]/n,
        "MRR":   rr_sum/n,
        "per_q": per_q,
    }

def main():
    print("Loading chunks...")
    nc = naive_chunks(); sc = structured_chunks_deduped()
    print(f"  naive={len(nc)}  structured={len(sc)}")

    print("Building baseline (naive + dense)...")
    t=time.time(); base = BaselineRetriever(nc); print(f"  built in {time.time()-t:.1f}s")
    print("Building hybrid (structured + dense + BM25)...")
    t=time.time()
    boost = {"reference":0.002, "regulation":0.001, "market_data":0.001, "listings":0.002, "listings_data":0.0}
    hyb = HybridRetriever(sc, dtype_boost=boost, dense_weight=0.7, sparse_weight=0.3); print(f"  built in {time.time()-t:.1f}s")
    print("Wrapping hybrid with cross-encoder reranker...")
    t=time.time(); reranked = RerankedRetriever(hyb, pool_size=20); print(f"  built in {time.time()-t:.1f}s")

    rb = evaluate(base); rh = evaluate(hyb); rr = evaluate(reranked)

    print("\n=== RETRIEVAL RESULTS (source-level, n=20) ===")
    print(f"{'metric':8} {'baseline':>10} {'hybrid':>10} {'+reranked':>10}")
    for m in ["hit@1","hit@3","hit@5","MRR"]:
        print(f"{m:8} {rb[m]:>10.3f} {rh[m]:>10.3f} {rr[m]:>10.3f}")

    print("\nPer-question first-correct rank (None = missed in top-5):")
    print(f"{'Q':5} {'baseline':>10} {'hybrid':>10} {'+reranked':>10}")
    for qb, qh, qr in zip(rb["per_q"], rh["per_q"], rr["per_q"]):
        print(f"{qb['id']:5} {str(qb['rank']):>10} {str(qh['rank']):>10} {str(qr['rank']):>10}")

    json.dump({"baseline":rb, "hybrid":rh, "reranked":rr},
              open(os.path.join(os.path.dirname(__file__),"..","data","processed","retrieval_results.json"),"w"),
              indent=2)
    print("\nsaved -> data/processed/retrieval_results.json")

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__),"..","data","processed"), exist_ok=True)
    main()