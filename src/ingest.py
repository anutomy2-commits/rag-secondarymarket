"""
Ingestion: parse the off-plan-resale study corpus into structured chunks.

Two chunking strategies live here so the eval can compare them:
  - naive_chunks():      fixed-size, overlap, no structure  (BASELINE)
  - structured_chunks(): structure-aware (tables kept whole, legal
                         articles as units, JSON records as rows) (IMPROVED)

Every chunk carries metadata: source, doc_type, page, strategy, chunk_id.
doc_type drives later metadata-filtered retrieval.
"""
import os, re, json, glob, hashlib
import pdfplumber

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW = os.path.abspath(RAW)

# --- the 10-file study set, with declared doc types -------------------------
CORPUS = {
    "OffPlan_Resale_Factors_Reference.pdf":            "reference",
    "DLD_Law13_2008_InterimRegister.pdf":              "regulation",
    "DLD_Law19_2017_Art11_ExplanatoryNotes.pdf":       "regulation",
    "DLD_RealEstate_Legislation_Compendium.pdf":       "regulation",
    "DLD_AnnualReport_2024.pdf":                        "market_data",
    "DLD_HousePriceIndex_Methodology.pdf":             "market_data",
    "PropertyFinder_OffPlan_Resale_Listings.pdf":      "listings",
    "PropertyFinder_OffPlan_Listings.pdf":             "listings",
    "PropertyFinder_OffPlan_Resale_source.json":       "listings_data",
    "PropertyFinder_OffPlan_source.json":              "listings_data",
}

def _cid(source, idx, text):
    h = hashlib.md5((source + str(idx) + text[:64]).encode()).hexdigest()[:10]
    return f"{source[:20]}-{idx}-{h}"

# ---------------------------------------------------------------------------
# raw extraction: page text + tables, per PDF
# ---------------------------------------------------------------------------
def _pdf_pages(path):
    """Yield (page_no, text, [table_as_text,...]) per page."""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tbls = []
            for t in page.extract_tables():
                rows = [" | ".join((c or "").strip() for c in row) for row in t if any(row)]
                if rows:
                    tbls.append("\n".join(rows))
            yield i, text, tbls

# ---------------------------------------------------------------------------
# BASELINE chunking: fixed-size sliding window over concatenated page text
# ---------------------------------------------------------------------------
def naive_chunks(size=800, overlap=120):
    out = []
    for fname, dtype in CORPUS.items():
        if fname.endswith(".json"):
            # baseline treats JSON as a flat text blob too (naive on purpose)
            data = json.load(open(os.path.join(RAW, fname)))
            blob = json.dumps(data)
            words = blob.split()
            step = size - overlap
            for j in range(0, len(words), step):
                seg = " ".join(words[j:j+size])
                if seg.strip():
                    out.append(_mk(seg, fname, dtype, None, "naive", len(out)))
            continue
        full = []
        for pno, text, tbls in _pdf_pages(os.path.join(RAW, fname)):
            full.append(text)
            full.extend(tbls)  # tables flattened into the stream (naive)
        words = "\n".join(full).split()
        step = size - overlap
        for j in range(0, len(words), step):
            seg = " ".join(words[j:j+size])
            if seg.strip():
                out.append(_mk(seg, fname, dtype, None, "naive", len(out)))
    return out

# ---------------------------------------------------------------------------
# IMPROVED chunking: structure-aware
# ---------------------------------------------------------------------------
_ARTICLE = re.compile(r'(Article\s*\(?\d+\)?)', re.I)

def _split_articles(text):
    """Split legal text on Article boundaries, keeping the heading."""
    parts = _ARTICLE.split(text)
    if len(parts) <= 1:
        return [text]
    chunks, buf = [], parts[0]
    for k in range(1, len(parts), 2):
        head = parts[k]
        body = parts[k+1] if k+1 < len(parts) else ""
        if buf.strip():
            chunks.append(buf.strip())
        buf = head + body
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

def structured_chunks(max_words=350):
    out = []
    for fname, dtype in CORPUS.items():
        path = os.path.join(RAW, fname)

        if fname.endswith(".json"):
            data = json.load(open(path))
            for rec in data:
                # one chunk per listing, as a readable line
                parts = [f"{k}: {v}" for k, v in rec.items()
                         if k not in ("id",) and v not in (None, "", [])]
                seg = f"Off-plan resale listing. " + "; ".join(str(p) for p in parts)
                out.append(_mk(seg, fname, dtype, None, "structured", len(out)))
            continue

        for pno, text, tbls in _pdf_pages(path):
            # tables: each kept as one whole chunk (rows intact)
            for t in tbls:
                seg = "Table:\n" + t
                out.append(_mk(seg, fname, dtype, pno, "structured", len(out)))

            body = text.strip()
            if not body:
                continue
            if dtype == "regulation":
                units = _split_articles(body)
            else:
                # split on blank lines / headings into paragraph units
                units = [u.strip() for u in re.split(r'\n{2,}', body) if u.strip()]
                if len(units) == 1:
                    units = [body]
            for u in units:
                # cap very long units by word window (preserving whole where small)
                words = u.split()
                if len(words) <= max_words:
                    out.append(_mk(u, fname, dtype, pno, "structured", len(out)))
                else:
                    for j in range(0, len(words), max_words):
                        seg = " ".join(words[j:j+max_words])
                        out.append(_mk(seg, fname, dtype, pno, "structured", len(out)))
    return out

def _mk(text, source, dtype, page, strategy, idx):
    return {
        "id": _cid(source, idx, text),
        "text": text,
        "source": source,
        "doc_type": dtype,
        "page": page if page is not None else -1,
        "strategy": strategy,
        "char_count": len(text),          # spec: store character count as metadata
        "chunk_index": idx,               # spec: store chunk index as metadata
    }


def structured_chunks_deduped(max_words=350, threshold=0.95):
    """Convenience: structured chunks with near-duplicates removed. This is
    what the live pipeline (API, retrieval) should use so it benefits from
    dedup, instead of the raw structured_chunks() which keeps duplicates.
    Returns just the kept chunks (drops the report)."""
    chunks = structured_chunks(max_words=max_words)
    kept, _dropped = deduplicate(chunks, threshold=threshold)
    return kept


# Version tag for the chunking logic. BUMP THIS whenever you change how
# chunks are produced (structured_chunks / dedup / metadata), so a persisted
# index built by the old logic is correctly detected as stale.
CHUNKING_VERSION = "structured_v1_dedup095"


def corpus_fingerprint(embedding_model="all-MiniLM-L6-v2"):
    """
    A compact signature of everything that, if changed, should invalidate a
    persisted index:
      - which source files exist + their sizes + last-modified times
      - the embedding model name (different model = incompatible vectors)
      - the chunking version (different chunking = different chunks)
    Returned as a short hash plus the raw parts (for logging/debugging why a
    rebuild was triggered).
    """
    file_parts = []
    for fname in sorted(CORPUS.keys()):
        path = os.path.join(RAW, fname)
        if os.path.exists(path):
            st = os.stat(path)
            file_parts.append(f"{fname}:{st.st_size}:{int(st.st_mtime)}")
        else:
            file_parts.append(f"{fname}:MISSING")
    raw = "|".join(file_parts) + f"|model={embedding_model}|chunking={CHUNKING_VERSION}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:16]
    return {"hash": digest, "embedding_model": embedding_model,
            "chunking_version": CHUNKING_VERSION, "n_source_files": len(file_parts)}


# ============================================================
# Deduplication: catch near-identical chunks before they're indexed.
#
# Why this matters (found for real, not hypothetical): pdfplumber's
# extract_text() and extract_tables() can BOTH capture the same table's
# content -- once as a clean structured table chunk, once as a messy
# flowing-paragraph chunk from the surrounding page text. Without dedup,
# both get indexed, wasting retrieval slots and letting the same fact
# outrank other content just by appearing twice.
#
# Method: embed every chunk, compare each new chunk against ones already
# accepted (not all pairs -- O(n) growing comparisons, not O(n^2) upfront),
# skip anything above the similarity threshold. Threshold matches the
# build-guide spec: cosine similarity > 0.95.
# ============================================================
def deduplicate(chunks, threshold=0.95, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        return chunks, []

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    kept, kept_embeddings, dropped = [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        if kept_embeddings:
            sims = np.array(kept_embeddings) @ emb
            best = float(sims.max())
            if best > threshold:
                dupe_of = kept[int(sims.argmax())]
                dropped.append({"chunk": chunk, "duplicate_of": dupe_of["id"], "similarity": best})
                continue
        kept.append(chunk)
        kept_embeddings.append(emb)

    return kept, dropped


if __name__ == "__main__":
    n = naive_chunks()
    s = structured_chunks()
    print(f"naive chunks:      {len(n)}")
    print(f"structured chunks: {len(s)}  (before dedup)")

    s_clean, dupes = deduplicate(s)
    print(f"structured chunks: {len(s_clean)}  (after dedup, threshold=0.95)")
    if dupes:
        print(f"\n{len(dupes)} near-duplicate chunk(s) removed:")
        for d in dupes[:5]:
            c = d["chunk"]
            print(f"   sim={d['similarity']:.3f}  {c['source'][:35]:35} p{c['page']}  "
                  f"{c['text'][:60].strip().replace(chr(10),' ')}...")
        if len(dupes) > 5:
            print(f"   ...and {len(dupes)-5} more")

    from collections import Counter
    print("\nstructured by doc_type (post-dedup):", dict(Counter(c['doc_type'] for c in s_clean)))
    print("structured by source (post-dedup):")
    for src, cnt in Counter(c['source'] for c in s_clean).most_common():
        print(f"   {cnt:4d}  {src}")
    # sample
    print("\nsample structured (regulation):")
    for c in s_clean:
        if c['doc_type']=='regulation' and 'Article' in c['text']:
            print("  ", c['text'][:120].replace("\n"," "), "...")
            break