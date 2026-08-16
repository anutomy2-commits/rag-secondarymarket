# Dubai Off-Plan Resale RAG

A retrieval-augmented generation service over a small corpus of Dubai
off-plan real-estate documents (DLD laws and regulations, DLD market-data
reports, and PropertyFinder listings). Answers are generated only from
retrieved context, cited by source, and scoped by the caller's access role.
Served over FastAPI with a single-page web UI.

## What it does

- **Hybrid retrieval** — dense embeddings (`sentence-transformers`,
  `all-MiniLM-L6-v2`) fused with BM25 keyword search via weighted
  reciprocal rank fusion (0.7 dense / 0.3 sparse), backed by a persistent
  ChromaDB collection.
- **Role-based access control** — every query is scoped to the doc_types
  the caller's role may see (`analyst`, `agent`, `legal`, `public`),
  enforced on the dense path (Chroma `where` filter), the sparse path
  (BM25 has no native filter, so results are checked against the policy
  explicitly), and again as a final pass over whatever comes back.
- **Grounded, cited generation** — Claude answers strictly from the
  retrieved context blocks and cites `[n]` per claim; citations that
  don't map to a real block are caught structurally.
- **Optional faithfulness verification** — an LLM-as-judge pass
  (`verify=true`) checks whether each cited claim is actually supported
  by the chunk it cites, feeding a composite confidence score
  (retrieval confidence + citation coverage + completeness).
- **Conversation memory** — follow-up questions ("and who pays it?") are
  rewritten into standalone queries using recent turns before retrieval,
  per session.
- **Web UI** (`src/static/index.html`) — ask questions, switch the access
  role live to see retrieval change, inspect citations/confidence/sources
  for each answer.

**Not part of the live API:** `src/retrievers.py` also implements a
cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and a
`RerankedRetriever`, but these are only used by the offline evaluation
script (`eval_retrieval.py`) to compare baseline vs. hybrid vs.
hybrid+reranked retrieval quality. The deployed API
(`retrievers_production.py`) retrieves via dense+BM25 hybrid fusion only,
no reranking step.

## Architecture

```
POST /v1/ask
  │
  ├─ 1. memory.py        rewrite follow-up into a standalone query (if use_memory)
  │
  ├─ 2. retrievers_production.py
  │       resolve role -> AccessPolicy (access_control.py)
  │       dense search (Chroma, where-filtered) + BM25 search (filtered in-process)
  │       weighted RRF fusion -> final policy re-check -> top-k chunks
  │
  ├─ 3. generate_llm.py  Claude answers from the retrieved chunks only, cites [n]
  │
  ├─ 4. verify.py         (optional) LLM-judge checks each citation, scores confidence
  │
  └─ 5. memory.py         save the turn for future follow-ups
```

### Corpus & doc_types

| Source | doc_type |
|---|---|
| `OffPlan_Resale_Factors_Reference.pdf` | `reference` |
| `DLD_Law13_2008_InterimRegister.pdf`, `DLD_Law19_2017_Art11_ExplanatoryNotes.pdf`, `DLD_RealEstate_Legislation_Compendium.pdf` | `regulation` |
| `DLD_AnnualReport_2024.pdf`, `DLD_HousePriceIndex_Methodology.pdf` | `market_data` |
| `PropertyFinder_OffPlan_Listings.pdf`, `PropertyFinder_OffPlan_Resale_Listings.pdf` | `listings` |
| `PropertyFinder_OffPlan_source.json`, `PropertyFinder_OffPlan_Resale_source.json` | `listings_data` |

### Roles (`src/access_control.py`)

| Role | Visible doc_types |
|---|---|
| `analyst` | all |
| `agent` | `listings`, `listings_data`, `reference` |
| `legal` | `regulation`, `reference` |
| `public` (default) | `reference`, `listings` |

Unknown roles are rejected (fail closed); a missing role falls back to
`public`. **Demo note (see `api.py`):** the role is taken from the
request body for this demo. In production it must come from the
authenticated caller (SSO/IAM claims), never a client-supplied value.

## Project structure

```
src/
  api.py                  FastAPI app: /health, /, /v1/documents, /v1/ask, /v1/reset
  ingest.py                PDF/JSON -> chunks, corpus fingerprinting
  access_control.py        Roles -> doc_type visibility, policy enforcement
  retrievers_production.py Persistent (Chroma+BM25) hybrid retriever used by the API
  retrievers.py             Baseline/Hybrid/Reranked retrievers used by eval only
  generate_llm.py           Grounded, cited answer generation (Claude)
  verify.py                 LLM-as-judge citation verification + confidence scoring
  memory.py                 Per-session conversation memory + query rewriting
  eval_retrieval.py         Offline: baseline vs hybrid vs hybrid+reranked retrieval eval
  run_pipeline.py           Offline: full retrieve->generate->verify run over eval questions
  static/index.html         Web UI
data/
  raw/                      Source PDFs/JSON + eval_set.json
  chroma_store/              Persisted Chroma collection + BM25 pickle (built at startup)
  processed/                 Output of eval_retrieval.py / run_pipeline.py
```

## Setup & run locally

Requires Python (tested with 3.14; see `requirements.txt` for pinned
versions) and an [Anthropic API key](https://console.anthropic.com).

```bash
git clone https://github.com/anutomy2-commits/rag-secondarymarket.git
cd rag-secondarymarket
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Run the API:

```bash
uvicorn src.api:app --reload
```

On first run (or whenever `data/raw/` changes), startup embeds the
corpus and builds the Chroma + BM25 index — this takes a little while.
Later runs detect the persisted index is still valid and load it
directly. Open `http://localhost:8000` for the web UI, or
`http://localhost:8000/docs` for the interactive API docs.

`/health` and `/v1/documents` work without an API key; `/v1/ask` needs
`ANTHROPIC_API_KEY` set.

### Offline scripts

```bash
python3 src/eval_retrieval.py     # baseline vs hybrid vs hybrid+reranked, Hit@k/MRR
python3 src/run_pipeline.py 5     # full retrieve->generate->verify over 5 eval questions
```

Both read `data/raw/eval_set.json` and write results to `data/processed/`.

## Deploying to Render

The repo includes `render.yaml`:

```yaml
services:
  - type: web
    name: rag-secondarymarket
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: 3.14.3
      - key: ANTHROPIC_API_KEY
        sync: false
```

Notes:

- **Single worker is required.** `SessionMemoryStore` (conversation
  memory) is held in process memory — more than one worker would break
  follow-up questions for users landing on a different worker.
- **`ANTHROPIC_API_KEY`** is intentionally left out of `render.yaml`
  (`sync: false`) — set it in the Render dashboard's Environment tab,
  never commit it. Set a spending cap on the key before sharing the URL,
  since anyone with it can trigger LLM calls.
- **No persistent disk**, by design: `data/raw/` ships in the repo, and
  the index is rebuilt from it on every cold start (a Chroma+BM25 rebuild
  over the corpus, roughly tens of seconds). This keeps the deploy on
  Render's free instance type. If cold-start latency becomes a problem,
  a persistent disk mounted at `./data` avoids the rebuild — but that
  needs a paid instance type, plus manually seeding `data/raw` onto the
  disk once, since a fresh `git checkout` resets file timestamps and
  would otherwise invalidate the fingerprinted index on every deploy
  anyway.

To deploy: push to GitHub, connect the repo on Render as a Blueprint
(reads `render.yaml` automatically), set `ANTHROPIC_API_KEY` in the
dashboard, and deploy.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + retriever-ready check |
| `/` | GET | Web UI |
| `/v1/documents` | GET | List corpus source documents and their doc_type |
| `/v1/ask` | POST | Ask a question — see fields below |
| `/v1/reset` | POST | Clear a session's conversation memory |

`POST /v1/ask` body:

| Field | Default | Description |
|---|---|---|
| `question` | — | The question to answer |
| `k` | `5` | Number of chunks to retrieve (1-20) |
| `verify` | `false` | Run LLM-as-judge faithfulness verification (slower, costs API calls) |
| `use_memory` | `true` | Resolve follow-ups using conversation history |
| `session_id` | `"default"` | Isolates each user's conversation memory |
| `role` | `"public"` | Access role scoping retrieval (demo only — see security note above) |
