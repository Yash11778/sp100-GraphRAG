---
title: SP100 GraphRAG
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Round 3 — Optimized TigerGraph GraphRAG (S&P-100 SEC Filings)

Three-pipeline comparison for the TigerGraph GraphRAG Hackathon Round 3
(Generalizable Context Optimization): **LLM-only** vs **Traditional RAG** vs
**Optimized GraphRAG** on 400 SEC filings (100 S&P-100 companies × 10-K,
DEF 14A, 2×8-K).

## Headline results (held-out 50 Qs, 3 independent runs, frozen config)

| Pipeline | Strict pass | Graded /3 | Citation /2 | BERTScore F1 | Avg tokens |
|---|---|---|---|---|---|
| LLM-only | 10.0% | 0.65 | 0.04 | −0.244 | 455 |
| Traditional RAG | 15.3% | 0.77 | 1.25 | −0.027 | 2,182 |
| **GraphRAG** | **62.0%** | **2.23** | **1.83** | **+0.219** | **1,190** |

**45.4% fewer total inference tokens than Traditional RAG (43.0% by the Gemini
API's own usage metadata) at 4.1× the strict pass rate.** Ablation: disabling
the graph drops GraphRAG to 22% while RAISING tokens ~49% — the graph is
simultaneously the accuracy and the efficiency mechanism. Full numbers, tiers,
and disclosures: [eval/RESULTS.md](eval/RESULTS.md).

## Architecture

```
                      ┌────────────────────────────────────────────┐
question ──► deterministic router (0 LLM tokens)                   │
             │  companies · audit firms · sectors · people ·      │
             │  $-amounts · dates · event types                   │
             ▼                                                    │
   ┌─────────────────────────── TigerGraph Savanna ─────────────┐ │
   │ typed traversals   AUDITED_BY / IN_SECTOR / MENTIONS_TOPIC │ │
   │                    NAMES_* / REPORTS_EVENT(declared)       │ │
   │ kwsearch           DB-side keyword scan over Chunk.text    │ │
   │ vsearch(_scoped)   HNSW over Chunk.emb (384-d, COSINE),    │ │
   │                    optionally ticker-filtered              │ │
   └────────────────────────────────────────────────────────────┘ │
             ▼                                                    │
   context optimizer: dedup · form-prior rerank · per-company     │
   balance · 1800-token evidence cap                              │
             ▼                                                    │
   Gemini 2.5 Flash (temp 0, thinking off) ──► answer + citations │
```

- **Graph schema**: Company, Sector, AuditFirm, Document, Event(declared),
  Chunk(+emb vector), Entity, Topic, Person; 12 edge types ([schema/schema.gsql](schema/schema.gsql))
- **Same embedding model both retrieval pipelines**: all-MiniLM-L6-v2 (384-d);
  RAG uses FAISS, GraphRAG uses Savanna HNSW over the same 86,552 chunks
- **Ingestion = 0 LLM tokens** (deterministic extraction + local embeddings):
  [eval/ingestion_costs.md](eval/ingestion_costs.md)

## Repo layout

```
config.py               paths + canonical names
ingestion/              parse_filings, extract_auditor, extract_events(declared),
                        extract_mentions, build_chunks — all deterministic
schema/schema.gsql      graph DDL (vector attr + kwsearch/vsearch installed at setup)
scripts/                setup_schema, load_core, load_mentions, load_chunks,
                        validate_graph_queries, tg.py (REST client)
pipelines/              pipeline1_llm, pipeline2_rag, pipeline3_graphrag,
                        graph_queries (traversal layer), resolver, utils
eval/                   evaluate.py (strict+graded+citation+BERTScore),
                        judge.py, aggregate_runs.py, results/ (3 runs + v1 archive),
                        RESULTS.md, FROZEN_CONFIG.md, ingestion_costs.md
api/app.py              FastAPI: POST /compare (live 3-pipeline), GET /results
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # fill TG_HOST / TG_SECRET / TG_GRAPH / GEMINI_API_KEY

.venv/bin/python scripts/setup_schema.py      # graph + schema
# add Chunk.emb vector attr + install kwsearch/vsearch/vsearch_scoped (see schema/)
.venv/bin/python scripts/load_core.py         # companies/docs/events/auditors
.venv/bin/python scripts/load_mentions.py     # entities/topics
.venv/bin/python scripts/load_chunks.py       # 86,552 chunks + embeddings
.venv/bin/python scripts/validate_graph_queries.py
```

## Run the benchmark

```bash
.venv/bin/python eval/evaluate.py --dev            # 18-Q dev set (tuning only)
.venv/bin/python eval/evaluate.py --run 1          # held-out 50, all 3 pipelines
GR_ABLATION=no_graph .venv/bin/python eval/evaluate.py --pipelines graphrag --run 81
.venv/bin/python eval/aggregate_runs.py eval/results/eval_run{1,2,3}.csv
```

## Run the app + dashboard

**Backend** (from `round3-sp100/`):

```bash
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- `POST /compare` — runs all three pipelines live on any question
- `GET /results` — official frozen-config benchmark (3 runs + ablations)
- `GET /health` — readiness check

**Frontend** (from the dashboard folder, e.g. `~/graphrag-dashboard/`):

```bash
npm install        # first time only
npm run dev        # http://localhost:5173
```

The Vite dev server proxies `/compare`, `/results`, `/health` to
`http://localhost:8000` (see `vite.config.js`) — no env var needed. For a
production build instead:

```bash
VITE_API_BASE=http://localhost:8000 npm run build && npx vite preview
```

The dashboard runs any question through all three pipelines live (tokens,
strict + graded + citation judges, BERTScore) and renders the official
frozen-config benchmark — 3-run aggregate, tier breakdown, ablations, and the
per-question table with failures — from `GET /results`.

## Experimental integrity

- Tuning used ONLY the 18-question dev set; config frozen before held-out runs
  (trail: [eval/FROZEN_CONFIG.md](eval/FROZEN_CONFIG.md); v1 iteration archived)
- Deterministic router → routing_tokens = 0 on every row; all LLM spend visible
- Judge blind to pipeline; same rubric for all; judge errors = 0 across runs
- BERTScore per spec: bert-score==0.3.13 · roberta-large ·
  rescale_with_baseline=True · idf=False
- Known key-vs-document discrepancies disclosed in RESULTS.md (answers follow
  the source filings, verifiably)
