# Round 3 — Final Results (Frozen Config v3)

Held-out 50 questions × 3 pipelines × 3 independent runs. Judge errors: 0 across
all runs. Raw per-question outputs: `eval/results/eval_run{1,2,3}.csv`
(iteration history archived: `v1/`, `v2-pre-instrumentation/`). Frozen config +
full tuning trail: `FROZEN_CONFIG.md`. One-time ingestion costs (0 LLM tokens):
`ingestion_costs.md`.

## Headline (mean of 3 runs, range in parens)

| Pipeline | Strict pass | Graded /3 | Citation /2 | BERTScore F1 | Avg tokens (tiktoken) | Avg tokens (Gemini API) |
|---|---|---|---|---|---|---|
| LLM-only | 10.0% (10–10) | 0.65 | 0.04 | −0.244 | 455 | 475 |
| Traditional RAG | 15.3% (14–16) | 0.77 | 1.25 | −0.027 | 2,182 | 2,420 |
| **GraphRAG (ours)** | **62.0% (60–64)** | **2.23** | **1.83** | **+0.219** | **1,190** | **1,379** |

**Token reduction vs Traditional RAG: 45.4% by tiktoken estimate, 43.0% by the
Gemini API's own usage_metadata (100% per-call coverage) — at 4.1× the strict
pass rate.** Both accountings are reported per row; the API numbers are actual
usage, as the brief requires when available.

## Pass rate by tier (mean of 3 runs)

| Tier | Traditional RAG | GraphRAG | Mechanism |
|---|---|---|---|
| A (single-hop) | 50.0% | **83.3%** | ticker-scoped statement-row retrieval |
| B (two-hop bridges) | 12.5% | **47.9%** | Event-vertex + named-entity/person kwsearch bridges |
| C (aggregation/intersection) | 8.3% | **78.3%** | typed traversals → exact sets in tens of tokens |
| D (cross-doc & temporal) | 0.0% | **16.7%** | chronology from Event dates; hardest tier |

## Ablations (held-out, frozen v3, graphrag only)

| Variant | Pass | Avg tokens | Δ vs full (62.0%) |
|---|---|---|---|
| no_graph (router off, global HNSW only) | 22.0% | 1,777 | **−40.0 pts, +49% tokens** |
| no_scope (never company-scoped) | 62.0% | 1,176 | 0.0 pts (within run noise) |
| no_opt (no dedup/prior/balance/cap) | 56.0% | 1,174 | −6.0 pts |

The graph is the dominant mechanism: removing it costs 40 points of pass rate
AND raises token usage ~49% — vector-only retrieval needs more context to do
less. Context optimization contributes ~6 points; company-scoping's effect was
within noise on this run set (disclosed honestly — its v2 measurement was
−3.3 pts).

## Benchmark-validity notes (disclosed, per brief)

- All 50 held-out questions evaluated in every run; no rows removed; failures
  remain in the CSVs and are browsable in the dashboard's per-question table.
- LLM-only never beats a retrieval pipeline on any tier (10% overall floor).
- Known data/key discrepancies (answers follow the SOURCE documents,
  verifiably):
  - Our graph has Citigroup→KPMG; C's DEF14A ratification section names KPMG
    explicitly (8+ mentions) though EQ25's key lists 11 KPMG companies
    without C.
  - Auditor totals: ours E&Y 32 / PwC 32 / D&T 24 / KPMG 12 vs key 35/29/25/11 —
    extraction is stable under two independent rules (cue-proximity and
    ratification-sentence) and is verifiable per company from the filings.
  - Buyback chronology uses 8-K FILING dates (labelled as such in evidence);
    the key's dates differ by ≤1 day for NFLX but ordering is identical.
- Tuning used only the 18-question dev set (100% dev pass at each freeze); the
  held-out set was never used to tune. All three protocol iterations
  (v1 → v2 → v3) are archived with their configs and raw outputs.

## Reproducibility

- 3 independent full runs; deterministic components (routing, retrieval sets)
  identical across runs; judge nondeterminism shown as ranges.
- Deterministic router: routing_tokens = 0 on every row (no hidden LLM calls).
- Actual Gemini API usage recorded per call (`api_prompt/output/total_tokens`)
  alongside the disclosed tiktoken estimate (delta ≈ 1–11% depending on
  content; both columns in every CSV row).
- Pinned deps (`requirements.txt`), brief-mandated BERTScore config
  (bert-score==0.3.13, roberta-large, rescale_with_baseline=True, idf=False).
- Savanna is both graph store and vector store (HNSW on Chunk.emb, 384-d,
  COSINE) plus DB-side keyword scan (`kwsearch`) and filtered vector search
  (`vsearch_scoped`) as installed GSQL queries.
