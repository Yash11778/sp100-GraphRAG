# How We Made GraphRAG Cheaper AND More Accurate — TigerGraph Hackathon Round 3

*Our Round 3 submission for the TigerGraph GraphRAG Hackathon: 4.1× the accuracy
of traditional RAG at 45% fewer tokens — with every claim traceable to raw,
per-question results.*

---

## The challenge

Round 3 of the TigerGraph GraphRAG Hackathon posed a deceptively hard question:
GraphRAG's high-recall retrieval finds more evidence than plain vector search —
but does the LLM actually need all of it? Teams were given the same
heterogeneous dataset — 400 SEC filings (10-Ks, DEF 14A proxies, 8-Ks) from 100
S&P-100 companies — and 50 held-out questions ranging from single-fact lookups
to corpus-wide aggregations and cross-document reasoning. The task: build a
context-optimization approach that makes GraphRAG retrieval **smaller, more
precise, and higher quality** than traditional RAG, without sacrificing
accuracy.

Every team ran the same three pipelines: an LLM-only baseline, a good-faith
traditional RAG (FAISS, top-k chunks), and an optimized GraphRAG on TigerGraph
Savanna — same generator (Gemini 2.5 Flash at temperature 0), same embedding
model for both retrieval pipelines, same blind LLM judge.

## Our thesis: the traversal IS the context optimization

Most context-optimization work treats retrieval and compression as separate
steps: retrieve a lot, then dedupe/rerank/summarize it down. We inverted that.
If the knowledge graph is modeled well, **a typed graph traversal returns the
answer set itself — not a pile of chunks to sift**.

One real example from our evaluation. The question: *"The company that
authorized a $4.1B share-repurchase program (a 2026 8-K) — who is its
auditor?"*

Traditional RAG retrieved eight chunks — 2,192 tokens — and still failed.
Our GraphRAG pipeline sent the model **two lines (~40 tokens)**:

```
Graph result — 8-K $4.1B share-repurchase authorization: FedEx (FDX), filed 2026-07-10
Graph result — FedEx (FDX)'s independent auditor: Ernst & Young LLP
```

That's an Event-vertex lookup (`value ≈ $4.1B`) followed by one `AUDITED_BY`
hop. No chunk ranking, no compression model, no LLM router — and an 89% token
reduction on that question with a correct, cited answer.

## How it works

**A deterministic router (0 LLM tokens).** Regex-and-dictionary resolvers
detect companies, audit firms, sectors, people, dollar amounts, and dates in
the question, then choose a strategy. No LLM calls means routing costs nothing
and is fully auditable — `routing_tokens = 0` on every result row.

**TigerGraph Savanna as graph + vector + keyword engine.** Our schema has nine
vertex types (Company, AuditFirm, Sector, Document, Event, Chunk with a 384-d
HNSW vector attribute, and more) and twelve edge types. Three retrieval
modalities run inside the same database:

- **Typed traversals** for aggregation/intersection questions ("Which
  KPMG-audited companies are in Consumer Staples?") — exact sets in tens of
  tokens
- **Filtered vector search** (`vsearch_scoped`) — HNSW over chunk embeddings,
  restricted to a company when the question names one
- **DB-side keyword scan** (`kwsearch`) — resolves people and named entities
  ("the company that appointed John Ternus", "…that acquired Apogee
  Therapeutics") straight from chunk text, generalizing the graph beyond
  pre-extracted edges

**Event vertices for bridge questions.** 8-K filings became typed Event
vertices (dividend / buyback / leadership, with values, dates, and a
`declared` flag distinguishing an actual dividend declaration from an announced
intention — a distinction the filings themselves make). Two-hop "bridge"
questions resolve their first hop in the graph, then answer the second hop with
a traversal or a scoped search.

**Context optimization on what remains.** For questions that still need text
evidence: chunk dedup, document-type priors (10-K for financials, DEF 14A for
governance, 8-K for events), per-company balancing for comparisons, and a hard
1,800-token evidence cap.

**Zero-LLM ingestion.** Parsing, auditor extraction, event extraction, and
topic tagging are all deterministic; embeddings are computed locally. One-time
ingestion cost: **0 LLM tokens** — so every token we report is inference, with
nothing hidden in preprocessing.

## Results (held-out 50 questions × 3 independent runs, frozen config)

| Pipeline | Strict pass | Graded /3 | Citation /2 | BERTScore F1 | Avg tokens |
|---|---|---|---|---|---|
| LLM-only | 10.0% | 0.65 | 0.04 | −0.244 | 455 |
| Traditional RAG | 15.3% | 0.77 | 1.25 | −0.027 | 2,182 |
| **GraphRAG (ours)** | **62.0%** | **2.23** | **1.83** | **+0.219** | **1,190** |

**45.4% fewer total inference tokens than traditional RAG at 4.1× the strict
pass rate.** We report tokens two ways — a tiktoken estimate and the Gemini
API's own `usage_metadata` (100% per-call coverage), which independently shows
a 43.0% reduction. The pass-rate range across three runs was 60–64%, with zero
judge errors.

The ablations tell the sharpest story. Turning the graph off (global vector
search only, same database, same embeddings) dropped the pass rate to **22%
while RAISING token usage 49%**. The graph isn't a garnish on RAG — it's
simultaneously the accuracy mechanism and the efficiency mechanism.

## What we learned

1. **Model the corpus, not the benchmark.** Our auditor extraction disagrees
   with the official answer key on a few companies — and we kept our data,
   because the filings themselves back it (Citigroup's proxy names KPMG as its
   auditor eight times). We disclosed every discrepancy with citations rather
   than fitting the key.
2. **Deterministic beats clever.** A regex router with 0 token cost outperformed
   what an LLM router would give us, and it's fully reproducible.
3. **Silent failures are the enemy of honest benchmarks.** Midway through, we
   discovered our BERTScore had been silently reading 0.0 for an entire run — a
   Python module-shadowing bug (`eval/evaluate.py` shadowed the HF `evaluate`
   package). Loud failures and per-row instrumentation are now our default.
4. **Freeze-and-archive makes iteration honest.** We tuned only on an
   18-question dev set, froze the config before each held-out evaluation, and
   archived every protocol iteration (34% → 45% → 57% → 62%) with raw outputs.

## Honest limitations

Cross-document temporal reasoning is still hard: our Tier-D pass rate is 16.7%
(traditional RAG and LLM-only both scored 0%). Two-hop numeric bridges land at
47.9%. Both are measured, reported per-question, and unsolved — for now.

## Try it

Our repo ships all three pipelines, the ingestion and evaluation code, a
FastAPI backend, and a dashboard that runs any question through all three
pipelines live — showing tokens, judges, BERTScore, and the official
per-question benchmark, failures included.

---

*Built for the **TigerGraph GraphRAG Hackathon (Round 3)** on TigerGraph
Savanna — used as both the graph database and the vector database — with
Gemini 2.5 Flash for generation and judging. Thanks to the TigerGraph team for
the dataset, credits, and a challenge that rewards rigor over hype.*

*Code & full results: [GitHub repo link] · Dashboard demo: [link] ·
#TigerGraph #GraphRAG #KnowledgeGraphs #RAG #LLM*
