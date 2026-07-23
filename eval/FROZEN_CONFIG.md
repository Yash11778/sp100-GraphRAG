# Frozen Pipeline Configuration — Round 3 Held-Out Evaluation

**Config v3 — frozen 2026-07-19 (final).** Two additions over v2, both validated
on the dev set (100% dev pass at freeze):
- Named-entity bridge: the person-bridge mechanism generalized to any external
  entity ('the company that acquired <Entity>' → DB-side kwsearch over 8-K
  chunks → company → second hop). Mechanism-level generalization, same code path
  class as the person bridge.
- Actual Gemini API token usage (`usage_metadata`) captured per call and
  reported per row (`api_prompt/output/total_tokens`) alongside the disclosed
  tiktoken estimate that remains the cross-pipeline comparison basis
  (live delta ≈1%). Earlier same-config runs archived:
  `eval/results/v2-pre-instrumentation/`.

**Config v2 — frozen 2026-07-19 (superseded; v1 runs & ablations archived as
`eval/results/v1/`).** v2 changes were motivated by v1 failure analysis at the
mechanism level and validated ONLY on the dev set (100% dev pass at freeze):
- `Event.declared` (BOOL): 8-K wording distinguishes a dividend DECLARATION from
  an announced intention/increase (document-derived, per-filing regex)
- Numeric-fact retrieval: ticker-scoped `kwsearch` over 10-K chunks for metric
  phrases, ranked by numeric density (financial-statement rows first)
- Bridge second hop appends the resolved company's 8-K events (graph facts)
- Auditor route: multi-ask questions keep the scoped evidence layer
- Officer-name chunks: role-phrase kwsearch when the question asks who/whom

Frozen: 2026-07-19, before the 3× held-out evaluation runs. No pipeline,
prompt, retrieval, or routing change was made after this point. Tuning used
ONLY the 18-question dev set (`data/qa/dev_questions.json`), disjoint from the
held-out 50.

## Fixed generation & evaluation conditions (all three pipelines)
- Generator: `gemini-2.5-flash`, temperature 0.0, thinking disabled, max 512 output tokens
- Embedding model (RAG + GraphRAG): `sentence-transformers/all-MiniLM-L6-v2` (384-d)
- Judge: same rubric prompts for every pipeline, judge blind to pipeline identity
  - strict PASS/FAIL + graded 0-3 + citation/evidence 0-2 (Gemini, `gemini_strict`)
- BERTScore: `bert-score==0.3.13`, `roberta-large`, `rescale_with_baseline=True`, `idf=False`
- Token accounting: tiktoken cl100k_base on exact prompt/answer strings; routing tokens
  reported per row (always 0 for the deterministic router)

## Pipeline 1 — LLM-only
Direct question → Gemini. No dataset access.

## Pipeline 2 — Traditional RAG (good-faith baseline)
FAISS over the same 86,552 chunks + same embedder, top-k=8 full chunks → Gemini.

## Pipeline 3 — Optimized TigerGraph GraphRAG (frozen settings)
- Deterministic router (0 LLM routing tokens): typed traversals over
  AUDITED_BY / IN_SECTOR / MENTIONS_TOPIC / NAMES_* / REPORTS_EVENT(Event
  vertices: dividend, buyback, leadership) + DB-side `kwsearch` (Chunk.text scan)
  + `vsearch_scoped` (ticker-filtered HNSW)
- Bridge resolution: $-amount → Event.value reverse lookup; person surname →
  8-K/DEF14A kwsearch; date phrase → 10-K kwsearch ∩ auditor set
- Context optimization: chunk-id dedup, form-prior rerank (10-K financials /
  DEF14A governance / 8-K events), per-company round-robin balance (≤4
  companies), evidence cap MAX_EVIDENCE_TOKENS=1800, top_k=6
- Company rendering: `Name (TICKER)` labels in graph evidence lines

## Dev-set tuning trail (all decisions, in order)
1. run 90: 88.9% pass — found (a) bare-ticker answers judged FAIL (ISRG vs
   "Intuitive Surgical"), (b) verbose caveat answers judged FAIL
2. Added Name (TICKER) labels to graph evidence + concise-answer instruction
3. run 91: 83.3% — concise instruction over-corrected (dropped company names);
   also found+fixed BERTScore module-shadowing bug (eval/evaluate.py shadowed
   the HF `evaluate` package → all scores 0.0; now uses bert_score.BERTScorer)
4. Final prompt: state every asked fact, name indirectly-identified companies,
   omit procedural conditions unless asked
5. run 92: 94.4% pass, graded 2.94/3, citation 2.00/2, bertF1 0.320, 562 avg
   tokens; last dev failure re-verified PASS after step 4 → FROZEN

## Disclosed pipeline differences (brief §Controlled Evaluation Conditions)
- GraphRAG's system prompt adds graph-evidence handling + company-naming
  instructions (disclosed above); core instructions identical in intent
- Graph-specific techniques (typed traversal, kwsearch, scoped HNSW) apply only
  to GraphRAG per the brief; RAG uses the same embedder/chunks in FAISS
- Ablations (`GR_ABLATION`: no_graph / no_scope / no_opt) quantify each
  mechanism's contribution on the held-out set with this frozen config
