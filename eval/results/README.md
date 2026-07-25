# Raw results — file manifest

Every number in `../RESULTS.md` traces to a file here. Nothing is filtered: all 50
held-out questions appear in every evaluation run, including failures.

Regenerate the reported aggregates with:

```bash
.venv/bin/python eval/adjusted_results.py     # all four result tiers, from these CSVs
```

## Held-out evaluation — the reported result (config v3, frozen)

| File | Contents | Pipelines | Rows |
|---|---|---|---|
| `eval_run1.csv` | Held-out run 1 | llm_only, basic_rag, graphrag | 150 |
| `eval_run2.csv` | Held-out run 2 | same | 150 |
| `eval_run3.csv` | Held-out run 3 | same | 150 |

Three independent executions of the same frozen configuration. Deterministic
components (routing, retrieval sets) are identical across runs; the variation is
generator/judge nondeterminism.

## Ablations (`GR_ABLATION`, graphrag only, **single run each**)

The CSVs carry no ablation column — the run number *is* the label. The mapping is
verifiable from the data, as noted in each row below.

| File | `GR_ABLATION` | What is disabled | Verifiable by |
|---|---|---|---|
| `eval_run81.csv` | `no_graph` | Router off; global HNSW vector search only | `graph_used=False` and empty `graph_path` on all 50 rows |
| `eval_run82.csv` | `no_scope` | Router on, but vector retrieval never company-scoped | `graph_used=True` on 37 rows; differs from run83 on 31 rows' evidence |
| `eval_run83.csv` | `no_opt` | No dedup / form-prior / balance / token cap | `graph_used=True` on 37 rows; raw top-k dump instead of optimised evidence |

These are single runs, not 3×. See the statistical-power caveat in `../RESULTS.md`.

## Supplementary checks (added after the audit; none modify the frozen pipeline)

| File | Produced by | Purpose |
|---|---|---|
| `crossjudge_run1_gemini-2-5-pro.csv` | `eval/crossjudge.py` | Re-scores run 1's stored answers with `gemini-2.5-pro` — a different model from the `gemini-2.5-flash` generator. 97.3% verdict agreement. |
| `fairness_rag_labelled.csv` | `eval/fairness_check_rag_labels.py` | Traditional RAG re-run with `[TICKER FORM]` source labels (the asymmetry our audit found). 16.0% → 18.0%. |

## Development set (tuning only — disjoint from the held-out 50)

| File | Pass | Note |
|---|---|---|
| `dev_run90.csv` | 88.9% | Found bare-ticker and verbose-caveat failure modes |
| `dev_run91.csv` | 83.3% | Concise-answer instruction over-corrected; BERTScore shadowing bug found + fixed |
| `dev_run92.csv` | 94.4% | Final prompt; config frozen after this |
| `dev_run93.csv` | 100.0% | Post-freeze validation |
| `dev_run94.csv` | 100.0% | Post-freeze validation |

18 questions, `data/qa/dev_questions.json`. Full trail: `../FROZEN_CONFIG.md`.

## Archived earlier protocol iterations

| Directory | Held-out pass | Why superseded |
|---|---|---|
| `v1/` | 45.3% | Pre-`Event.declared`, pre-numeric-fact retrieval |
| `v2-pre-instrumentation/` | 57.3% | Same config as v3 but before actual Gemini API token capture |

Retained so the 34% → 45.3% → 57.3% → 62.0% progression is auditable rather than
asserted. Each directory holds its own `eval_run1/2/3` plus ablations `81/82/83`.
