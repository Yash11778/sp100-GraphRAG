# One-Time Ingestion / Extraction / Embedding / Indexing Costs

Reported separately from per-question inference costs, as the Round 3 brief requires.

## Headline

**The entire ingestion pipeline consumed 0 LLM tokens.** Every extraction step is
deterministic (regex/keyword/structured parsing), and embeddings are computed
locally with an open-source model. One-time cost is purely local compute.

| Step | Method | LLM tokens | Output |
|---|---|---:|---|
| Parsing (`ingestion/parse_filings.py`) | deterministic text/section parsing | 0 | 400 parsed docs (100 companies × 10-K, DEF14A, 2×8-K) |
| Auditor extraction (`extract_auditor.py`) | canonical-name matching over DEF14A | 0 | 100 company→auditor pairs |
| Event extraction (`extract_events.py`) | cue-phrase + regex over 8-Ks | 0 | 149 events (133 leadership, 9 dividend, 7 buyback) |
| Mentions/topics (`extract_mentions.py`) | keyword rules over 10-Ks | 0 | TSMC/NVIDIA entities, climate/GLP-1 topics |
| Chunking (`build_chunks.py`) | fixed-size chunking | 0 | 86,552 chunks |
| Embeddings | local `all-MiniLM-L6-v2` (384-d), ~25.7M tokens of text embedded | 0 (local compute) | 86,552 × 384 vectors |
| Graph + vector load (`scripts/load_*.py`) | REST++ upserts to Savanna | 0 | 664 core vertices, 86,552 Chunk vertices, ~174K edges |
| Index build | Savanna HNSW (COSINE, 384-d) on `Chunk.emb` | 0 | 1 vector index |

Wall-clock (one machine, one pass): parsing+extraction ≈ minutes; embedding ≈
tens of minutes CPU; Savanna load ≈ 2 min (chunks) + seconds (core).

## Why this matters for the comparison

Because ingestion is deterministic and LLM-free, **all** LLM spend shows up in the
per-question inference accounting — there is no hidden token cost shifted into
preprocessing. The same chunk store + embedding model serves both Traditional RAG
(FAISS) and GraphRAG (Savanna HNSW), isolating graph retrieval + context
optimization as the experimental variables.
