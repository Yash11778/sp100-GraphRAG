# PPT Slide Content — Mapped to Your Reference Design
### Optimized TigerGraph GraphRAG for Context Optimization over S&P-100 SEC Filings

> **How to read this file.** Each slide below mirrors the *exact box layout* of
> your reference deck (the purple-header template). For every box I give the
> label + the text to paste. For every diagram I give a **🎨 Napkin AI prompt** —
> paste that prompt into Napkin AI to generate the visual in the same style.
> Design constants to keep on every slide: **purple header bar**, white body,
> **orange→purple gradient footer bar**, lavender/grey content boxes, serif
> numerals (01/02/03).

**Slide order:** Introduction · Problem Statement · Proposed Title · Objectives ·
Literature Survey · Methodology · Expected Results · Timeline · Conclusion ·
References.

---

## SLIDE 1 — Introduction

**Header:** `Introduction`

**📦 Box (top, blue-tint) — "The Numbers Speak"**
> Enterprises store millions of documents, yet large language models (ChatGPT,
> Gemini) cannot read a *private* corpus — so they hallucinate confident, unsourced
> answers.
>
> Traditional RAG retrieves more and more text to compensate, making answers slow,
> expensive, and still wrong on questions that need facts combined across many
> documents.

**📦 Box (left, grey) — "Current Approach"** *(serif numerals 01–04)*
- **01** Vector-only RAG retrieves *similar-looking* text chunks
- **02** High-recall chunk dumps → thousands of tokens per question
- **03** Fails on aggregation, intersection & multi-hop questions
- **04** No exact sets, weak citations, high cost & latency

**🔴 Bottom tagline (red italic, centred):**
> *Need for a Token-Efficient, Explainable GraphRAG for Relational Question
> Answering over Documents*

**🎨 Napkin AI prompt (right-side diagram — replaces the "Recycling Cycle"):**
```
Create a clean horizontal 5-step cycle diagram titled "Document QA Pipeline"
with numbered circular icons:
1. Documents — 400 SEC filings collected
2. Knowledge Graph — entities & relationships extracted
3. Retrieve — graph traversal + vector + keyword search
4. Optimize — small, precise evidence set
5. Answer — cited, accurate response by the LLM
Use a modern flat icon style, blue/green/purple accent colors, curved arrows
connecting the steps in a loop.
```

---

## SLIDE 2 — Problem Statement

**Header:** `Problem Statement`

**📦 Box (top-left, lavender) — "WHAT Is the Problem???"**
- Relational questions need facts *combined across many documents* (auditors,
  sectors, board members, corporate events).
- Vector RAG retrieves *similar text*, not *facts* — it cannot compute exact sets.
- To compensate it stuffs more chunks → more tokens, lower precision.
- Result: expensive answers that are still wrong (15% pass, 2,182 tokens/question).

**📦 Box (bottom-left, lavender) — "WHY This Problem Important???"**
- Analysts, auditors & compliance teams need **exact, cited** answers.
- Wrong or unsourced answers are costly and fail audits.
- "More retrieval" raises cost & latency without fixing accuracy.
- Enterprises need a **token-efficient, auditable** way to query private corpora.

**📦 Box (right, plain) — "Real-World Relevance"** *(bullets)*
- LLMs alone hallucinate on private data — no corpus access, no citations.
- Traditional RAG can't answer aggregation / multi-hop questions reliably.
- LLM-based GraphRAG builds the graph by calling an LLM on every document → huge
  hidden cost.
- A **deterministic, graph-based retrieval** system is needed for accurate,
  cheap, citable QA.

**🎨 Napkin AI prompt (bottom-right — replaces the Venn diagram):**
```
Create a 2-circle Venn diagram titled "Bridging the Gap in Document QA".
Left circle: "Low Cost & Fewer Tokens". Right circle: "High Accuracy &
Citations". Overlap label: "Optimized GraphRAG". Soft pastel purple and blue
circles, minimal flat style, small icons inside each circle (a coin/token icon
left, a checkmark/target icon right, a knowledge-graph node icon in the overlap).
```

---

## SLIDE 3 — Proposed Title

**Header:** `Proposed Title`

**🏷️ Title banner (torn-paper style, centred):**
> Optimized TigerGraph GraphRAG: Typed-Traversal Context Optimization for
> Token-Efficient, Citable Question Answering over Heterogeneous SEC Filings

**📦 Box (left, grey-blue) — "Explanation of Title"**
- **Optimized GraphRAG**
  - Retrieval is a knowledge graph, tuned to return answers, not raw chunks.
- **Typed-Traversal Context Optimization**
  - A graph query *is* the compression — it returns the exact answer set.
- **Token-Efficient & Citable**
  - Fewer tokens per correct answer, every answer backed by a source filing.
- **Heterogeneous SEC Filings**
  - Works across three document types: 10-K, DEF 14A, 8-K.

**📦 Box (right, grey-blue) — "Scope of the Project"** *(bullets)*
- Knowledge graph over **400 filings / 100 S&P-100 companies**
- **9 vertex types, 12 edge types** (Company, Auditor, Sector, Event, Chunk…)
- **Deterministic router** — 0 LLM tokens, fully reproducible
- **3 retrieval modes** in one DB: traversal · keyword scan · vector (HNSW)
- **Context optimization**: dedup · form-priors · balance · token cap
- **FastAPI backend + live dashboard**; uses only text + local embeddings

**🎨 Napkin AI prompt (optional supporting visual):**
```
Create a simple word-breakdown diagram: a central pill labeled "Optimized
GraphRAG" with four labeled branches radiating out — "Knowledge Graph",
"Typed Traversal", "Token-Efficient", "Citable Answers" — each branch with a
small matching flat icon. Purple and teal accents, clean minimal style.
```

---

## SLIDE 4 — Objectives of the Project

**Header:** `Objectives of the Project`

**📦 Box (top-left, lavender) — "Main Objective"**
- Build an explainable, **token-efficient** GraphRAG pipeline
- Answer relational questions over private SEC filings
- Return **exact sets + citations**, not fuzzy passages
- Reduce inference tokens **without** losing accuracy

**📦 Box (mid-left, lavender) — "Specific Objectives"**
- Build a 9-vertex / 12-edge knowledge graph in **TigerGraph Savanna**
- Extract structure with **0 LLM tokens** (deterministic regex/keyword rules)
- Design a **deterministic router** → typed traversal · kwsearch · scoped HNSW
- Add **context optimization** + a **3-pipeline controlled comparison**

**📦 Box (bottom-left, lavender) — "Expected Outcomes"**
- Working GraphRAG system on live Savanna graph
- Exact-set answers for aggregation & bridge questions
- **62% accuracy at ~45% fewer tokens** than traditional RAG
- Ablations proving the graph is the key mechanism
- Live demo (FastAPI + dashboard) + auditable token accounting

**🎨 Napkin AI prompt (right-side pipeline diagram):**
```
Create a horizontal AI pipeline diagram titled "GRAPHRAG PIPELINE FOR DOCUMENT QA"
with 4 boxed stages connected by arrows:
1. QUESTION — user asks a question (chat icon)
2. DETERMINISTIC ROUTER — 0 LLM tokens; detects companies, auditors, sectors,
   amounts, dates (filter/route icon)
3. TIGERGRAPH SAVANNA — 3 retrieval modes: Typed Traversal, Keyword Scan,
   Vector Search (database + graph-nodes icon)
4. GEMINI 2.5 FLASH — cited answer (spark/answer icon)
Below stage 3 show three small branch labels: "Traversal", "kwsearch",
"vsearch". Below stage 4 add a caption box: "Context Optimizer: dedup,
form-priors, balance, 1800-token cap". Purple/blue/green flat style.
```

---

## SLIDE 5 — Literature Survey
*(Not in your sample images, but required in your list — same purple-header +
table style. Keep the table on the slide; put notes in speaker script.)*

**Header:** `Literature Survey`

**📋 Table (blue header row, exactly the columns from your screenshot):**

| Sr. No. | Paper Title and its Author | Details of Publication with Year | Methods Used | Findings | Limitation |
|---|---|---|---|---|---|
| 1 | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — Lewis et al. | NeurIPS 2020 | Parametric LLM + dense retriever (DPR) over Wikipedia | Introduced RAG; grounding LLMs in retrieved text beats closed-book QA | Single-hop; no multi-hop or corpus-wide aggregation |
| 2 | *Retrieval-Augmented Generation for LLMs: A Survey* — Gao et al. | arXiv 2023 | Naïve / Advanced / Modular RAG taxonomy | Names retrieval precision & context length as the core bottlenecks | Confirms: more retrieval ≠ better answers; context bloat hurts |
| 3 | *From Local to Global: A Graph RAG Approach…* — Edge et al. (Microsoft) | arXiv 2024 | LLM-built entity graph + community summaries + global map-reduce | Graph structure unlocks global/aggregation questions | **Graph built by an LLM per document → very high hidden token cost** |
| 4 | *Graph Retrieval-Augmented Generation: A Survey* — Peng et al. | arXiv 2024 | Survey of graph construction, graph-guided retrieval, generation | Establishes GraphRAG; typed structure improves relational QA | Lacks token-cost-aware, reproducible evaluation |
| 5 | *G-Retriever: RAG for Textual Graph QA* — He et al. | NeurIPS 2024 | Retrieve a relevant subgraph (Steiner tree) + LLM | Subgraph retrieval improves multi-hop QA, cuts hallucination | Needs a pre-existing graph; subgraph selection is heavy |
| 6 | *RAPTOR: Recursive Abstractive Tree Retrieval* — Sarthi et al. | ICLR 2024 | Recursive cluster + summarize into a retrieval tree | Multi-level summaries help broad-context questions | LLM summarization = token cost; tree ≠ typed relations |
| 7 | *Sentence-BERT* — Reimers & Gurevych | EMNLP 2019 | Siamese BERT sentence embeddings | Fast, high-quality dense retrieval (basis of MiniLM used here) | Similarity only; no relations, sets, or logic |
| 8 | *Approximate NN Search using HNSW* — Malkov & Yashunin | IEEE TPAMI 2018 | Hierarchical Navigable Small World graphs | Fast, accurate vector search (index used in Savanna & FAISS) | Geometric similarity only |
| 9 | *Billion-Scale Similarity Search (FAISS)* — Johnson et al. | IEEE Big Data 2019 | GPU-accelerated ANN vector search | Scalable vector-search backend (our RAG baseline) | Vector-only; same multi-hop weakness |
| 10 | *BERTScore* — Zhang et al. | ICLR 2020 | Contextual-embedding cosine similarity of candidate vs reference | Correlates with human judgement (our quality metric) | A similarity metric, not a factuality check |

**🎨 Napkin AI prompt (optional visual instead of a plain table):**
```
Create a horizontal research-evolution timeline titled "From RAG to Optimized
GraphRAG" with 5 milestone nodes: "2020 RAG (Lewis)", "2023 RAG Survey (Gao)",
"2024 GraphRAG (Microsoft)", "2024 GraphRAG Survey (Peng)", "2024 G-Retriever".
End the arrow at a highlighted node "Our Work: 0-token GraphRAG + Auditable
Eval". Flat minimal style, purple accent, small icons per node.
```

---

## SLIDE 6 — Methodology

**Header:** `Methodology`

**📦 Left column box (purple header) — "Approach"** *(bold label + line each)*
- **Replace fuzzy chunk-dump** — with typed graph traversal
- **TigerGraph Savanna** — graph + vector + keyword in one DB
- **Deterministic Router** — 0 LLM tokens, fully reproducible
- **Event Vertices** — resolve multi-hop bridge questions
- **Context Optimizer** — dedup / form-priors / balance / cap
- **Zero-token Ingestion** — deterministic extraction + local embeddings

**📦 Right column box (purple header) — "Tech Stack"** *(grouped labels)*
- **Graph + Vector DB:** TigerGraph Savanna (GSQL, HNSW, REST++)
- **Embeddings:** all-MiniLM-L6-v2 (384-d); FAISS (RAG baseline)
- **Generator + Judge:** Gemini 2.5 Flash (temp 0)
- **Backend:** FastAPI + uvicorn
- **Frontend:** Vite dashboard (live 3-pipeline compare)
- **Libraries:** Python, NumPy, Pandas, tiktoken, BERTScore
- **Dataset:** 400 SEC filings — 100 S&P-100 companies

**🎨 Napkin AI prompt (centre — the big methodology flow):**
```
Create a detailed horizontal methodology flow titled "METHODOLOGY" with 6
numbered stages in rounded boxes connected by arrows:
01 INGESTION — parse 400 SEC filings, deterministic extraction, 86,552 chunks,
   local embeddings (0 LLM tokens)
02 KNOWLEDGE GRAPH — build 9 vertices / 12 edges in TigerGraph Savanna + HNSW
   vector index
03 DETERMINISTIC ROUTER — detect companies, auditors, sectors, amounts, dates;
   pick a strategy (0 LLM tokens)
04 RETRIEVAL — Typed Traversal | Keyword Scan (kwsearch) | Vector Search
   (vsearch / scoped)
05 CONTEXT OPTIMIZER — dedup, form-prior rerank, per-company balance,
   1800-token cap
06 GENERATION — Gemini 2.5 Flash produces a cited answer
Add a bottom ribbon with tags: "0 LLM-token ingestion • Deterministic •
Auditable • Token-Efficient • Citable". Modern flat style, purple/blue/green.
```

---

## SLIDE 7 — Expected Results

**Header:** `Expected Results`

**📦 Box (left, lavender) — "01 WHAT WE AIM TO ACHIEVE"**
> Build a token-efficient, explainable GraphRAG system that answers relational
> questions over private SEC filings by traversing a knowledge graph — returning
> exact sets and citations while using fewer tokens than traditional RAG.

**📦 Box (right, lavender) — "02 OUTPUT OF THE SYSTEM"**
> For any question, the system routes deterministically, retrieves via graph
> traversal / keyword scan / scoped vector search, optimizes the evidence, and
> returns a **cited answer** — recording tokens, graph path, and judge scores per
> question.

**🏷️ "03 BENEFITS AT A GLANCE"** *(6 numbered coloured cards)*
1. **High Accuracy** — 62% pass vs 15% RAG (4.1×)
2. **Token-Efficient** — ~45% fewer tokens than RAG
3. **Exact Aggregation** — 8% → 78% on Tier-C questions
4. **Auditable & Cited** — 0-token router, source citations
5. **One-DB Retrieval** — graph + vector + keyword in Savanna
6. **Generalizable** — swap extraction rules for any corpus

**🎨 Napkin AI prompt (optional results bar chart):**
```
Create a grouped bar chart titled "Accuracy vs Token Cost" comparing three
pipelines: "LLM-only" (10% pass, 455 tokens), "Traditional RAG" (15% pass,
2182 tokens), "GraphRAG (Ours)" (62% pass, 1190 tokens). Show two bars per
pipeline — one for pass rate (%), one for average tokens — with GraphRAG
highlighted in purple. Clean flat style, clear legend.
```

---

## SLIDE 8 — Timeline

**Header:** `Timeline`

**📦 Left-top — "Project Phases"** *(phase icons Phase 1–6)*
- **Phase 1** — Literature Survey & Problem Scoping
- **Phase 2** — Dataset & Ingestion (parse, extract, chunk, embed)
- **Phase 3** — Knowledge-Graph Build (schema + Savanna load + indexes)
- **Phase 4** — Pipelines (router, traversals, RAG & LLM baselines)
- **Phase 5** — Tuning (dev-set only) → Freeze Config
- **Phase 6** — Evaluation, Ablations, Demo & Documentation

**📦 Left-bottom — "Milestones"** *(flow boxes, same as sample)*
Problem Fixed → Corpus Ingested → Graph Live on Savanna → 3 Pipelines Running →
Config Frozen (100% dev) → 62% Held-out + Ablations → Demo & Docs Done

**📅 Right — "Completion Plan" (Gantt)** — tasks × months:

| Task | Span |
|---|---|
| Literature Survey | Month 1 |
| Dataset & Ingestion | Month 1–2 |
| Knowledge-Graph Build | Month 2 |
| Router + Traversal Layer | Month 3 |
| RAG + LLM Baselines | Month 3 |
| Context Optimization + Tuning | Month 4 |
| Held-out Eval + Ablations | Month 5 |
| FastAPI + Dashboard | Month 5 |
| Documentation & Final Presentation | Month 6 |

**🎨 Napkin AI prompt (Gantt / completion plan):**
```
Create a Gantt-style completion plan titled "Completion Plan". Rows (tasks):
Literature Survey; Dataset & Ingestion; Knowledge-Graph Build; Router +
Traversal; RAG + LLM Baselines; Context Optimization + Tuning; Held-out Eval +
Ablations; FastAPI + Dashboard; Documentation & Final Presentation. Columns are
6 monthly buckets. Show purple horizontal bars staggered across the months, with
a "Deadline" marker at the end. Clean flat style.
```
*(Also reuse the sample's "Project Phases" horizontal icon-timeline and the
"Milestones" rounded-box flow — just relabel with the phase/milestone text above.)*

---

## SLIDE 9 — Conclusion

**Header:** `Conclusion`

**📦 Top banner (purple, italic):**
> This project delivers a token-efficient, explainable GraphRAG framework that
> answers relational questions over private SEC filings by traversing a knowledge
> graph — returning exact, cited answers while using ~45% fewer tokens than
> traditional RAG.

**🃏 Three cards (01 / 02 / 03):**

- **01 — Problem Addressed** *(purple)*
  > Vector-only RAG can't answer aggregation, intersection, and multi-hop
  > questions; it stuffs more chunks, raising cost and lowering precision, and
  > still fails.

- **02 — Proposed Solution** *(blue)*
  > A deterministic router + TigerGraph Savanna (graph + vector + keyword) + a
  > context optimizer + Gemini 2.5 Flash. The graph traversal *is* the context
  > optimization — it returns answers, not evidence to sift.

- **03 — Impact & Results** *(green)*
  > **62% accuracy at ~45% fewer tokens (4.1× RAG)**; ablation proves the graph is
  > essential (−40 pts & +49% tokens without it). Fully auditable, 0-token
  > ingestion, generalizable to any document corpus.

**🟧 Bottom keyword bar:**
> • Typed-Traversal Retrieval | • 0-Token Deterministic Router | • Token-Efficient
> & Cited | • Generalizable GraphRAG

**🎨 Napkin AI prompt (optional closing visual):**
```
Create a simple 3-pillar summary graphic with icons: Pillar 1 "Accurate" (target
icon, 62% pass), Pillar 2 "Efficient" (token/coin icon, -45% tokens), Pillar 3
"Auditable" (checklist/citation icon, 0-token ingestion). Purple, blue, green
columns, minimal flat style, one headline stat under each.
```

---

## SLIDE 10 — References

**Header:** `References`

1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* **NeurIPS 2020.**
2. Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* **arXiv:2312.10997.**
3. Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* **arXiv:2404.16130 (Microsoft).**
4. Peng, B. et al. (2024). *Graph Retrieval-Augmented Generation: A Survey.* **arXiv:2408.08921.**
5. He, X. et al. (2024). *G-Retriever: RAG for Textual Graph Understanding and QA.* **NeurIPS 2024.**
6. Sarthi, P. et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* **ICLR 2024.**
7. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT.* **EMNLP 2019.**
8. Malkov, Y. & Yashunin, D. (2018). *Approximate Nearest Neighbor Search using HNSW.* **IEEE TPAMI.**
9. Johnson, J., Douze, M. & Jégou, H. (2019). *Billion-Scale Similarity Search with GPUs (FAISS).* **IEEE Big Data.**
10. Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* **ICLR 2020.**
11. TigerGraph. *TigerGraph Savanna & GSQL Documentation.* docs.tigergraph.com.
12. Google. *Gemini 2.5 Flash Documentation.* ai.google.dev.
13. U.S. SEC EDGAR — corporate filings (10-K, DEF 14A, 8-K). sec.gov/edgar.

---

## Quick reference — every diagram you need in Napkin AI

| Slide | Diagram | Napkin prompt location |
|---|---|---|
| Introduction | 5-step "Document QA Pipeline" cycle | Slide 1 |
| Problem Statement | 2-circle Venn "Bridging the Gap" | Slide 2 |
| Proposed Title | word-breakdown branches (optional) | Slide 3 |
| Objectives | 4-stage GraphRAG pipeline | Slide 4 |
| Literature Survey | RAG→GraphRAG evolution timeline (optional) | Slide 5 |
| Methodology | 6-stage methodology flow + tag ribbon | Slide 6 |
| Expected Results | accuracy-vs-tokens bar chart (optional) | Slide 7 |
| Timeline | Gantt "Completion Plan" + phase icons + milestones | Slide 8 |
| Conclusion | 3-pillar summary (optional) | Slide 9 |

**Design constants (repeat on every slide):** purple header bar with white bold
title · orange→purple gradient footer bar · lavender / grey-blue rounded content
boxes · serif numerals for 01–04 · red italic tagline only on the Introduction
slide.
