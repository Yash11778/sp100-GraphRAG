# Project PPT — Slide-by-Slide Content & Speaker Notes
### Optimized TigerGraph GraphRAG for Context Optimization over S&P-100 SEC Filings

> How to use this file: each `## Slide N` is one PowerPoint slide. The **bullets**
> are what you put *on* the slide (keep them short on screen). The **"Detailed
> explanation / speaker notes"** block is what you *say* while presenting or write
> in the report — the deep version. Copy the bullets onto the slide, speak the
> notes.

**Deck order:** Introduction → Problem Statement → Proposed Title → Objectives →
Literature Survey → Methodology → Expected Results → Timeline → Conclusion →
References.

---

## Slide 1 — Title

**On the slide:**
- **Optimized GraphRAG for Context Optimization over S&P-100 SEC Filings**
- A Knowledge-Graph-Augmented Retrieval System on TigerGraph Savanna
- Your name · Guide name · Institution · Year
- *62% accuracy at ~45% fewer tokens than traditional RAG*

**Detailed explanation / speaker notes:**
This project builds an AI question-answering system over a large collection of
corporate financial filings. Instead of the usual "retrieve similar text and hope
the model figures it out," we build a **knowledge graph** of the companies and let
the system *traverse relationships* to answer questions precisely and cheaply. The
one-line result to anchor the whole talk: **4.1× the accuracy of traditional RAG
while using fewer tokens** — accuracy and efficiency improved *together*.

---

## Slide 2 — Introduction

**On the slide:**
- LLMs (ChatGPT, Gemini) don't know your *private* documents → they hallucinate
- **RAG** = retrieve passages from your corpus, then let the LLM answer from them
- **GraphRAG** = the retrieval layer is a *knowledge graph* of typed
  entities & relationships, not just a bag of text passages
- Domain: **400 SEC filings** of the **100 largest U.S. public companies**
- Goal: answer *relational* questions (aggregation, intersection, multi-hop)
  **accurately, with citations, and cheaply**

**Detailed explanation / speaker notes:**
Large Language Models are trained on a frozen snapshot of the public internet, so
they cannot answer questions about a specific private corpus — and when they don't
know, they confidently make things up. **Retrieval-Augmented Generation (RAG)**
solves half of this: before answering, we fetch relevant text from the corpus and
place it in the prompt, so the model answers *from the documents*.

The standard approach — **vector RAG** — embeds every passage as a number-vector
and returns the passages most similar to the question. That works for simple
lookups, but it struggles with questions that require *combining facts across many
documents*. **GraphRAG** addresses this by first extracting the documents into a
**knowledge graph** — nodes for companies, auditors, sectors, people, events, and
edges for the relationships between them — so questions can be answered by
*walking the graph* rather than sifting through text. Our domain is U.S. SEC
regulatory filings (annual reports, proxy statements, event filings) of the S&P
100, where questions are inherently relational (*who audits whom, which companies
are in which sector, which company did event X*) and where **exact answers and
citations are mandatory**.

---

## Slide 3 — Problem Statement

**On the slide:**
- Vector RAG retrieves *similar text*, not *facts* → fails on:
  - **Aggregation** ("how many companies does each Big-Four firm audit?")
  - **Intersection** ("KPMG-audited **AND** in Consumer Staples")
  - **Multi-hop bridges** ("the company that authorized a \$4.1B buyback — who
    audits it?")
- To compensate, it stuffs **more chunks** → **more tokens, higher cost, lower
  precision**, still wrong
- **Core question:** GraphRAG finds *more* evidence than vector search — but does
  the LLM *need* all of it?
- **Challenge:** make retrieval **smaller, more precise, higher quality** than
  traditional RAG **without losing accuracy**

**Detailed explanation / speaker notes:**
The specific technical problem: vector similarity is good at "find me passages
that look like this question," but the valuable questions over a filing corpus are
**relational**. You cannot reliably *retrieve* the answer to "which companies are
audited by KPMG **and** in Consumer Staples" by similarity — the answer isn't in
any single passage; it's a *set* you have to compute over the corpus. Vector RAG's
only lever is to retrieve *more* chunks and hope the LLM reasons over them, which
costs thousands of tokens and *still* frequently fails.

This is exactly the challenge posed: GraphRAG's high-recall retrieval finds more
evidence than plain vector search — **but does the model actually need all of
it?** The task is a **context-optimization** problem: produce a *smaller, more
precise, higher-quality* evidence set than traditional RAG, while keeping — ideally
improving — accuracy. The measurable failure we target: on our benchmark, plain
vector RAG scores only **15%** and spends **2,182 tokens** per question.

---

## Slide 4 — Proposed Title

**On the slide:**
- **"Optimized TigerGraph GraphRAG: Typed-Traversal Context Optimization for
  Token-Efficient, Citable Question Answering over Heterogeneous SEC Filings"**
- Short form: **Optimized GraphRAG for Context Optimization on S&P-100 Filings**
- Central thesis: **The graph traversal *is* the context optimization**

**Detailed explanation / speaker notes:**
The title names the three pillars of the contribution: (1) it's built on
**TigerGraph** (a native graph *and* vector database), (2) the novelty is
**typed-traversal context optimization** — using graph queries to shrink context
rather than a separate compression model, and (3) the outcome is
**token-efficient, citable QA** on a **heterogeneous** corpus (three different
document types). The thesis in one line — *the traversal is the context
optimization* — means: if the graph is modeled well, a typed query returns the
**answer set itself**, not a pile of chunks to sift. Most context-optimization
work retrieves a lot and *then* compresses; we invert it — retrieve *precisely* so
there's almost nothing to compress.

---

## Slide 5 — Objectives of the Project

**On the slide:**
1. Build a **knowledge graph** (9 vertex types, 12 edge types) from 400 SEC
   filings — with **zero LLM tokens** (fully deterministic extraction)
2. Use **TigerGraph Savanna** as one engine for **graph + vector + keyword** retrieval
3. Design a **deterministic router (0 LLM tokens)** that picks the right retrieval
   strategy per question
4. Answer relational questions by **typed traversal** → exact sets in *tens* of tokens
5. Add **context optimization** (dedup, form-priors, per-company balance, token cap)
6. **Prove** the gains: 3-pipeline controlled comparison + ablations + auditable
   token accounting
7. Ship a **live demo** (FastAPI + dashboard) running all pipelines on any question

**Detailed explanation / speaker notes:**
The objectives map one-to-one to the deliverables. First, build the graph
*without* spending LLM tokens on ingestion, so that every token we later report is
genuinely inference (nothing hidden in preprocessing). Second, exploit that
TigerGraph Savanna is simultaneously a graph store, a vector store (HNSW index on
chunk embeddings), and a keyword scanner — three retrieval modalities in one
database. Third, route each question deterministically with regex + dictionaries
(free and fully reproducible) rather than an LLM router. Fourth and fifth are the
core mechanism: answer aggregation/intersection/bridge questions by graph
traversal, and lightly optimize whatever text evidence remains. Sixth is
scientific rigor — a fair three-pipeline comparison and ablations that *prove*
each mechanism's contribution. Seventh makes it tangible: a dashboard that runs
any question through all three pipelines live.

---

## Slide 6 — Literature Survey

**On the slide (use the table format shown):**

| Sr. No. | Paper Title and its Author | Details of Publication with Year | Methods Used | Findings | Limitation |
|---|---|---|---|---|---|
| 1 | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — Lewis et al. (Facebook AI) | NeurIPS 2020 | Combine a parametric LLM with a non-parametric dense retriever (DPR) over Wikipedia | Introduced RAG; grounding LLMs in retrieved passages beats closed-book models on open-domain QA | Single-hop dense retrieval; cannot do multi-hop or corpus-wide aggregation reasoning |
| 2 | *Retrieval-Augmented Generation for LLMs: A Survey* — Gao et al. | arXiv 2023 | Survey of Naïve/Advanced/Modular RAG; taxonomy of retrieval, augmentation, generation | Systematizes RAG; identifies retrieval precision & context length as central bottlenecks | Confirms open problem: more retrieval ≠ better answers; context bloat hurts |
| 3 | *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Edge et al. (Microsoft) | arXiv 2024 (Microsoft GraphRAG) | LLM builds an entity knowledge graph + community summaries; global map-reduce over graph | Graph structure enables *global*/aggregation questions vector RAG cannot answer | **Graph is built by calling an LLM on every document → very high one-time token cost** |
| 4 | *Graph Retrieval-Augmented Generation: A Survey* — Peng et al. | arXiv 2024 | Survey of Graph-RAG: graph construction, graph-guided retrieval, graph-enhanced generation | Establishes GraphRAG as a field; typed structure improves multi-hop & relational QA | Notes lack of standardized, token-cost-aware evaluation and reproducible baselines |
| 5 | *G-Retriever: Retrieval-Augmented Generation for Textual Graph QA* — He et al. | NeurIPS 2024 | Retrieve a relevant subgraph (Prize-Collecting Steiner Tree) + LLM over it | Subgraph retrieval improves multi-hop QA and reduces hallucination on graph QA | Requires a pre-existing textual graph; subgraph selection is compute-heavy |
| 6 | *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* — Sarthi et al. (Stanford) | ICLR 2024 | Recursively cluster + summarize chunks into a tree; retrieve at multiple abstraction levels | Multi-level summaries help questions needing broad context | Summarization is LLM-driven (token cost); tree ≠ typed relational structure |
| 7 | *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* — Reimers & Gurevych | EMNLP 2019 | Siamese/triplet BERT to produce semantically meaningful sentence embeddings | Enables fast, high-quality dense retrieval (basis of MiniLM used here) | Pure similarity retrieval; no notion of relations, sets, or multi-hop logic |
| 8 | *Efficient and Robust Approximate Nearest Neighbor Search using HNSW* — Malkov & Yashunin | IEEE TPAMI 2018 (2016) | Hierarchical Navigable Small World graphs for ANN vector search | Fast, accurate approximate vector search at scale (the index used in Savanna & FAISS) | Indexes vectors only — no semantics beyond geometric similarity |
| 9 | *Billion-scale Similarity Search with GPUs (FAISS)* — Johnson, Douze, Jégou (Facebook AI) | IEEE Big Data 2019 | GPU-accelerated ANN library for dense vector search | Standard, scalable vector-search backend (our RAG baseline) | Vector-only; identical multi-hop/aggregation weakness as all dense RAG |
| 10 | *BERTScore: Evaluating Text Generation with BERT* — Zhang et al. (Cornell) | ICLR 2020 | Token-level contextual-embedding cosine similarity between candidate & reference | Correlates with human judgement better than BLEU/ROUGE (our answer-quality metric) | A similarity metric, not a correctness/factuality check on its own |

**Detailed explanation / speaker notes:**
Read the survey as a *story that leads to our design*. Papers 1–2 establish RAG
and name its bottleneck — retrieval precision and context length. Papers 3–5 are
the GraphRAG lineage: they show that a **typed graph structure** unlocks the
global/relational questions vector RAG can't do — but they also reveal the two
gaps we exploit: **(a) Microsoft-style GraphRAG builds the graph by calling an LLM
on every document (huge hidden token cost)**, and **(b) the field lacks
token-cost-aware, reproducible evaluation.** Papers 6 (RAPTOR) and the summary
surveys show context-optimization approaches that still rely on LLM
summarization. Papers 7–9 are the retrieval/indexing building blocks we actually
use (MiniLM embeddings, HNSW, FAISS). Paper 10 is our evaluation metric.

**Our positioning (say this explicitly):** we take the GraphRAG idea from
papers 3–5 but (i) build the graph with **0 LLM tokens** using deterministic
extraction — fixing gap (a); (ii) add **auditable, two-way token accounting and
ablations** — fixing gap (b); and (iii) use the traversal *itself* as the
context-optimizer instead of an LLM summarizer as in paper 6.

---

## Slide 7 — Methodology

**On the slide (architecture diagram + stages):**
```
Question
  → Deterministic Router (0 LLM tokens): detects companies, auditors,
    sectors, people, $-amounts, dates, event types
  → TigerGraph Savanna (one DB, three modalities):
        • Typed traversals   (AUDITED_BY, IN_SECTOR, REPORTS_EVENT, …)
        • kwsearch           (DB-side keyword scan of Chunk.text)
        • vsearch / _scoped  (HNSW vector search, 384-d, optionally ticker-filtered)
  → Context Optimizer: dedup · form-prior rerank · per-company balance · 1,800-token cap
  → Gemini 2.5 Flash (temp 0)  →  Answer + Citations
```
- **Ingestion (0 LLM tokens):** parse → deterministic extraction (auditor, events,
  topics) → 86,552 chunks → local MiniLM embeddings → load to Savanna → HNSW index
- **Graph schema:** 9 vertices (Company, Sector, AuditFirm, Person, Entity, Topic,
  Document, Event, Chunk+emb), 12 typed edges
- **3 pipelines compared:** LLM-only · Traditional RAG (FAISS top-8) · GraphRAG

**Detailed explanation / speaker notes:**
Walk the pipeline end-to-end.

**(1) Building the graph (one-time, 0 LLM tokens).** We parse each filing, then
extract structure *deterministically*: the **auditor** is found by scoring
Big-Four firm names by how often they sit within ~200 characters of an
"independent registered public accounting firm" cue (co-location beats an
incidental name-drop); **events** are pulled from 8-Ks by cue-phrase + regex
(dividend \$/share, buyback \$bn, leadership change), with a `declared` flag that
distinguishes an actual dividend declaration from an announced intention;
**topics/entities** by keyword rules. We chunk the text into 86,552 chunks, embed
each *locally* with all-MiniLM-L6-v2 (384-d), and upsert everything into
TigerGraph Savanna, which builds an HNSW vector index on the chunk embeddings.
Because none of this calls an LLM, the graph costs **0 tokens** to build — so
every token we later report is genuinely inference.

**(2) Answering a question.** A **deterministic router** (pure regex +
dictionaries, 0 tokens) reads the question and detects the entities in it, then
picks a strategy: aggregation/intersection → a **typed traversal** that returns
the exact company set in tens of tokens; a bridge question → resolve hop 1 from an
**Event vertex** (e.g. \$4.1B → FedEx), then hop 2 by another traversal (auditor)
or a company-scoped vector search; a people/open question → a **DB-side keyword
scan**; a remaining factual question → **company-scoped HNSW vector search**.
Whatever text evidence remains is passed through the **context optimizer** (dedup,
document-type priors, per-company balancing, a hard 1,800-token cap) and handed to
**Gemini 2.5 Flash** at temperature 0, which produces the answer with citations.

**(3) Fair comparison.** All three pipelines share the same generator; both
retrieval pipelines share the same embedder and the same 86,552 chunks — so the
*only* variable is retrieval + context optimization, and a blind LLM judge scores
every answer with the same rubric.

---

## Slide 8 — Expected Results

**On the slide:**

| Pipeline | Strict pass | Graded /3 | Citation /2 | BERTScore F1 | Avg tokens |
|---|---|---|---|---|---|
| LLM-only | 10.0% | 0.65 | 0.04 | −0.244 | 455 |
| Traditional RAG | 15.3% | 0.77 | 1.25 | −0.027 | 2,182 |
| **GraphRAG (ours)** | **62.0%** | **2.23** | **1.83** | **+0.219** | **1,190** |

- **4.1× RAG's accuracy at ~45% fewer tokens** (43% by Gemini's own usage metadata)
- **By tier:** Aggregation 8%→**78%**, Bridges 12%→**48%**, Single-hop 50%→**83%**
- **Ablation (the proof):** graph OFF → **22% pass AND +49% tokens** — the graph is
  *both* the accuracy and the efficiency mechanism

**Detailed explanation / speaker notes:**
The headline: on 50 held-out questions across 3 independent runs, GraphRAG reaches
**62% strict pass** versus 15% for traditional RAG and 10% for the LLM alone — and
it does so with **fewer tokens** (1,190 vs 2,182), a ~45% reduction. So accuracy
and cost improved *together*, not traded off. The tier breakdown shows *why*: the
biggest gains are on aggregation (8%→78%) and bridge (12%→48%) questions — exactly
the relational questions vector RAG can't do. The single most convincing number is
the **ablation**: if we switch the graph off and fall back to global vector search
(same database, same embeddings), pass rate collapses to **22%** *while token
usage rises 49%*. That's the whole thesis, measured: vector-only retrieval needs
*more* context to do *less*. We also report tokens two independent ways (a tiktoken
estimate and the Gemini API's own usage metadata) and they agree.

---

## Slide 9 — Timeline

**On the slide (Gantt-style / phased):**

| Phase | Weeks | Activity | Milestone |
|---|---|---|---|
| 1. Study & scoping | 1–2 | Literature survey; understand corpus & question tiers | Problem + design fixed |
| 2. Ingestion | 3–4 | Parse filings; deterministic extraction; chunk + embed | 400 docs, 86,552 chunks, 0 LLM tokens |
| 3. Graph build | 5 | Schema (9V/12E); load to Savanna; HNSW + kwsearch/vsearch | Live graph on TigerGraph Savanna |
| 4. Pipelines | 6–7 | Router + traversals; RAG + LLM-only baselines | 3 pipelines runnable |
| 5. Tuning (dev set only) | 8 | Tune on 18-Q dev set; freeze config | Frozen config (100% dev pass) |
| 6. Evaluation | 9 | 3 held-out runs + ablations; judge + BERTScore | 62% result, ablations |
| 7. Demo & docs | 10 | FastAPI + dashboard; report, deck, blog | Live demo + documentation |

**Detailed explanation / speaker notes:**
The project runs in seven phases over roughly ten weeks. We begin with the
literature survey and understanding the corpus and the four question tiers, then
build the *zero-token* ingestion pipeline (parsing, deterministic extraction,
chunking, local embeddings). Next we define the graph schema and load it into
Savanna with the HNSW index and the keyword/vector queries. We then implement the
router and traversal layer plus the two baselines. Crucially, we **tune only on a
small 18-question dev set and freeze the configuration before touching the held-out
questions** — this keeps the evaluation honest. Then come three independent
held-out runs plus ablations, and finally the live demo and documentation. Each
phase ends in a concrete, checkable milestone.

*(Adjust the week numbers to match your actual academic schedule.)*

---

## Slide 10 — Conclusion

**On the slide:**
- Built a **GraphRAG** system that answers relational SEC-filing questions by
  **traversing a typed knowledge graph** instead of dumping similar text
- **The traversal *is* the context optimization** → answers, not evidence to sift
- **62% accuracy at ~45% fewer tokens** — accuracy *and* efficiency improved together
- **Proven by ablation:** removing the graph costs **−40 pts AND +49% tokens**
- **Honest & auditable:** 0-token ingestion, two-way token accounting, disclosed
  data/key discrepancies, archived iterations
- **Generalizable:** swap the extraction rules → same engine serves contracts,
  medical records, research corpora
- **Future work:** cross-document temporal reasoning (Tier D), richer numeric
  second-hops, learned (vs regex) event extraction

**Detailed explanation / speaker notes:**
To conclude: we set out to make GraphRAG retrieval *smaller, more precise, and
higher quality* than traditional RAG without losing accuracy — and we did, by a
wide margin. The key insight, which the whole project validates, is that a
well-modeled graph traversal returns the *answer* rather than evidence to sift, so
it simultaneously improves accuracy and cuts tokens. We proved this with a
controlled three-pipeline comparison and an ablation that shows the graph is
indispensable. The system is deliberately honest — zero-token ingestion, two
independent token counts, and disclosed discrepancies where our data follows the
source documents rather than the answer key. The mechanisms are document-agnostic,
so the same architecture generalizes to other domains by changing only the
extraction rules. Remaining hard problems — cross-document temporal reasoning and
numeric multi-hop bridges — are measured, disclosed, and the natural next steps.

---

## Slide 11 — References

**On the slide:**
1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive
   NLP Tasks.* **NeurIPS 2020.**
2. Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language
   Models: A Survey.* **arXiv:2312.10997.**
3. Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to
   Query-Focused Summarization.* **arXiv:2404.16130 (Microsoft GraphRAG).**
4. Peng, B. et al. (2024). *Graph Retrieval-Augmented Generation: A Survey.*
   **arXiv:2408.08921.**
5. He, X. et al. (2024). *G-Retriever: Retrieval-Augmented Generation for Textual
   Graph Understanding and Question Answering.* **NeurIPS 2024.**
6. Sarthi, P. et al. (2024). *RAPTOR: Recursive Abstractive Processing for
   Tree-Organized Retrieval.* **ICLR 2024.**
7. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
   Siamese BERT-Networks.* **EMNLP 2019.**
8. Malkov, Y. & Yashunin, D. (2018). *Efficient and Robust Approximate Nearest
   Neighbor Search Using Hierarchical Navigable Small World Graphs.* **IEEE TPAMI.**
9. Johnson, J., Douze, M. & Jégou, H. (2019). *Billion-Scale Similarity Search
   with GPUs (FAISS).* **IEEE Transactions on Big Data.**
10. Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT.*
    **ICLR 2020.**
11. TigerGraph. *TigerGraph Savanna & GSQL Documentation.* docs.tigergraph.com.
12. Google. *Gemini 2.5 Flash Model Documentation.* ai.google.dev.
13. U.S. SEC EDGAR — corporate filings (10-K, DEF 14A, 8-K). sec.gov/edgar.

**Detailed explanation / speaker notes:**
References 1–6 are the RAG/GraphRAG research lineage that motivates the design;
7–10 are the retrieval, indexing, and evaluation building blocks we use directly;
11–13 are the platform and data sources (TigerGraph Savanna, Gemini, and SEC
EDGAR). Verify the exact arXiv IDs / page numbers against the originals before
final submission, and format them in your institution's required citation style
(IEEE / APA).

---

### Appendix — quick facts to have ready for Q&A

- **Dataset:** 100 companies × 4 filings = 400 docs; 10-K + DEF 14A + 2×8-K;
  ≈ 90 MB, ≈ 20.5 M Gemini tokens; 86,552 chunks.
- **Graph:** 9 vertex types, 12 edge types; 664 core vertices + 86,552 chunks;
  ~174K edges. Embeddings: all-MiniLM-L6-v2, 384-d, HNSW/COSINE.
- **Why 0-token ingestion matters:** no hidden preprocessing cost → every reported
  token is inference → the RAG-vs-GraphRAG comparison is fair.
- **Why deterministic router:** free, instant, fully reproducible, auditable
  (`routing_tokens = 0` on every row) — vs an LLM router's cost/latency/nondeterminism.
- **Biggest single number:** graph OFF → 22% pass **and** +49% tokens.
- **Honesty note:** where our extracted data disagrees with the answer key (e.g.
  Citigroup→KPMG), we follow the source filing and disclose it with citations.
