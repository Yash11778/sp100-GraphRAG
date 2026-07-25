"""Baseline-fairness check: does Traditional RAG improve if its chunks carry the
same [TICKER FORM] source labels that GraphRAG's evidence carries?

Why this exists. Our audit found an UNDISCLOSED asymmetry between the pipelines:

  pipeline3 (GraphRAG) builds evidence as   f"[{ticker} {form}] {text}"   -> 78% of
    its evidence blocks are source-labelled
  pipeline2 (Traditional RAG) builds it as  "\n\n---\n\n".join(c["text"])  -> 0%

So the baseline received naked chunk text with no indication of which company or
filing each chunk came from, while GraphRAG did. On multi-company questions
(EQ13/14/15/23) that is not a retrieval-quality difference — it removes the
baseline's ability to attribute a number to a company at all. The brief requires
Traditional RAG to be "a competent, good-faith baseline" and requires every
material difference to be disclosed, so this needed measuring rather than
explaining away.

The chunk records already carry `ticker` and `form`; pipeline2 simply never used
them. Labels cost ~5 tokens/chunk (~40 tokens on top of ~2,046, i.e. ~2%).

This script does NOT modify pipeline2 or any reported number. It re-runs the
baseline with labels as a clearly-marked supplementary experiment and reports the
delta, so the size of the unfairness is on the record either way.

    .venv/bin/python eval/fairness_check_rag_labels.py            # all 50
    .venv/bin/python eval/fairness_check_rag_labels.py --limit 10

Output: eval/results/fairness_rag_labelled.csv
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pipelines.utils import count_tokens, gemini_generate, setup_gemini  # noqa: E402
from eval.judge import llm_judge_with_source  # noqa: E402

FAISS = ROOT / "data/chunks/rag_index.faiss"
CHUNKS = ROOT / "data/chunks/chunks.pkl"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Verbatim from pipelines/pipeline2_rag.py -- unchanged, so the ONLY difference
# between the reported baseline and this run is the chunk source label.
PROMPT = (
    "You are an expert financial-filings assistant. Use ONLY the context below to "
    "answer the question. Be precise with names, numbers, dates. If the context lists "
    "multiple entities, list all that apply. Do not say 'the context does not state' "
    "if the answer is present.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=8)   # identical to pipeline2
    a = ap.parse_args()

    import faiss
    import numpy as np
    from fastembed import TextEmbedding

    embedder = TextEmbedding(EMBED_MODEL)
    index = faiss.read_index(str(FAISS))
    chunks = pickle.load(open(CHUNKS, "rb"))
    client = setup_gemini()

    qa = json.loads((ROOT / "data/qa/eval_questions.json").read_text())
    if a.limit:
        qa = qa[:a.limit]

    base = pd.read_csv(ROOT / "eval/results/eval_run1.csv")
    base = base[base.pipeline == "basic_rag"].set_index("qid")

    rows = []
    for item in tqdm(qa, desc="RAG + source labels"):
        q, gt, qid = item["question"], item["answer"], item.get("id", "")
        v = np.array(list(embedder.embed([q]))[0], dtype=np.float32).reshape(1, -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        _, idxs = index.search(v, a.top_k)
        got = [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]

        # THE ONLY CHANGE: prefix each chunk with its source, exactly as pipeline3 does.
        ctx = "\n\n---\n\n".join(f"[{c['ticker']} {c['form']}] {c['text']}" for c in got)

        prompt = PROMPT.format(context=ctx, question=q)
        t0 = time.time()
        ans = gemini_generate(client, prompt, max_tokens=512)
        lat = round(time.time() - t0, 3)
        verdict, _ = llm_judge_with_source(q, gt, ans)

        rows.append({
            "qid": qid, "tier": item.get("tier", ""),
            "judge_labelled": verdict,
            "judge_reported": base.loc[qid, "judge"] if qid in base.index else None,
            "ctx_tokens_labelled": count_tokens(ctx),
            "ctx_tokens_reported": base.loc[qid, "context_tokens"] if qid in base.index else None,
            "total_tokens_labelled": count_tokens(prompt) + count_tokens(ans),
            "total_tokens_reported": base.loc[qid, "total_tokens"] if qid in base.index else None,
            "latency_s": lat, "answer": ans, "ground_truth": gt, "question": q,
        })
        pd.DataFrame(rows).to_csv(ROOT / "eval/results/fairness_rag_labelled.csv", index=False)

    df = pd.DataFrame(rows)
    rep = 100 * df.judge_reported.eq("PASS").mean()
    lab = 100 * df.judge_labelled.eq("PASS").mean()
    print("\n" + "=" * 70)
    print(f"Traditional RAG, {len(df)} held-out questions, top_k={a.top_k}")
    print(f"  as reported (no source labels) : {rep:5.1f}% pass   "
          f"{df.total_tokens_reported.mean():6.0f} tokens")
    print(f"  with [TICKER FORM] labels      : {lab:5.1f}% pass   "
          f"{df.total_tokens_labelled.mean():6.0f} tokens")
    print(f"  DELTA                          : {lab - rep:+5.1f} pts   "
          f"{df.total_tokens_labelled.mean() - df.total_tokens_reported.mean():+6.0f} tokens")
    flip = df[df.judge_reported.ne(df.judge_labelled)]
    if len(flip):
        print(f"\n  verdict changed on {len(flip)} questions:")
        for r in flip.itertuples():
            print(f"    {r.qid} ({r.tier}): {r.judge_reported} -> {r.judge_labelled}")
    print("\nWrote eval/results/fairness_rag_labelled.csv")


if __name__ == "__main__":
    main()
