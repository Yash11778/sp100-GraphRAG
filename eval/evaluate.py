"""Round 3 offline benchmark — run all three pipelines on the eval set, judge each
answer (strict LLM-as-judge + BERTScore), and record full token accounting.

Every field the brief's Required Results section asks for is written per row:
system+question+context tokens, routing tokens, output tokens, total inference
tokens, retrieved chunk ids (citations), graph usage, latency, judge, BERTScore.

    python eval/evaluate.py                 # held-out 50, one run
    python eval/evaluate.py --dev           # dev set instead
    python eval/evaluate.py --run 2         # tag output as run 2
    python eval/evaluate.py --pipelines graphrag   # subset

Needs GEMINI_API_KEY in .env (generation + judge).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pipelines.pipeline1_llm import pipeline1  # noqa: E402
from pipelines.pipeline2_rag import pipeline2  # noqa: E402
from pipelines.pipeline3_graphrag import pipeline3  # noqa: E402
from eval.judge import (llm_judge_with_source, compute_bertscore,  # noqa: E402
                        graded_judge, citation_judge)

ALL_PIPELINES = {"llm_only": pipeline1, "basic_rag": pipeline2, "graphrag": pipeline3}


def run(qa_file: Path, out_file: Path, pipelines: list[str]) -> None:
    qa = json.loads(qa_file.read_text())
    print(f"Loaded {len(qa)} questions from {qa_file.name}; pipelines={pipelines}")

    rows = []
    for item in tqdm(qa, desc="Evaluating"):
        q, gt = item["question"], item["answer"]
        qid, tier = item.get("id", ""), item.get("tier", "")
        for name in pipelines:
            fn = ALL_PIPELINES[name]
            try:
                r = fn(q)
            except Exception as e:
                print(f"  [{name}] ERROR: {e}", flush=True)
                r = {"answer": "", "prompt_tokens": 0, "completion_tokens": 0,
                     "total_tokens": 0, "latency_s": 0}

            judge, jsrc = "FAIL", "error"
            for _ in range(2):
                try:
                    judge, jsrc = llm_judge_with_source(q, gt, r.get("answer", ""))
                    break
                except Exception as e:
                    print(f"  [{name}] judge error: {e}", flush=True)
                    time.sleep(3)
            # Brief-required graded (0-3) + citation/evidence (0-2) scores.
            grade, cit_score = None, None
            try:
                grade = graded_judge(q, gt, r.get("answer", ""))
            except Exception as e:
                print(f"  [{name}] graded-judge error: {e}", flush=True)
            try:
                cit_score = citation_judge(q, r.get("answer", ""), r.get("evidence", ""))
            except Exception as e:
                print(f"  [{name}] citation-judge error: {e}", flush=True)
            try:
                # Brief-mandated config: roberta-large, rescale_with_baseline=True, idf=False.
                bs = compute_bertscore([r.get("answer", "") or ""], [gt])
                bert = bs.get("rescaled_f1")
            except Exception:
                bert = None

            rows.append({
                "qid": qid, "tier": tier, "pipeline": name,
                "judge": judge, "judge_source": jsrc, "graded_score": grade,
                "citation_score": cit_score, "bertscore_f1": bert,
                "prompt_tokens": r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
                "context_tokens": r.get("retrieved_context_tokens", 0),
                "routing_tokens": r.get("routing_tokens", 0),
                "total_tokens": r.get("total_tokens", 0),
                # Actual Gemini API usage (brief: report actual when available;
                # tiktoken totals above are the disclosed cross-pipeline estimate)
                "api_prompt_tokens": r.get("api_prompt_tokens"),
                "api_output_tokens": r.get("api_output_tokens"),
                "api_total_tokens": r.get("api_total_tokens"),
                "latency_s": r.get("latency_s", 0),
                "graph_used": r.get("graph_used", ""),
                "graph_path": r.get("graph_path", ""),
                "sources": "; ".join(str(s) for s in r.get("sources", [])),
                "answer": r.get("answer", ""),
                "ground_truth": gt, "question": q,
                "evidence": r.get("evidence", ""),
            })

    df = pd.DataFrame(rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"\nWrote {out_file}  ({len(df)} rows)")

    # Quick summary
    for name in pipelines:
        sub = df[df.pipeline == name]
        passes = (sub.judge == "PASS").mean() * 100
        toks = sub.total_tokens.mean()
        bert = sub.bertscore_f1.dropna().mean() if sub.bertscore_f1.notna().any() else float("nan")
        grade = sub.graded_score.dropna().mean() if sub.graded_score.notna().any() else float("nan")
        cit = sub.citation_score.dropna().mean() if sub.citation_score.notna().any() else float("nan")
        print(f"  {name:10s} pass={passes:5.1f}%  graded={grade:.2f}/3  cit={cit:.2f}/2  "
              f"avg_total_tokens={toks:7.0f}  bertF1={bert:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--pipelines", nargs="*", default=list(ALL_PIPELINES))
    a = ap.parse_args()
    qa_file = ROOT / ("data/qa/dev_questions.json" if a.dev else "data/qa/eval_questions.json")
    tag = "dev" if a.dev else "eval"
    out = ROOT / f"eval/results/{tag}_run{a.run}.csv"
    run(qa_file, out, a.pipelines)


if __name__ == "__main__":
    main()
