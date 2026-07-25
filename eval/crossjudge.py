"""Independent-judge cross-check — re-score stored answers with a DIFFERENT model.

The brief asks that the judge "use a different model from the generator where
practical". Our reported runs were judged by `gemini-2.5-flash`, the same model
that generated the answers. This script re-judges the SAME stored answers with a
different, stronger model (default `gemini-2.5-pro`) using the IDENTICAL strict
rubric from eval/judge.py, and reports how far the two judges agree.

It re-scores stored text only — it does not touch the pipelines, retrieval, or the
frozen v3 configuration, so every reported number stays traceable. Output is a
separate CSV; nothing existing is overwritten.

    .venv/bin/python eval/crossjudge.py                    # run 1, gemini-2.5-pro
    .venv/bin/python eval/crossjudge.py --run 2 --model gemini-3-pro-preview
    .venv/bin/python eval/crossjudge.py --pipelines graphrag basic_rag

Resumable: re-running reuses verdicts already written to the output CSV, so a
rate-limit interruption costs nothing.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from eval.judge import STRICT_JUDGE_PROMPT  # noqa: E402  (same rubric, verbatim)


def judge_with(model: str, prompt: str) -> str:
    from google import genai
    from google.genai import types
    global _client
    try:
        _client
    except NameError:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # gemini-2.5-pro is a reasoning model: thinking tokens are drawn from the
    # output budget, so a tight max_output_tokens (we started at 8) leaves NO
    # room for the verdict -- response.text comes back None and every row
    # silently scores FAIL. Give the budget headroom and cap thinking instead.
    cfg = dict(temperature=0.0, max_output_tokens=2048,
               thinking_config=types.ThinkingConfig(thinking_budget=128))
    for attempt in range(6):
        try:
            r = _client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(**cfg))
            txt = ""
            try:
                txt = (r.text or "").strip()
            except Exception:  # thinking-token responses raise instead of returning None
                parts = r.candidates[0].content.parts
                txt = " ".join(p.text for p in parts
                               if getattr(p, "text", None)
                               and not getattr(p, "thought", False)).strip()
            return txt.upper()
        except Exception as e:
            s = str(e)
            if any(k in s for k in ("429", "quota", "RESOURCE_EXHAUSTED",
                                    "503", "UNAVAILABLE", "500")) and attempt < 5:
                wait = min(60, 5 * (2 ** attempt))
                print(f"  [retry {attempt+1}/6 in {wait}s] {s[:90]}", flush=True)
                time.sleep(wait)
            else:
                raise
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--pipelines", nargs="*",
                    default=["llm_only", "basic_rag", "graphrag"])
    a = ap.parse_args()

    src = ROOT / f"eval/results/eval_run{a.run}.csv"
    out = ROOT / f"eval/results/crossjudge_run{a.run}_{a.model.replace('.', '-')}.csv"
    df = pd.read_csv(src)
    df = df[df.pipeline.isin(a.pipelines)].copy()

    done = {}
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.qid, r.pipeline): r.judge_cross for r in prev.itertuples()}
        print(f"Resuming: {len(done)} verdicts already stored in {out.name}")

    rows = []
    for r in tqdm(list(df.itertuples()), desc=f"cross-judging with {a.model}"):
        key = (r.qid, r.pipeline)
        if key in done:
            verdict = done[key]
        else:
            prompt = STRICT_JUDGE_PROMPT.format(
                q=r.question, correct=r.ground_truth, answer=str(r.answer or ""))
            raw = judge_with(a.model, prompt)
            verdict = "PASS" if "PASS" in raw else "FAIL"
        rows.append({"qid": r.qid, "tier": r.tier, "pipeline": r.pipeline,
                     "judge_primary": r.judge, "judge_cross": verdict,
                     "agree": r.judge == verdict})
        pd.DataFrame(rows).to_csv(out, index=False)   # checkpoint every row

    res = pd.DataFrame(rows)
    print(f"\nWrote {out}  ({len(res)} rows)\n")
    print(f"{'pipeline':11s} {'primary(2.5-flash)':>19s} {'cross('+a.model+')':>26s} {'agree':>7s}")
    for p in a.pipelines:
        s = res[res.pipeline == p]
        if s.empty:
            continue
        print(f"{p:11s} {100*s.judge_primary.eq('PASS').mean():18.1f}% "
              f"{100*s.judge_cross.eq('PASS').mean():25.1f}% "
              f"{100*s.agree.mean():6.1f}%")
    print(f"\nOverall judge agreement: {100*res.agree.mean():.1f}% "
          f"({int(res.agree.sum())}/{len(res)} verdicts identical)")


if __name__ == "__main__":
    main()
