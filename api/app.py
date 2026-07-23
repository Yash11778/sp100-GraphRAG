"""Round 3 three-pipeline comparison API + official-results endpoint.

POST /compare  {question, ground_truth?} -> runs LLM-only / Traditional RAG /
               GraphRAG side by side with live token accounting; when a ground
               truth is given, judges every pipeline (strict + graded + citation)
               and BERTScores the GraphRAG answer.
GET  /results  -> frozen-config held-out results: 3-run aggregate per pipeline,
               tier breakdown, ablations, and per-question rows (run 1).

    .venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pipelines.pipeline1_llm import pipeline1  # noqa: E402
from pipelines.pipeline2_rag import pipeline2  # noqa: E402
from pipelines.pipeline3_graphrag import pipeline3  # noqa: E402
from eval.judge import (llm_judge_with_source, compute_bertscore,  # noqa: E402
                        graded_judge, citation_judge)

app = FastAPI(title="Round 3 GraphRAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_pipeline_executor = ThreadPoolExecutor(max_workers=3)
_eval_executor = ThreadPoolExecutor(max_workers=4)

RESULTS_DIR = ROOT / "eval" / "results"


class QueryRequest(BaseModel):
    question: str
    ground_truth: str = ""


@app.post("/compare")
def compare(req: QueryRequest):
    """Run all three pipelines in parallel; judge + BERTScore when GT provided."""
    futures = {
        "llm_only":  _pipeline_executor.submit(pipeline1, req.question),
        "basic_rag": _pipeline_executor.submit(pipeline2, req.question),
        "graphrag":  _pipeline_executor.submit(pipeline3, req.question),
    }
    results = {}
    for name, fut in futures.items():
        try:
            results[name] = fut.result(timeout=120)
        except Exception as e:
            return JSONResponse(status_code=500,
                                content={"detail": f"{name} pipeline failed: {e}"})

    p1, p2, p3 = results["llm_only"], results["basic_rag"], results["graphrag"]
    basic_rag_ok = p2.get("status") != "index_unavailable"
    token_reduction = round((1 - p3["total_tokens"] / max(p2["total_tokens"], 1)) * 100, 1) \
        if basic_rag_ok else None
    cost_reduction = round((1 - p3["cost_usd"] / max(p2["cost_usd"], 1e-9)) * 100, 1) \
        if basic_rag_ok else None

    out = {
        "llm_only": p1, "basic_rag": p2, "graphrag": p3,
        "graphrag_status": "ok" if p3.get("graph_used") else "vector_only",
        "token_reduction_pct": token_reduction,
        "cost_reduction_pct": cost_reduction,
    }

    if req.ground_truth:
        ev = {
            "judge_llm_only":  _eval_executor.submit(llm_judge_with_source, req.question, req.ground_truth, p1["answer"]),
            "judge_basic_rag": _eval_executor.submit(llm_judge_with_source, req.question, req.ground_truth, p2["answer"]),
            "judge_graphrag":  _eval_executor.submit(llm_judge_with_source, req.question, req.ground_truth, p3["answer"]),
            "graded_graphrag": _eval_executor.submit(graded_judge, req.question, req.ground_truth, p3["answer"]),
            "citation_graphrag": _eval_executor.submit(citation_judge, req.question, p3["answer"], p3.get("evidence", "")),
            "bertscore":       _eval_executor.submit(compute_bertscore, [p3["answer"]], [req.ground_truth]),
        }
        for key, fut in ev.items():
            try:
                value = fut.result(timeout=90)
                if key.startswith("judge_"):
                    out[key], out[f"{key}_source"] = value
                else:
                    out[key] = value
            except Exception:
                # Honest failure — no silent PASS.
                if key.startswith("judge_"):
                    out[key], out[f"{key}_source"] = "ERROR", "error"
                else:
                    out[key] = "ERROR"
    return out


def _run_files():
    return [RESULTS_DIR / f"eval_run{i}.csv" for i in (1, 2, 3)]


@app.get("/results")
def results():
    """Frozen-config held-out results: aggregates, tiers, ablations, per-question."""
    import pandas as pd
    runs = [pd.read_csv(p) for p in _run_files() if p.exists()]
    if not runs:
        return JSONResponse(status_code=404, content={"detail": "no result files"})
    df = pd.concat(runs)

    agg = {}
    for name, sub in df.groupby("pipeline"):
        agg[name] = {
            "pass_pct": round((sub.judge == "PASS").mean() * 100, 1),
            "graded": round(sub.graded_score.dropna().mean(), 2),
            "citation": round(sub.citation_score.dropna().mean(), 2),
            "bert_f1": round(sub.bertscore_f1.dropna().mean(), 3),
            "avg_tokens": round(sub.total_tokens.mean()),
            "avg_latency_s": round(sub.latency_s.mean(), 2),
        }
    rag, gr = agg.get("basic_rag", {}), agg.get("graphrag", {})
    reduction = round((1 - gr.get("avg_tokens", 0) / max(rag.get("avg_tokens", 1), 1)) * 100, 1)

    tiers = {}
    for (tier, name), sub in df.groupby(["tier", "pipeline"]):
        tiers.setdefault(tier, {})[name] = round((sub.judge == "PASS").mean() * 100, 1)

    per_run = [{
        "run": i + 1,
        "graphrag_pass": round((r[r.pipeline == "graphrag"].judge == "PASS").mean() * 100, 1),
        "basic_rag_pass": round((r[r.pipeline == "basic_rag"].judge == "PASS").mean() * 100, 1),
        "llm_only_pass": round((r[r.pipeline == "llm_only"].judge == "PASS").mean() * 100, 1),
    } for i, r in enumerate(runs)]

    ablations = {}
    for tag, fn in (("no_graph", "eval_run81.csv"), ("no_scope", "eval_run82.csv"),
                    ("no_opt", "eval_run83.csv")):
        p = RESULTS_DIR / fn
        if p.exists():
            a = pd.read_csv(p)
            a = a[a.pipeline == "graphrag"]
            ablations[tag] = {
                "pass_pct": round((a.judge == "PASS").mean() * 100, 1),
                "avg_tokens": round(a.total_tokens.mean()),
            }

    r1 = runs[0]
    per_question = [{
        "qid": r.qid, "tier": r.tier, "pipeline": r.pipeline, "judge": r.judge,
        "graded": None if pd.isna(r.graded_score) else int(r.graded_score),
        "citation": None if pd.isna(r.citation_score) else int(r.citation_score),
        "bert_f1": None if pd.isna(r.bertscore_f1) else round(r.bertscore_f1, 3),
        "total_tokens": int(r.total_tokens),
        "graph_path": None if pd.isna(r.graph_path) else r.graph_path,
        "question": r.question,
    } for _, r in r1.iterrows()]

    return {"aggregate": agg, "token_reduction_pct": reduction, "tiers": tiers,
            "per_run": per_run, "ablations": ablations, "runs": len(runs),
            "per_question": per_question}


@app.get("/")
def root():
    return {"status": "ok", "message": "Round 3 GraphRAG API — POST /compare, GET /results"}


@app.get("/health")
def health():
    return {"status": "ok"}
