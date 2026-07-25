"""HuggingFace Spaces entrypoint (Gradio SDK).

Why this file exists: the Space SDKs that run a plain Dockerfile are a paid
feature, and the Gradio SDK is free. Gradio is itself built on FastAPI, so
rather than rewriting the service we mount OUR FastAPI app (`api/app.py`) and
hang a small Gradio UI off it. Every REST route the dashboard uses keeps its
exact path:

    GET  /health   GET  /ready    GET  /results    POST /compare   POST /query

...and a human-browsable UI is served at `/ui` so a judge who opens the Space
URL sees something useful instead of a JSON 404.

Spaces runs `python app.py`, so the __main__ block below is the real entrypoint;
it serves the combined ASGI app on the port Spaces expects (7860).

Run locally exactly as Spaces does:
    .venv/bin/python app.py
"""
import os

import gradio as gr

from api.app import app as fastapi_app

SPACE_PORT = int(os.getenv("PORT", "7860"))
_DOCS = "https://github.com/Yash11778/sp100-GraphRAG"


def _results_markdown() -> str:
    """Render the frozen-config results + self-audit cuts straight from the API
    layer, so this UI can never drift from what /results serves."""
    try:
        from api.app import results as _results
        d = _results()
        if not isinstance(d, dict):
            return "Results are not available in this deployment."
    except Exception as e:  # a broken results file must not take the UI down
        return f"Results unavailable: {e}"

    agg = d.get("aggregate", {})
    rows = ["| Pipeline | Strict pass | Graded /3 | Citation /2 | BERTScore F1 | Avg tokens |",
            "|---|---|---|---|---|---|"]
    for key, label in (("llm_only", "LLM-only"), ("basic_rag", "Traditional RAG"),
                       ("graphrag", "**GraphRAG (ours)**")):
        a = agg.get(key)
        if a:
            rows.append(f"| {label} | {a['pass_pct']}% | {a['graded']} | {a['citation']} "
                        f"| {a['bert_f1']} | {a['avg_tokens']:,} |")
    out = [f"### Held-out benchmark — {d.get('runs', '?')} independent runs, frozen config",
           "\n".join(rows),
           f"\n**Token reduction vs Traditional RAG: {d.get('token_reduction_pct')}%**"]

    integ = d.get("integrity") or {}
    if integ.get("cuts"):
        out.append("\n---\n### ⚠ Research integrity — self-disclosed\n")
        out.append(integ.get("disclosure", ""))
        cut_rows = ["\n| Result cut | Qs | Trad. RAG | GraphRAG | Tokens | Ratio |",
                    "|---|---|---|---|---|---|"]
        for c in integ["cuts"]:
            star = " ⭐" if "headline claim" in c["label"] else ""
            cut_rows.append(f"| {c['label']}{star} | {c['n_questions']} | {c['basic_rag_pass']}% "
                            f"| **{c['graphrag_pass']}%** | −{c['token_reduction_pct']}% | {c['ratio']}× |")
        out.append("\n".join(cut_rows))
        cj, bf = integ.get("cross_judge"), integ.get("baseline_fairness")
        if cj:
            out.append(f"\n**Independent judge:** `{cj['judge_model']}` re-scoring answers written "
                       f"by `{cj['generator_model']}` agrees on **{cj['agreement_pct']}%** of verdicts.")
        if bf:
            out.append(f"**Baseline fairness:** Traditional RAG with source labels "
                       f"{bf['rag_as_reported_pass']}% → {bf['rag_with_labels_pass']}% "
                       f"({bf['delta_pts']:+} pts). {bf['note']}")
        out.append(f"\nFull disclosure: `{integ.get('details_doc', 'eval/RESULTS.md')}` in the repo.")
    return "\n\n".join(out)


def compare(question: str, ground_truth: str):
    """Run all three pipelines on one question — the judge-facing demo."""
    from api.app import QueryRequest, compare as _compare
    q = (question or "").strip()
    if not q:
        return "Enter a question first.", "", ""
    try:
        r = _compare(QueryRequest(question=q, ground_truth=(ground_truth or "").strip()))
    except Exception as e:
        return f"Error: {e}", "", ""
    if not isinstance(r, dict):
        return f"Error: {r}", "", ""

    def block(key: str, label: str) -> str:
        p = r.get(key) or {}
        if p.get("status") and p["status"] != "ok":
            return f"### {label}\n\n_Unavailable: {p['status']}_"
        toks = p.get("total_tokens", 0)
        head = f"### {label}\n\n**{toks:,} tokens** · {p.get('latency_s', 0)}s"
        if p.get("judge"):
            head += f" · judge: **{p['judge']}**"
        return f"{head}\n\n{p.get('answer', '(no answer)')}"

    summary = ""
    if r.get("token_reduction_pct") is not None:
        summary = f"## GraphRAG used {r['token_reduction_pct']}% fewer tokens than Traditional RAG\n\n"
    ev = (r.get("graphrag") or {}).get("evidence") or ""
    src = (r.get("graphrag") or {}).get("sources") or []
    eviden = "### Evidence supplied to GraphRAG\n\n"
    if src:
        eviden += "**Citations:** " + ", ".join(f"`{s}`" for s in src[:12]) + "\n\n"
    eviden += f"```\n{ev[:4000]}\n```" if ev else "_none_"

    return (summary + block("llm_only", "1 · LLM-only")
            + "\n\n---\n\n" + block("basic_rag", "2 · Traditional RAG")
            + "\n\n---\n\n" + block("graphrag", "3 · Optimized GraphRAG"),
            eviden, _results_markdown())


with gr.Blocks(title="SP100 GraphRAG — Round 3") as demo:
    gr.Markdown(
        "# Round 3 — Optimized TigerGraph GraphRAG\n"
        "Three-pipeline comparison over 400 SEC filings (100 S&P-100 companies × "
        "10-K, DEF 14A, 2×8-K). **LLM-only** vs **Traditional RAG** vs "
        "**Optimized GraphRAG** on TigerGraph Savanna.\n\n"
        f"Source, full results and self-audit: {_DOCS}"
    )
    with gr.Row():
        q_in = gr.Textbox(label="Question", scale=3,
                          placeholder="e.g. Which KPMG-audited companies are in the Financials sector?")
        gt_in = gr.Textbox(label="Reference answer (optional — enables judging)", scale=2)
    run = gr.Button("Run all three pipelines", variant="primary")
    answers = gr.Markdown()
    with gr.Accordion("Evidence & citations", open=False):
        evidence = gr.Markdown()
    with gr.Accordion("Official held-out benchmark & integrity disclosure", open=True):
        results_md = gr.Markdown(_results_markdown)

    run.click(compare, [q_in, gt_in], [answers, evidence, results_md])
    gr.Markdown(
        "### REST API\n"
        "`GET /health` · `GET /ready` · `GET /results` · `POST /compare` · `POST /query`\n\n"
        "The evaluation dashboard consumes these directly."
    )

# Mount the Gradio UI at /ui ON TOP OF our FastAPI app, so the REST routes keep
# their original paths and the dashboard needs no change.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SPACE_PORT)
