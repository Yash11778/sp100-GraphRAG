"""HuggingFace Spaces entrypoint (Gradio SDK).

Why this file exists: the Space SDKs that run a plain Dockerfile are a paid
feature, and the Gradio SDK is free. Gradio is itself built on FastAPI, so
rather than rewriting the service we serve our existing FastAPI app
(`api/app.py`) from Gradio's own server. Every REST route the dashboard uses
keeps its exact path:

    GET /health    GET /results    POST /compare

(The dashboard's vite proxy also lists /ready, /query and /debug, but api/app.py
never defined them -- they are stale entries and nothing calls them.)

The Gradio UI is served at `/` so a judge who opens the Space URL sees something
useful instead of a JSON 404. Gradio owns `/`, so our root route is skipped
rather than fighting it.

HOW IT BINDS THE PORT -- this bit is load-bearing. An earlier version mounted
Gradio inside our FastAPI app and ran `uvicorn.run(...)` directly. On Spaces that
crashed with:

    ERROR: [Errno 98] error while attempting to bind on ('0.0.0.0', 7860):
           address already in use

because the Spaces/ZeroGPU runtime is built around Gradio's own `launch()` and
already holds 7860 -- our second bind had nowhere to go. So we now let Gradio
launch (the platform-supported path) and graft our API routes onto the server it
creates, instead of standing up a competing one. `prevent_thread_lock=True`
returns control so the routes can be attached, then `block_thread()` keeps the
process alive the way `launch()` normally would.

Run locally exactly as Spaces does:
    .venv/bin/python app.py
"""
import os

import gradio as gr

from api.app import app as fastapi_app

SPACE_PORT = int(os.getenv("PORT", "7860"))
_DOCS = "https://github.com/Yash11778/sp100-GraphRAG"

# ZeroGPU hardware refuses to start a Space with "No @spaces.GPU function detected
# during startup". This service is CPU-only -- it does graph traversal, vector
# search and Gemini API calls, none of which touch a GPU -- so there is no honest
# GPU function to offer. We register one no-op purely to satisfy that startup
# check; it is never called and allocates nothing. On CPU-basic hardware the
# `spaces` package is absent and this block is simply skipped.
#
# If the Space is switched to CPU basic in Settings -> Hardware (which is what
# this workload actually wants), this stub becomes dead code and can be removed.
try:
    import spaces  # noqa: E402

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():  # pragma: no cover - never invoked
        """Present only so ZeroGPU's startup detector finds a decorated function."""
        return None
except Exception:  # not on ZeroGPU, or the package is unavailable
    pass


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
        "`GET /health` · `GET /results` · `POST /compare`\n\n"
        "The evaluation dashboard consumes these directly."
    )

# Kept so the module can still be served by a plain ASGI runner (e.g. a local
# `uvicorn app:app`), where nothing else owns the port. Spaces does NOT use this
# object -- see the __main__ block for how the port is actually bound there.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")


def _attach_api_routes(server_app) -> int:
    """Graft our REST routes onto the FastAPI instance Gradio built.

    Inserted at the FRONT of the route list so they resolve before any Gradio
    catch-all, which keeps /health, /results and /compare on their original
    paths -- the dashboard needs no change and judges' curl commands still work.
    """
    from fastapi.routing import APIRoute
    ours = [r for r in fastapi_app.routes if isinstance(r, APIRoute)]
    existing = {(r.path, tuple(sorted(r.methods or ()))) for r in server_app.routes
                if isinstance(r, APIRoute)}
    new = [r for r in ours if (r.path, tuple(sorted(r.methods or ()))) not in existing]
    server_app.router.routes[:0] = new
    return len(new)


if __name__ == "__main__":
    # Gradio owns the port on Spaces; launching any other server here is what
    # produced the "address already in use" crash. prevent_thread_lock returns
    # the FastAPI instance so we can attach our routes to THAT server.
    # ssr_mode=False is required, not cosmetic. Gradio 6 defaults to SSR, which
    # needs a Node front proxy; the Spaces image has no Node, so Gradio logged
    # "Cannot start Node server on any port in the range 7860-7860" and moved the
    # Python server to 7861 -- the port Spaces does NOT route to. Disabling SSR
    # keeps everything on 7860 in pure Python.
    # NB: no show_api= argument -- Gradio 6 removed it.
    server_app, _, _ = demo.launch(
        server_name="0.0.0.0", server_port=SPACE_PORT,
        prevent_thread_lock=True, ssr_mode=False,
    )
    n = _attach_api_routes(server_app)
    print(f"[startup] attached {n} REST routes to the Gradio server "
          f"on port {SPACE_PORT}", flush=True)
    demo.block_thread()
