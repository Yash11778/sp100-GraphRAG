"""Pipeline 2 — Traditional RAG baseline (good-faith).

Same embedding model as GraphRAG (all-MiniLM-L6-v2), sensible top-k over a FAISS
index of the whole SP100 corpus. No graph, no reranking — the honest baseline the
brief requires (no deliberate over/under-retrieval).

Returns full token accounting via make_result + retrieved chunk ids for citations.
"""
import os
import pickle
import time
from pathlib import Path

from pipelines.utils import count_tokens, gemini_generate, make_result, setup_gemini

_ROOT = Path(__file__).parent.parent.resolve()
_FAISS = _ROOT / "data/chunks/rag_index.faiss"
_CHUNKS = _ROOT / "data/chunks/chunks.pkl"
_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embedder = _index = _chunks = _client = None
_load_error = ""


def _load():
    global _embedder, _index, _chunks, _client, _load_error
    if _index is not None:
        return
    if not _FAISS.exists() or not _CHUNKS.exists():
        _load_error = "RAG index not built. Run ingestion/build_chunks.py first."
        return
    try:
        import faiss
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(_EMBED_MODEL)
        _index = faiss.read_index(str(_FAISS))
        with open(_CHUNKS, "rb") as f:
            _chunks = pickle.load(f)
        _client = setup_gemini()
        _load_error = ""
    except Exception as e:
        _load_error = str(e)


def _embed(text: str):
    import numpy as np
    emb = list(_embedder.embed([text]))[0]
    emb = np.array(emb, dtype=np.float32).reshape(1, -1)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)


def pipeline2(question: str, top_k: int = 8) -> dict:
    _load()
    if _load_error:
        res = make_result("basic_rag", f"Basic RAG unavailable: {_load_error}", 0, 0, 0.0)
        res["sources"] = []
        res["status"] = "index_unavailable"
        return res

    emb = _embed(question)
    _, idxs = _index.search(emb, top_k)
    retrieved = [_chunks[i] for i in idxs[0] if 0 <= i < len(_chunks)]
    context = "\n\n---\n\n".join(c["text"] for c in retrieved)
    sources = [c["chunk_id"] for c in retrieved]

    prompt = (
        "You are an expert financial-filings assistant. Use ONLY the context below to "
        "answer the question. Be precise with names, numbers, dates. If the context lists "
        "multiple entities, list all that apply. Do not say 'the context does not state' "
        "if the answer is present.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    start = time.time()
    answer = gemini_generate(_client, prompt, max_tokens=512)
    latency = round(time.time() - start, 3)

    res = make_result("basic_rag", answer, count_tokens(prompt), count_tokens(answer), latency)
    from pipelines.utils import LAST_USAGE
    res.update(LAST_USAGE)   # actual Gemini API usage (brief: report when available)
    res["sources"] = sources
    res["retrieved_context_tokens"] = count_tokens(context)
    res["evidence"] = context
    res["top_k"] = top_k
    return res
