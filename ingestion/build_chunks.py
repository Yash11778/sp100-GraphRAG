"""Chunk every parsed filing, embed locally (all-MiniLM-L6-v2), build the FAISS
index for Basic RAG, and persist chunks + embeddings for the Savanna load.

Shared embedding model across RAG + GraphRAG (brief requirement). No API.

Chunk size ~256 tokens (MiniLM's max window) with 40-token overlap so embeddings
are faithful (larger chunks would be silently truncated by the model).

    python ingestion/build_chunks.py            # full build
    python ingestion/build_chunks.py --count    # just size the job, no embedding
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import PARSED_DIR, CHUNKS_DIR  # noqa: E402
import json  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 256
OVERLAP = 40
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_doc(rec: dict) -> list[dict]:
    toks = ENC.encode(rec["text"])
    step = CHUNK_TOKENS - OVERLAP
    out = []
    for seq, start in enumerate(range(0, len(toks), step)):
        piece = toks[start:start + CHUNK_TOKENS]
        if not piece:
            break
        text = ENC.decode(piece).strip()
        if len(text) < 20:
            continue
        out.append({
            "chunk_id": f"{rec['doc_id']}::c{seq}",
            "doc_id": rec["doc_id"],
            "ticker": rec["ticker"],
            "form": rec["form"],
            "seq": seq,
            "text": text,
            # 'source' used by the Basic RAG prompt/citation.
            "source": f"{rec['ticker']} {rec['form']} {rec['filing_date']}",
        })
        if start + CHUNK_TOKENS >= len(toks):
            break
    return out


def build_all_chunks() -> list[dict]:
    index = json.loads((PARSED_DIR / "_index.json").read_text())
    chunks = []
    for r in index:
        rec = json.loads((PARSED_DIR / f"{r['doc_id']}.json").read_text())
        chunks.extend(chunk_doc(rec))
    return chunks


def main(count_only: bool = False) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_all_chunks()
    by_form = {}
    for c in chunks:
        by_form[c["form"]] = by_form.get(c["form"], 0) + 1
    print(f"Total chunks: {len(chunks)}  by form: {by_form}")
    if count_only:
        return

    print(f"Embedding with {EMBED_MODEL} (CPU) ...")
    from fastembed import TextEmbedding
    embedder = TextEmbedding(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    vecs = np.array(list(embedder.embed(texts, batch_size=256)), dtype=np.float32)
    # L2-normalise so inner-product == cosine.
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
    print("Embeddings shape:", vecs.shape)

    import faiss
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, str(CHUNKS_DIR / "rag_index.faiss"))
    with open(CHUNKS_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(CHUNKS_DIR / "embeddings.npy", vecs)
    print(f"Wrote {CHUNKS_DIR}/rag_index.faiss, chunks.pkl, embeddings.npy")


if __name__ == "__main__":
    main(count_only="--count" in sys.argv)
