"""Batch-load all Chunk vertices (with 384-d embeddings) + edges into sp100.

  Chunk vertex: chunk_id, doc_id, ticker, form, seq, text, emb(vector)
  HAS_CHUNK  (Document -> Chunk)
  CHUNK_OF   (Chunk -> Company)

Vectors go in as {"emb": {"value": [floats]}} (confirmed searchable via the
installed `vsearch` query). Batched so each REST++ upsert stays small.

    python scripts/load_chunks.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tg  # noqa: E402

ROOT = Path(__file__).parent.parent
BATCH = 1500


def main() -> None:
    chunks = pickle.load(open(ROOT / "data/chunks/chunks.pkl", "rb"))
    vecs = np.load(ROOT / "data/chunks/embeddings.npy")
    n = len(chunks)
    assert vecs.shape[0] == n, (vecs.shape, n)
    print(f"Loading {n} chunks in batches of {BATCH} ...", flush=True)

    total_v = total_e = 0
    t0 = time.time()
    for b0 in range(0, n, BATCH):
        b1 = min(b0 + BATCH, n)
        V = {"Chunk": {}}
        E = {"Document": {}, "Chunk": {}}
        for i in range(b0, b1):
            c = chunks[i]
            cid = c["chunk_id"]
            V["Chunk"][cid] = {
                "doc_id": {"value": c["doc_id"]},
                "ticker": {"value": c["ticker"]},
                "form": {"value": c["form"]},
                "seq": {"value": int(c["seq"])},
                "text": {"value": c["text"]},
                "emb": {"value": vecs[i].tolist()},
            }
            E["Document"].setdefault(c["doc_id"], {}).setdefault("HAS_CHUNK", {}).setdefault("Chunk", {})[cid] = {}
            E["Chunk"].setdefault(cid, {}).setdefault("CHUNK_OF", {}).setdefault("Company", {})[c["ticker"]] = {}

        r = tg.restpp("/restpp/graph/sp100", method="POST", graph="sp100",
                      data=json.dumps({"vertices": V, "edges": E}),
                      headers={"Content-Type": "application/json"}, timeout=300)
        j = r.json()["results"][0]
        total_v += j.get("accepted_vertices", 0)
        total_e += j.get("accepted_edges", 0)
        elapsed = time.time() - t0
        print(f"  {b1}/{n}  (+{j.get('accepted_vertices',0)}v {j.get('accepted_edges',0)}e)  "
              f"{elapsed:.0f}s", flush=True)

    print(f"\nDone. accepted_vertices={total_v} accepted_edges={total_e} in {time.time()-t0:.0f}s")
    resp = tg.restpp("/restpp/graph/sp100/vertices/Chunk?count_only=true", graph="sp100")
    print("live Chunk count:", resp.json()["results"][0]["count"])


if __name__ == "__main__":
    main()
