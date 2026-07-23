"""Upsert the keyword-derived Entity/Topic vertices + relationship edges into sp100.

  Entity "TSMC"   (kind=foundry)     <- NAMES_FOUNDRY   from Company
  Entity "NVIDIA" (kind=competitor)  <- NAMES_COMPETITOR from Company
  Topic  "climate change", "GLP-1"   <- MENTIONS_TOPIC   from Company (form=10-K)

    python scripts/load_mentions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tg  # noqa: E402

GCSV = Path(__file__).parent.parent / "data" / "graph_csv"


def v(x):
    return {"value": x}


def main() -> None:
    m = json.loads((GCSV / "mentions.json").read_text())

    vertices = {"Entity": {}, "Topic": {}}
    edges = {"Company": {}}

    def cedge(ticker, etype, tgt_type, tgt_id, attrs=None):
        edges["Company"].setdefault(ticker, {}).setdefault(etype, {}).setdefault(tgt_type, {})[tgt_id] = attrs or {}

    # Foundry
    f = m["NAMES_FOUNDRY"]
    vertices["Entity"][f["entity"]] = {"kind": v(f["kind"])}
    for t in f["tickers"]:
        cedge(t, "NAMES_FOUNDRY", "Entity", f["entity"])

    # Competitor
    c = m["NAMES_COMPETITOR"]
    vertices["Entity"][c["entity"]] = {"kind": v(c["kind"])}
    for t in c["tickers"]:
        cedge(t, "NAMES_COMPETITOR", "Entity", c["entity"])

    # Topics
    for topic, tickers in m["MENTIONS_TOPIC"].items():
        vertices["Topic"][topic] = {}
        for t in tickers:
            cedge(t, "MENTIONS_TOPIC", "Topic", topic, {"form": {"value": "10-K"}})

    payload = {"vertices": vertices, "edges": edges}
    r = tg.restpp("/restpp/graph/sp100", method="POST", graph="sp100",
                  data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=120)
    print("Upsert status:", r.status_code)
    print(r.text[:300])

    for vt in ["Entity", "Topic"]:
        resp = tg.restpp(f"/restpp/graph/sp100/vertices/{vt}?count_only=true", graph="sp100")
        try:
            print(f"  live {vt}: {resp.json()['results'][0]['count']}")
        except Exception:
            print(f"  live {vt}: {resp.text[:80]}")


if __name__ == "__main__":
    main()
