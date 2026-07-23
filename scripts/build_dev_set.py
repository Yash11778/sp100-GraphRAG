"""Generate a development / tuning question set, DISJOINT from the held-out 50 EQs.

Answers are derived from our own validated deterministic extraction (auditor,
events, sector, mentions), so the dev set is self-consistent and correct without
touching the held-out questions. Used only to tune retrieval/optimization params;
frozen configs are then evaluated on eval_questions.json.

    python scripts/build_dev_set.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARSED = ROOT / "data" / "parsed"
GCSV = ROOT / "data" / "graph_csv"
OUT = ROOT / "data" / "qa" / "dev_questions.json"

NAMES = {}  # ticker -> company name


def main() -> None:
    index = json.loads((PARSED / "_index.json").read_text())
    for r in index:
        NAMES[r["ticker"]] = r["company"]
    sector = {r["ticker"]: r["sector"] for r in index}
    auditor = json.loads((GCSV / "company_auditor.json").read_text())
    events = json.loads((GCSV / "events.json").read_text())
    mentions = json.loads((GCSV / "mentions.json").read_text())

    # Reverse indexes
    by_auditor = defaultdict(list)
    for t, f in auditor.items():
        by_auditor[f].append(t)
    by_sector = defaultdict(list)
    for t, s in sector.items():
        by_sector[s].append(t)
    dividends = {e["ticker"]: e["value"] for e in events if e["event_type"] == "dividend"}
    buybacks = {e["ticker"]: e["value"] for e in events if e["event_type"] == "buyback"}

    dev = []

    def add(qid, tier, hop, q, a):
        dev.append({"id": qid, "tier": tier, "hop": hop, "question": q, "answer": a, "dev": True})

    # --- Tier A shape: single-fact auditor lookups (companies not focal in EQs) ---
    for i, t in enumerate(["GS", "BA", "DIS", "MMM", "GM", "TXN"], 1):
        if t in auditor:
            add(f"DV-A{i:02d}", "A", 1, f"Who is {NAMES[t]}'s independent registered public accounting firm?", auditor[t])

    # --- Tier A shape: single-fact dividend lookups (not in EQ34's named set) ---
    for i, t in enumerate(["BNY", "USB", "INTU", "MS"], 1):
        if t in dividends:
            add(f"DV-A1{i}", "A", 1, f"What quarterly cash dividend per share did {NAMES[t]} declare in its 2026 8-K?", dividends[t])

    # --- Tier B shape: 2-hop bridge (event -> auditor) ---
    for i, t in enumerate(["FDX", "ISRG", "PYPL"], 1):
        if t in buybacks and t in auditor:
            add(f"DV-B{i:02d}", "B", 2,
                f"The company that authorized a {buybacks[t]} share-repurchase program (a 2026 8-K) — who is its auditor?",
                f"{NAMES[t]}; {auditor[t]}")

    # --- Tier C shape: aggregation over a typed edge ---
    deloitte = sorted(by_auditor.get("Deloitte & Touche LLP", []))
    add("DV-C01", "C", 3, "List all dataset companies audited by Deloitte & Touche.", ", ".join(deloitte))
    add("DV-C02", "C", 3, "Which companies authorized a share-repurchase program via an 8-K in the dataset?",
        ", ".join(sorted(buybacks)))
    energy = sorted(by_sector.get("Energy", []))
    add("DV-C03", "C", 3, "Which companies are in the Energy sector?", ", ".join(energy))

    # --- Tier C/D shape: intersection (auditor x sector) ---
    ey = set(by_auditor.get("Ernst & Young LLP", []))
    it = set(by_sector.get("Information Technology", []))
    add("DV-C04", "C", 3, "Which Ernst & Young-audited companies are in the Information Technology sector?",
        ", ".join(sorted(ey & it)))
    deloitte_ind = set(deloitte) & set(by_sector.get("Industrials", []))
    add("DV-C05", "C", 3, "Which Deloitte-audited companies are in the Industrials sector?",
        ", ".join(sorted(deloitte_ind)))

    OUT.write_text(json.dumps(dev, indent=2), encoding="utf-8")
    print(f"Wrote {len(dev)} dev questions -> {OUT}")
    for d in dev:
        print(f"  {d['id']} [{d['tier']}] {d['question'][:70]}  -> {d['answer'][:60]}")


if __name__ == "__main__":
    main()
