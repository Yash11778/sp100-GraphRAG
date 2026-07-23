"""Load the deterministic core into the sp100 graph via REST++ upsert.

Loads (no API needed):
  Company, Sector, AuditFirm, Document, Event vertices
  IN_SECTOR, AUDITED_BY, FILED, REPORTS_EVENT edges

Board / competitor / foundry / topic edges + Chunk vertices are loaded by later
steps. Uses the standard REST++ upsert payload so the whole load is one call.

    python scripts/load_core.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tg  # noqa: E402

ROOT = Path(__file__).parent.parent
PARSED = ROOT / "data" / "parsed"
GCSV = ROOT / "data" / "graph_csv"


def v(value):
    return {"value": value}


def main() -> None:
    index = json.loads((PARSED / "_index.json").read_text())
    auditor = json.loads((GCSV / "company_auditor.json").read_text())
    events = json.loads((GCSV / "events.json").read_text())

    companies = {}   # ticker -> {name, sector}
    for r in index:
        companies.setdefault(r["ticker"], {"name": r["company"], "sector": r["sector"]})

    vertices: dict = {"Company": {}, "Sector": {}, "AuditFirm": {}, "Document": {}, "Event": {}}
    edges: dict = {"Company": {}, "Document": {}}

    def add_company_edge(ticker, etype, tgt_type, tgt_id, attrs=None):
        edges["Company"].setdefault(ticker, {}).setdefault(etype, {}).setdefault(tgt_type, {})[tgt_id] = attrs or {}

    def add_doc_edge(doc_id, etype, tgt_type, tgt_id, attrs=None):
        edges["Document"].setdefault(doc_id, {}).setdefault(etype, {}).setdefault(tgt_type, {})[tgt_id] = attrs or {}

    # Company + Sector + IN_SECTOR
    for t, c in companies.items():
        vertices["Company"][t] = {"name": v(c["name"]), "sector": v(c["sector"])}
        vertices["Sector"][c["sector"]] = {}
        add_company_edge(t, "IN_SECTOR", "Sector", c["sector"])

    # AuditFirm + AUDITED_BY
    for t, firm in auditor.items():
        vertices["AuditFirm"][firm] = {}
        add_company_edge(t, "AUDITED_BY", "AuditFirm", firm)

    # Document + FILED
    for r in index:
        vertices["Document"][r["doc_id"]] = {
            "ticker": v(r["ticker"]), "form": v(r["form"]),
            "filing_date": v(r["filing_date"]), "accession": v(r["accession"]),
            "source_url": v(r["source_url"]),
        }
        add_company_edge(r["ticker"], "FILED", "Document", r["doc_id"])

    # Event + REPORTS_EVENT
    for e in events:
        vertices["Event"][e["event_id"]] = {
            "event_type": v(e["event_type"]), "value": v(e["value"]),
            "event_date": v(e["filing_date"]), "summary": v(e["summary"][:512]),
        }
        add_doc_edge(e["doc_id"], "REPORTS_EVENT", "Event", e["event_id"])

    payload = {"vertices": vertices, "edges": edges}
    counts = {k: len(x) for k, x in vertices.items()}
    print("Upserting vertices:", counts)

    r = tg.restpp("/restpp/graph/sp100", method="POST", graph="sp100",
                  data=json.dumps(payload),
                  headers={"Content-Type": "application/json"}, timeout=180)
    print("Upsert status:", r.status_code)
    print(r.text[:400])

    # Verify counts live from the graph.
    for vt in ["Company", "Sector", "AuditFirm", "Document", "Event"]:
        resp = tg.restpp(f"/restpp/graph/sp100/vertices/{vt}?count_only=true", graph="sp100")
        try:
            n = resp.json()["results"][0]["count"]
        except Exception:
            n = resp.text[:80]
        print(f"  live {vt}: {n}")


if __name__ == "__main__":
    main()
