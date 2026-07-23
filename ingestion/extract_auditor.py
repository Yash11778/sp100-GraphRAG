"""Deterministic auditor extraction from DEF 14A filings.

Every S&P-100 proxy statement ratifies its "independent registered public
accounting firm" and repeats the firm's name many times in the audit-fee
section. We score each Big-Four surface form by how often it co-occurs with the
auditor phrase and pick the winner -> one canonical AuditFirm per company.

No API. Validated against EQ25 (11 KPMG companies) and EQ44 (E&Y 35 / PwC 29 /
Deloitte 25 / KPMG 11).

    python ingestion/extract_auditor.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import PARSED_DIR, AUDIT_FIRMS, AUDIT_FIRM_CANONICAL  # noqa: E402

AUDITOR_CUE = re.compile(
    r"independent\s+(registered\s+public\s+accounting|auditing|auditor)", re.I
)


def score_firm(text: str, firm: str) -> int:
    """Count firm mentions that sit within ~200 chars of an auditor cue phrase,
    plus a small weight for total mentions. Co-location with the cue is what
    separates the real auditor from an incidental name-drop."""
    pattern = re.compile(re.escape(firm), re.I)
    total = 0
    near = 0
    for m in pattern.finditer(text):
        total += 1
        window = text[max(0, m.start() - 200): m.end() + 200]
        if AUDITOR_CUE.search(window):
            near += 1
    return near * 100 + total  # cue-proximity dominates; total breaks ties


def extract_auditor(text: str) -> str | None:
    scores = {}
    for firm in AUDIT_FIRMS:
        s = score_firm(text, firm)
        if s:
            canon = AUDIT_FIRM_CANONICAL[firm]
            # Keep the max across surface forms mapping to the same canonical firm.
            scores[canon] = max(scores.get(canon, 0), s)
    if not scores:
        return None
    return max(scores, key=scores.get)


def main() -> None:
    index = json.loads((PARSED_DIR / "_index.json").read_text())
    def14a = [r for r in index if r["form"] == "DEF14A"]

    results = {}   # ticker -> canonical auditor
    unresolved = []
    for r in def14a:
        rec = json.loads((PARSED_DIR / f"{r['doc_id']}.json").read_text())
        firm = extract_auditor(rec["text"])
        if firm:
            results[r["ticker"]] = firm
        else:
            unresolved.append(r["ticker"])

    # Distribution — compare to EQ44: E&Y 35 / PwC 29 / Deloitte 25 / KPMG 11.
    dist = Counter(results.values())
    print(f"Resolved auditor for {len(results)}/{len(def14a)} companies "
          f"({len(unresolved)} unresolved: {unresolved})\n")
    print("Auditor distribution (compare EQ44 -> E&Y 35 / PwC 29 / Deloitte 25 / KPMG 11):")
    for firm, n in dist.most_common():
        print(f"  {n:3d}  {firm}")

    kpmg = sorted(t for t, f in results.items() if f == "KPMG LLP")
    print(f"\nKPMG companies ({len(kpmg)}) -> {kpmg}")
    print("EQ25 expected  (11) -> ['ACN', 'ADBE', 'BNY', 'COST', 'EMR', 'GD', "
          "'HD', 'PEP', 'PFE', 'V', 'WFC']")

    out = PARSED_DIR.parent / "graph_csv" / "company_auditor.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
