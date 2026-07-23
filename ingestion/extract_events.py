"""Deterministic 8-K event extraction: dividends, buybacks, leadership changes.

8-Ks are short and formulaic. We classify each 8-K by cue phrases and pull the
key value (dividend $/share, buyback $bn) with regex. One Event per (doc, type).

No API. Validated against:
  EQ34 dividends -> AIG $0.50, CAT $1.63, COST $1.47, ORCL, PG
  EQ35 buybacks  -> Adobe $25B, Netflix $25B
  EQ50           -> COST $1.47/share

    python ingestion/extract_events.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import PARSED_DIR, GRAPH_CSV_DIR  # noqa: E402

# --- Cue phrases per event type --------------------------------------------
DIVIDEND_CUE = re.compile(r"\bdividend\b", re.I)
BUYBACK_CUE = re.compile(r"repurchase|buyback|repurchase program", re.I)
LEADERSHIP_CUE = re.compile(
    r"chief executive officer|chief financial officer|\bCEO\b|\bCFO\b|"
    r"appoint|resign|retire|succeed|transition from (his|her|their) role|"
    r"executive chair|chief legal officer",
    re.I,
)

# --- Value extractors -------------------------------------------------------
# "$1.63 per share", "of ... ($1.47) per share", "$0.50 per share"
DIV_VALUE = re.compile(r"\$(\d+(?:\.\d+)?)\s*(?:\)?\s*)?per share", re.I)
# "$25 billion", "up to $25 billion", "$1.5 billion"
BUYBACK_VALUE = re.compile(r"\$(\d+(?:\.\d+)?)\s*billion", re.I)
# an appointment/transition target role
ROLE = re.compile(
    r"as (?:its |our |the )?((?:Chief [A-Z][a-z]+ Officer)|President and Chief Executive Officer|"
    r"Chief Executive Officer|Chief Financial Officer|Chief Legal Officer|President)",
    re.I,
)


def classify(text: str) -> list[dict]:
    """Return a list of event dicts found in this 8-K (usually 0-1, sometimes 2)."""
    events = []
    low = text

    # Dividend: require the value pattern AND that the "$X per share" match is a real
    # dividend, not the cover-page "par value $0.001 per share" boilerplate. Accept the
    # first match whose left context mentions 'dividend' and does NOT say 'par value'.
    if DIVIDEND_CUE.search(low):
        for m in DIV_VALUE.finditer(low):
            left = low[max(0, m.start() - 60): m.start()]
            if "par value" in left.lower() or "stated value" in left.lower():
                continue
            if not re.search(r"dividend", low[max(0, m.start() - 160): m.end() + 40], re.I):
                continue
            # 'declared': the 8-K reports an actual declaration, vs an announced
            # intention/plan to increase the dividend (different corporate act —
            # the two read differently in the filings and answer differently).
            declared = bool(re.search(
                r"declared a (quarterly )?(cash )?dividend|board.{0,80}declared", low, re.I))
            events.append({"event_type": "dividend", "value": f"${m.group(1)}/share",
                           "declared": declared})
            break

    # Buyback / repurchase program with a $bn authorization.
    if BUYBACK_CUE.search(low):
        m = BUYBACK_VALUE.search(low)
        if m:
            events.append({
                "event_type": "buyback",
                "value": f"${m.group(1)}B",
            })

    # Leadership change (CEO/CFO/CLO appointment or transition).
    if LEADERSHIP_CUE.search(low):
        # Only tag as a leadership *event* when an appointment/transition verb is present,
        # not merely a signature block naming an officer.
        if re.search(r"appoint|resign|retire|succeed|transition|elect", low, re.I):
            role_m = ROLE.search(low)
            events.append({
                "event_type": "leadership",
                "value": (role_m.group(1) if role_m else "leadership change"),
            })

    return events


def snippet(text: str, event_type: str) -> str:
    """A short evidence snippet around the triggering phrase (for the Event.summary)."""
    cue = {"dividend": DIVIDEND_CUE, "buyback": BUYBACK_CUE, "leadership": LEADERSHIP_CUE}[event_type]
    m = cue.search(text)
    if not m:
        return ""
    s = max(0, m.start() - 60)
    return re.sub(r"\s+", " ", text[s:m.end() + 220]).strip()


def main() -> None:
    index = json.loads((PARSED_DIR / "_index.json").read_text())
    eights = [r for r in index if r["form"] == "8-K"]

    all_events = []
    by_type = {}
    for r in eights:
        rec = json.loads((PARSED_DIR / f"{r['doc_id']}.json").read_text())
        for ev in classify(rec["text"]):
            et = ev["event_type"]
            by_type[et] = by_type.get(et, 0) + 1
            all_events.append({
                "event_id": f"{r['doc_id']}::{et}",
                "doc_id": r["doc_id"],
                "ticker": r["ticker"],
                "filing_date": r["filing_date"],
                "event_type": et,
                "value": ev["value"],
                "declared": ev.get("declared", True),
                "summary": snippet(rec["text"], et),
            })

    GRAPH_CSV_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_CSV_DIR / "events.json").write_text(json.dumps(all_events, indent=2), encoding="utf-8")

    print(f"Extracted {len(all_events)} events from {len(eights)} 8-Ks. By type: {by_type}\n")

    def show(et, expected):
        rows = sorted({(e["ticker"], e["value"]) for e in all_events if e["event_type"] == et})
        print(f"{et} ({len(rows)}):")
        for t, v in rows:
            print(f"    {t:6s} {v}")
        print(f"  expected -> {expected}\n")

    show("dividend", "EQ34: AIG $0.50, CAT $1.63, COST $1.47, ORCL, PG")
    show("buyback", "EQ35: ADBE $25B, NFLX $25B  (VZ/others may also authorize)")
    print(f"Wrote {GRAPH_CSV_DIR / 'events.json'}")


if __name__ == "__main__":
    main()
