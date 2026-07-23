"""GraphRAG structured-traversal layer for the sp100 graph.

These typed traversals are the token-reduction engine: a Tier-C aggregation /
intersection question is answered by a small graph query returning an exact
company set, instead of a full-corpus RAG sweep. Pipeline 3 calls these to build
a tiny, precise evidence set before generation.

Uses interpreted GSQL (no install step) via the global token. All patterns start
FROM Company and follow forward edges, so no reverse edges are needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import tg  # noqa: E402


def _tickers(result_json: dict, key: str = "R") -> list[str]:
    """Pull ticker strings out of a PRINT R[R.ticker] result block."""
    for block in result_json.get("results", []):
        for k, v in block.items():
            if isinstance(v, list):
                out = []
                for item in v:
                    if isinstance(item, dict):
                        attrs = item.get("attributes", {})
                        out.append(attrs.get(f"{key}.ticker") or item.get("v_id"))
                if out:
                    return sorted(x for x in out if x)
    return []


def _run(gsql_body: str) -> dict:
    import json
    q = f"USE GRAPH sp100\nINTERPRET QUERY () FOR GRAPH sp100 {{\n{gsql_body}\n}}"
    out = tg.gsql(q)
    # Response has a "Using graph" line then JSON.
    start = out.find("{")
    return json.loads(out[start:])


# --- Query builders ---------------------------------------------------------
def companies_by_auditor(firm: str) -> list[str]:
    return _tickers(_run(
        f'R = SELECT c FROM Company:c -(AUDITED_BY:e)-> AuditFirm:a WHERE a.name == "{firm}";'
        f'\n  PRINT R[R.ticker];'))


def companies_by_entity(entity: str, edge: str) -> list[str]:
    return _tickers(_run(
        f'R = SELECT c FROM Company:c -({edge}:e)-> Entity:x WHERE x.name == "{entity}";'
        f'\n  PRINT R[R.ticker];'))


def companies_by_topic_and_sector(topic: str, sector: str) -> list[str]:
    return _tickers(_run(
        f'R = SELECT c FROM Company:c -(MENTIONS_TOPIC:e)-> Topic:t '
        f'WHERE t.name == "{topic}" AND c.sector == "{sector}";'
        f'\n  PRINT R[R.ticker];'))


def companies_by_auditor_and_sector(firm: str, sector: str) -> list[str]:
    return _tickers(_run(
        f'R = SELECT c FROM Company:c -(AUDITED_BY:e)-> AuditFirm:a '
        f'WHERE a.name == "{firm}" AND c.sector == "{sector}";'
        f'\n  PRINT R[R.ticker];'))


def companies_by_entity_and_auditor(entity: str, edge: str, firm: str) -> list[str]:
    return _tickers(_run(
        f'S = SELECT c FROM Company:c -({edge}:e)-> Entity:x WHERE x.name == "{entity}";'
        f'\n  R = SELECT c FROM S:c -(AUDITED_BY:e2)-> AuditFirm:a WHERE a.name == "{firm}";'
        f'\n  PRINT R[R.ticker];'))


def auditor_counts() -> dict:
    """EQ44: how many companies each Big-Four firm audits. Simple: count per firm."""
    firms = ["Ernst & Young LLP", "PricewaterhouseCoopers LLP",
             "Deloitte & Touche LLP", "KPMG LLP"]
    return {f: len(companies_by_auditor(f)) for f in firms}


def auditor_of(ticker: str) -> str | None:
    """The audit firm for one company (EQ10/12/17/18/48/49 bridge tail)."""
    data = _run(
        f'R = SELECT a FROM Company:c -(AUDITED_BY:e)-> AuditFirm:a WHERE c.ticker == "{ticker}";'
        f'\n  PRINT R[R.name];')
    for block in data.get("results", []):
        for v in block.values():
            if isinstance(v, list) and v:
                return v[0].get("attributes", {}).get("R.name") or v[0].get("v_id")
    return None


def companies_by_event(event_type: str) -> list[dict]:
    """Companies with an 8-K event of a type, with the event value (EQ34/35).
    Event.event_id is '<TICKER>_<form>_<date>::<type>', so the ticker is derivable.
    Returns [{ticker, value, event_date}]."""
    import json
    q = (f'USE GRAPH sp100\nINTERPRET QUERY () FOR GRAPH sp100 {{\n'
         f'  E = SELECT ev FROM Event:ev WHERE ev.event_type == "{event_type}";\n'
         f'  PRINT E[E.event_id AS event_id, E.value AS value, E.event_date AS event_date];\n}}')
    out = tg.gsql(q)
    data = json.loads(out[out.find("{"):])
    rows = []
    for block in data.get("results", []):
        for v in block.values():
            if isinstance(v, list):
                for item in v:
                    at = item.get("attributes", {})
                    eid = at.get("event_id", "")
                    ticker = eid.split("_", 1)[0] if eid else None
                    rows.append({"ticker": ticker, "value": at.get("value"),
                                 "event_date": at.get("event_date")})
    return rows


def events_of(ticker: str, event_type: str | None = None) -> list[dict]:
    """All 8-K events for one company (optionally one type), with value/date/summary.
    Answers per-company factual event questions (dividend $/share, buyback size,
    leadership changes) straight from the graph."""
    import json
    cond = f'ev.event_id LIKE "{ticker}_%"'
    if event_type:
        cond += f' AND ev.event_type == "{event_type}"'
    q = (f'USE GRAPH sp100\nINTERPRET QUERY () FOR GRAPH sp100 {{\n'
         f'  E = SELECT ev FROM Event:ev WHERE {cond};\n'
         f'  PRINT E[E.event_id AS event_id, E.event_type AS event_type, '
         f'E.value AS value, E.event_date AS event_date, E.summary AS summary];\n}}')
    out = tg.gsql(q)
    data = json.loads(out[out.find("{"):])
    rows = []
    for block in data.get("results", []):
        for v in block.values():
            if isinstance(v, list):
                for item in v:
                    at = item.get("attributes", {})
                    rows.append({k.split(".")[-1]: val for k, val in at.items()})
    return rows


def events_detailed(event_type: str) -> list[dict]:
    """companies_by_event + summary field, sorted by event_date (chronology answers)."""
    import json
    q = (f'USE GRAPH sp100\nINTERPRET QUERY () FOR GRAPH sp100 {{\n'
         f'  E = SELECT ev FROM Event:ev WHERE ev.event_type == "{event_type}";\n'
         f'  PRINT E[E.event_id AS event_id, E.value AS value, '
         f'E.event_date AS event_date, E.summary AS summary, E.declared AS declared];\n}}')
    out = tg.gsql(q)
    data = json.loads(out[out.find("{"):])
    rows = []
    for block in data.get("results", []):
        for v in block.values():
            if isinstance(v, list):
                for item in v:
                    at = item.get("attributes", {})
                    eid = at.get("event_id", "")
                    rows.append({"ticker": eid.split("_", 1)[0] if eid else None,
                                 "event_id": eid, "value": at.get("value"),
                                 "event_date": at.get("event_date"),
                                 "declared": at.get("declared", True),
                                 "summary": at.get("summary", "")})
    return sorted(rows, key=lambda r: r.get("event_date") or "")


def event_by_value(event_type: str, amount: str) -> list[dict]:
    """Reverse bridge: '$4.1B buyback' -> which company. Matches Event.value
    containing the amount string (e.g. '4.1')."""
    return [r for r in events_detailed(event_type)
            if amount in (r.get("value") or "")]


def find_leadership(person_kw: str) -> list[dict]:
    """Leadership events whose summary mentions a person -> resolves
    'the company that appointed X' bridges from the graph."""
    return [r for r in events_detailed("leadership")
            if person_kw.lower() in (r.get("summary") or "").lower()]


def kwsearch(keyword: str, form: str = "", limit: int = 40, ticker: str = "") -> list[dict]:
    """DB-side keyword scan over Chunk.text via the installed `kwsearch` query.
    Generalizes 'which companies mention X' beyond pre-extracted topics, finds
    people (directors/officers) in DEF14A text, and — ticker-scoped — retrieves
    numeric-fact chunks ('total net sales') from one company's 10-K. Returns
    [{ticker, chunk_id, form, snippet, text}] — snippet centred on the keyword."""
    import json
    r = tg.restpp("/restpp/query/sp100/kwsearch", method="POST", graph="sp100",
                  data=json.dumps({"kw": keyword, "form_filter": form, "top": limit,
                                   "tk": ticker}),
                  headers={"Content-Type": "application/json"}, timeout=120)
    hits = r.json().get("results", [{}])[0].get("v", [])
    rows = []
    for h in hits:
        at = h.get("attributes", {})
        text = at.get("text", "")
        i = text.lower().find(keyword.lower())
        lo = max(0, i - 130); hi = min(len(text), i + len(keyword) + 130)
        rows.append({"ticker": at.get("ticker"), "chunk_id": h.get("v_id"),
                     "form": at.get("form"), "snippet": text[lo:hi].strip(),
                     "text": text})
    return rows


def vector_search(query_vec: list, k: int = 6, ticker: str = "") -> list[dict]:
    """Semantic retrieval over Chunk.emb via the installed `vsearch` HNSW query.
    Optional DB-side ticker scoping via `vsearch_scoped` (filtered HNSW)."""
    import json
    if ticker:
        r = tg.restpp("/restpp/query/sp100/vsearch_scoped", method="POST", graph="sp100",
                      data=json.dumps({"query_vector": query_vec, "k": k, "tk": ticker}),
                      headers={"Content-Type": "application/json"}, timeout=60)
    else:
        r = tg.restpp("/restpp/query/sp100/vsearch", method="POST", graph="sp100",
                      data=json.dumps({"query_vector": query_vec, "k": k}),
                      headers={"Content-Type": "application/json"}, timeout=60)
    hits = r.json().get("results", [{}])[0].get("v", [])
    return [h.get("attributes", {}) for h in hits]
