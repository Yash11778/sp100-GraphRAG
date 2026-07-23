"""Pipeline 3 — Optimized TigerGraph GraphRAG.

Strategy (the token-reduction engine):
  1. A DETERMINISTIC router (0 LLM routing tokens) classifies the question and
     resolves entities (companies, audit firms, sectors, people, $ amounts).
  2. Aggregation/intersection questions -> a small typed graph traversal returns
     the EXACT answer set as a compact evidence line (tens of tokens) instead of
     a top-k chunk dump (thousands).
  3. Bridge questions ("the company that <did X> — <ask Y>") -> the graph resolves
     hop 1 from Event vertices (8-K events: dividends, buybacks, leadership), then
     hop 2 is answered by another traversal (auditor) or a company-scoped vector
     search (financial facts) — so retrieval is precise, not corpus-wide.
  4. People/keyword questions -> DB-side keyword scan over Chunk.text (kwsearch),
     generalizing "which companies mention X" beyond pre-extracted topics.
  5. Remaining factual questions -> semantic vector search over Chunk.emb
     (Savanna HNSW), company-scoped per named company so multi-company
     comparisons get balanced evidence.
  6. Context optimization before generation: chunk dedup, form-prior reranking
     (10-K for financials, DEF14A for governance, 8-K for events), per-company
     balance, and a hard evidence-token cap.

Graph contribution is explicit: `graph_used` + `graph_path` are recorded per answer.
"""
import os
import re
import time
from pathlib import Path

from pipelines.utils import count_tokens, gemini_generate, make_result, setup_gemini
from pipelines import graph_queries as gq
from pipelines import resolver as rz

_ROOT = Path(__file__).parent.parent.resolve()
_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder = None
_client = None

MAX_EVIDENCE_TOKENS = 1800   # frozen after dev-set tuning
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")

# Words that look like capitalized names but never are people.
_NAME_STOP = {
    "The", "Which", "What", "Who", "When", "Where", "Why", "How", "Among",
    "Besides", "Company", "Companies", "Inc", "Corp", "CEO", "CFO", "Chief",
    "Officer", "Executive", "Chair", "Chairman", "Board", "Director",
    "April", "May", "June", "July", "January", "February", "March", "August",
    "September", "October", "November", "December", "Big", "Four", "Data",
    "Center", "Health", "Care", "Energy", "Consumer", "Staples", "Financials",
    "Utilities", "Information", "Technology", "Tier", "Fiscal", "Year",
}


def _embed(text: str):
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(_EMBED_MODEL)
    import numpy as np
    v = np.array(list(_embedder.embed([text]))[0], dtype=np.float32)
    return (v / (np.linalg.norm(v) + 1e-10)).tolist()


def _person_names(question: str) -> list[str]:
    """Capitalized 2-4 token sequences that plausibly name a person, excluding
    company names and stop words. Handles 'F. William McNabb III', 'John Ternus'."""
    comps = {c.lower() for c in re.findall(r"[A-Za-z][\w.&\- ]+", question)}
    # token = capitalized word, allowing camel-case surnames (McNabb, DiMartino)
    _tok = r"[A-Z][a-z]+(?:[A-Z][a-z]+)?"
    cands = re.findall(
        rf"((?:[A-Z]\.\s+)?{_tok}(?:\s+(?:[A-Z]\.|{_tok})){{1,3}}"
        rf"(?:\s+(?:III|Jr\.?|Sr\.?))?)", question)
    out = []
    known_companies = set(rz.NAME2TICKER)
    for c in cands:
        toks = c.replace(".", "").split()
        # real person names never contain role/section words
        if any(t in _NAME_STOP for t in toks):
            continue
        low = re.sub(r"\s+", " ", c.lower().replace(".", "").strip())
        if low in known_companies or rz.detect_companies(c):
            continue
        if c.strip() not in out:
            out.append(c.strip())
    return out


def _entity_names(question: str) -> list[str]:
    """Multi-word proper-noun phrases that name an external entity ('Apogee
    Therapeutics', 'BT Group plc') — the generalization of _person_names to any
    named entity, allowing all-caps tokens (BT) and corporate suffixes."""
    tok = r"(?:[A-Z][a-z]+(?:[A-Z][a-z]+)?|[A-Z]{2,5})"
    cands = re.findall(rf"\b({tok}(?:\s+{tok}){{1,3}}(?:\s+(?:plc|Inc\.?|Corp\.?|LLC|Ltd\.?))?)", question)
    out = []
    for c in cands:
        toks = c.split()
        if any(t in _NAME_STOP for t in toks):
            continue
        if rz.detect_companies(c):          # dataset companies route elsewhere
            continue
        if len(toks) >= 2 and c.strip() not in out:
            out.append(c.strip())
    return out


def _find_amount(q: str) -> str | None:
    """'$4.1B', '$25 billion', '$2.0B' -> '4.1' / '25' / '2.0' (Event.value match key)."""
    m = re.search(r"\$\s*([\d]+(?:\.\d+)?)\s*(?:b\b|bn\b|billion)", q, re.I)
    return m.group(1) if m else None


def _find_date_phrase(q: str) -> str | None:
    m = re.search(rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}}),\s+(\d{{4}})", q)
    return f"{m.group(1)} {m.group(2)}, {m.group(3)}" if m else None


def _second_hop_kind(q: str) -> str:
    if "audit" in q or "accounting firm" in q:
        return "auditor"
    return "vector"


# ---------------------------------------------------------------------------
# Deterministic router.
# Returns dict(evidence=[..], path=str, tickers=[..], mode="exact"|"bridge"|None)
# ---------------------------------------------------------------------------
def _route(question: str):
    q = question.lower()
    firm = rz.detect_firm(question)
    sector = rz.detect_sector(question)
    comps = rz.detect_companies(question)
    R = lambda ev, path, tk, mode: {"evidence": ev if isinstance(ev, list) else [ev],
                                    "path": path, "tickers": tk, "mode": mode}

    # -- Big-Four audit counts --
    if ("how many" in q or "each" in q) and ("big four" in q or "audit" in q):
        counts = gq.auditor_counts()
        ev = "Graph result — companies audited per firm: " + \
             ", ".join(f"{k}={v}" for k, v in counts.items())
        return R(ev, "auditor_counts", [], "exact")

    # -- chronological ordering of events (buybacks/dividends) --
    if "chronolog" in q or ("order" in q and ("buyback" in q or "repurchase" in q or "dividend" in q)):
        etype = "buyback" if ("buyback" in q or "repurchase" in q) else "dividend"
        rows = gq.events_detailed(etype)
        if comps:
            rows = [r for r in rows if r["ticker"] in set(comps)]
        ev = f"Graph result — {etype} events in chronological order (by filing date): " + \
             "; ".join(f"{r['ticker']} {r['event_date']} ({r['value']})" for r in rows)
        return R(ev, f"REPORTS_EVENT({etype}) ORDER BY date", [r["ticker"] for r in rows], "exact")

    # -- '$X buyback/repurchase' value bridge: which company / its auditor --
    amt = _find_amount(q)
    if amt and ("repurchase" in q or "buyback" in q) and not comps:
        rows = gq.event_by_value("buyback", amt)
        if rows:
            tks = sorted({r["ticker"] for r in rows})
            ev = [f"Graph result — 8-K ${amt}B share-repurchase authorization: " +
                  ", ".join(f"{rz.label(r['ticker'])} — {r['value']}, filed {r['event_date']}" for r in rows)]
            path = f"REPORTS_EVENT(buyback, value~{amt}B)"
            if _second_hop_kind(q) == "auditor":
                for t in tks:
                    a = gq.auditor_of(t)
                    ev.append(f"Graph result — {rz.label(t)}'s independent auditor: {a}")
                path += " -> AUDITED_BY"
                return R(ev, path, tks, "exact")
            return R(ev, path, tks, "bridge")

    # -- leadership bridge: 'the company that appointed <person>...' --
    # Person surname -> DB-side keyword scan over 8-K chunks resolves the company
    # (Event summaries are truncated; chunk text is complete).
    persons = _person_names(question)
    if persons and any(w in q for w in ("appoint", "named", "ceo", "cfo", "chair",
                                        "chief", "officer", "departing", "incoming",
                                        "retire", "transition")):
        for p in persons:
            surname = next((t for t in reversed(p.replace(".", "").split())
                            if len(t) > 3 and t not in ("III", "Jr", "Sr")), p)
            hits = gq.kwsearch(surname, form="8-K", limit=20)
            if hits:
                tks = sorted({h["ticker"] for h in hits})
                ev = [f"Graph result — 8-K filings mentioning {p}: {', '.join(rz.label(t) for t in tks)}"]
                for h in hits[:3]:
                    ev.append(f"[{h['ticker']} 8-K] …{h['snippet']}…")
                path = f"kwsearch({surname}, 8-K)"
                if _second_hop_kind(q) == "auditor":
                    for t in tks:
                        ev.append(f"Graph result — {rz.label(t)}'s independent auditor: {gq.auditor_of(t)}")
                    path += " -> AUDITED_BY"
                    return R(ev, path, tks, "exact")
                return R(ev, path, tks, "bridge")

    # -- person on boards: '<person> serves on the boards of which companies' --
    if persons and ("board" in q or "director" in q):
        for p in persons:
            surname = next((t for t in reversed(p.replace(".", "").split())
                            if len(t) > 3 and t not in ("III", "Jr", "Sr")), p)
            hits = gq.kwsearch(surname, form="DEF14A", limit=30)
            if hits:
                tks = sorted({h["ticker"] for h in hits})
                ev = [f"Graph result — DEF14A filings mentioning {p}: {', '.join(rz.label(t) for t in tks)}"]
                for h in hits[:4]:
                    ev.append(f"[{h['ticker']} DEF14A] …{h['snippet']}…")
                return R(ev, f"kwsearch({surname}, DEF14A)", tks, "exact")

    # -- fiscal-year-end date x auditor (bridge via keyword scan) --
    dph = _find_date_phrase(question)
    if dph and firm and "fiscal year" in q:
        hits = gq.kwsearch(dph, form="10-K", limit=40)
        audited = set(gq.companies_by_auditor(firm))
        tks = sorted({h["ticker"] for h in hits} & audited)
        ev = [f"Graph result — {firm}-audited companies whose 10-K mentions "
              f"'{dph}': {', '.join(rz.label(t) for t in tks) or '(none found)'}"]
        for h in [h for h in hits if h["ticker"] in tks][:3]:
            ev.append(f"[{h['ticker']} 10-K] …{h['snippet']}…")
        return R(ev, f"kwsearch({dph},10-K) ∩ AUDITED_BY({firm})", tks, "exact")

    # -- TSMC foundry --
    if "tsmc" in q or "taiwan semiconductor" in q or "foundry" in q:
        if firm:
            tk = gq.companies_by_entity_and_auditor("TSMC", "NAMES_FOUNDRY", firm)
            path = f"NAMES_FOUNDRY(TSMC) ∩ AUDITED_BY({firm})"
        else:
            tk = gq.companies_by_entity("TSMC", "NAMES_FOUNDRY")
            path = "NAMES_FOUNDRY(TSMC)"
        ev = "Graph result — companies naming TSMC as a foundry" + \
             (f" audited by {firm}" if firm else "") + f": {', '.join(tk)}"
        mode = "bridge" if any(w in q for w in ("revenue", "highest", "income")) else "exact"
        return R(ev, path, tk, mode)

    # -- NVIDIA competitor --
    if "nvidia" in q and "compet" in q:
        if firm:
            tk = gq.companies_by_entity_and_auditor("NVIDIA", "NAMES_COMPETITOR", firm)
            path = f"NAMES_COMPETITOR(NVIDIA) ∩ AUDITED_BY({firm})"
        else:
            tk = gq.companies_by_entity("NVIDIA", "NAMES_COMPETITOR")
            path = "NAMES_COMPETITOR(NVIDIA)"
        ev = "Graph result — companies naming NVIDIA as a competitor" + \
             (f" audited by {firm}" if firm else "") + f": {', '.join(tk)}"
        return R(ev, path, tk, "exact")

    # -- climate risk x sector --
    if "climate" in q and sector:
        tk = gq.companies_by_topic_and_sector("climate change", sector)
        return R(f"Graph result — {sector} companies discussing climate: {', '.join(tk)}",
                 f"MENTIONS_TOPIC(climate) ∩ sector={sector}", tk, "exact")

    # -- GLP-1 / weight-loss x sector/firm --
    if "glp" in q or "weight loss" in q or "weight-loss" in q or "obesity" in q:
        if firm and sector:
            base = gq.companies_by_topic_and_sector("GLP-1", sector)
            aud = set(gq.companies_by_auditor(firm))
            tk = sorted(set(base) & aud)
            return R(f"Graph result — {sector} companies mentioning GLP-1 audited by {firm}: {', '.join(tk)}",
                     f"MENTIONS_TOPIC(GLP-1) ∩ sector={sector} ∩ AUDITED_BY({firm})", tk, "exact")
        if sector:
            tk = gq.companies_by_topic_and_sector("GLP-1", sector)
            return R(f"Graph result — {sector} companies mentioning GLP-1/weight-loss drugs: {', '.join(tk)}",
                     f"MENTIONS_TOPIC(GLP-1) ∩ sector={sector}", tk, "exact")

    # -- sector membership: 'which companies are in the X sector' --
    if sector and ("which companies" in q or "list" in q) and "audit" not in q \
       and "mention" not in q and "climate" not in q:
        tk = sorted(t for t, s in rz.SECTOR.items() if s == sector)
        return R(f"Graph result — {sector} sector companies: {', '.join(tk)}",
                 f"IN_SECTOR({sector})", tk, "exact")

    # -- audited by FIRM (+optional sector) --
    if firm and ("audit" in q or "accounting firm" in q) and \
       ("which" in q or "list" in q or "companies" in q or sector):
        if sector:
            tk = gq.companies_by_auditor_and_sector(firm, sector)
            return R(f"Graph result — {sector} companies audited by {firm}: {', '.join(tk)}",
                     f"AUDITED_BY({firm}) ∩ sector={sector}", tk, "exact")
        tk = gq.companies_by_auditor(firm)
        return R(f"Graph result — companies audited by {firm}: {', '.join(tk)}",
                 f"AUDITED_BY({firm})", tk, "exact")

    # -- who is COMPANY's auditor (multi-ask questions keep the vector layer) --
    if ("audit" in q or "accounting firm" in q) and comps:
        t = comps[0]
        a = gq.auditor_of(t)
        if a:
            # "…who is he, and who is X's auditor?" — auditor is only one part;
            # bridge mode adds scoped evidence for the other ask(s).
            multi = any(s in q for s in (" and ", "who is he", "who is she", "whom",
                                         "name it", "what were", "what was",
                                         "how much", "which growth", "revised"))
            return R(f"Graph result — {rz.label(t)}'s independent auditor: {a}",
                     f"AUDITED_BY of {t}", [t], "bridge" if multi else "exact")

    # -- per-company 8-K event facts (dividend $/share, buyback size, leadership) --
    if comps and ("8-k" in q or "8k" in q or "dividend" in q or "buyback" in q
                  or "repurchase" in q or "cfo" in q or "ceo" in q or "appoint" in q
                  or "chief" in q or "officer" in q or "restructur" in q):
        t = comps[0]
        etype = ("dividend" if "dividend" in q else
                 "buyback" if ("buyback" in q or "repurchase" in q) else
                 "leadership" if ("cfo" in q or "ceo" in q or "appoint" in q
                                  or "chief" in q or "officer" in q) else None)
        rows = gq.events_of(t, etype)
        if rows:
            ev = [f"Graph result — {rz.label(t)} 8-K events" + (f" ({etype})" if etype else "") + ": " +
                  "; ".join(f"[{r['event_type']}] {r['value']} filed {r['event_date']} — {r['summary'][:180]}"
                            for r in rows)]
            return R(ev, f"REPORTS_EVENT({etype or '*'}) of {t}", [t], "bridge")

    # -- dividends declared via 8-K (corpus-wide) --
    # The filings distinguish an actual DECLARATION from an announced intention/
    # increase; the Event.declared flag (extracted from the 8-K wording) keeps
    # the two apart so the answer reflects what each filing actually did.
    if "dividend" in q and ("which" in q or "companies" in q or "list" in q):
        rows = gq.events_detailed("dividend")
        dec = [r for r in rows if r.get("declared")]
        ann = [r for r in rows if not r.get("declared")]
        ev = "Graph result — companies whose 8-K DECLARED a cash dividend: " + \
             ", ".join(f"{rz.label(r['ticker'])} ({r['value']}, {r['event_date']})" for r in dec)
        if ann:
            ev += ". Separately, 8-Ks that only ANNOUNCED a dividend increase (not a declaration): " + \
                  ", ".join(f"{r['ticker']} ({r['value']})" for r in ann)
        return R(ev, "REPORTS_EVENT(dividend, declared)", [r["ticker"] for r in dec], "exact")

    # -- buybacks via 8-K (corpus-wide) --
    if ("repurchase" in q or "buyback" in q) and ("which" in q or "companies" in q or "billion" in q):
        rows = gq.events_detailed("buyback")
        if "$25" in q or "25 billion" in q:
            rows = [r for r in rows if "25" in (r["value"] or "")]
        ev = "Graph result — companies that authorized a share buyback via 8-K: " + \
             ", ".join(f"{r['ticker']} ({r['value']}, {r['event_date']})" for r in rows)
        return R(ev, "REPORTS_EVENT(buyback)", [r["ticker"] for r in rows], "exact")

    # -- generic named-entity bridge: 'the company that <did X with> <Entity>' --
    # Same mechanism as the person bridge, generalized: resolve the company via a
    # DB-side scan for the entity name in 8-K chunks (acquisitions, JVs, deals).
    if ("company that" in q or "which company" in q) and not comps:
        for ent in _entity_names(question):
            hits = gq.kwsearch(ent, form="8-K", limit=12)
            if not hits:
                continue
            tks = sorted({h["ticker"] for h in hits})
            ev = [f"Graph result — 8-K filings mentioning {ent}: " +
                  ", ".join(rz.label(t) for t in tks)]
            for h in hits[:2]:
                ev.append(f"[{h['ticker']} 8-K] {h['text']}")
            path = f"kwsearch({ent}, 8-K)"
            if _second_hop_kind(q) == "auditor":
                for t in tks[:3]:
                    ev.append(f"Graph result — {rz.label(t)}'s independent auditor: {gq.auditor_of(t)}")
                path += " -> AUDITED_BY"
                return R(ev, path, tks, "exact")
            return R(ev, path, tks, "bridge")

    return {"evidence": [], "path": None, "tickers": [], "mode": None}


# ---------------------------------------------------------------------------
# Context-optimized retrieval layer
# ---------------------------------------------------------------------------
# Financial-metric phrases: question keyword -> exact phrases to scan for in the
# company's 10-K. Statement rows containing these phrases carry the number the
# question wants; plain vector search often surfaces narrative chunks instead.
_METRIC_PHRASES = [
    (("net sales",), ["total net sales", "net sales"]),
    (("net operating revenues", "operating revenues"), ["net operating revenues"]),
    (("revenues and other income",), ["total revenues and other income"]),
    (("total revenue", "revenue"), ["total revenue", "total revenues"]),
    (("net income",), ["net income"]),
    (("r&d", "research and development"), ["research and development"]),
]
_NUM_PAT = re.compile(r"\$?\d[\d,]{2,}")


def _numeric_evidence(question: str, tickers: list[str]) -> tuple[list, list]:
    """Ticker-scoped DB-side phrase scan over 10-K chunks, ranked by how many
    large numbers the chunk carries (financial-statement rows rank first)."""
    ql = question.lower()
    phrases = next((ph for keys, ph in _METRIC_PHRASES
                    if any(k in ql for k in keys)), None)
    if not phrases or not tickers:
        return [], []
    parts, cits = [], []
    for t in tickers[:2]:
        hits = []
        for ph in phrases:
            hits = gq.kwsearch(ph, form="10-K", limit=12, ticker=t)
            if hits:
                break
        if not hits:
            continue
        hits.sort(key=lambda h: -len(_NUM_PAT.findall(h.get("text", ""))))
        for h in hits[:2]:
            parts.append(f"[{h['ticker']} 10-K] {h['text']}")
            cits.append(h["chunk_id"])
    return parts, cits


def _form_prior(q: str) -> str | None:
    ql = q.lower()
    if any(w in ql for w in ("net sales", "revenue", "net income", "r&d",
                             "research and development", "fiscal year",
                             "operating", "total")):
        return "10-K"
    if any(w in ql for w in ("auditor", "board", "director", "governance",
                             "chief executive officer", "proxy")):
        return "DEF14A"
    if any(w in ql for w in ("8-k", "dividend", "buyback", "repurchase",
                             "appointed", "restructuring")):
        return "8-K"
    return None


def _vector_evidence(question: str, scope: list[str], k: int,
                     exclude: set | None = None) -> tuple[list, list]:
    """Company-scoped, deduplicated, form-prior-reranked chunk retrieval."""
    qv = _embed(question)
    prior = _form_prior(question)
    hits = []
    if scope:
        per = max(2, k // max(1, len(scope[:4])))
        for t in scope[:4]:
            try:
                got = gq.vector_search(qv, k=per + 2, ticker=t)
            except Exception:
                got = [h for h in gq.vector_search(qv, k=k * 8) if h.get("ticker") == t]
            hits.extend(got[:per + 2])
    else:
        hits = gq.vector_search(qv, k=k * 2)

    # dedup + form-prior rerank (stable: prior form first, then original order)
    seen, uniq = set(exclude or ()), []
    for h in hits:
        cid = h.get("chunk_id") or h.get("text", "")[:80]
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(h)
    if prior:
        uniq.sort(key=lambda h: 0 if h.get("form") == prior else 1)
    if scope:
        # round-robin rebalance so each named company keeps evidence
        by_t = {}
        for h in uniq:
            by_t.setdefault(h.get("ticker"), []).append(h)
        merged, i = [], 0
        while len(merged) < len(uniq):
            added = False
            for t in scope[:4]:
                lst = by_t.get(t, [])
                if i < len(lst):
                    merged.append(lst[i]); added = True
            if not added:
                break
            i += 1
        uniq = merged or uniq

    parts, cits, budget = [], [], MAX_EVIDENCE_TOKENS
    for h in uniq[:k]:
        text = f"[{h.get('ticker')} {h.get('form')}] {h.get('text','')}"
        cost = count_tokens(text)
        if cost > budget:
            continue
        budget -= cost
        parts.append(text)
        cits.append(h.get("chunk_id"))
    return parts, cits


def _get_client():
    global _client
    if _client is None:
        _client = setup_gemini()
    return _client


def pipeline3(question: str, top_k: int = 6) -> dict:
    """GR_ABLATION env var (read per call) supports the brief-required ablations:
      no_graph — router disabled: global Savanna HNSW vector search only
      no_scope — router on, but vector layer never company-scoped
      no_opt   — no dedup / form-prior / balance / token cap (raw top-k dump)
    """
    t0 = time.time()
    ablation = os.getenv("GR_ABLATION", "").strip().lower()

    if ablation == "no_graph":
        route = {"evidence": [], "path": None, "tickers": [], "mode": None}
    else:
        route = _route(question)      # deterministic, 0 LLM tokens
    evidence_parts = list(route["evidence"])
    citations = [f"graph://{route['path']}"] if route["path"] else []
    graph_used = bool(route["path"])

    # Vector layer: skip when the graph set IS the answer; scoped for bridges.
    mode = route["mode"]
    if mode != "exact":
        scope = [] if ablation in ("no_graph", "no_scope") \
            else (rz.detect_companies(question) or route["tickers"])

        # Bridge second hop: the resolved company's 8-K events are graph facts
        # the question may need (buyback amount, leadership dates) — cheap and exact.
        if mode == "bridge" and route["tickers"] and ablation != "no_graph":
            ql = question.lower()
            for t in route["tickers"][:2]:
                evs = gq.events_of(t)
                if evs:
                    evidence_parts.append(
                        f"Graph result — {rz.label(t)} 8-K events: " +
                        "; ".join(f"[{e['event_type']}] {e['value']} filed {e['event_date']}"
                                  for e in evs))
                    citations.append(f"graph://REPORTS_EVENT(*) of {t}")
                # Officer names live in the 8-K text, not the Event vertex —
                # pull the announcement chunk when the question asks who/whom.
                if any(w in ql for w in ("cfo", "ceo", "chief", "officer",
                                         "whom", "who is", "appoint")):
                    role = ("Chief Financial Officer" if "cfo" in ql or "financial officer" in ql
                            else "Chief Executive Officer" if "ceo" in ql or "executive officer" in ql
                            else "Chief Legal Officer" if "legal officer" in ql
                            else "Officer")
                    khits = gq.kwsearch(role, form="8-K", limit=4, ticker=t)
                    for h in khits[:1]:
                        evidence_parts.append(f"[{h['ticker']} 8-K] {h['text']}")
                        citations.append(h["chunk_id"])

        if ablation == "no_opt":
            qv = _embed(question)
            hits = gq.vector_search(qv, k=top_k)
            for h in hits:
                evidence_parts.append(f"[{h.get('ticker')} {h.get('form')}] {h.get('text','')}")
                citations.append(h.get("chunk_id"))
        else:
            # Numeric-fact chunks first (financial-statement rows), then semantic.
            n_parts, n_cits = ([], []) if ablation == "no_graph" \
                else _numeric_evidence(question, scope[:2])
            evidence_parts.extend(n_parts)
            citations.extend(n_cits)
            v_parts, v_cits = _vector_evidence(question, scope[:4],
                                               top_k - min(2, len(n_parts)),
                                               exclude=set(n_cits))
            evidence_parts.extend(v_parts)
            citations.extend(v_cits)

    context = "\n\n---\n\n".join(evidence_parts)
    prompt = (
        "You are an expert SEC-filings assistant. Use ONLY the evidence below to answer. "
        "When the evidence is a 'Graph result' list, treat it as the authoritative set and "
        "report it precisely (all members). Be exact with names, numbers, dates. "
        "Answer directly and completely: state every fact the question asks for — if the "
        "question identifies a company indirectly, name that company explicitly, then give "
        "the asked fact(s). No hedging; omit procedural conditions from the source "
        "(e.g. 'subject to approval', 'intention to') unless the question asks about them.\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    client = _get_client()
    answer = gemini_generate(client, prompt, max_tokens=512)
    latency = round(time.time() - t0, 3)

    res = make_result("graphrag", answer, count_tokens(prompt), count_tokens(answer), latency)
    from pipelines.utils import LAST_USAGE
    res.update(LAST_USAGE)   # actual Gemini API usage (brief: report when available)
    res["sources"] = citations
    res["retrieved_context_tokens"] = count_tokens(context)
    res["evidence"] = context
    res["routing_tokens"] = 0            # deterministic router — no LLM cost
    res["graph_used"] = graph_used
    res["graph_path"] = route["path"]
    return res
