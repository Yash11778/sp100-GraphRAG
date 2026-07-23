"""Parse the SP100 corpus into clean, structured per-document records.

Manifest-driven: every document in manifest.csv becomes one JSON record with
normalised metadata + lightly-cleaned text. No API, no network — pure local work.

Output:
  data/parsed/<doc_id>.json      one record per filing
  data/parsed/_index.json        list of all doc_ids + metadata (no full text)

    python ingestion/parse_filings.py
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config import CORPUS, MANIFEST, PARSED_DIR, FORM_MAP  # noqa: E402


def clean_text(raw: str) -> str:
    """Collapse the blank-line-heavy EDGAR text dump into readable paragraphs
    without destroying structure. We keep the content verbatim (numbers, names,
    tables-as-text) and only normalise whitespace so chunking is stable."""
    # Normalise unicode punctuation that shows up in filings.
    raw = raw.replace("’", "'").replace("‘", "'")
    raw = raw.replace("“", '"').replace("”", '"')
    raw = raw.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    # Collapse 3+ newlines to a paragraph break; single stray newlines within a
    # paragraph stay so sentence order is preserved.
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def doc_id_for(ticker: str, form_norm: str, date: str, seen: dict) -> str:
    """Stable, unique id. Second same-day filing of a form gets a _2 suffix
    (mirrors the corpus, e.g. JPM has two 8-Ks dated 2026-07-14)."""
    base = f"{ticker}_{form_norm}_{date}"
    n = seen.get(base, 0) + 1
    seen[base] = n
    return base if n == 1 else f"{base}_{n}"


def resolve_path(ticker: str, local_path: str) -> Path:
    """The manifest's local_path is repo-relative ('sp100_dataset/AAPL/..').
    Map it to the actual file under CORPUS by taking the filename and ticker."""
    fname = Path(local_path).name
    return CORPUS / ticker / fname


def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    index = []
    seen: dict = {}
    missing = 0
    with open(MANIFEST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip()
            form_raw = row["form"].strip()
            form_norm = FORM_MAP.get(form_raw, form_raw.replace(" ", ""))
            date = row["filing_date"].strip()
            path = resolve_path(ticker, row["local_path"])
            if not path.exists():
                missing += 1
                print(f"  [missing] {ticker} {form_raw} {date} -> {path}")
                continue
            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            did = doc_id_for(ticker, form_norm, date, seen)
            record = {
                "doc_id": did,
                "ticker": ticker,
                "company": row["name"].strip(),
                "sector": row["sector"].strip(),
                "form": form_norm,
                "form_raw": form_raw,
                "filing_date": date,
                "accession": row["accession"].strip(),
                "source_url": row["source_url"].strip(),
                "n_chars": len(text),
                "text": text,
            }
            (PARSED_DIR / f"{did}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            index.append({k: v for k, v in record.items() if k != "text"})

    (PARSED_DIR / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    forms = {}
    for r in index:
        forms[r["form"]] = forms.get(r["form"], 0) + 1
    print(f"\nParsed {len(index)} documents ({missing} missing).")
    print("By form:", forms)
    print(f"Companies: {len({r['ticker'] for r in index})}")
    print(f"Index -> {PARSED_DIR / '_index.json'}")


if __name__ == "__main__":
    main()
