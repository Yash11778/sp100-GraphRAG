"""Central paths + constants for the Round 3 SP100 GraphRAG project.

Everything downstream (parsing, extraction, chunking, pipelines, eval) imports
from here so there is exactly one place that knows where the corpus lives.
"""
import os
from pathlib import Path

# --- Repo layout ------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
DATA = ROOT / "data"
PARSED_DIR = DATA / "parsed"          # cleaned per-document JSON
GRAPH_CSV_DIR = DATA / "graph_csv"    # vertex/edge CSVs ready for Savanna load
CHUNKS_DIR = DATA / "chunks"          # FAISS index + chunk pickle
QA_DIR = DATA / "qa"                  # held-out EQ set + dev set

# --- Corpus location --------------------------------------------------------
# The raw SP100 dataset lives inside the repo folder (gitignored — not pushable).
# Override with SP100_DATASET env var if the folder moves.
DEFAULT_CORPUS = (
    ROOT
    / "sp100_dataset-20260717T162023Z-1-001"
    / "sp100_dataset"
)
CORPUS = Path(os.getenv("SP100_DATASET", str(DEFAULT_CORPUS)))
MANIFEST = CORPUS / "manifest.csv"

# --- Known audit firms (canonical names) ------------------------------------
# Ordered longest-first so "Ernst & Young" matches before a bare "Young", etc.
AUDIT_FIRMS = [
    "PricewaterhouseCoopers LLP",
    "PricewaterhouseCoopers",
    "Ernst & Young LLP",
    "Ernst & Young",
    "KPMG LLP",
    "KPMG",
    "Deloitte & Touche LLP",
    "Deloitte & Touche",
    "Deloitte",
]
# Map every surface form to a single canonical firm node.
AUDIT_FIRM_CANONICAL = {
    "PricewaterhouseCoopers LLP": "PricewaterhouseCoopers LLP",
    "PricewaterhouseCoopers": "PricewaterhouseCoopers LLP",
    "Ernst & Young LLP": "Ernst & Young LLP",
    "Ernst & Young": "Ernst & Young LLP",
    "KPMG LLP": "KPMG LLP",
    "KPMG": "KPMG LLP",
    "Deloitte & Touche LLP": "Deloitte & Touche LLP",
    "Deloitte & Touche": "Deloitte & Touche LLP",
    "Deloitte": "Deloitte & Touche LLP",
}

# --- Form-type normalisation ------------------------------------------------
FORM_MAP = {
    "10-K": "10-K",
    "8-K": "8-K",
    "DEF 14A": "DEF14A",
    "DEF14A": "DEF14A",
}
