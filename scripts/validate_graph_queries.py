"""Validate the GraphRAG traversal layer against the official EQ answer keys.

Proves the graph answers Tier-C aggregation/intersection questions with exact
small traversals (no LLM, no vector search). Run any time — reads the live graph.

    python scripts/validate_graph_queries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines import graph_queries as gq  # noqa: E402

EY = "Ernst & Young LLP"
PWC = "PricewaterhouseCoopers LLP"
DEL = "Deloitte & Touche LLP"
KPMG = "KPMG LLP"


def check(label, got, expected_str, expected_set):
    got_s = set(got)
    exp = set(expected_set)
    ok = exp <= got_s   # every expected ticker present (extras noted, keys under-count)
    extra = got_s - exp
    flag = "OK " if ok else "MISS"
    print(f"[{flag}] {label}")
    print(f"       got={sorted(got_s)}")
    print(f"       key={expected_str}" + (f"  (+extra {sorted(extra)})" if ok and extra else ""))
    if not ok:
        print(f"       MISSING {sorted(exp - got_s)}")
    return ok


def main():
    n = 0
    ok = 0

    def c(*a):
        nonlocal n, ok
        n += 1
        ok += 1 if check(*a) else 0

    # EQ25 KPMG (key 11; we also find Citi -> 12)
    c("EQ25 audited by KPMG", gq.companies_by_auditor(KPMG),
      "ACN,ADBE,BNY,COST,EMR,GD,HD,PEP,PFE,V,WFC",
      ["ACN","ADBE","BNY","COST","EMR","GD","HD","PEP","PFE","V","WFC"])
    # EQ26 TSMC foundry
    c("EQ26 name TSMC as foundry", gq.companies_by_entity("TSMC","NAMES_FOUNDRY"),
      "AMD,AVGO,INTC,NVDA,QCOM", ["AMD","AVGO","INTC","NVDA","QCOM"])
    # EQ42 NVIDIA competitor
    c("EQ42 name NVIDIA as competitor", gq.companies_by_entity("NVIDIA","NAMES_COMPETITOR"),
      "AMD,CSCO,INTC,QCOM", ["AMD","CSCO","INTC","QCOM"])
    # EQ30 climate x Energy
    c("EQ30 climate x Energy", gq.companies_by_topic_and_sector("climate change","Energy"),
      "CVX,XOM,COP", ["CVX","XOM","COP"])
    # EQ31 climate x Utilities
    c("EQ31 climate x Utilities", gq.companies_by_topic_and_sector("climate change","Utilities"),
      "DUK,NEE,SO", ["DUK","NEE","SO"])
    # EQ32 GLP-1 x Health Care
    c("EQ32 GLP-1 x Health Care", gq.companies_by_topic_and_sector("GLP-1","Health Care"),
      "ABBV,AMGN,CVS,ISRG,LLY,MDT,MRK,PFE", ["ABBV","AMGN","CVS","ISRG","LLY","MDT","MRK","PFE"])
    # EQ36 KPMG x Consumer Staples
    c("EQ36 KPMG x Consumer Staples", gq.companies_by_auditor_and_sector(KPMG,"Consumer Staples"),
      "PEP,COST", ["PEP","COST"])
    # EQ41 Deloitte x Health Care
    c("EQ41 Deloitte x Health Care", gq.companies_by_auditor_and_sector(DEL,"Health Care"),
      "BMY,UNH", ["BMY","UNH"])
    # EQ38 TSMC x E&Y
    c("EQ38 TSMC x E&Y", gq.companies_by_entity_and_auditor("TSMC","NAMES_FOUNDRY",EY),
      "AMD,INTC", ["AMD","INTC"])
    # EQ39 TSMC x PwC
    c("EQ39 TSMC x PwC", gq.companies_by_entity_and_auditor("TSMC","NAMES_FOUNDRY",PWC),
      "AVGO,NVDA,QCOM", ["AVGO","NVDA","QCOM"])
    # EQ43 NVIDIA-competitor x PwC
    c("EQ43 NVIDIA-competitor x PwC", gq.companies_by_entity_and_auditor("NVIDIA","NAMES_COMPETITOR",PWC),
      "CSCO,QCOM", ["CSCO","QCOM"])

    print("\n--- EQ44 auditor counts (key: E&Y 35 / PwC 29 / Deloitte 25 / KPMG 11) ---")
    print("   ", gq.auditor_counts())

    print(f"\n{ok}/{n} Tier-C traversals contain the full official key set.")


if __name__ == "__main__":
    main()
