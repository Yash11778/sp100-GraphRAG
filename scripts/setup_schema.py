"""Create the fresh `sp100` graph and install the base schema on Savanna.

Non-destructive: creates a NEW graph alongside the existing Transaction_Fraud /
LegalGraph graphs; touches neither. Idempotent-ish: if sp100 already exists the
CREATE GRAPH line errors harmlessly and we proceed to (re)apply the schema.

The Chunk vector attribute is added separately at embedding-build time (once the
exact Savanna vector DDL is confirmed against the live instance).

    python scripts/setup_schema.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tg  # noqa: E402

SCHEMA_JOB = """
USE GRAPH sp100
CREATE SCHEMA_CHANGE JOB sp100_init FOR GRAPH sp100 {
  ADD VERTEX Company (PRIMARY_ID ticker STRING, name STRING, sector STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Sector (PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX AuditFirm (PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Person (PRIMARY_ID person_id STRING, name STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Entity (PRIMARY_ID name STRING, kind STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Topic (PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Document (PRIMARY_ID doc_id STRING, ticker STRING, form STRING, filing_date STRING, accession STRING, source_url STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Event (PRIMARY_ID event_id STRING, event_type STRING, value STRING, event_date STRING, summary STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Chunk (PRIMARY_ID chunk_id STRING, doc_id STRING, ticker STRING, form STRING, seq INT, text STRING) WITH primary_id_as_attribute="true";

  ADD DIRECTED EDGE IN_SECTOR (FROM Company, TO Sector);
  ADD DIRECTED EDGE AUDITED_BY (FROM Company, TO AuditFirm);
  ADD DIRECTED EDGE HAS_DIRECTOR (FROM Company, TO Person);
  ADD DIRECTED EDGE HAS_OFFICER (FROM Company, TO Person, role STRING);
  ADD DIRECTED EDGE NAMES_COMPETITOR (FROM Company, TO Entity);
  ADD DIRECTED EDGE NAMES_FOUNDRY (FROM Company, TO Entity);
  ADD DIRECTED EDGE MENTIONS_TOPIC (FROM Company, TO Topic, form STRING);
  ADD DIRECTED EDGE FILED (FROM Company, TO Document);
  ADD DIRECTED EDGE REPORTS_EVENT (FROM Document, TO Event);
  ADD DIRECTED EDGE HAS_CHUNK (FROM Document, TO Chunk);
  ADD DIRECTED EDGE CHUNK_OF (FROM Chunk, TO Company);
}
RUN SCHEMA_CHANGE JOB sp100_init
DROP JOB sp100_init
"""


def main() -> None:
    print("Graphs before:", tg.graphs())

    if "sp100" not in tg.graphs():
        print("Creating graph sp100 ...")
        print(" ", tg.gsql("CREATE GRAPH sp100()").strip()[:200])
    else:
        print("Graph sp100 already exists; applying schema.")

    print("Installing schema ...")
    out = tg.gsql(SCHEMA_JOB)
    print(out.strip()[-600:])

    print("\nGraphs after:", tg.graphs())
    print("sp100 schema:")
    print(tg.gsql("USE GRAPH sp100\nls")[:800])


if __name__ == "__main__":
    main()
