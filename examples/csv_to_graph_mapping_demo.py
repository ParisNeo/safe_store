"""
Example demonstrating SafeStore's Dual CSV-to-Graph Ingestion Pipelines:

1. Workflow A: Data CSV + Mapping Specification (Manual / File-based)
   - Ingests CSV using declarative mapping rules or mapping CSV table.
2. Workflow B: Data CSV + Autonomous AI Mapping Proposal (LLM)
   - Inspects CSV samples, synthesizes mapping rules using the LLM,
     verifies the rules, and executes ingestion.
3. Visualizes TBox Schema vs ABox Content.
"""

from pathlib import Path
import json
from safe_store import SafeStore, GraphStore, TBoxManager, TabularMapper, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "csv_mapping_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore Tabular CSV-to-Graph & TBox/ABox Separation Demo ")
    print("=" * 80)

    # 1. Prepare Sample Data CSV
    sample_csv_text = """emp_id,full_name,job_title,dept_id,dept_name,office_city
E101,Alice Smith,Chief Architect,D01,Core Systems,Zurich
E102,Bob Jones,Senior Scientist,D01,Core Systems,Zurich
E103,Carol White,Security Officer,D02,Cyber Shield,Paris
"""
    csv_file = Path("employees_sample.csv")
    csv_file.write_text(sample_csv_text, encoding='utf-8')

    store = SafeStore(db_path=db_file, vectorizer_name="st", log_level=LogLevel.INFO)

    with store:
        graph = GraphStore(store=store)

        # ---------------------------------------------------------------------
        # Workflow 1: AI-Assisted Mapping Proposal from Data CSV
        # ---------------------------------------------------------------------
        print("\n[Step 1] Synthesizing Mapping Rules via AI from CSV Data Sample...")
        mapper = TabularMapper(store=store, tbox=graph.tbox)

        # Ingest with AI proposal (uses store's configured LLM or fallback heuristic)
        proposed_rules = mapper.propose_mapping_with_llm(
            csv_data_or_path=csv_file,
            user_guidance="Map emp_id to Person, dept_id to Company, and connect them with worksFor."
        )

        print("\nProposed Declarative Rules (Ready for User Verification):")
        print(json.dumps(proposed_rules, indent=2))

        # Apply verified rules
        print("\n[Step 2] Applying Verified Mapping to Knowledge Graph...")
        summary = mapper.map_csv(csv_file, mapping_rules=proposed_rules)
        print(f"✓ Created {summary['entities_created']} entities and {summary['triples_generated']} triples.")

        # ---------------------------------------------------------------------
        # Workflow 2: TBox vs ABox Graph Separation
        # ---------------------------------------------------------------------
        print("\n[Step 3] Inspecting Graph Scopes:")

        # ABox Graph (Instances only)
        abox = graph.get_abox_graph()
        print(f"• ABox Content (Instances): {len(abox['nodes'])} nodes, {len(abox['edges'])} edges")
        for n in abox['nodes'][:3]:
            print(f"  - [{n['category']}] {n['label']}")

        # TBox Graph (Schema only)
        tbox = graph.get_tbox_graph()
        print(f"\n• TBox Schema (Classes): {len(tbox['nodes'])} classes, {len(tbox['edges'])} ontological axioms")
        for n in tbox['nodes'][:4]:
            print(f"  - {n['label']}")

        # Unified Graph (Both connected by rdf:type)
        unified = graph.get_unified_graph()
        print(f"\n• Unified Graph: {len(unified['nodes'])} total nodes, {len(unified['edges'])} total links")

    store.close()
    cleanup_db(db_file)
    csv_file.unlink(missing_ok=True)
    print("\n" + "=" * 80)
    print(" CSV mapping and TBox/ABox separation demonstration complete! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()