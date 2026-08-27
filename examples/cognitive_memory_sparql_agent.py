"""
Example demonstrating LLM Cognitive Memory, SPARQL 1.1 Update Reorganization,
Episodic Event Logging, Associative Pathways, and Chunk Grounding.
"""

from pathlib import Path
from safe_store import SafeStore, GraphStore, LogLevel


def cleanup_db(db_file: str):
    import gc
    gc.collect()
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "cognitive_agent_memory.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore LLM Cognitive Memory & SPARQL 1.1 Reorganization Engine ")
    print("=" * 80)

    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=40,
        chunk_overlap=5,
        log_level=LogLevel.INFO
    )

    with store:
        # Step 1: Ingest source evidence documents
        print("\n[Step 1] Ingesting Source Text Evidence...")
        doc1 = "Dr. Elena Rostova discovered the anomalous resonance in Subsystem Beta at 04:00 UTC."
        store.add_text("telemetry_log_doc", doc1, metadata={"system": "Telemetry"})

        graph = GraphStore(store=store)

        # Step 2: LLM executes SPARQL 1.1 Update to construct initial ontology and ABox triples
        print("\n[Step 2] LLM Reorganizing Thoughts via SPARQL 1.1 UPDATE (INSERT DATA)...")
        sparql_update = """
        PREFIX ex: <http://example.org/>
        PREFIX ont: <http://example.org/ontology/>
        INSERT DATA {
            ex:ElenaRostova a ont:Scientist ;
                            ont:name "Dr. Elena Rostova" ;
                            ont:specialty "Quantum Telemetry" ;
                            ont:investigates ex:SubsystemBeta .
            ex:SubsystemBeta a ont:Subsystem ;
                             ont:name "Subsystem Beta" ;
                             ont:status "Degraded" .
        }
        """
        update_result = graph.execute_sparql_update(sparql_update)
        print(f"  • SPARQL Update Status: {update_result['status']} ({update_result['nodes_synchronized']} nodes synced)")

        # Step 3: Record an Episodic Event & Ground it to Chunk 1
        print("\n[Step 3] Recording Episodic Memory with Evidence Provenance...")
        episode_id = graph.memory.record_episode(
            title="Anomalous Resonance Incident",
            description="Elena identified irregular frequency harmonics in Subsystem Beta.",
            participants=["Dr. Elena Rostova"],
            outcome="Triggered diagnostic investigation",
            source_chunk_ids=[1]
        )
        print(f"  • Created Episode ID: {episode_id}")

        # Step 4: Associative Memory Recall
        print("\n[Step 4] Querying Associative Memory Pathways for 'Dr. Elena Rostova'...")
        associative_view = graph.memory.recall_associative("Dr. Elena Rostova", max_hops=2)
        print(f"  • Found {len(associative_view['associated_entities'])} associated entities:")
        for ent in associative_view['associated_entities']:
            props = ent.get('properties', {})
            print(f"    - [{ent.get('label')}] {props.get('name') or props.get('title')}")

        print(f"\n  • Grounded Source Chunks ({len(associative_view['grounded_chunks'])}):")
        for chk in associative_view['grounded_chunks']:
            print(f"    Evidence: \"{chk['chunk_text']}\"")

        # Step 5: LLM Tool Interface Discovery
        print("\n[Step 5] Exposing LLM Function Calling Tools:")
        tools = graph.get_tool_definitions()
        for t in tools:
            fn = t["function"]
            print(f"  • Tool: {fn['name']} - {fn['description'][:70]}...")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" Cognitive memory agent demo completed successfully. ")
    print("=" * 80)


if __name__ == "__main__":
    main()