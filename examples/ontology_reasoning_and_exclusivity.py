"""
Example demonstrating SafeStore's RDFS/OWL Semantic Reasoning & Logical Exclusivity Engine.

Demonstrates:
1. Subclass Transitive Inheritance (RDFS reasoning).
2. Inverse Property Generation (e.g. worksFor <-> employs).
3. Symmetric Property Inferences (e.g. collaboratesWith).
4. Transitive Property Chaining (e.g. partOf).
5. Disjointness & Exclusivity Constraint Verification (owl:disjointWith).
6. Detecting Logical Inconsistencies and Conflicts in Knowledge Graphs.
"""

from pathlib import Path
import json
from safe_store import SafeStore, GraphStore, TBoxManager, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "reasoning_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore OWL/RDFS Deductive Reasoning & Exclusivity Engine Demo ")
    print("=" * 80)

    store = SafeStore(db_path=db_file, vectorizer_name="st", log_level=LogLevel.INFO)

    with store:
        graph = GraphStore(store=store)

        # ---------------------------------------------------------------------
        # Step 1: Populate Graph with Asserted Entities and Relationships
        # ---------------------------------------------------------------------
        print("\n[Step 1] Adding Entities to Graph...")
        alice_id = graph.add_node("Person", {"name": "Alice Smith", "identifying_value": "Alice Smith"})
        acme_id = graph.add_node("Company", {"name": "Acme Robotics", "identifying_value": "Acme Robotics"})
        subsys_id = graph.add_node("Concept", {"name": "Vision Subsystem", "identifying_value": "Vision Subsystem"})
        core_id = graph.add_node("Concept", {"name": "Core Autonomy Suite", "identifying_value": "Core Autonomy Suite"})

        # Relationships
        graph.add_relationship(alice_id, acme_id, "worksFor")
        graph.add_relationship(subsys_id, core_id, "partOf")

        print("✓ Created: Alice (Person) --[worksFor]--> Acme Robotics (Company)")
        print("✓ Created: Vision Subsystem (Concept) --[partOf]--> Core Autonomy Suite (Concept)")

        # ---------------------------------------------------------------------
        # Step 2: Querying Without Reasoning vs With Reasoning (Inverse Inferences)
        # ---------------------------------------------------------------------
        print("\n[Step 2] Testing Inverse Property Inference (worksFor <-> employs):")
        # Notice we never created an 'employs' edge in Step 1!
        sparql_inverse = """
        PREFIX ont: <http://example.org/ontology/>
        SELECT ?company ?employee WHERE {
            ?company ont:employs ?employee .
        }
        """

        unreasoned = graph.query_sparql(sparql_inverse, enable_reasoning=False)
        print(f"• Query without reasoning: {len(unreasoned['results']['bindings'])} results (Edge was never explicitly stored).")

        reasoned = graph.query_sparql(sparql_inverse, enable_reasoning=True)
        print(f"• Query WITH reasoning: {len(reasoned['results']['bindings'])} result(s) inferred!")
        for b in reasoned['results']['bindings']:
            print(f"  → Inferred: {b['company']['value'].split('/')[-1]} employs {b['employee']['value'].split('/')[-1]}")

        # ---------------------------------------------------------------------
        # Step 3: Subclass Hierarchy Reasoning
        # ---------------------------------------------------------------------
        print("\n[Step 3] Testing Subclass Inheritance (Company is an Organization and Agent):")
        sparql_agents = """
        PREFIX ont: <http://example.org/ontology/>
        SELECT ?agent WHERE {
            ?agent a ont:Agent .
        }
        """
        agent_results = graph.query_sparql(sparql_agents, enable_reasoning=True)
        print(f"• Inferred instances of ont:Agent: {len(agent_results['results']['bindings'])}")
        for b in agent_results['results']['bindings']:
            print(f"  → Found Agent: {b['agent']['value'].split('/')[-1]}")

        # ---------------------------------------------------------------------
        # Step 4: Logical Consistency & Exclusivity Verification
        # ---------------------------------------------------------------------
        print("\n[Step 4] Checking Logical Consistency on Valid Graph:")
        report_valid = graph.check_consistency()
        print(f"• Is Consistent: {report_valid['is_consistent']} (Violations: {report_valid['violation_count']})")

        # ---------------------------------------------------------------------
        # Step 5: Triggering an Exclusivity / Disjointness Conflict
        # ---------------------------------------------------------------------
        print("\n[Step 5] Deliberately creating a Disjoint Class Contradiction...")
        # Core ontology defines: Person owl:disjointWith Organization
        # Let's create an entity declared as BOTH Person and Organization:
        contradictory_node_id = graph.add_node("Organization", {
            "name": "Alice Smith",
            "identifying_value": "Alice Smith" # Same identifier will fuse with Alice
        })

        print("• Checking logical consistency after conflicting assertion...")
        report_conflict = graph.check_consistency()
        print(f"• Is Consistent: {report_conflict['is_consistent']}")
        print(f"• Violations Count: {report_conflict['violation_count']}")
        for v in report_conflict['violations']:
            print(f"  ⚠️ [{v['severity']}] {v['type']}: {v['conflict']}")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" Ontology reasoning and exclusivity demonstration complete! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()