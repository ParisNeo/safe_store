"""
Example demonstrating SafeStore's W3C SPARQL 1.1 Engine, TBox/ABox Management,
and Declarative Tabular-to-Graph Mapping (CSV / SQLite tables to RDF Knowledge Graphs).

Covers:
1. Loading and inspecting an OWL/RDFS TBox domain ontology.
2. Declaratively mapping CSV and SQLite tables into an interconnected RDF ABox graph.
3. Executing W3C SPARQL 1.1 SELECT queries (Multi-hop relational joins).
4. Executing W3C SPARQL 1.1 ASK queries (Boolean verification).
5. Executing W3C SPARQL 1.1 CONSTRUCT queries (Knowledge graph transformation / inference).
6. Executing W3C SPARQL 1.1 DESCRIBE queries (Resource inspection).
"""

from pathlib import Path
import json
import shutil
import sqlite3

from safe_store import SafeStore, GraphStore, TBoxManager, TabularMapper, LogLevel


def cleanup_demo_files(db_file: str, work_dir: Path):
    """Cleans up database and workspace directory artifacts."""
    for ext in ["", ".lock", "-wal", "-shm"]:
        p = Path(f"{db_file}{ext}")
        p.unlink(missing_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def create_sample_sqlite_source(sqlite_path: Path):
    """Creates a sample source SQLite database to demonstrate map_sqlite_table."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE legacy_suppliers (
            supplier_id TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            country TEXT NOT NULL,
            rating REAL NOT NULL
        )
    """)
    cursor.executemany(
        "INSERT INTO legacy_suppliers VALUES (?, ?, ?, ?)",
        [
            ("SUP-01", "Acme Robotics", "France", 4.9),
            ("SUP-02", "CyberShield Labs", "Germany", 4.7),
            ("SUP-03", "Hyperion Dynamics", "Switzerland", 4.8),
        ]
    )
    conn.commit()
    conn.close()


def main():
    db_file = "sparql_ontology_demo.db"
    work_dir = Path("temp_sparql_demo")
    cleanup_demo_files(db_file, work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print(" SafeStore SPARQL 1.1 Engine & Ontology Knowledge Base Demo ")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Define and Load Domain TBox Ontology (Turtle Format)
    # -------------------------------------------------------------------------
    ontology_ttl = """
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix ex: <http://example.org/ontology/> .

    ex:Agent a owl:Class .
    ex:Person a owl:Class ; rdfs:subClassOf ex:Agent .
    ex:Organization a owl:Class ; rdfs:subClassOf ex:Agent .
    ex:Company a owl:Class ; rdfs:subClassOf ex:Organization .
    ex:Project a owl:Class .

    ex:worksFor a owl:ObjectProperty ;
        rdfs:domain ex:Person ;
        rdfs:range ex:Company .

    ex:leadsProject a owl:ObjectProperty ;
        rdfs:domain ex:Person ;
        rdfs:range ex:Project .

    ex:hasName a owl:DatatypeProperty ;
        rdfs:domain ex:Agent ;
        rdfs:range xsd:string .

    ex:hasRole a owl:DatatypeProperty ;
        rdfs:domain ex:Person ;
        rdfs:range xsd:string .

    ex:hasBudget a owl:DatatypeProperty ;
        rdfs:domain ex:Project ;
        rdfs:range xsd:decimal .
    """
    ontology_path = work_dir / "domain.ttl"
    ontology_path.write_text(ontology_ttl, encoding="utf-8")

    tbox = TBoxManager()
    tbox.load_ontology(ontology_path, format="turtle")

    print(f"\n[Step 1] Loaded TBox Schema ({len(tbox.get_classes())} Classes):")
    for cls in tbox.get_classes():
        print(f"  - Class: {cls}")

    # -------------------------------------------------------------------------
    # 2. Prepare Sample Tabular CSV Data
    # -------------------------------------------------------------------------
    csv_content = """person_id,full_name,role,company_id,company_name,project_id,project_name,budget
P1,Alice Smith,Principal Architect,C1,Acme Robotics,PRJ-901,Autonomous Drone,1500000
P2,Bob Jones,Lead Scientist,C1,Acme Robotics,PRJ-902,Vision AI Engine,850000
P3,Charlie Brown,Security Lead,C2,CyberShield Labs,PRJ-903,Quantum Firewall,2200000
"""
    csv_path = work_dir / "enterprise_data.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    # Prepare external SQLite database table
    sqlite_source_path = work_dir / "external_source.db"
    create_sample_sqlite_source(sqlite_source_path)

    # -------------------------------------------------------------------------
    # 3. Initialize SafeStore and Ingest Data via TabularMapper
    # -------------------------------------------------------------------------
    store = SafeStore(db_path=db_file, vectorizer_name="st", log_level=LogLevel.INFO)

    with store:
        mapper = TabularMapper(store=store, tbox=tbox)

        # Declarative CSV Mapping Rules
        csv_mapping_rules = {
            "entity_mappings": [
                {
                    "class": "http://example.org/ontology/Person",
                    "subject_template": "http://example.org/person/{person_id}",
                    "properties": {
                        "full_name": "http://example.org/ontology/hasName",
                        "role": "http://example.org/ontology/hasRole"
                    }
                },
                {
                    "class": "http://example.org/ontology/Company",
                    "subject_template": "http://example.org/company/{company_id}",
                    "properties": {
                        "company_name": "http://example.org/ontology/hasName"
                    }
                },
                {
                    "class": "http://example.org/ontology/Project",
                    "subject_template": "http://example.org/project/{project_id}",
                    "properties": {
                        "project_name": "http://example.org/ontology/hasName",
                        "budget": "http://example.org/ontology/hasBudget"
                    }
                }
            ],
            "relationship_mappings": [
                {
                    "predicate": "http://example.org/ontology/worksFor",
                    "source_template": "http://example.org/person/{person_id}",
                    "target_template": "http://example.org/company/{company_id}"
                },
                {
                    "predicate": "http://example.org/ontology/leadsProject",
                    "source_template": "http://example.org/person/{person_id}",
                    "target_template": "http://example.org/project/{project_id}"
                }
            ]
        }

        csv_summary = mapper.map_csv(file_path=csv_path, mapping_rules=csv_mapping_rules)
        print(f"\n[Step 2] Mapped CSV -> ABox RDF Triples: {csv_summary}")

        # Declarative SQLite Mapping Rules
        sqlite_mapping_rules = {
            "entity_mappings": [
                {
                    "class": "http://example.org/ontology/Company",
                    "subject_template": "http://example.org/supplier/{supplier_id}",
                    "properties": {
                        "company_name": "http://example.org/ontology/hasName"
                    }
                }
            ]
        }
        sqlite_summary = mapper.map_sqlite_table(
            db_path=sqlite_source_path,
            table_name="legacy_suppliers",
            mapping_rules=sqlite_mapping_rules
        )
        print(f"[Step 3] Mapped SQLite Table -> ABox RDF Triples: {sqlite_summary}")

        # ---------------------------------------------------------------------
        # 4. SPARQL 1.1 SELECT: Multi-Hop Relational Traversal
        # -------------------------------------------------------------------------
        graph_store = GraphStore(store=store)

        print("\n" + "-" * 70)
        print("[SPARQL 1.1 SELECT] Multi-Hop Join (Person -> Role -> Company -> Project -> Budget):")
        print("-" * 70)
        sparql_select = """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?personName ?role ?companyName ?projectName ?budget WHERE {
            ?person a ex:Person ;
                    ex:hasName ?personName ;
                    ex:hasRole ?role ;
                    ex:worksFor ?company ;
                    ex:leadsProject ?project .
            ?company ex:hasName ?companyName .
            ?project ex:hasName ?projectName ;
                     ex:hasBudget ?budget .
        }
        """
        select_results = graph_store.query_sparql(sparql_select)
        for b in select_results["results"]["bindings"]:
            print(f"  • {b['personName']['value']} ({b['role']['value']}) @ {b['companyName']['value']}")
            print(f"    Project: {b['projectName']['value']} (Budget: ${float(b['budget']['value']):,.2f})")

        # ---------------------------------------------------------------------
        # 5. SPARQL 1.1 ASK: Boolean Constraint Validation
        # -------------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("[SPARQL 1.1 ASK] Verification: Does Acme Robotics lead an Autonomous Drone project?")
        print("-" * 70)
        sparql_ask = """
        PREFIX ex: <http://example.org/ontology/>
        ASK {
            ?person ex:worksFor ?company ;
                    ex:leadsProject ?project .
            ?company ex:hasName "Acme Robotics" .
            ?project ex:hasName "Autonomous Drone" .
        }
        """
        ask_result = graph_store.query_sparql(sparql_ask)
        print(f"  Boolean Answer: {ask_result['boolean']}")

        # ---------------------------------------------------------------------
        # 6. SPARQL 1.1 CONSTRUCT: Graph Subgraph Transformation
        # -------------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("[SPARQL 1.1 CONSTRUCT] Inferring Direct Company-to-Project Sponsorship Triples:")
        print("-" * 70)
        sparql_construct = """
        PREFIX ex: <http://example.org/ontology/>
        CONSTRUCT {
            ?company ex:sponsorsProject ?project .
        }
        WHERE {
            ?person ex:worksFor ?company ;
                    ex:leadsProject ?project .
        }
        """
        construct_result = graph_store.query_sparql(sparql_construct)
        for triple in construct_result["triples"]:
            subj = triple['subject']['value'].split('/')[-1]
            pred = triple['predicate']['value'].split('/')[-1]
            obj = triple['object']['value'].split('/')[-1]
            print(f"  Constructed Triple: ({subj}) --[{pred}]--> ({obj})")

        # ---------------------------------------------------------------------
        # 7. SPARQL 1.1 DESCRIBE: Resource Introspection
        # -------------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("[SPARQL 1.1 DESCRIBE] Inspecting Triples Related to Alice Smith:")
        print("-" * 70)
        sparql_describe = """
        PREFIX ex: <http://example.org/ontology/>
        DESCRIBE ?person WHERE {
            ?person a ex:Person ;
                    ex:hasName "Alice Smith" .
        }
        """
        describe_result = graph_store.query_sparql(sparql_describe)
        for triple in describe_result["triples"]:
            print(f"  • {triple['subject']['value']} -> {triple['predicate']['value']} -> {triple['object']['value']}")

    # Final cleanup
    store.close()
    cleanup_demo_files(db_file, work_dir)
    print("\n" + "=" * 70)
    print(" SPARQL and Ontology Knowledge Base demo completed successfully. ")
    print("=" * 70)


if __name__ == "__main__":
    main()