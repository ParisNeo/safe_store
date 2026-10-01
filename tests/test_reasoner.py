import pytest
from pathlib import Path
from safe_store import SafeStore, GraphStore, TBoxManager, SemanticReasoner, LogLevel


@pytest.fixture
def reasoning_store(tmp_path: Path) -> SafeStore:
    db_path = tmp_path / "test_reasoner.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )
    return store


class TestSemanticReasoner:

    def test_inverse_property_reasoning(self, reasoning_store: SafeStore):
        """Test that A worksFor B infers B employs A via owl:inverseOf."""
        with reasoning_store:
            graph = GraphStore(store=reasoning_store)

            alice_id = graph.add_node("Person", {"name": "Alice", "identifying_value": "Alice"})
            acme_id = graph.add_node("Company", {"name": "Acme", "identifying_value": "Acme"})
            graph.add_relationship(alice_id, acme_id, "worksFor")

            # Without reasoning: 0 rows for employs
            sparql = """
            PREFIX ont: <http://example.org/ontology/>
            SELECT ?company ?person WHERE {
                ?company ont:employs ?person .
            }
            """
            unreasoned = graph.query_sparql(sparql, enable_reasoning=False)
            assert len(unreasoned["results"]["bindings"]) == 0

            # With reasoning: 1 row inferred
            reasoned = graph.query_sparql(sparql, enable_reasoning=True)
            assert len(reasoned["results"]["bindings"]) == 1
            binding = reasoned["results"]["bindings"][0]
            assert "Acme" in binding["company"]["value"]
            assert "Alice" in binding["person"]["value"]

    def test_subclass_hierarchy_reasoning(self, reasoning_store: SafeStore):
        """Test that Company inherits Organization and Agent classes."""
        with reasoning_store:
            graph = GraphStore(store=reasoning_store)

            acme_id = graph.add_node("Company", {"name": "Acme", "identifying_value": "Acme"})

            # Query for general Agent class
            sparql = """
            PREFIX ont: <http://example.org/ontology/>
            SELECT ?agent WHERE {
                ?agent a ont:Agent .
            }
            """
            results = graph.query_sparql(sparql, enable_reasoning=True)
            assert len(results["results"]["bindings"]) >= 1
            assert any("Acme" in b["agent"]["value"] for b in results["results"]["bindings"])

    def test_symmetric_property_reasoning(self, reasoning_store: SafeStore):
        """Test that A collaboratesWith B infers B collaboratesWith A."""
        with reasoning_store:
            graph = GraphStore(store=reasoning_store)

            bob_id = graph.add_node("Person", {"name": "Bob", "identifying_value": "Bob"})
            carol_id = graph.add_node("Person", {"name": "Carol", "identifying_value": "Carol"})
            # Directed edge from Bob to Carol
            graph.add_relationship(bob_id, carol_id, "collaboratesWith")

            # Query reverse edge (Carol collaboratesWith Bob)
            sparql = """
            PREFIX ont: <http://example.org/ontology/>
            SELECT ?x ?y WHERE {
                <http://example.org/Carol> ont:collaboratesWith ?y .
            }
            """
            results = graph.query_sparql(sparql, enable_reasoning=True)
            assert len(results["results"]["bindings"]) == 1
            assert "Bob" in results["results"]["bindings"][0]["y"]["value"]

    def test_transitive_property_reasoning(self, reasoning_store: SafeStore):
        """Test that A partOf B and B partOf C infers A partOf C."""
        with reasoning_store:
            graph = GraphStore(store=reasoning_store)

            sensor = graph.add_node("Tool", {"name": "Sensor", "identifying_value": "Sensor"})
            module = graph.add_node("Tool", {"name": "Module", "identifying_value": "Module"})
            system = graph.add_node("Tool", {"name": "System", "identifying_value": "System"})

            graph.add_relationship(sensor, module, "partOf")
            graph.add_relationship(module, system, "partOf")

            sparql = """
            PREFIX ont: <http://example.org/ontology/>
            ASK {
                <http://example.org/Sensor> ont:partOf <http://example.org/System> .
            }
            """
            ask_res = graph.query_sparql(sparql, enable_reasoning=True)
            assert ask_res["boolean"] is True

    def test_disjointness_exclusivity_conflict_detection(self, reasoning_store: SafeStore):
        """Test detecting mutually exclusive class conflicts (Person vs Organization)."""
        with reasoning_store:
            graph = GraphStore(store=reasoning_store)

            # Valid graph
            report_clean = graph.check_consistency()
            assert report_clean["is_consistent"] is True
            assert report_clean["violation_count"] == 0

            # Add node and declare custom disjoint classes
            graph.tbox.declare_disjoint("http://example.org/ontology/Human", "http://example.org/ontology/Robot")
            nid = graph.add_node("Human", {"name": "Cyborg-01", "identifying_value": "Cyborg-01"})
            # Inject contradictory second type
            graph.conn.execute("INSERT INTO graph_nodes (node_label, node_properties, unique_signature) VALUES (?, ?, ?)",
                               ("Robot", '{"name": "Cyborg-01", "identifying_value": "Cyborg-01"}', "Robot:Cyborg-01"))
            graph.conn.commit()

            report_dirty = graph.check_consistency()
            assert report_dirty["is_consistent"] is False
            assert report_dirty["violation_count"] >= 1
            assert any(v["type"] == "disjoint_class_violation" for v in report_dirty["violations"])