import pytest
from pathlib import Path
from unittest.mock import MagicMock

from safe_store import SafeStore, GraphStore, LogLevel


@pytest.fixture
def memory_store(tmp_path: Path) -> GraphStore:
    """Sets up a SafeStore with GraphStore and Cognitive Memory initialized."""
    db_path = tmp_path / "test_memory.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )

    # Ingest document chunks for grounding
    doc_text = "Dr. Alice Smith led Project Hyperion in Zurich, demonstrating quantum entanglement sensors."
    store.add_text("hyperion_brief", doc_text, metadata={"topic": "Quantum"})

    graph_store = GraphStore(store=store, llm_executor_callback=MagicMock())
    return graph_store


class TestCognitiveMemoryAndSparqlUpdate:

    def test_sparql_insert_and_select_update(self, memory_store: GraphStore):
        """Test reconstructing graph memory using SPARQL 1.1 INSERT DATA and querying via SELECT."""
        insert_cmd = """
        PREFIX ex: <http://example.org/>
        PREFIX ont: <http://example.org/ontology/>
        INSERT DATA {
            ex:Alice a ont:Person ;
                     ont:name "Alice Smith" ;
                     ont:worksOn ex:ProjectHyperion .
            ex:ProjectHyperion a ont:Project ;
                               ont:name "Project Hyperion" ;
                               ont:location "Zurich" .
        }
        """
        update_res = memory_store.execute_sparql_update(insert_cmd)
        assert update_res["status"] == "success"
        assert update_res["nodes_synchronized"] >= 2

        # Verify via SPARQL SELECT
        select_query = """
        PREFIX ont: <http://example.org/ontology/>
        SELECT ?personName ?projectName ?loc WHERE {
            ?p ont:name ?personName ;
               ont:worksOn ?proj .
            ?proj ont:name ?projectName ;
                  ont:location ?loc .
        }
        """
        res = memory_store.query_sparql(select_query)
        bindings = res["results"]["bindings"]
        assert len(bindings) == 1
        assert bindings[0]["personName"]["value"] == "Alice Smith"
        assert bindings[0]["projectName"]["value"] == "Project Hyperion"
        assert bindings[0]["loc"]["value"] == "Zurich"

    def test_record_episodic_memory_and_provenance(self, memory_store: GraphStore):
        """Test recording episodic events and linking them to participants and text chunks."""
        episode_id = memory_store.memory.record_episode(
            title="Sensor Lab Deployment",
            description="Installed quantum sensor array at Alpine testing facility.",
            participants=["Alice Smith"],
            outcome="Successful Calibration",
            source_chunk_ids=[1]
        )

        assert episode_id is not None

        # Recall associative memory for Alice
        recalled = memory_store.memory.recall_associative("Alice Smith", max_hops=2)
        assert len(recalled["associated_entities"]) >= 1
        assert any("EpisodicMemory" in str(e) or "Sensor Lab" in str(e) for e in recalled["associated_entities"])

    def test_link_individual_to_chunks_and_grounded_evidence(self, memory_store: GraphStore):
        """Test explicit chunk grounding for an entity."""
        memory_store.memory.link_individual_to_chunks("Alice Smith", [1])
        evidence = memory_store.memory.get_grounded_evidence("Alice Smith")
        
        assert len(evidence) >= 1
        assert "Project Hyperion" in evidence[0]["chunk_text"]

    def test_llm_tool_dispatching(self, memory_store: GraphStore):
        """Test LLM function calling tool dispatcher."""
        tools = memory_store.get_tool_definitions()
        assert len(tools) >= 5
        tool_names = [t["function"]["name"] for t in tools]
        assert "execute_sparql_query" in tool_names
        assert "execute_sparql_update" in tool_names
        assert "record_episodic_memory" in tool_names
        assert "recall_associative_memory" in tool_names

        # Dispatch an episodic memory creation via the tool interface
        result = memory_store.dispatch_tool("record_episodic_memory", {
            "title": "Quantum Key Protocol Review",
            "description": "Reviewed QKD security protocol draft.",
            "participants": ["Alice Smith"]
        })
        assert isinstance(result, int)