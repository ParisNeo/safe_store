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

        # Recall associative memory for Alice in the same store where the episode was recorded
        recalled = memory_store.memory.recall_associative("Alice Smith", max_hops=2)
        assert len(recalled["associated_entities"]) >= 1
        assert any("EpisodicMemory" in str(e) or "Sensor Lab" in str(e) for e in recalled["associated_entities"])

    def test_database_info_and_graph_diagnostics(self, memory_store: GraphStore):
        """Test store.get_database_info() and graph_store.get_graph_info()."""
        # Insert a sample relationship and verify diagnostics
        p_id = memory_store.add_node("Person", {"name": "Bob", "identifying_value": "Bob"})
        c_id = memory_store.add_node("Organization", {"name": "CERN", "identifying_value": "CERN"})
        memory_store.add_relationship(p_id, c_id, "AFFILIATED_WITH")

        graph_info = memory_store.get_graph_info()
        assert graph_info["total_nodes"] >= 2
        assert graph_info["total_relationships"] >= 1
        assert "Person" in graph_info["nodes_by_label"]

        db_info = memory_store.store.get_database_info()
        assert db_info["documents"]["total_documents"] >= 1
        assert db_info["documents"]["total_chunks"] >= 1
        assert db_info["knowledge_graph"]["total_nodes"] >= 2
        assert len(db_info["documents"]["list"]) >= 1
        assert db_info["documents"]["list"][0]["chunk_count"] >= 1

        # Check info alias returns identical structure
        info_alias = memory_store.store.info(print_summary=False)
        assert info_alias["store_name"] == db_info["store_name"]

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

    def test_custom_lollms_client_injection(self, tmp_path: Path):
        """Test injecting a custom lollms_client instance directly into SafeStore and GraphStore."""
        mock_client = MagicMock()
        mock_client.generate_code.return_value = '{"nodes": [{"label": "Concept", "properties": {"identifying_value": "Telemetry", "name": "Telemetry"}}], "relationships": []}'

        db_path = tmp_path / "test_client_injection.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            chunk_size=50,
            chunk_overlap=5,
            lollms_client=mock_client
        )

        assert store.lollms_client is mock_client

        # Test GraphStore inherits lollms_client from SafeStore
        graph_store = store.get_graph_store()
        assert graph_store.lollms_client is mock_client

        # Verify executor automatically routes to the client
        prompt = "Extract knowledge graph from: Telemetry controller online."
        raw_res = graph_store.llm_generator(prompt, json_mode=True)
        assert "Telemetry" in raw_res
        mock_client.generate_code.assert_called_once()

        # Test updating the client dynamically via set_lollms_client
        new_mock_client = MagicMock()
        new_mock_client.generate_code.return_value = '{"nodes": [], "relationships": []}'
        store.set_lollms_client(new_mock_client)

        assert store.lollms_client is new_mock_client
        assert graph_store.lollms_client is new_mock_client

        store.close()

    def test_custom_callable_llm_generator(self, tmp_path: Path):
        """Test providing an arbitrary custom callable generator to SafeStore and GraphStore."""
        # 1. Rich signature: (prompt, system_prompt=None, json_mode=False, **kwargs)
        recorded_calls = []

        def rich_generator(prompt: str, system_prompt: str = None, json_mode: bool = False, **kwargs) -> str:
            recorded_calls.append({"prompt": prompt, "system_prompt": system_prompt, "json_mode": json_mode})
            if json_mode:
                return '{"nodes": [{"label": "Module", "properties": {"identifying_value": "Engine", "name": "Engine"}}], "relationships": []}'
            return "SELECT ?s WHERE { ?s ?p ?o }"

        db_path = tmp_path / "test_callable_gen.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            llm_generator=rich_generator
        )

        # Ingest text so that chunk #1 exists in the database
        store.add_text("test_doc", "Test text content for chunk grounding.")

        graph = store.graph

        # Extract using the rich callable
        nodes, rels = graph._extract_and_insert_graph("Test text content", chunk_ids=[1])
        assert nodes == 1
        assert len(recorded_calls) >= 1
        assert recorded_calls[0]["json_mode"] is True
        assert recorded_calls[0]["system_prompt"] is not None

        # 2. Simple single-argument lambda: lambda prompt: ...
        lambda_called = []
        simple_lambda = lambda p: (lambda_called.append(p) or '{"is_same": true}')

        store.set_llm_generator(simple_lambda)
        res = graph.llm_generator("Test prompt")
        assert '{"is_same": true}' in res
        assert len(lambda_called) == 1

        store.close()