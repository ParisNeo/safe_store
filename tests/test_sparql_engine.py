import pytest
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock

from safe_store import SafeStore, LogLevel
from safe_store.graph.graph_store import GraphStore
from safe_store.core.exceptions import QueryError, GraphError


@pytest.fixture
def store_with_graph(tmp_path: Path) -> SafeStore:
    """Provides a SafeStore instance initialized with sample documents and graph support."""
    db_path = tmp_path / "test_sparql.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=100,
        chunk_overlap=10,
        log_level=LogLevel.DEBUG
    )
    return store


@pytest.fixture
def populated_graph_store(store_with_graph: SafeStore) -> GraphStore:
    """
    Sets up a GraphStore with known RDF triples / nodes and relationships:
    - Alice (Person) worksFor AcmeCorp (Company)
    - Bob (Person) worksFor AcmeCorp (Company)
    - AcmeCorp locatedIn Paris (City)
    - Alice knows Bob
    - Alice age 30, Bob age 25
    """
    mock_llm = MagicMock(return_value='{"nodes": [], "relationships": []}')
    graph_store = GraphStore(store=store_with_graph, llm_executor_callback=mock_llm)

    # Insert test data into graph
    alice_id = graph_store.add_node("Person", {
        "identifying_value": "Alice",
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com"
    })
    bob_id = graph_store.add_node("Person", {
        "identifying_value": "Bob",
        "name": "Bob",
        "age": 25,
        "email": "bob@example.com"
    })
    acme_id = graph_store.add_node("Company", {
        "identifying_value": "AcmeCorp",
        "name": "AcmeCorp",
        "industry": "Technology"
    })
    paris_id = graph_store.add_node("City", {
        "identifying_value": "Paris",
        "name": "Paris",
        "country": "France"
    })

    graph_store.add_relationship(alice_id, acme_id, "worksFor", {"role": "Lead Engineer"})
    graph_store.add_relationship(bob_id, acme_id, "worksFor", {"role": "Data Scientist"})
    graph_store.add_relationship(acme_id, paris_id, "locatedIn", {"since": 2010})
    graph_store.add_relationship(alice_id, bob_id, "knows", {"closeness": "colleague"})

    return graph_store


class TestSparqlSelectQueries:

    def test_sparql_select_simple_pattern(self, populated_graph_store: GraphStore):
        """Test a basic SELECT query matching subjects, predicates, and objects."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person ?company WHERE {
            ?person ex:worksFor ?company .
        }
        """
        result = populated_graph_store.query_sparql(query)
        
        assert "head" in result
        assert "vars" in result["head"]
        assert set(result["head"]["vars"]) == {"person", "company"}
        
        bindings = result["results"]["bindings"]
        assert len(bindings) == 2
        
        persons = {b["person"]["value"] for b in bindings}
        companies = {b["company"]["value"] for b in bindings}
        
        assert "Alice" in str(persons) or any("Alice" in str(v) for v in persons)
        assert "Bob" in str(persons) or any("Bob" in str(v) for v in persons)
        assert "AcmeCorp" in str(companies) or any("AcmeCorp" in str(v) for v in companies)

    def test_sparql_select_two_hop_join(self, populated_graph_store: GraphStore):
        """Test multi-hop relational join: Person -> worksFor -> Company -> locatedIn -> City."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person ?city WHERE {
            ?person ex:worksFor ?company .
            ?company ex:locatedIn ?city .
        }
        """
        result = populated_graph_store.query_sparql(query)
        bindings = result["results"]["bindings"]
        
        assert len(bindings) == 2
        for binding in bindings:
            assert "person" in binding
            assert "city" in binding
            assert "Paris" in str(binding["city"]["value"])

    def test_sparql_select_with_filter_equality(self, populated_graph_store: GraphStore):
        """Test SELECT query with FILTER clause."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person WHERE {
            ?person ex:worksFor ?company .
            FILTER(?person = "Alice" || CONTAINS(STR(?person), "Alice"))
        }
        """
        result = populated_graph_store.query_sparql(query)
        bindings = result["results"]["bindings"]
        
        assert len(bindings) == 1
        assert "Alice" in str(bindings[0]["person"]["value"])

    def test_sparql_select_optional_clause(self, populated_graph_store: GraphStore):
        """Test SELECT query with OPTIONAL pattern."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?person ?friend WHERE {
            ?person ex:worksFor ?company .
            OPTIONAL { ?person ex:knows ?friend }
        }
        """
        result = populated_graph_store.query_sparql(query)
        bindings = result["results"]["bindings"]
        
        assert len(bindings) >= 2
        # Alice knows Bob, but Bob may not have outgoing knows
        alice_bindings = [b for b in bindings if "Alice" in str(b["person"]["value"])]
        assert len(alice_bindings) == 1
        assert "friend" in alice_bindings[0]

    def test_sparql_select_aggregation_count(self, populated_graph_store: GraphStore):
        """Test aggregation COUNT with GROUP BY."""
        query = """
        PREFIX ex: <http://example.org/>
        SELECT ?company (COUNT(?person) AS ?employeeCount) WHERE {
            ?person ex:worksFor ?company .
        }
        GROUP BY ?company
        """
        result = populated_graph_store.query_sparql(query)
        bindings = result["results"]["bindings"]
        
        assert len(bindings) == 1
        assert "employeeCount" in bindings[0]
        assert int(bindings[0]["employeeCount"]["value"]) == 2


class TestSparqlAskQueries:

    def test_sparql_ask_true(self, populated_graph_store: GraphStore):
        """Test ASK query returning True for existing pattern."""
        query = """
        PREFIX ex: <http://example.org/>
        ASK {
            ?person ex:worksFor ?company .
            ?company ex:locatedIn ?city .
        }
        """
        result = populated_graph_store.query_sparql(query)
        assert isinstance(result, dict)
        assert result.get("boolean") is True

    def test_sparql_ask_false(self, populated_graph_store: GraphStore):
        """Test ASK query returning False for non-existent pattern."""
        query = """
        PREFIX ex: <http://example.org/>
        ASK {
            ?person ex:worksFor ?company .
            ?company ex:locatedIn <http://example.org/city/Tokyo> .
        }
        """
        result = populated_graph_store.query_sparql(query)
        assert isinstance(result, dict)
        assert result.get("boolean") is False


class TestSparqlConstructQueries:

    def test_sparql_construct_subgraph(self, populated_graph_store: GraphStore):
        """Test CONSTRUCT query creating a transformed RDF graph."""
        query = """
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        CONSTRUCT {
            ?person foaf:workplaceHomepage ?company .
        }
        WHERE {
            ?person ex:worksFor ?company .
        }
        """
        result = populated_graph_store.query_sparql(query)
        
        assert "graph" in result or "triples" in result
        triples = result.get("graph") or result.get("triples")
        assert len(triples) == 2
        
        predicates = [str(t.get("predicate", {}).get("value", t.get("predicate", ""))) for t in triples]
        assert any("workplaceHomepage" in p for p in predicates)


class TestSparqlDescribeQueries:

    def test_sparql_describe_node(self, populated_graph_store: GraphStore):
        """Test DESCRIBE query returning all associated triples for a resource."""
        query = """
        PREFIX ex: <http://example.org/>
        DESCRIBE ?company WHERE {
            ?person ex:worksFor ?company .
            FILTER(CONTAINS(STR(?company), "AcmeCorp"))
        }
        """
        result = populated_graph_store.query_sparql(query)
        
        assert "graph" in result or "triples" in result
        triples = result.get("graph") or result.get("triples")
        assert len(triples) > 0


class TestSparqlSyntaxAndValidation:

    def test_sparql_invalid_syntax_raises_error(self, populated_graph_store: GraphStore):
        """Test that invalid SPARQL syntax raises a QueryError."""
        invalid_query = "SELECT INVALID SYNTAX FROM NOWHERE {"
        with pytest.raises(QueryError):
            populated_graph_store.query_sparql(invalid_query)