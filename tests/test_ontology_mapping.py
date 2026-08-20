import pytest
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock

from safe_store import SafeStore, LogLevel
from safe_store.graph.ontology.tbox import TBoxManager
from safe_store.graph.mapping.tabular_mapper import TabularMapper
from safe_store.core.exceptions import ConfigurationError, SafeStoreError


SAMPLE_TURTLE_ONTOLOGY = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.org/ontology/> .

ex:Agent a owl:Class .
ex:Person a owl:Class ;
    rdfs:subClassOf ex:Agent .
ex:Organization a owl:Class ;
    rdfs:subClassOf ex:Agent .
ex:Company a owl:Class ;
    rdfs:subClassOf ex:Organization .

ex:worksFor a owl:ObjectProperty ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Organization .

ex:hasName a owl:DatatypeProperty ;
    rdfs:domain ex:Agent ;
    rdfs:range xsd:string .

ex:hasRevenue a owl:DatatypeProperty ;
    rdfs:domain ex:Company ;
    rdfs:range xsd:float .
"""


@pytest.fixture
def sample_ontology_file(tmp_path: Path) -> Path:
    ttl_path = tmp_path / "domain_ontology.ttl"
    ttl_path.write_text(SAMPLE_TURTLE_ONTOLOGY, encoding="utf-8")
    return ttl_path


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    csv_content = (
        "id,name,role,company_id,company_name,revenue\n"
        "1,Alice Smith,Engineer,C101,Acme Robotics,5000000\n"
        "2,Bob Jones,Analyst,C101,Acme Robotics,5000000\n"
        "3,Charlie Brown,Director,C102,Beta Labs,1200000\n"
    )
    csv_path = tmp_path / "employees.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture
def safe_store_with_graph(tmp_path: Path) -> SafeStore:
    db_path = tmp_path / "test_ontology_store.db"
    return SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=100,
        chunk_overlap=10,
        log_level=LogLevel.DEBUG
    )


class TestTBoxManager:

    def test_load_ontology_from_turtle(self, sample_ontology_file: Path):
        """Test loading Turtle ontology and introspecting classes and properties."""
        tbox = TBoxManager()
        tbox.load_ontology(sample_ontology_file, format="turtle")

        classes = tbox.get_classes()
        assert "http://example.org/ontology/Person" in classes
        assert "http://example.org/ontology/Company" in classes
        assert "http://example.org/ontology/Organization" in classes

        # Test subclass hierarchy
        subclasses_of_agent = tbox.get_subclasses("http://example.org/ontology/Agent", recursive=True)
        assert "http://example.org/ontology/Person" in subclasses_of_agent
        assert "http://example.org/ontology/Organization" in subclasses_of_agent
        assert "http://example.org/ontology/Company" in subclasses_of_agent

    def test_get_domain_and_range(self, sample_ontology_file: Path):
        """Test property domain and range inspection."""
        tbox = TBoxManager()
        tbox.load_ontology(sample_ontology_file, format="turtle")

        domain, range_ = tbox.get_property_domain_range("http://example.org/ontology/worksFor")
        assert domain == "http://example.org/ontology/Person"
        assert range_ == "http://example.org/ontology/Organization"

    def test_export_ontology_to_llm_prompt_schema(self, sample_ontology_file: Path):
        """Test serializing TBox into structured guidance for LLM graph extraction."""
        tbox = TBoxManager()
        tbox.load_ontology(sample_ontology_file, format="turtle")

        schema_repr = tbox.to_prompt_schema()
        assert "Person" in schema_repr
        assert "worksFor" in schema_repr
        assert "Company" in schema_repr


class TestTabularMapper:

    def test_map_csv_to_rdf_abox(self, safe_store_with_graph: SafeStore, sample_csv_file: Path, sample_ontology_file: Path):
        """Test mapping a CSV file into ABox RDF triples and inserting into SafeStore."""
        tbox = TBoxManager()
        tbox.load_ontology(sample_ontology_file, format="turtle")

        mapping_rules = {
            "entity_mappings": [
                {
                    "class": "http://example.org/ontology/Person",
                    "subject_template": "http://example.org/person/{id}",
                    "properties": {
                        "name": "http://example.org/ontology/hasName"
                    }
                },
                {
                    "class": "http://example.org/ontology/Company",
                    "subject_template": "http://example.org/company/{company_id}",
                    "properties": {
                        "company_name": "http://example.org/ontology/hasName",
                        "revenue": "http://example.org/ontology/hasRevenue"
                    }
                }
            ],
            "relationship_mappings": [
                {
                    "predicate": "http://example.org/ontology/worksFor",
                    "source_template": "http://example.org/person/{id}",
                    "target_template": "http://example.org/company/{company_id}"
                }
            ]
        }

        mapper = TabularMapper(store=safe_store_with_graph, tbox=tbox)
        result = mapper.map_csv(file_path=sample_csv_file, mapping_rules=mapping_rules)

        assert result["triples_generated"] > 0
        assert result["entities_created"] >= 5  # 3 Persons + 2 Companies

        # Verify through SPARQL
        from safe_store.graph.graph_store import GraphStore
        graph_store = GraphStore(store=safe_store_with_graph, llm_executor_callback=MagicMock())
        
        sparql_query = """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?personName ?companyName WHERE {
            ?person a ex:Person ;
                    ex:hasName ?personName ;
                    ex:worksFor ?company .
            ?company ex:hasName ?companyName .
        }
        """
        res = graph_store.query_sparql(sparql_query)
        bindings = res["results"]["bindings"]
        
        assert len(bindings) == 3
        names = {b["personName"]["value"] for b in bindings}
        assert "Alice Smith" in names
        assert "Bob Jones" in names
        assert "Charlie Brown" in names

    def test_map_sqlite_table_to_rdf(self, safe_store_with_graph: SafeStore, sample_ontology_file: Path, tmp_path: Path):
        """Test mapping an external SQLite table directly into the ABox graph."""
        src_db_path = tmp_path / "source_data.db"
        src_conn = sqlite3.connect(src_db_path)
        src_conn.execute("CREATE TABLE products (sku TEXT PRIMARY KEY, title TEXT, price REAL);")
        src_conn.execute("INSERT INTO products VALUES ('SKU-001', 'Wireless Sensor', 49.99);")
        src_conn.execute("INSERT INTO products VALUES ('SKU-002', 'Gateway Hub', 199.99);")
        src_conn.commit()
        src_conn.close()

        tbox = TBoxManager()
        tbox.load_ontology(sample_ontology_file, format="turtle")

        mapping_rules = {
            "entity_mappings": [
                {
                    "class": "http://example.org/ontology/Product",
                    "subject_template": "http://example.org/product/{sku}",
                    "properties": {
                        "title": "http://example.org/ontology/hasName"
                    }
                }
            ]
        }

        mapper = TabularMapper(store=safe_store_with_graph, tbox=tbox)
        result = mapper.map_sqlite_table(
            db_path=src_db_path,
            table_name="products",
            mapping_rules=mapping_rules
        )

        assert result["triples_generated"] >= 4
        assert result["entities_created"] == 2