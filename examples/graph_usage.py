"""
SafeStore Knowledge Graph Engine Demo.

Demonstrates:
1. Document ingestion and automatic knowledge graph construction using LLM extraction.
2. Ontology-driven entity fusion and relational graph indexing.
3. W3C SPARQL 1.1 querying (SELECT, ASK, CONSTRUCT).
4. Natural language graph traversal with neighborhood expansion.
5. Tri-Modal Unified Hybrid Retrieval (query_graph_hybrid) combining Dense Vectors, BM25, and Graph.
6. Programmatic graph manipulation (CRUD for nodes and relationships).
"""

from pathlib import Path
import json
import shutil
import sqlite3
from typing import Dict, List, Any, Optional

import safe_store
from safe_store import GraphStore, SafeStore, LogLevel
from ascii_colors import ASCIIColors, trace_exception

# --- Configuration ---
DB_FILE = "graph_example_store.db"
DOC_DIR = Path("temp_docs_graph_example")

# --- Structured Ontology Definition ---
DETAILED_ONTOLOGY = {
    "nodes": {
        "Person": {
            "description": "A human individual, researcher, or executive.",
            "properties": {"name": "string", "title": "string", "identifying_value": "string"}
        },
        "Company": {
            "description": "A commercial organization or tech business.",
            "properties": {"name": "string", "location": "string", "identifying_value": "string"}
        },
        "Product": {
            "description": "A software platform or hardware product.",
            "properties": {"name": "string", "identifying_value": "string"}
        },
        "ResearchPaper": {
            "description": "An academic paper or scientific publication.",
            "properties": {"title": "string", "identifying_value": "string"}
        }
    },
    "relationships": {
        "WORKS_AT": {"description": "Person is employed by Company.", "source": "Person", "target": "Company"},
        "CEO_OF": {"description": "Person is the CEO of Company.", "source": "Person", "target": "Company"},
        "PRODUCES": {"description": "Company creates or sells Product.", "source": "Company", "target": "Product"},
        "AUTHOR_OF": {"description": "Person authored a ResearchPaper.", "source": "Person", "target": "ResearchPaper"},
        "COMPETITOR_OF": {"description": "Company competes with another Company.", "source": "Company", "target": "Company"}
    }
}


def print_header(title: str) -> None:
    print("\n" + "=" * 30 + f" {title} " + "=" * 30)


def cleanup() -> None:
    print_header("Cleaning Up Previous Run")
    paths = [
        Path(DB_FILE),
        Path(f"{DB_FILE}.lock"),
        Path(f"{DB_FILE}-wal"),
        Path(f"{DB_FILE}-shm"),
        DOC_DIR
    ]
    for p in paths:
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
                print(f"- Removed file: {p}")
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                print(f"- Removed directory: {p}")
        except OSError as e:
            print(f"- Warning: Could not remove {p}: {e}")


def get_llm_callback():
    """
    Returns an LLM extraction callback.
    Attempts to connect to a local Lollms/Ollama client, falling back to a deterministic
    semantic parser if no local LLM endpoint is active.
    """
    try:
        import pipmaster as pm
        pm.ensure_packages(["lollms_client"])
        from lollms_client import LollmsClient
        client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"host_address": "http://localhost:11434", "model_name": "mistral:latest"})
        if client.llm:
            ASCIIColors.success("Connected to live LollmsClient LLM backend.")
            return lambda prompt: client.generate_code(prompt, language="json", temperature=0.05)
    except Exception:
        pass

    ASCIIColors.info("Using built-in deterministic extraction engine for graph demonstration.")

    def fallback_extractor(prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Seed node and query guidance parser
        if "seed_nodes" in prompt or "identify main entities" in prompt.lower():
            if "evelyn reed" in prompt_lower:
                return json.dumps({
                    "seed_nodes": [{"label": "Person", "identifying_property_key": "name", "identifying_property_value": "Dr. Evelyn Reed"}],
                    "target_relationships": [{"type": "WORKS_AT", "direction": "any"}, {"type": "CEO_OF", "direction": "any"}],
                    "target_node_labels": ["Company", "Product"],
                    "max_depth": 2
                })
            if "acme" in prompt_lower:
                return json.dumps({
                    "seed_nodes": [{"label": "Company", "identifying_property_key": "name", "identifying_property_value": "Acme Innovations"}],
                    "target_relationships": [{"type": "PRODUCES", "direction": "any"}],
                    "target_node_labels": ["Product"],
                    "max_depth": 2
                })
            return json.dumps({"seed_nodes": []})

        # Document 1 extraction: Acme Innovations & Evelyn Reed
        if "acme innovations" in prompt_lower:
            return json.dumps({
                "nodes": [
                    {"label": "Company", "properties": {"identifying_value": "Acme Innovations", "name": "Acme Innovations", "location": "Silicon Valley"}},
                    {"label": "Person", "properties": {"identifying_value": "Dr. Evelyn Reed", "name": "Dr. Evelyn Reed", "title": "CEO"}},
                    {"label": "Person", "properties": {"identifying_value": "John Doe", "name": "John Doe", "title": "Senior Engineer"}},
                    {"label": "Product", "properties": {"identifying_value": "NovaCore", "name": "NovaCore"}},
                    {"label": "Company", "properties": {"identifying_value": "Beta Solutions", "name": "Beta Solutions"}}
                ],
                "relationships": [
                    {"source_node_label": "Person", "source_node_identifying_value": "Dr. Evelyn Reed", "target_node_label": "Company", "target_node_identifying_value": "Acme Innovations", "type": "CEO_OF"},
                    {"source_node_label": "Person", "source_node_identifying_value": "John Doe", "target_node_label": "Company", "target_node_identifying_value": "Acme Innovations", "type": "WORKS_AT"},
                    {"source_node_label": "Company", "source_node_identifying_value": "Acme Innovations", "target_node_label": "Product", "target_node_identifying_value": "NovaCore", "type": "PRODUCES"},
                    {"source_node_label": "Company", "source_node_identifying_value": "Acme Innovations", "target_node_label": "Company", "target_node_identifying_value": "Beta Solutions", "type": "COMPETITOR_OF"}
                ]
            })

        # Document 2 extraction: Research Paper & Quantum theories
        if "quantum entanglement" in prompt_lower:
            return json.dumps({
                "nodes": [
                    {"label": "ResearchPaper", "properties": {"identifying_value": "Quantum Entanglement in Nanostructures", "title": "Quantum Entanglement in Nanostructures"}},
                    {"label": "Person", "properties": {"identifying_value": "Dr. Alice Smith", "name": "Dr. Alice Smith", "title": "Lead Researcher"}},
                    {"label": "Person", "properties": {"identifying_value": "Dr. Evelyn Reed", "name": "Dr. Evelyn Reed", "title": "Consultant"}}
                ],
                "relationships": [
                    {"source_node_label": "Person", "source_node_identifying_value": "Dr. Alice Smith", "target_node_label": "ResearchPaper", "target_node_identifying_value": "Quantum Entanglement in Nanostructures", "type": "AUTHOR_OF"}
                ]
            })

        return json.dumps({"nodes": [], "relationships": []})

    return fallback_extractor


def main():
    cleanup()
    ASCIIColors.set_log_level(LogLevel.INFO)

    print_header("Phase 1: Preparing Documents & Building Vector Base")
    DOC_DIR.mkdir(exist_ok=True, parents=True)

    doc1_content = (
        "Acme Innovations, led by CEO Dr. Evelyn Reed, is a premier tech company based in Silicon Valley. "
        "Their flagship product, 'NovaCore', was launched in 2023 for AI acceleration. "
        "John Doe works as a Senior Engineer at Acme Innovations and reports to Dr. Reed. "
        "Acme Innovations is a key competitor of Beta Solutions."
    )
    (DOC_DIR / "company_info.txt").write_text(doc1_content.strip(), encoding="utf-8")

    doc2_content = (
        "The research paper 'Quantum Entanglement in Nanostructures' by Dr. Alice Smith cites foundational work "
        "by Dr. Evelyn Reed on early quantum theories. Dr. Reed is widely known for her leadership at Acme Innovations."
    )
    (DOC_DIR / "research_paper.txt").write_text(doc2_content.strip(), encoding="utf-8")

    llm_callback = get_llm_callback()

    with SafeStore(db_path=DB_FILE, vectorizer_name="st", log_level=LogLevel.INFO) as store:
        # 1. Ingest Documents into SafeStore
        store.add_document(DOC_DIR / "company_info.txt", metadata={"source": "Enterprise Docs"})
        store.add_document(DOC_DIR / "research_paper.txt", metadata={"source": "Academic Press"})

        # 2. Initialize Knowledge Graph Store
        print_header("Phase 2: Extracting Knowledge Graph from Documents")
        graph_store = GraphStore(
            store=store,
            llm_executor_callback=llm_callback,
            ontology=DETAILED_ONTOLOGY
        )

        build_stats = graph_store.build_graph_for_all_documents()
        ASCIIColors.success(f"Graph build completed: {build_stats}")

        # 3. Inspect Extracted Graph
        all_nodes = graph_store.get_all_nodes(limit=20)
        all_rels = graph_store.get_all_relationships(limit=20)
        print(f"\n- Extracted {len(all_nodes)} Nodes:")
        for n in all_nodes:
            print(f"  • [{n['label']}] ID={n['node_id']}: {n['properties']}")

        print(f"\n- Extracted {len(all_rels)} Relationships:")
        for r in all_rels:
            print(f"  • (ID:{r['source_node_id']}) --[{r['type']}]--> (ID:{r['target_node_id']})")

        # 4. W3C SPARQL 1.1 Queries
        print_header("Phase 3: W3C SPARQL 1.1 Multi-Hop Queries")

        # SPARQL SELECT
        sparql_select = """
        PREFIX ex: <http://example.org/ontology/>
        SELECT ?person ?role ?company ?product WHERE {
            ?p ex:CEO_OF ?c ;
               ex:name ?person ;
               ex:title ?role .
            ?c ex:name ?company ;
               ex:PRODUCES ?prod .
            ?prod ex:name ?product .
        }
        """
        print("Executing SPARQL SELECT query:")
        res_select = graph_store.query_sparql(sparql_select)
        for b in res_select["results"]["bindings"]:
            print(f"  Result: {b['person']['value']} ({b['role']['value']}) -> {b['company']['value']} produces {b['product']['value']}")

        # SPARQL ASK (Boolean validation)
        sparql_ask = """
        PREFIX ex: <http://example.org/ontology/>
        ASK {
            ?c ex:name "Acme Innovations" ;
               ex:COMPETITOR_OF ?competitor .
        }
        """
        res_ask = graph_store.query_sparql(sparql_ask)
        print(f"\nSPARQL ASK (Does Acme Innovations have registered competitors?): {res_ask['boolean']}")

        # SPARQL CONSTRUCT (Subgraph Transformation)
        sparql_construct = """
        PREFIX ex: <http://example.org/ontology/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        CONSTRUCT {
            ?person foaf:workplaceHomepage ?company .
        }
        WHERE {
            ?person ex:CEO_OF ?company .
        }
        """
        res_construct = graph_store.query_sparql(sparql_construct)
        print(f"\nSPARQL CONSTRUCT generated {len(res_construct['triples'])} new triples:")
        for t in res_construct["triples"]:
            print(f"  Constructed: {t['subject']['value']} -> {t['predicate']['value']} -> {t['object']['value']}")

        # 5. Natural Language Graph Traversal Query
        print_header("Phase 4: Natural Language Graph Query")
        nl_query = "Who is Dr. Evelyn Reed and what companies is she associated with?"
        print(f"Query: '{nl_query}'")
        nl_res = graph_store.query_graph(nl_query, output_mode="full")
        print(f"Discovered Subgraph Nodes: {len(nl_res['graph']['nodes'])}")
        print(f"Discovered Subgraph Relationships: {len(nl_res['graph']['relationships'])}")
        print(f"Linked Provenance Chunks: {len(nl_res['chunks'])}")

        # 6. Tri-Modal Unified Hybrid Retrieval (query_graph_hybrid)
        print_header("Phase 5: Tri-Modal Hybrid Retrieval (Dense + BM25 + Graph RRF)")
        hybrid_query = "NovaCore AI acceleration flagship product details"
        print(f"Hybrid Query: '{hybrid_query}'")
        hybrid_response = graph_store.query_graph_hybrid(
            query_text=hybrid_query,
            top_k=3,
            dense_weight=0.4,
            bm25_weight=0.3,
            graph_weight=0.3
        )

        for i, chunk in enumerate(hybrid_response["ranked_chunks"], 1):
            print(f"\n  Rank {i} (Fused Score: {chunk.get('fused_score', 0):.4f}):")
            print(f"  Document: {chunk.get('file_path')}")
            print(f"  Chunk Preview: {chunk.get('chunk_text', '')[:100]}...")

        # 7. Programmatic Graph Manipulation (CRUD)
        print_header("Phase 6: Programmatic Graph Manipulation")
        acme_node = next((n for n in all_nodes if "acme" in n["properties"].get("name", "").lower()), None)
        if acme_node:
            acme_id = acme_node["node_id"]
            # Add new product node
            new_prod_id = graph_store.add_node("Product", {"identifying_value": "ChronoLeap", "name": "ChronoLeap", "version": "2.0"})
            rel_id = graph_store.add_relationship(acme_id, new_prod_id, "PRODUCES", {"launched": 2024})
            ASCIIColors.success(f"Added product ChronoLeap (Node ID: {new_prod_id}) with relationship ID: {rel_id}")

            # Discover neighbors
            neighbors = graph_store.find_neighbors(acme_id, direction="outgoing")
            print(f"\nOutgoing neighbors of {acme_node['properties'].get('name')}:")
            for nb in neighbors:
                print(f"  -> [{nb['label']}] {nb['properties'].get('name', nb['properties'])}")

    print_header("Graph Usage Demo Completed Successfully")
    cleanup()


if __name__ == "__main__":
    main()