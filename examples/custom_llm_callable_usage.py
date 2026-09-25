"""
Example demonstrating how to plug in ANY custom LLM generator callable into SafeStore.

The calling application can use ANY tool or provider (OpenAI, Ollama, Anthropic,
LangChain, vLLM, local function, or simple lambda).

Standard Callable Protocol Signature:
    def my_llm_generator(
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs: Any
    ) -> str:
        ...

Also supports simple single-argument callables:
    my_llm_generator = lambda prompt: my_model.generate(prompt)
"""

import json
from pathlib import Path
from typing import Optional, Any
from safe_store import SafeStore, GraphStore, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# 1. Example Generator: Custom Function / Dispatcher
# -----------------------------------------------------------------------------
def custom_llm_generator(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    **kwargs: Any
) -> str:
    """
    Simulates a calling application's custom LLM backend (e.g. OpenAI SDK,
    requests to an internal microservice, Ollama, or local model).
    """
    print(f"\n[Custom LLM Invoked] (json_mode={json_mode})")
    if system_prompt:
        print(f"  • System: {system_prompt[:60]}...")
    print(f"  • Prompt Preview: {prompt.strip()[:80].replace(chr(10), ' ')}...")

    prompt_lower = prompt.lower()

    # Case A: SPARQL 1.1 Generation
    if "sparql" in prompt_lower or "prefix" in prompt_lower:
        return """```sparql
PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?tool ?creator WHERE {
    ?tool a ont:Tool ;
          ont:createdBy ?creator .
}
```"""

    # Case B: Entity Fusion
    if "entity a properties" in prompt_lower:
        return json.dumps({
            "is_same": True,
            "reasoning": "Both entities share the identical canonical identifier."
        })

    # Case C: Natural Language Query Parsing
    if "seed_nodes" in prompt_lower or "identify main entities" in prompt_lower:
        return json.dumps({
            "seed_nodes": [{"label": "Tool", "identifying_property_key": "name", "identifying_property_value": "SafeStore"}],
            "target_relationships": [{"type": "USES", "direction": "any"}],
            "max_depth": 1
        })

    # Case D: Knowledge Graph Extraction
    return json.dumps({
        "nodes": [
            {
                "label": "Tool",
                "properties": {
                    "identifying_value": "SafeStore",
                    "name": "SafeStore",
                    "description": "Local sovereign vector and graph database."
                }
            },
            {
                "label": "Person",
                "properties": {
                    "identifying_value": "ParisNeo",
                    "name": "ParisNeo",
                    "role": "Creator"
                }
            }
        ],
        "relationships": [
            {
                "source_node_label": "Tool",
                "source_node_identifying_value": "SafeStore",
                "target_node_label": "Person",
                "target_node_identifying_value": "ParisNeo",
                "type": "CREATED_BY",
                "properties": {"year": 2025}
            }
        ]
    })


def main():
    db_file = "custom_callable_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore Custom LLM Callable Generator Demonstration ")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Option 1: Pass callable directly to SafeStore
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing SafeStore with custom LLM generator callable...")
    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        llm_generator=custom_llm_generator,
        chunk_size=60,
        chunk_overlap=10,
        log_level=LogLevel.INFO
    )

    with store:
        # Ingest text document
        store.add_text(
            unique_id="overview_doc",
            text="SafeStore was created by ParisNeo as an ultra-fast local database combining vectors and knowledge graphs."
        )

        # Access graph store directly from store.graph (inherits llm_generator!)
        graph = store.graph

        print("\n[Step 2] Building knowledge graph using custom LLM callable...")
        stats = graph.build_graph_for_all_documents()
        print(f"✓ Extraction complete: {stats['nodes_created']} node(s), {stats['relationships_created']} relationship(s).")

        # Execute natural language SPARQL query generation
        print("\n[Step 3] Translating Natural Language to SPARQL using custom LLM callable...")
        sparql_query = graph.generate_sparql("Find all tools and who created them")
        print("\nGenerated SPARQL Query:\n" + sparql_query)

        # ---------------------------------------------------------------------
        # Option 2: Also works with a simple 1-line lambda!
        # ---------------------------------------------------------------------
        print("\n[Step 4] Dynamically updating generator to a simple single-argument lambda...")
        simple_lambda = lambda prompt: json.dumps({
            "nodes": [{"label": "Concept", "properties": {"identifying_value": "LambdaExtraction", "name": "LambdaExtraction"}}],
            "relationships": []
        })
        store.set_llm_generator(simple_lambda)

        # Test single-argument lambda handles prompt execution cleanly
        test_res = graph.llm_generator("Extract: Simple test text")
        print(f"✓ Single-argument lambda response: {test_res}")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" Custom LLM callable demonstration finished successfully! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()