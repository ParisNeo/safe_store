===========
Graph Store
===========

The ``GraphStore`` module manages an interconnected semantic knowledge graph within a SafeStore SQLite database.
It unifies:

1. **High-Context Graph Extraction**: Extract entities (nodes) and directed relationships (triplets) using an LLM in entire document passes or batched chunk windows.
2. **W3C SPARQL 1.1 Query Engine**: Full standard ``SELECT``, ``ASK``, ``CONSTRUCT``, and ``DESCRIBE`` query execution.
3. **SPARQL 1.1 Update Engine**: Real-time graph modification and concept reorganization via ``INSERT DATA`` and ``DELETE DATA``.
4. **Declarative Tabular Mapping**: Instant zero-LLM transformation of CSV, Excel, and SQLite tables into grounded RDF knowledge graphs.
5. **Tri-Modal Unified Retrieval**: Fusing graph traversal, dense vector similarity, and sparse BM25 lexical matches via Reciprocal Rank Fusion.
6. **Cognitive Memory & LLM Tool Calling**: Episodic event logging, associative recall, and grounded text chunk provenance.

Initialization
--------------

.. code-block:: python

   from safe_store import SafeStore, GraphStore

   store = SafeStore(db_path="knowledge.db", vectorizer_name="st")
   graph_store = GraphStore(
       store=store,
       llm_executor_callback=my_llm_callback,  # Optional custom callback
       ontology=my_tbox_ontology              # Optional strict TBox schema or dict
   )

Fast Graph Extraction Across Large Context Windows
--------------------------------------------------

Modern LLMs have context windows of 32k to 128k+ tokens. Rather than making slow sequential calls for every single small chunk, ``GraphStore`` provides three extraction modes:

.. code-block:: python

   # 1. Mode 'document' (Default & Recommended - 1 LLM call per document)
   # Up to 20x faster than chunk-by-chunk extraction and captures cross-chunk relationships.
   stats = graph_store.build_graph_for_all_documents(
       mode='document',
       guidance="Focus on software microservices, APIs, and team owners."
   )
   print(f"Extracted {stats['nodes_created']} nodes and {stats['relationships_created']} relationships.")

   # 2. Mode 'batch_chunks' (Balanced - groups N chunks per LLM call)
   stats = graph_store.build_graph_for_all_documents(
       mode='batch_chunks',
       chunks_per_batch=10
   )

   # 3. Mode 'chunk' (Granular - processes each chunk in isolation)
   stats = graph_store.build_graph_for_all_documents(mode='chunk')

Graph Diagnostics
-----------------

.. py:method:: get_graph_info() -> Dict[str, Any]

   Returns real-time diagnostics: total nodes, total relationships, breakdown by label and relationship type, and chunk provenance link counts.

W3C SPARQL 1.1 Querying
-----------------------

.. py:method:: generate_sparql(natural_language_query: str, guidance: Optional[str] = None) -> str

   Translates plain English questions into executable W3C SPARQL 1.1 queries using LOLLMS, grounded in the database's live entity classes, relationships, and schema.

.. code-block:: python

   # Ask in natural language
   query_str = graph_store.generate_sparql("Find all tools that require Python and show their authors")
   print("Generated Query:\n", query_str)

   # Execute generated query immediately
   results = graph_store.query_sparql(query_str)

W3C SPARQL 1.1 Querying
-----------------------

.. py:method:: query_sparql(sparql_query: str) -> Dict[str, Any]

   Executes standards-compliant W3C SPARQL 1.1 queries across the graph.

.. code-block:: python

   # Multi-Hop Relational Join (SELECT)
   query = """
   PREFIX ex: <http://example.org/>
   PREFIX ont: <http://example.org/ontology/>
   SELECT ?personName ?companyName ?projectName WHERE {
       ?person a ont:Person ;
               ont:name ?personName ;
               ont:worksFor ?company ;
               ont:leadsProject ?project .
       ?company ont:name ?companyName .
       ?project ont:name ?projectName .
   }
   """
   results = graph_store.query_sparql(query)
   for b in results["results"]["bindings"]:
       print(f"{b['personName']['value']} works at {b['companyName']['value']} on {b['projectName']['value']}")

SPARQL 1.1 Updates (Knowledge Reorganization)
---------------------------------------------

.. py:method:: execute_sparql_update(sparql_update: str) -> Dict[str, Any]

   Executes standard W3C SPARQL 1.1 update commands (``INSERT DATA``, ``DELETE DATA``, ``DELETE WHERE``) and synchronizes SQLite graph tables atomically.

.. code-block:: python

   graph_store.execute_sparql_update("""
   PREFIX ex: <http://example.org/>
   PREFIX ont: <http://example.org/ontology/>
   INSERT DATA {
       ex:Alice a ont:Architect ;
                ont:name "Alice Smith" ;
                ont:leadsProject ex:ProjectPhoenix .
   }
   """)

Declarative Tabular-to-Graph Mapping (Zero-LLM)
-----------------------------------------------

Transform structured tables directly into grounded RDF knowledge graphs in milliseconds without consuming LLM tokens:

.. code-block:: python

   from safe_store import TabularMapper, TBoxManager

   tbox = TBoxManager()
   tbox.load_ontology("domain.ttl", format="turtle")

   mapper = TabularMapper(store=store, tbox=tbox)
   summary = mapper.map_csv("inventory.csv", mapping_rules={
       "entity_mappings": [
           {
               "class": "http://example.org/ontology/Product",
               "subject_template": "http://example.org/product/{sku}",
               "properties": {"name": "http://example.org/ontology/hasName"}
           }
       ],
       "relationship_mappings": [
           {
               "predicate": "http://example.org/ontology/suppliedBy",
               "source_template": "http://example.org/product/{sku}",
               "target_template": "http://example.org/supplier/{supplier_id}"
           }
       ]
   })

Tri-Modal Unified Graph Search
------------------------------

.. py:method:: query_graph_hybrid(query_text: str, top_k: int = 5, dense_weight: float = 0.4, bm25_weight: float = 0.3, graph_weight: float = 0.3, min_relevance_percent: float = 0.0) -> Dict[str, Any]

   Executes tri-modal retrieval combining graph neighborhood exploration, dense vector similarity, and sparse BM25 lexical search using Reciprocal Rank Fusion.

Cognitive Memory & LLM Tool Calling
-----------------------------------

.. py:method:: memory.record_episode(title: str, description: str, participants: List[str], source_chunk_ids: List[int], ...) -> int

   Records an episodic memory event grounded in physical document text chunks.

.. py:method:: memory.recall_associative(concept_or_entity: str, max_hops: int = 2) -> Dict[str, Any]

   Traverses associative graph pathways and retrieves grounded text evidence.

.. py:method:: get_tool_definitions() -> List[Dict[str, Any]]

   Returns standardized JSON schemas for LLM function calling tools (OpenAI, Anthropic, Ollama, Lollms).