===========
Graph Store
===========

The ``GraphStore`` module manages knowledge graphs within the SafeStore database. It supports dynamic and ontology-constrained LLM graph extraction, W3C SPARQL 1.1 query & update engines, declarative tabular mapping, and cognitive memory.

Initialization
--------------

.. code-block:: python

   from safe_store import SafeStore, GraphStore

   store = SafeStore(db_path="graph_kb.db", vectorizer_name="st")
   graph_store = GraphStore(
       store=store,
       llm_executor_callback=my_llm_callback,  # Optional for automatic extraction
       ontology=my_ontology_schema            # Optional TBox/dict schema
   )

Graph Diagnostics
-----------------

.. py:method:: get_graph_info() -> Dict[str, Any]

   Returns diagnostic information about the graph store, including total nodes, total relationships, breakdown by label/type, and provenance links.

Node Management
---------------

.. py:method:: add_node(label: str, properties: Dict[str, Any]) -> int

   Adds a new node to the graph and computes its vector embedding for semantic search.

   :param label: The type/label of the node (e.g., "Person", "Company", "Concept").
   :param properties: A dictionary of properties (must include an `identifying_value` or `name`).
   :return: The ID of the newly created node.

.. py:method:: get_node_details(node_id: int) -> Optional[Dict[str, Any]]

   Retrieves node details by ID.

.. py:method:: update_node(node_id: int, label: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool

   Updates an existing node's label and properties.

.. py:method:: delete_node(node_id: int) -> bool

   Deletes a node and its associated relationships.

Relationship Management
-----------------------

.. py:method:: add_relationship(source_node_id: int, target_node_id: int, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> int

   Creates a directed relationship between two nodes.

.. py:method:: delete_relationship(relationship_id: int) -> bool

   Deletes a specific relationship by ID.

W3C SPARQL 1.1 Query & Update
-----------------------------

.. py:method:: query_sparql(sparql_query: str) -> Dict[str, Any]

   Executes standard W3C SPARQL 1.1 queries (``SELECT``, ``ASK``, ``CONSTRUCT``, ``DESCRIBE``).

.. py:method:: execute_sparql_update(sparql_update: str) -> Dict[str, Any]

   Executes standard W3C SPARQL 1.1 update commands (``INSERT DATA``, ``DELETE DATA``, ``DELETE WHERE``) and synchronizes SQLite graph tables.

Tri-Modal Unified Graph Search
------------------------------

.. py:method:: query_graph_hybrid(query_text: str, top_k: int = 5, dense_weight: float = 0.4, bm25_weight: float = 0.3, graph_weight: float = 0.3, min_relevance_percent: float = 0.0) -> Dict[str, Any]

   Executes tri-modal retrieval combining graph neighborhood exploration, dense vector similarity, and sparse BM25 lexical search using Reciprocal Rank Fusion.

Cognitive Memory System (``graph_store.memory``)
------------------------------------------------

.. py:method:: memory.record_episode(title: str, description: str, participants: Optional[List[str]] = None, source_chunk_ids: Optional[List[int]] = None, ...) -> int

   Records an episodic event with temporal metadata and source text chunk grounding.

.. py:method:: memory.recall_associative(concept_or_entity: str, max_hops: int = 2) -> Dict[str, Any]

   Explores associative memory pathways originating from an entity or concept.

.. py:method:: memory.get_llm_tool_definitions() -> List[Dict[str, Any]]

   Returns standard JSON tool schemas for LLM function calling.

.. py:method:: memory.dispatch_llm_tool(tool_name: str, arguments: Dict[str, Any]) -> Any

   Executes an LLM tool call and returns serializable results.