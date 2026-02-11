Graph Store
===========

The ``GraphStore`` module provides capabilities to build and manage a knowledge graph within the SafeStore database. It supports automatic graph extraction from documents using LLMs, as well as manual management of nodes and relationships.

Initialization
--------------

.. code-block:: python

   from safe_store.graph import GraphStore

   # Assuming 'store' is an initialized SafeStore instance
   graph_store = GraphStore(
       store=store,
       llm_executor_callback=my_llm_function,
       ontology=my_ontology_dict  # Optional
   )

Node Management
---------------

.. py:method:: add_node(label: str, properties: Dict[str, Any]) -> int

   Adds a new node to the graph.

   :param label: The type/label of the node (e.g., "Person", "Company").
   :param properties: A dictionary of properties for the node.
   :return: The ID of the newly created node.

.. py:method:: get_node_details(node_id: int) -> Optional[Dict[str, Any]]

   Retrieves the details of a specific node.

   :param node_id: The ID of the node to retrieve.
   :return: A dictionary containing the node's details, or None if not found.

.. py:method:: update_node(node_id: int, label: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool

   Updates an existing node.

   :param node_id: The ID of the node to update.
   :param label: (Optional) A new label for the node.
   :param properties: (Optional) A dictionary of properties to update (merges with existing).
   :return: True if successful.

.. py:method:: delete_node(node_id: int) -> bool

   Deletes a node and all its connected relationships.

   :param node_id: The ID of the node to delete.
   :return: True if successful.

Relationship Management
-----------------------

.. py:method:: add_relationship(source_node_id: int, target_node_id: int, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> int

   Creates a relationship between two nodes.

   :param source_node_id: The ID of the source node.
   :param target_node_id: The ID of the target node.
   :param rel_type: The type of relationship (e.g., "WORKS_FOR").
   :param properties: (Optional) Properties for the relationship.
   :return: The ID of the newly created relationship.

.. py:method:: get_relationship(relationship_id: int) -> Optional[Dict[str, Any]]

   Retrieves details of a specific relationship.

   :param relationship_id: The ID of the relationship.
   :return: A dictionary with relationship details or None.

.. py:method:: update_relationship(relationship_id: int, rel_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool

   Updates an existing relationship.

   :param relationship_id: The ID of the relationship to update.
   :param rel_type: (Optional) New type for the relationship.
   :param properties: (Optional) New properties to update.
   :return: True if successful.

.. py:method:: delete_relationship(relationship_id: int) -> bool

   Deletes a specific relationship.

   :param relationship_id: The ID of the relationship to delete.
   :return: True if successful.

Graph Building & Querying
-------------------------

.. py:method:: build_graph_for_document(doc_id: int, ...)

   extracts graph data from the chunks of a specific document using the configured LLM.

.. py:method:: query_graph(natural_language_query: str, ...)

   Performs a semantic search to find relevant nodes and then traverses the graph to answer the query.
