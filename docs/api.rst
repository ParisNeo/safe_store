===
API
===

This section provides detailed reference documentation for the ``safe_store`` library's public API.

Core Store
----------

.. automodule:: safe_store.store
   :members: SafeStore, LogLevel, DEFAULT_LOCK_TIMEOUT
   :undoc-members:
   :show-inheritance:

Knowledge Graph & SPARQL
------------------------

.. automodule:: safe_store.graph.graph_store
   :members: GraphStore
   :undoc-members:

.. automodule:: safe_store.graph.cognitive_memory
   :members: CognitiveMemoryStore
   :undoc-members:

.. automodule:: safe_store.graph.ontology.tbox
   :members: TBoxManager
   :undoc-members:

.. automodule:: safe_store.graph.mapping.tabular_mapper
   :members: TabularMapper
   :undoc-members:

Search & Rank Fusion
--------------------

.. automodule:: safe_store.search.bm25
   :members: BM25Retriever
   :undoc-members:

.. automodule:: safe_store.search.fusion
   :members: reciprocal_rank_fusion, weighted_score_fusion
   :undoc-members:

Semantic Datalake
-----------------

.. automodule:: safe_store.datalake.viewer
   :members: DatalakeViewer
   :undoc-members:

Exceptions
----------

.. automodule:: safe_store.core.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Vectorization
-------------

.. automodule:: safe_store.vectorization.base
   :members: BaseVectorizer
   :undoc-members:

.. automodule:: safe_store.vectorization.manager
   :members: VectorizationManager
   :undoc-members: