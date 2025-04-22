===
API
===

This section provides detailed documentation for the ``safe_store`` library's public API.

Core Class
----------

.. automodule:: safe_store.store
   :members: safe_store, LogLevel
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. automodule:: safe_store.core.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Vectorizers
-----------

.. automodule:: safe_store.vectorization.base
   :members: BaseVectorizer
   :undoc-members:

.. automodule:: safe_store.vectorization.methods.sentence_transformer
   :members: SentenceTransformerVectorizer
   :undoc-members:

.. automodule:: safe_store.vectorization.methods.tfidf
   :members: TfidfVectorizerWrapper
   :undoc-members:

Utilities
---------
While primarily used internally, the ``ascii_colors`` library is exposed for configuration.

.. automodule:: ascii_colors
   :members: ASCIIColors, LogLevel, FileHandler, Formatter, JSONFormatter
   :undoc-members:

(Add other modules/classes as needed)
