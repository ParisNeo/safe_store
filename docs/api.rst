===
API
===

This section provides detailed documentation for the ``safe_store`` library's public API.

Core Class
----------

.. automodule:: SafeStore.store
   :members: SafeStore, LogLevel
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. automodule:: SafeStore.core.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Vectorizers
-----------

.. automodule:: SafeStore.vectorization.base
   :members: BaseVectorizer
   :undoc-members:

.. automodule:: SafeStore.vectorization.methods.sentence_transformer
   :members: SentenceTransformerVectorizer
   :undoc-members:

.. automodule:: SafeStore.vectorization.methods.tfidf
   :members: TfidfVectorizerWrapper
   :undoc-members:

Utilities
---------
While primarily used internally, the ``ascii_colors`` library is exposed for configuration.

.. automodule:: ascii_colors
   :members: ASCIIColors, LogLevel, FileHandler, Formatter, JSONFormatter
   :undoc-members:

(Add other modules/classes as needed)
