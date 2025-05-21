# safe_store/__init__.py
"""
safe_store: Simple SQLite Vector Store for RAG.

A Python utility library providing a lightweight, efficient, and file-based
vector database using SQLite. Optimized for easy integration into
Retrieval-Augmented Generation (RAG) pipelines for Large Language Models (LLMs).
Includes optional encryption, concurrency control, and graph data capabilities.
"""

from .store import SafeStore, LogLevel, TEMP_FILE_DB_INDICATOR, IN_MEMORY_DB_INDICATOR, DEFAULT_LOCK_TIMEOUT
from .graph.graph_store import GraphStore # Added GraphStore
from .core.exceptions import ( # Expose exceptions for users
    SafeStoreError,
    DatabaseError,
    FileHandlingError,
    ParsingError,
    IndexingError,  
    VectorizationError,
    QueryError,
    ConfigurationError,
    ConcurrencyError,
    EncryptionError,
    # Graph specific exceptions
    GraphError,
    GraphDBError,
    GraphProcessingError,
    LLMCallbackError,
)
from .indexing.parser import SAFE_STORE_SUPPORTED_FILE_EXTENSIONS, parse_document 
from .indexing.chunking import chunk_text, chunk_text_by_tokens
from ascii_colors import ASCIIColors # Expose for user configuration convenience

__version__ = "2.2.0" # Assuming this will be the version for this feature

__all__ = [
    "SafeStore",
    "GraphStore", # Added GraphStore
    "ASCIIColors",
    "LogLevel",
    # Exceptions
    "SafeStoreError",
    "DatabaseError",
    "FileHandlingError",
    "ParsingError",
    "IndexingError",
    "VectorizationError",
    "QueryError",
    "ConfigurationError",
    "ConcurrencyError",
    "EncryptionError",
    "GraphError", # Added GraphError
    "GraphDBError", # Added GraphDBError
    "GraphProcessingError", # Added GraphProcessingError
    "LLMCallbackError", # Added LLMCallbackError
    # globals
    "SAFE_STORE_SUPPORTED_FILE_EXTENSIONS",
    "TEMP_FILE_DB_INDICATOR",
    "IN_MEMORY_DB_INDICATOR",
    "DEFAULT_LOCK_TIMEOUT",
    # utilities
    "parse_document",
    "chunk_text",
    "chunk_text_by_tokens"
]