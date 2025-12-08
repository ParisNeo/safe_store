"""
safe_store: Simple SQLite Vector Store for RAG.

A Python utility library providing a lightweight, efficient, and file-based
vector database using SQLite. Optimized for easy integration into
Retrieval-Augmented Generation (RAG) pipelines for Large Language Models (LLMs).
Includes optional encryption, concurrency control, and graph data capabilities.
"""

from .store import SafeStore, LogLevel, TEMP_FILE_DB_INDICATOR, IN_MEMORY_DB_INDICATOR, DEFAULT_LOCK_TIMEOUT
from .graph.graph_store import GraphStore
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
from .processing.text_cleaning import basic_text_cleaner # Expose the basic cleaner as a utility
from ascii_colors import ASCIIColors # Expose for user configuration convenience

__version__ = "3.3.2" # Version bump to reflect API changes

__all__ = [
    "SafeStore",
    "GraphStore",
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
    "GraphError",
    "GraphDBError",
    "GraphProcessingError",
    "LLMCallbackError",
    # globals
    "SAFE_STORE_SUPPORTED_FILE_EXTENSIONS",
    "TEMP_FILE_DB_INDICATOR",
    "IN_MEMORY_DB_INDICATOR",
    "DEFAULT_LOCK_TIMEOUT",
    # utilities
    "parse_document",
    "basic_text_cleaner"
]