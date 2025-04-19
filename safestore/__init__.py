# safestore/__init__.py
"""
SafeStore: Simple SQLite Vector Store for RAG.

A Python utility library providing a lightweight, efficient, and file-based
vector database using SQLite. Optimized for easy integration into
Retrieval-Augmented Generation (RAG) pipelines for Large Language Models (LLMs).
"""

from .store import SafeStore, LogLevel
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
)
from ascii_colors import ASCIIColors # Expose for user configuration convenience

__version__ = "1.3.0" # <-- BUMPED VERSION for Phase 4

__all__ = [
    "SafeStore",
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
]