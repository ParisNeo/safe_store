# safe_store/__init__.py
"""
safe_store: Simple SQLite Vector Store for RAG.

A Python utility library providing a lightweight, efficient, and file-based
vector database using SQLite. Optimized for easy integration into
Retrieval-Augmented Generation (RAG) pipelines for Large Language Models (LLMs).
Includes optional encryption and concurrency control.
"""

from .store import safe_store, LogLevel
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
)
from ascii_colors import ASCIIColors # Expose for user configuration convenience

__version__ = "1.4.0" # <-- BUMPED VERSION

__all__ = [
    "safe_store",
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
    "EncryptionError", # Added EncryptionError
]