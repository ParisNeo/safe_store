# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-04-16

This is the initial public release of SafeStore! 🎉 This version establishes the core foundation for indexing and vectorizing text documents.

### Added

*   **Core Functionality:**
    *   Introduced the main `SafeStore` class for managing the vector store.
    *   Established SQLite3 as the database backend, creating the file if it doesn't exist.
    *   Defined the core database schema including tables for `documents`, `chunks`, `vectorization_methods`, and `vectors`, with appropriate relationships and indexes.
    *   Implemented robust database connection handling (`connect_db`) and schema initialization (`initialize_schema`).
    *   Enabled SQLite's WAL (Write-Ahead Logging) mode for potential future concurrency improvements.

*   **Indexing Pipeline:**
    *   Created the primary `add_document` method to handle the end-to-end ingestion of files.
    *   Added support for parsing plain text (`.txt`) files using `safestore.indexing.parser`.
    *   Implemented configurable text chunking (`chunk_text`) based on character count (`chunk_size`) and overlap (`chunk_overlap`).
    *   Included tracking and storage of chunk start/end character positions relative to the original document.
    *   Stored the full, unmodified text of ingested documents in the `documents` table to enable future re-chunking or re-indexing with different parameters.
    *   Integrated file hashing (SHA256) to detect content changes between `add_document` calls for the same file path.
    *   Implemented logic within `add_document` to handle different scenarios:
        *   **New Documents:** Parses, chunks, vectorizes, and stores all related data.
        *   **Unchanged Documents:** Checks file hash; if unchanged *and* vectors for the specified method exist, skips reprocessing efficiently.
        *   **Changed Documents:** Detects hash mismatch, automatically deletes old chunks/vectors for that document, updates the document record, and then re-parses, re-chunks, and re-vectorizes the new content.

*   **Vectorization:**
    *   Laid the foundation for supporting multiple vectorization methods via `VectorizationManager`.
    *   Added initial support for vectorization using `sentence-transformers` models (using `st:model-name` format, e.g., `st:all-MiniLM-L6-v2` as default).
    *   Implemented storage of NumPy vector embeddings as SQLite BLOBs using custom adapters.
    *   Created mechanisms (`add_or_get_vectorization_method`) to register and retrieve vectorization method details (name, type, dimensions, data type) in the `vectorization_methods` table.

*   **Utilities & Development:**
    *   Integrated `ascii_colors` for clear, leveled, and colorful console logging and user feedback throughout the library's operations. Configured default log level via `SafeStore` initialization.
    *   Established the basic project structure using a flat layout (`safestore` directory at the root).
    *   Developed an initial test suite using `pytest` and `unittest.mock`, covering:
        *   Unit tests for text chunking logic with various edge cases.
        *   Integration tests for the `add_document` workflow verifying correct behavior and database state for new, unchanged, and changed documents.
        *   Mocking of external dependencies (`sentence-transformers`) for testing.
        *   Verification of logging calls using mock objects.
    *   Configured project metadata, dependencies, and optional dependencies (`[sentence-transformers]`) in `pyproject.toml`.