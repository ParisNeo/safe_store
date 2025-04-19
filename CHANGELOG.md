# CHANGELOG.md
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-04-18 <!-- Update Date -->

### Added

*   **Concurrency Handling:**
    *   Added `filelock` dependency for inter-process concurrency control.
    *   Implemented exclusive file-based locking (`.db.lock` file) around database write operations (`add_document`, `add_vectorization`, `remove_vectorization`). This prevents race conditions when multiple processes access the same `SafeStore` database file.
    *   Added `lock_timeout` parameter to `SafeStore.__init__` (default 60 seconds) to control how long to wait for the lock.
    *   Added basic `threading.RLock` for intra-process thread safety around critical sections and connection handling.
    *   Added logging (`ascii_colors`) for lock acquisition attempts, success, release, and timeouts.
    *   Refactored write methods (`add_document`, `add_vectorization`, `remove_vectorization`) into internal `_impl` methods that assume the lock is held.
*   **Parsing Infrastructure (Foundation):**
    *   Added optional dependencies for parsing PDF (`pypdf`), DOCX (`python-docx`), and HTML (`beautifulsoup4`, `lxml`) files via the `safestore[parsing]` extra.
    *   Added placeholder functions (`parse_pdf`, `parse_docx`, `parse_html`) in `safestore.indexing.parser`.
    *   Updated the `parse_document` dispatcher in `safestore.indexing.parser` to recognize `.pdf`, `.docx`, and `.html` extensions (implementation pending).

### Changed

*   `safestore.store.SafeStore`:
    *   `__init__`: Now accepts `lock_timeout`, resolves database path, creates `.lock` file path, connects and initializes DB within a lock, initializes threading and file locks.
    *   `close()`: Now safer with instance lock and better error handling.
    *   `__enter__` / `__exit__`: Ensure connection exists and handle closure.
    *   Write methods now acquire instance and file locks before calling internal implementation methods.
    *   `query()`: Uses instance lock for thread safety; relies on SQLite WAL mode for read concurrency (no explicit file lock for reads currently). Added connection check.
*   `pyproject.toml`: Bumped version to 1.2.0. Added `filelock`, `pypdf`, `python-docx`, `beautifulsoup4`, `lxml`, `cryptography` dependencies. Defined `[parsing]`, `[encryption]`, `[all]` extras.

### Fixed

*   Minor fix in `VectorizationManager.update_method_params` to correctly format the SQL update statement when `new_dim` is provided.
*   Corrected assertion in `test_add_vectorization_tfidf_all_docs` (`test_store_phase2.py`) related to TF-IDF fitting log message.
*   Corrected mock target and assertion in `test_remove_vectorization` (`test_store_phase2.py`) related to cache removal log message.


## [1.1.0] - 2025-04-17

### Added

*   **Querying:**
    *   Implemented the `SafeStore.query()` method for retrieving document chunks based on semantic similarity to a query text.
    *   Uses cosine similarity (`safestore.search.similarity.cosine_similarity`) for ranking.
    *   Supports specifying the `vectorizer_name` to use for the query, ensuring consistency with indexed vectors.
    *   Returns `top_k` results containing chunk text, similarity score, document path, and positional information.
    *   Includes logging (`ascii_colors`) for query steps (vectorization, loading, similarity calculation).
    *   Initial implementation loads all candidate vectors for the specified method into memory for comparison (potential performance bottleneck for very large datasets noted).

*   **Multiple Vectorization Methods:**
    *   Added support for TF-IDF vectorization using `scikit-learn`.
        *   Requires `pip install safestore[tfidf]`.
        *   Added `safestore.vectorization.methods.tfidf.TfidfVectorizerWrapper`.
        *   Handles fitting the TF-IDF model:
            *   During `add_document`: Fits only on the chunks of the *current* document if the method is new and unfitted (with a warning).
            *   During `add_vectorization`: Fits on chunks from *all* specified documents (or the entire store) if the method is new and unfitted.
        *   Stores fitted state (vocabulary, IDF weights) in the `vectorization_methods.params` JSON column in the database.
        *   Loads fitted state when the vectorizer is requested via `VectorizationManager`.
    *   Updated `VectorizationManager` (`get_vectorizer`) to handle different vectorizer types (`st:`, `tfidf:`) and manage their state (including loading/saving TF-IDF fitted parameters).
    *   Added `safestore.vectorization.manager.VectorizationManager.update_method_params` to update DB record after fitting TF-IDF.
    *   Added `[tfidf]` and `[all-vectorizers]` optional dependencies in `pyproject.toml`.

*   **Vectorizer Management Methods:**
    *   Implemented `SafeStore.add_vectorization()`: Allows adding embeddings for a new vectorization method to documents already in the store, without re-parsing/re-chunking. Handles fitting for TF-IDF if needed. Supports targeting all documents or a specific document.
    *   Implemented `SafeStore.remove_vectorization()`: Deletes a specific vectorization method and all associated vector embeddings from the database and cache.

*   **Testing:**
    *   Added new test file `tests/test_store_phase2.py`.
    *   Included integration tests for `query()`, covering basic usage, non-existent vectorizers, and methods with no vectors.
    *   Added tests for using TF-IDF during `add_document`.
    *   Added tests for `add_vectorization()` with both Sentence Transformers and TF-IDF.
    *   Added tests for `remove_vectorization()`.
    *   Enhanced mocking for `sentence-transformers` and added mocking for `scikit-learn`'s `TfidfVectorizer` to allow testing without the dependencies necessarily installed.

### Changed

*   `safestore.vectorization.manager.VectorizationManager`: Now caches the database parameters alongside the instance and method ID. Invalidates cache entries when parameters are updated (e.g., TF-IDF fitting). Includes helper `_get_method_details_from_db`.
*   `safestore.store.SafeStore`:
    *   `add_document()` now checks for existing vectors for the *specific* vectorizer being used when deciding whether to skip processing an unchanged document. It also handles the initial fitting of TF-IDF if necessary. Accepts optional `vectorizer_params`.
*   `safestore.search.similarity.cosine_similarity`: Added handling for case where `vectors` input is 1D (representing a single vector). Added type checking.
*   `pyproject.toml`: Bumped version to 1.1.0. Added `scikit-learn` to `[tfidf]` extra. Added `[all-vectorizers]` extra. Added Python 3.12 classifier.

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