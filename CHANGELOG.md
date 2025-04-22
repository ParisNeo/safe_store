# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2025-04-20
### Added

*   **Encryption:**
    *   Added optional encryption at rest for `chunk_text` using `cryptography` (Fernet/AES-128-CBC). Requires `safe_store[encryption]`.
    *   Added `encryption_key` parameter to `safe_store.__init__`.
    *   Encryption/decryption is handled automatically during `add_document` and `query` if the key is present.
    *   `add_vectorization` now attempts decryption when fitting TF-IDF on potentially encrypted chunks.
    *   Added `safe_store.core.exceptions.EncryptionError`.
    *   Implemented `safe_store.security.encryption.Encryptor` class for handling key derivation (PBKDF2) and encryption/decryption. Uses a fixed salt (documented limitation).
    *   Added tests for encryption logic (`test_encryption.py`, `test_store_encryption.py`).
    *   Added `examples/encryption_usage.py`.
    *   Added documentation (`encryption.rst`) explaining the feature, usage, and security considerations.
*   **Documentation:**
    *   Created Sphinx documentation structure under `docs/`. (Moved from 1.3.0 plan to here as it aligns with final polish).
    *   Added content for `conf.py`, `index.rst`, `installation.rst`, `quickstart.rst`, `api.rst`, `logging.rst`, `encryption.rst`.
    *   Added `docs/requirements.txt`.
    *   Added `sphinx` and `sphinx-rtd-theme` to `[dev]` extras in `pyproject.toml`.
*   **Examples:**
    *   Implemented `examples/custom_logging.py` demonstrating global `ascii_colors` configuration.
*   **Testing:**
    *   Added comprehensive tests for encryption functionality.
    *   Added tests for store closure (`close()`) and context manager (`__enter__`/`__exit__`) behavior. (Moved from 1.3.0 plan).
    *   Added tests for `list_documents` and `list_vectorization_methods`. (Moved from 1.3.0 plan).
    *   Ensured tests relying on optional dependencies are skipped correctly or use mocks when dependencies are unavailable (refined mock setup in `conftest.py`).

### Changed

*   `pyproject.toml`: Bumped version to 1.4.0. Added `cryptography` dependency and `[encryption]` extra. Updated `[all]` and `[dev]` extras. Updated classifiers.
*   `README.md`: Significantly updated with encryption feature, detailed logging section, refined quick start, installation instructions, and feature list.
*   `safe_store/__init__.py`: Bumped version. Exposed `EncryptionError`.
*   `safe_store/store.py`: Integrated `Encryptor`. Modified indexing and querying methods to handle encryption/decryption. Added related error handling.
*   `tests/conftest.py`: Improved conditional mocking setup for optional dependencies (`sentence-transformers`, `scikit-learn`, `cryptography`) using `pytest.mark.skipif` in relevant test files/modules where mocks aren't sufficient. Applied mocks more selectively.
*   Refined type hints and docstrings across modules for better clarity and consistency.

### Fixed

*   Corrected assertions in `test_store_phase2.py` related to TF-IDF fitting log and cache removal log messages. (Moved fix confirmation from 1.2.0 change list as it was related to test refinement).
*   Ensured `TfidfVectorizerWrapper.load_fitted_state` robustly reconstructs the internal state required for `transform` to work after loading, including setting the internal IDF diagonal matrix.
*   Addressed potential `TypeError` if encrypted chunk data was somehow stored as string instead of bytes during decryption.
*   Fixed potential state inconsistencies in `TfidfVectorizerWrapper` if fitting failed or was performed on empty text.


## [1.3.0] - 2025-04-19

### Added

*   **API Polish & Type Hinting:**
    *   Added comprehensive type hints across the library (`safe_store`, `db.py`, vectorizers, parsers, etc.) using `typing`.
    *   Improved docstrings for public classes and methods, explaining parameters, return values, and potential exceptions.
*   **Error Handling:**
    *   Consistently raise specific custom exceptions defined in `safe_store.core.exceptions` (e.g., `DatabaseError`, `FileHandlingError`, `ParsingError`, `ConfigurationError`, `VectorizationError`, `QueryError`, `ConcurrencyError`) instead of generic exceptions.
    *   Ensured proper exception chaining using `raise ... from e`.
    *   Improved error messages for clarity.
*   **Documentation Structure:**
    *   Created basic Sphinx documentation structure under `docs/`.
    *   Added placeholder files for `conf.py`, `index.rst`, `installation.rst`, `quickstart.rst`, `api.rst`, `logging.rst`.
    *   Added `docs/requirements.txt`. (Full content writing pending).
*   **Examples:**
    *   Added `examples/basic_usage.py` demonstrating core indexing and querying workflow.
    *   Added `examples/custom_logging.py` showing how users can configure `ascii_colors` globally (e.g., set level, log to file).
*   **Helper Methods:**
    *   Added `safe_store.list_documents()` to retrieve metadata about stored documents.
    *   Added `safe_store.list_vectorization_methods()` to retrieve details about registered vectorizers.
*   **Testing:**
    *   Added basic tests for `safe_store.close()` and context manager (`__enter__`/`__exit__`) behavior.
    *   Added tests for new `list_documents` and `list_vectorization_methods`.
    *   Refined existing tests for clarity and robustness.

### Changed

*   **Dependencies:** Finalized optional dependencies and extras in `pyproject.toml`. Added `[dev]` extra for testing/linting/building/docs dependencies.
*   `pyproject.toml`: Bumped version to 1.3.0. Added project keywords, refined classifiers. Added URLs for Docs/Issues. Configured `hatch` for version management. Added `black` config.
*   `README.md`: Significantly updated with latest features, improved explanations (concurrency, logging), clearer installation instructions, refined Quick Start, and links to examples/repo.
*   `safe_store.core.db`: Enabled SQLite `PRAGMA foreign_keys = ON` for better data integrity. Set `check_same_thread=False` for `sqlite3.connect` as external locking (`filelock`) is used. Improved error handling in DB functions.
*   `safe_store.indexing.parser`: Improved error handling for specific file I/O and parsing library exceptions.
*   `safe_store.vectorization.manager`: Improved error handling, especially around missing dependencies and DB interactions. Added `remove_from_cache_by_id` helper.
*   `safe_store.vectorization.methods`: Improved error handling for model loading and vectorization failures. Ensured vectorizers handle empty input lists gracefully. Refined TF-IDF state loading/saving logic for robustness.
*   `safe_store.search.similarity`: Added handling for empty input `vectors` matrix. Improved validation messages.
*   `safe_store.__init__`: Exposed core exceptions for user convenience.

### Fixed

*   Corrected potential race condition in `VectorizationManager.get_vectorizer` when multiple processes try to add the same new method concurrently (now handles UNIQUE constraint error).
*   Ensured `TfidfVectorizerWrapper.load_fitted_state` correctly reconstructs the internal state required for `transform` to work after loading.
*   Fixed `TfidfVectorizerWrapper.vectorize` dimension check logic and error message for clarity.
*   Ensured `safe_store` context manager (`__enter__`) properly re-initializes connection if closed previously.
*   Corrected minor inconsistencies in log messages and error handling across various modules.

## [1.2.0] - 2025-04-18

### Added

*   **Concurrency Handling:**
    *   Added `filelock` dependency for inter-process concurrency control.
    *   Implemented exclusive file-based locking (`.db.lock` file) around database write operations (`add_document`, `add_vectorization`, `remove_vectorization`).
    *   Added `lock_timeout` parameter to `safe_store.__init__` (default 60 seconds).
    *   Added basic `threading.RLock` for intra-process thread safety.
    *   Added logging (`ascii_colors`) for lock acquisition/release/timeouts.
    *   Refactored write methods into internal `_impl` methods assuming lock is held.
*   **Parsing Infrastructure (Implemented):**
    *   Added optional dependencies for parsing PDF (`pypdf`), DOCX (`python-docx`), and HTML (`beautifulsoup4`, `lxml`) via the `safe_store[parsing]` extra.
    *   Implemented `parse_pdf`, `parse_docx`, `parse_html` in `safe_store.indexing.parser` using respective libraries.
    *   Updated the `parse_document` dispatcher to correctly call implemented parsers for `.pdf`, `.docx`, `.html`, `.htm` extensions. Added error handling for parsing failures and missing dependencies.

### Changed

*   `safe_store.store.safe_store`:
    *   `__init__`: Now accepts `lock_timeout`, resolves paths, connects/initializes DB within a lock, initializes threading/file locks.
    *   `close()`: Now safer with instance lock and better error handling.
    *   `__enter__` / `__exit__`: Ensure connection exists and handle closure.
    *   Write methods now acquire instance and file locks before calling internal `_impl` methods.
    *   `query()`: Uses instance lock; relies on SQLite WAL mode for read concurrency. Added connection check.
*   `pyproject.toml`: Bumped version to 1.2.0. Added `filelock`, `pypdf`, `python-docx`, `beautifulsoup4`, `lxml`. Defined `[parsing]`, `[all]` extras.
*   `safe_store.core.db`: Now uses `Union[str, Path]` for path types.

### Fixed

*   Minor fix in `VectorizationManager.update_method_params` SQL statement format.
*   Corrected assertions in `test_store_phase2.py` related to TF-IDF fitting log and cache removal log.
*   Resolved issues in parser implementations and dispatcher logic found during testing.
*   Fixed fixture copying in `conftest.py` for reliability.

## [1.1.0] - 2025-04-17

### Added

*   **Querying:** Implemented `safe_store.query()` using cosine similarity. Loads candidate vectors into memory.
*   **Multiple Vectorization Methods:** Added TF-IDF support (`tfidf:` prefix) using `scikit-learn`. Handles fitting during `add_document` (local fit) or `add_vectorization` (global/targeted fit). Stores fitted state in DB. Requires `safe_store[tfidf]`.
*   **Vectorizer Management:** Implemented `safe_store.add_vectorization()` and `safe_store.remove_vectorization()`.
*   **Testing:** Added `tests/test_store_phase2.py` covering query, TF-IDF, add/remove vectorization. Added mocking for `scikit-learn`.

### Changed

*   `VectorizationManager`: Caches DB parameters; invalidates cache on param update (e.g., TF-IDF fit).
*   `safe_store.add_document()`: Checks for specific vectorizer existence for skipping; handles initial TF-IDF fit.
*   `search.similarity.cosine_similarity`: Handles 1D `vectors` input. Added type checking.
*   `pyproject.toml`: Bumped version to 1.1.0. Added `scikit-learn` to `[tfidf]` extra. Added `[all-vectorizers]` extra.

## [1.0.0] - 2025-04-16

Initial public release.

### Added

*   Core `safe_store` class and SQLite backend setup (WAL mode enabled).
*   Database schema (`documents`, `chunks`, `vectorization_methods`, `vectors`).
*   Indexing pipeline (`add_document`):
    *   Parses `.txt` files.
    *   Stores full text.
    *   Configurable chunking (`chunk_text`) with position tracking.
    *   File hashing (SHA256) for change detection.
    *   Handles new, unchanged (skip), and changed (re-index) documents.
*   Vectorization foundation:
    *   `VectorizationManager`.
    *   Initial support for `sentence-transformers` (`st:` prefix). Requires `safe_store[sentence-transformers]`.
    *   Stores NumPy vectors as BLOBs.
*   Utilities: `ascii_colors` for logging.
*   Testing: `pytest` suite (`test_chunking.py`, `test_store_phase1.py`) with mocking.
*   Packaging: `pyproject.toml` setup.
```

### Create `examples/` directory and files:

**`examples/basic_usage.py`**

```python
# examples/basic_usage.py
import safe_store
from pathlib import Path
import time
import shutil

# --- Configuration ---
DB_FILE = "basic_usage_store.db"
DOC_DIR = Path("temp_docs_basic")
USE_ST = True       # Set to False if sentence-transformers not installed
USE_TFIDF = True    # Set to False if scikit-learn not installed
USE_PARSING = True  # Set to False if parsing libs not installed

# --- Helper Functions ---
def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def cleanup():
    print_header("Cleaning Up")
    db_path = Path(DB_FILE)
    lock_path = Path(f"{DB_FILE}.lock")
    wal_path = Path(f"{DB_FILE}-wal")
    shm_path = Path(f"{DB_FILE}-shm")

    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
        print(f"- Removed directory: {DOC_DIR}")
    if db_path.exists():
        db_path.unlink()
        print(f"- Removed database: {db_path}")
    if lock_path.exists():
        lock_path.unlink(missing_ok=True)
        print(f"- Removed lock file: {lock_path}")
    if wal_path.exists():
        wal_path.unlink(missing_ok=True)
        print(f"- Removed WAL file: {wal_path}")
    if shm_path.exists():
        shm_path.unlink(missing_ok=True)
        print(f"- Removed SHM file: {shm_path}")

# --- Main Script ---
if __name__ == "__main__":
    cleanup() # Start fresh

    # --- 1. Prepare Sample Documents ---
    print_header("Preparing Documents")
    DOC_DIR.mkdir(exist_ok=True)

    # Document 1 (Text)
    doc1_path = DOC_DIR / "intro.txt"
    doc1_content = """
    safe_store is a Python library for local vector storage.
    It uses SQLite as its backend, making it lightweight and file-based.
    Key features include concurrency control and support for multiple vectorizers.
    """
    doc1_path.write_text(doc1_content.strip(), encoding='utf-8')
    print(f"- Created: {doc1_path.name}")

    # Document 2 (HTML) - Requires [parsing]
    doc2_path = DOC_DIR / "web_snippet.html"
    doc2_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Example Page</title></head>
    <body>
        <h1>Information Retrieval</h1>
        <p>Efficient retrieval is crucial for RAG pipelines.</p>
        <p>safe_store helps manage embeddings for semantic search.</p>
    </body>
    </html>
    """
    if USE_PARSING:
        doc2_path.write_text(doc2_content.strip(), encoding='utf-8')
        print(f"- Created: {doc2_path.name}")
    else:
        print(f"- Skipping {doc2_path.name} (requires [parsing])")

    # Document 3 (Text) - Will be added later to show updates
    doc3_path = DOC_DIR / "update_later.txt"
    doc3_content_v1 = "Initial content for update testing."
    doc3_path.write_text(doc3_content_v1, encoding='utf-8')
    print(f"- Created: {doc3_path.name}")

    print(f"Documents prepared in: {DOC_DIR.resolve()}")

    # --- 2. Initialize safe_store ---
    print_header("Initializing safe_store")
    # Use INFO level for less verbose output in basic example
    store = safe_store.safe_store(DB_FILE, log_level=safe_store.LogLevel.INFO)

    # --- 3. Indexing Documents ---
    try:
        with store:
            print_header("Indexing Documents")
            # --- Index doc1 with Sentence Transformer (default) ---
            if USE_ST:
                print(f"\nIndexing {doc1_path.name} with ST...")
                try:
                    store.add_document(
                        doc1_path,
                        vectorizer_name="st:all-MiniLM-L6-v2", # Default, but explicit here
                        chunk_size=80,
                        chunk_overlap=15,
                        metadata={"source": "manual", "topic": "introduction"}
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not index with Sentence Transformer: {e}")
                    USE_ST = False # Disable ST for later steps if failed here
            else:
                 print(f"\nSkipping ST indexing for {doc1_path.name}")

            # --- Index doc2 (HTML) ---
            if USE_PARSING:
                print(f"\nIndexing {doc2_path.name} with ST...")
                if USE_ST:
                    try:
                        store.add_document(doc2_path, metadata={"source": "web", "language": "en"})
                    except safe_store.ConfigurationError as e:
                         print(f"  [SKIP] Could not index {doc2_path.name} with ST: {e}")
                else:
                    print(f"  [SKIP] ST vectorizer not available.")
            else:
                print(f"\nSkipping HTML indexing for {doc2_path.name}")

            # --- Index doc3 (initial version) ---
            print(f"\nIndexing {doc3_path.name} (v1) with ST...")
            if USE_ST:
                try:
                    store.add_document(doc3_path, metadata={"version": 1})
                except safe_store.ConfigurationError as e:
                     print(f"  [SKIP] Could not index {doc3_path.name} with ST: {e}")
            else:
                print(f"  [SKIP] ST vectorizer not available.")


            # --- Add TF-IDF Vectorization to all docs ---
            if USE_TFIDF:
                print_header("Adding TF-IDF Vectorization")
                try:
                    store.add_vectorization(
                        vectorizer_name="tfidf:my_tfidf",
                        # Let safe_store handle fitting on all docs
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not add TF-IDF vectorization: {e}")
                    USE_TFIDF = False # Disable TFIDF if failed
            else:
                print_header("Skipping TF-IDF Vectorization")


            # --- 4. Querying ---
            print_header("Querying")
            query_text = "vector database features"

            if USE_ST:
                print("\nQuerying with Sentence Transformer...")
                results_st = store.query(query_text, vectorizer_name="st:all-MiniLM-L6-v2", top_k=2)
                if results_st:
                    for i, res in enumerate(results_st):
                        print(f"  ST Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping ST Query.")

            if USE_TFIDF:
                print("\nQuerying with TF-IDF...")
                results_tfidf = store.query(query_text, vectorizer_name="tfidf:my_tfidf", top_k=2)
                if results_tfidf:
                    for i, res in enumerate(results_tfidf):
                        print(f"  TFIDF Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping TF-IDF Query.")


            # --- 5. File Updates & Re-indexing ---
            print_header("Updating and Re-indexing")
            print(f"Updating content of {doc3_path.name}...")
            doc3_content_v2 = "This content has been significantly updated for testing re-indexing."
            doc3_path.write_text(doc3_content_v2, encoding='utf-8')
            time.sleep(0.1) # Ensure file timestamp changes

            print(f"Running add_document again for {doc3_path.name}...")
            if USE_ST:
                store.add_document(doc3_path, metadata={"version": 2}) # Should detect change and re-index with ST
            else:
                 print("  Skipping re-indexing (ST not available).")
            # Note: TF-IDF vectors for the old chunks of doc3 were deleted by the re-index.
            # If we wanted TF-IDF for the *new* chunks, we'd need to run add_vectorization again.
            # Or, ideally, add_document could optionally re-vectorize for *all* methods. (Future enhancement)

            # --- 6. Listing ---
            print_header("Listing Contents")
            print("\n--- Documents ---")
            docs = store.list_documents()
            for doc in docs:
                print(f"- ID: {doc['doc_id']}, Path: {Path(doc['file_path']).name}, Hash: {doc['file_hash'][:8]}..., Meta: {doc.get('metadata')}")

            print("\n--- Vectorization Methods ---")
            methods = store.list_vectorization_methods()
            for method in methods:
                print(f"- ID: {method['method_id']}, Name: {method['method_name']}, Type: {method['method_type']}, Dim: {method['vector_dim']}, Fitted: {method.get('params',{}).get('fitted', 'N/A')}")

            # --- 7. Removing a Vectorization ---
            if USE_TFIDF:
                 print_header("Removing Vectorization")
                 print("Removing TF-IDF vectors...")
                 store.remove_vectorization("tfidf:my_tfidf")

                 print("\n--- Vectorization Methods After Removal ---")
                 methods_after = store.list_vectorization_methods()
                 for method in methods_after:
                      print(f"- ID: {method['method_id']}, Name: {method['method_name']}")
            else:
                 print_header("Skipping Vectorization Removal")


    except safe_store.ConfigurationError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Please install the required extras (e.g., pip install safe_store[all])")
    except safe_store.ConcurrencyError as e:
        print(f"\n[ERROR] Lock timeout or concurrency issue: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc() # Print traceback for unexpected errors
    finally:
        # Connection is closed automatically by 'with' statement
        print("\n--- End of Script ---")
        # Optional: uncomment cleanup() to remove files after run
        # cleanup()