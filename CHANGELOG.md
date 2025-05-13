# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [1.6.0] - 2025-05-08
### Added

*   **File formats**
    *    Added more text based files format
## [1.6.0] - 2025-05-08
### Fixed
*   Big change: the main class `safe_store` is now called `SafeStore`.
*   Corrected the error message string raised by `add_vectorization` when TF-IDF fitting is attempted on encrypted chunks without providing the necessary `encryption_key`. The message now aligns with test expectations, stating: "Cannot fit TF-IDF on encrypted chunks without the correct encryption key."

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