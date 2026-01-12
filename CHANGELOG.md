# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

#
- feat: bump version to 3.3.4 and refine graph processing query# [Unreleased]

- feat(vectorization): add detailed ASCII info for vectorizer initialization

## [2026-01-12 01:05]

- feat(graph): add robust chunk-level error handling

## [2.6.5] - 2025-11-06

### Added

*   **Custom Chunk Processing:** Introduced an optional `chunk_processor` callable to the `add_document` and `add_text` methods. This function `(chunk_text: str, metadata: dict) -> str` allows for on-the-fly transformation of chunk content before it is vectorized and stored. This enables advanced RAG workflows like summarization, keyword injection, or reformatting.
*   Updated `examples/basic_usage.py` to include a demonstration of the new `chunk_processor` feature.

### Changed

*   **Query Result Structure:** The dictionary returned for each result in the `query()` method now distinctly provides both `document_metadata` (the raw dictionary) and `chunk_text` (which includes the prepended metadata context). This offers greater flexibility by giving access to both the ready-to-use fused text for RAG prompts and the separate, structured metadata for application logic.
*   **Documentation (`README.md`):** Updated the "Core Concepts for Advanced RAG" section to explain and provide an example for the new `chunk_processor` functionality.

## [2.6.0] - 2025-10-20

### Added

*   **Public API Exports:** Exposed core exceptions, `LogLevel`, and utility functions like `parse_document` directly under the `safe_store` namespace in `__init__.py` for easier user access and import.

### Changed

*   **Internal Refactoring:** Minor refactoring in the `store.py` module to improve code clarity and maintainability around the content addition pipeline.
*   **Project Version:** Bumped version to `2.6.0`.

## [2.5.0] - 2025-09-15

### Added

*   **Metadata-Aware Vectorization:** Added a `vectorize_with_metadata` boolean parameter to `add_document` and `add_text`. When `True`, the document's metadata is prepended to each chunk *before* vectorization, enriching the semantic meaning of the resulting vector. The stored chunk text remains unchanged.

### Fixed

*   Corrected an issue where an empty chunk list could cause an error during vectorization instead of being gracefully skipped.

## [2.4.0] - 2025-08-22

### Added

*   **Point Cloud Visualization Export:** Implemented the `export_point_cloud()` method in `SafeStore`. This feature performs PCA on all vectors in the store and exports the 2D coordinates along with metadata, enabling visualization of semantic clusters.
*   Added `examples/point_cloud_and_api.py`, a complete, runnable example that demonstrates how to use `export_point_cloud` to power an interactive web-based visualization with an API for inspecting chunks.

### Changed

*   **Dependencies:** `scikit-learn` and `pandas` are now optional dependencies required for the `export_point_cloud` feature.

## [2.3.0] - 2025-07-30

### Changed

*   **Vectorizer Manager Refactor:** The `VectorizationManager` now uses a more robust dynamic module loading system (`importlib`) to discover and instantiate both built-in and custom vectorizers, reducing hardcoded paths.
*   **Custom Vectorizers:** The process for adding custom vectorizers is now standardized: create a folder with `__init__.py` and `description.yaml`.

### Fixed

*   Fixed a bug where the `st` vectorizer alias was pointing to a misspelled folder name (`sentense_transformer` instead of `sentence_transformer`). The alias now correctly points to the right location, ensuring Sentence Transformer models load correctly.

## [2.1.0] - 2025-06-10

### Changed

*   **Graph Query Performance:** Optimized the graph traversal logic in `GraphStore.query_graph` by improving the underlying SQL queries for fetching neighbor nodes, reducing the number of database calls.

### Fixed

*   Resolved an issue in `GraphStore` where entity fusion could fail on documents containing complex JSON-like strings, by improving the robustness of the JSON parsing from the LLM response.
*   Fixed a bug where `GraphStore.build_graph_for_all_documents` would not correctly resume if interrupted.

## [2.0.0] - 2025-05-16 


### Added

*   **Comprehensive Graph Database Functionality (`GraphStore`):**
    *   Introduced the new `GraphStore` class, enabling `safe_store` to function as a hybrid vector and graph database, utilizing the same underlying SQLite backend. This allows users to build, manage, and query knowledge graphs extracted from their text data.
    *   **LLM-Powered Graph Construction:**
        *   `GraphStore` integrates with Large Language Models (LLMs) via a user-supplied `llm_executor_callback: Callable[[str_prompt], str_response]`.
        *   **Internalized Prompt Engineering:** `GraphStore` now manages and provides optimized, detailed prompts to the `llm_executor_callback` for:
            *   Extracting structured graph data (nodes with labels, properties, and a `unique_id_key`; relationships with source/target identifiers, types, and properties) from text chunks. Prompts instruct the LLM to return JSON within markdown code blocks.
        *   **Node De-duplication and Property Updates:** Implemented `add_or_update_graph_node` logic in the database layer, which identifies existing nodes via a `unique_signature` (derived from label and a normalized unique ID property) and updates their properties if new information is extracted, or creates new nodes otherwise.
        *   **Chunk-to-Graph Linking:** Extracted graph nodes are linked back to their originating text chunks in the `node_chunk_links` table, enabling traceability.
    *   **Graph Building API:**
        *   `GraphStore.process_chunk_for_graph(chunk_id)`: Processes a single text chunk to extract and store graph elements.
        *   `GraphStore.build_graph_for_document(doc_id)`: Builds graph data for all chunks within a specified document.
        *   `GraphStore.build_graph_for_all_documents()`: Iteratively processes all unprocessed chunks in the database to build or extend the graph, operating in batches.
        *   Processed chunks are timestamped (`graph_processed_at`) to avoid redundant processing.
    *   **LLM-Powered Natural Language Graph Querying:**
        *   `GraphStore.query_graph(natural_language_query, output_mode, ...)`:
            *   Utilizes the `llm_executor_callback` with an internal, sophisticated query-parsing prompt to translate natural language questions into structured parameters for graph traversal (e.g., identifying seed nodes, target relationship types, desired neighbor labels, traversal depth).
            *   Performs graph traversal (currently a Breadth-First Search-like approach up to a specified depth) based on the LLM-parsed query parameters.
            *   Offers flexible `output_mode`s:
                *   `"graph_only"`: Returns the relevant subgraph (nodes and relationships).
                *   `"chunks_summary"`: Returns a list of text chunk summaries linked to the nodes in the resulting subgraph, augmented with information about the linking graph nodes.
                *   `"full"`: Combines both graph data and linked chunk summaries.
    *   **Direct Graph Access API:**
        *   `get_node_details(node_id)`: Retrieves full details for a specific node.
        *   `get_nodes_by_label(label, limit)`: Fetches nodes matching a given label.
        *   `get_relationships(node_id, relationship_type, direction, limit)`: Retrieves relationships connected to a node, with filtering options.
        *   `find_neighbors(node_id, relationship_type, direction, limit)`: Finds neighbor nodes.
        *   `get_chunks_for_node(node_id, limit)`: Retrieves text chunks associated with a specific graph node.
    *   **Database Schema Enhancements:**
        *   Integrated new tables (`graph_nodes`, `graph_relationships`, `node_chunk_links`, `store_metadata`) into the SQLite schema.
        *   Added `graph_processed_at` column to the `chunks` table for tracking graph processing status.
        *   The `db.initialize_schema` function now creates and manages this extended schema.
    *   **Customizable Prompts:** Users can optionally provide their own `graph_extraction_prompt_template` and `query_parsing_prompt_template` during `GraphStore` initialization to override the library's defaults.
    *   **Concurrency and Encryption:** `GraphStore` respects the existing file-locking mechanism for concurrent write safety and can utilize the `Encryptor` (if an `encryption_key` is provided) to decrypt chunk text for LLM processing.
*   **New Graph-Specific Exceptions:** Introduced `GraphError`, `GraphDBError`, `GraphProcessingError`, and `LLMCallbackError` in `safe_store.core.exceptions`.
*   **Comprehensive Example:** Added `examples/graph_usage.py` to demonstrate the setup and usage of `GraphStore`, including graph building from documents and natural language querying, using `lollms-client` as an example LLM interface.

### Changed

*   **Primary `GraphStore` Callback:** The mechanism for LLM interaction in `GraphStore` has been unified. Instead of separate callbacks for graph extraction and query parsing, `GraphStore.__init__` now expects a single `llm_executor_callback: Callable[[str_prompt], str_response]`. `GraphStore` is now responsible for crafting the specific prompts for different LLM tasks (graph extraction, query parsing) and passes these complete prompts to this executor callback.
*   **Project Version:** Upgraded to `2.0.0` to signify this major functional expansion and the change in the primary LLM callback pattern for `GraphStore`.
*   **Documentation (`README.md`):** Extensively updated to cover the new `GraphStore` class, its features, the dual vector/graph database nature of the library, and revised examples.
*   **`safe_store/__init__.py`:** Exports the new `GraphStore` class and associated graph exceptions.

## [1.7.0] - 2025-05-15 
*(This version was in pyproject.toml but not explicitly in the previous changelog. Assuming it was an internal step or the version before this graph work)*
### Changed
*   Version bump for internal development or minor adjustments. *(Placeholder - adjust if specific changes were made for 1.7.0)*

## [1.6.0] - 2025-05-08
### Added
*   **File formats**
    *    Added more text based files format (as per `safe_store/indexing/parser.py` update)

### Fixed
*   Big change: the main class `SafeStore` is now called `SafeStore` (was `safe_store` before, this is a class name capitalization fix). *(This was listed as a fix for 1.6.0, but the class name was `SafeStore` in `store.py` for a while. Clarifying if this was a different instance or a documentation fix)*
*   Corrected the error message string raised by `add_vectorization` when TF-IDF fitting is attempted on encrypted chunks without providing the necessary `encryption_key`. The message now aligns with test expectations, stating: "Cannot fit TF-IDF on encrypted chunks without the correct encryption key."

## [1.4.0] - 2025-04-20
### Added
*   **Encryption:**
    *   Added optional encryption at rest for `chunk_text` using `cryptography` (Fernet/AES-128-CBC). Requires `safe_store[encryption]`.
    *   Added `encryption_key` parameter to `SafeStore.__init__`.
    *   Encryption/decryption is handled automatically during `add_document` and `query` if the key is present.
    *   `add_vectorization` now attempts decryption when fitting TF-IDF on potentially encrypted chunks.
    *   Added `safe_store.core.exceptions.EncryptionError`.
    *   Implemented `safe_store.security.encryption.Encryptor` class for handling key derivation (PBKDF2) and encryption/decryption. Uses a fixed salt (documented limitation).
    *   Added tests for encryption logic (`test_encryption.py`, `test_store_encryption.py`).
    *   Added `examples/encryption_usage.py`.
    *   Added documentation (`encryption.rst`) explaining the feature, usage, and security considerations.
*   **Documentation:**
    *   Created Sphinx documentation structure under `docs/`.
    *   Added content for `conf.py`, `index.rst`, `installation.rst`, `quickstart.rst`, `api.rst`, `logging.rst`, `encryption.rst`.
    *   Added `docs/requirements.txt`.
    *   Added `sphinx` and `sphinx-rtd-theme` to `[dev]` extras in `pyproject.toml`.
*   **Examples:**
    *   Implemented `examples/custom_logging.py` demonstrating global `ascii_colors` configuration.
*   **Testing:**
    *   Added comprehensive tests for encryption functionality.
    *   Added tests for store closure (`close()`) and context manager (`__enter__`/`__exit__`) behavior.
    *   Added tests for `list_documents` and `list_vectorization_methods`.
    *   Ensured tests relying on optional dependencies are skipped correctly or use mocks when dependencies are unavailable.

### Changed
*   `pyproject.toml`: Bumped version to 1.4.0. Added `cryptography` dependency and `[encryption]` extra. Updated `[all]` and `[dev]` extras. Updated classifiers.
*   `README.md`: Significantly updated with encryption feature, detailed logging section, refined quick start, installation instructions, and feature list.
*   `safe_store/__init__.py`: Bumped version. Exposed `EncryptionError`.
*   `safe_store/store.py`: Integrated `Encryptor`. Modified indexing and querying methods to handle encryption/decryption. Added related error handling.
*   `tests/conftest.py`: Improved conditional mocking setup for optional dependencies.
*   Refined type hints and docstrings across modules for better clarity and consistency.

### Fixed
*   Corrected assertions in `test_store_phase2.py` related to TF-IDF fitting log and cache removal log messages.
*   Ensured `TfidfVectorizerWrapper.load_fitted_state` robustly reconstructs the internal state required for `transform` to work after loading.
*   Addressed potential `TypeError` if encrypted chunk data was somehow stored as string instead of bytes during decryption.
*   Fixed potential state inconsistencies in `TfidfVectorizerWrapper` if fitting failed or was performed on empty text.

## [1.3.0] - 2025-04-19
### Added
*   **API Polish & Type Hinting:** Comprehensive type hints and improved docstrings.
*   **Error Handling:** Consistent use of custom exceptions from `safe_store.core.exceptions` with proper chaining and clearer messages.
*   **Documentation Structure:** Basic Sphinx setup.
*   **Examples:** Added `examples/basic_usage.py`.
*   **Helper Methods:** `SafeStore.list_documents()`, `SafeStore.list_vectorization_methods()`.
*   **Testing:** Basic tests for `close()`, context manager, and new list methods.

### Changed
*   **Dependencies:** Finalized optional dependencies and extras in `pyproject.toml`. Added `[dev]` extra.
*   `pyproject.toml`: Bumped version to 1.3.0. Added project keywords, refined classifiers, URLs. Configured `hatch` for version management. Added `black` config.
*   `README.md`: Significantly updated with latest features, improved explanations.
*   `safe_store.core.db`: Enabled SQLite `PRAGMA foreign_keys = ON`. Set `check_same_thread=False`. Improved error handling.
*   Improved error handling in `parser.py`, `vectorization/manager.py`, `vectorization/methods`, `search/similarity.py`.
*   `safe_store/__init__.py`: Exposed core exceptions.

### Fixed
*   Corrected potential race condition in `VectorizationManager.get_vectorizer`.
*   Ensured `TfidfVectorizerWrapper.load_fitted_state` correctness.
*   Fixed `TfidfVectorizerWrapper.vectorize` dimension check.
*   Ensured `SafeStore` context manager properly re-initializes connection.

## [1.2.0] - 2025-04-18
### Added
*   **Concurrency Handling:** Implemented `filelock` for inter-process write safety (`.db.lock`). Added `lock_timeout` to `SafeStore.__init__`. Basic `threading.RLock` for intra-process safety.
*   **Parsing Infrastructure (Implemented):** Added optional dependencies and parsers for PDF (`pypdf`), DOCX (`python-docx`), HTML (`beautifulsoup4`, `lxml`). Updated `parse_document` dispatcher.

### Changed
*   `safe_store.store.SafeStore`: Integrated locking into `__init__`, `close`, context manager, and write methods.
*   `pyproject.toml`: Bumped version to 1.2.0. Added new parsing dependencies and `[parsing]`, `[all]` extras.
*   `safe_store.core.db`: Path types now `Union[str, Path]`.

### Fixed
*   Minor fix in `VectorizationManager.update_method_params` SQL.
*   Corrected assertions in `test_store_phase2.py`.
*   Resolved issues in parser implementations.

## [1.1.0] - 2025-04-17
### Added
*   **Querying:** Implemented `SafeStore.query()` using cosine similarity.
*   **Multiple Vectorization Methods:** Added TF-IDF support (`tfidf:` prefix) using `scikit-learn`. Handles fitting and stores state in DB. Requires `safe_store[tfidf]`.
*   **Vectorizer Management:** Implemented `SafeStore.add_vectorization()` and `SafeStore.remove_vectorization()`.
*   **Testing:** Added `tests/test_store_phase2.py`.

### Changed
*   `VectorizationManager`: Caches DB parameters; invalidates on update.
*   `SafeStore.add_document()`: Handles initial TF-IDF fit.
*   `search.similarity.cosine_similarity`: Handles 1D `vectors` input.
*   `pyproject.toml`: Bumped version to 1.1.0. Added `scikit-learn` to `[tfidf]` extra.

## [1.0.0] - 2025-04-16
### Added
*   Initial public release.
*   Core `SafeStore` class and SQLite backend (WAL mode).
*   Database schema (`documents`, `chunks`, `vectorization_methods`, `vectors`).
*   Indexing pipeline (`add_document`): `.txt` parsing, full text storage, chunking, file hashing, change detection.
*   Vectorization foundation: `VectorizationManager`, initial `sentence-transformers` support (`st:` prefix).
*   Utilities: `ascii_colors` for logging.
*   Testing: `pytest` suite.
*   Packaging: `pyproject.toml` setup.