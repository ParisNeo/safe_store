
**Project Name:** safe_store

**Goal:** A Python library providing an SQLite3-based vector database optimized for LLM RAG integration. It features on-demand vector loading, support for multiple document types, multiple vectorization methods, optional encryption, concurrency considerations, re-indexing capabilities, and utilizes `ascii_colors` for rich, leveled logging and console feedback.

**Core Principles:**

1.  **Simplicity:** Easy to install, configure, and use API. Leverages `ascii_colors` for straightforward, colorful console feedback by default.
2.  **Efficiency:** Optimized vector loading and search (NumPy), mindful memory usage.
3.  **Flexibility:** Support for various document types, chunking strategies, and vectorization models.
4.  **Robustness:** Comprehensive testing, error handling, clear logging via `ascii_colors`, and concurrency awareness.
5.  **Security:** Optional data encryption at rest.
6.  **Maintainability:** Well-structured code, clear documentation, standard packaging.

---

**Phase 0: Setup & Foundational Design**

*   **Objective:** Set up the project structure, development environment, basic database schema, initial configuration, and integrate `ascii_colors` for logging.
*   **Tasks:**
    1.  **Project Initialization:**
        *   Create the main project directory (`safe_store-project`).
        *   Initialize Git repository (`git init`).
        *   Create `pyproject.toml` using a modern build backend (e.g., `hatch`, `flit`). Define project metadata, dependencies (`sqlite3`, `numpy`, `ascii_colors`), and optional dependencies.
        *   Create `.gitignore`.
        *   Set up virtual environment (`python -m venv .venv`, `source .venv/bin/activate`).
        *   Install initial dependencies (`pip install -e .`).
    2.  **Directory Structure:** (Remains the same as the previous plan)
        ```
        safe_store-project/
        ├── safe_store/
        │       ├── __init__.py       # Public API exports
        │       ├── core/             # Core functionalities
        │       │   ├── __init__.py
        │       │   ├── db.py         # Database connection, schema, core CRUD
        │       │   ├── models.py     # Pydantic or dataclass models for DB objects
        │       │   └── exceptions.py # Custom exceptions
        │       ├── indexing/         # Document processing, chunking
        │       │   ├── __init__.py
        │       │   ├── parser.py     # Document type parsing logic
        │       │   └── chunking.py   # Text splitting logic
        │       ├── vectorization/    # Vectorization logic
        │       │   ├── __init__.py
        │       │   ├── base.py       # Base vectorizer class/interface
        │       │   ├── methods/      # Specific implementations (tfidf, sentence_transformer, etc.)
        │       │   │   └── __init__.py
        │       │   └── manager.py    # Manages different vectorizers
        │       ├── search/           # Search/query logic
        │       │   ├── __init__.py
        │       │   └── similarity.py # Vector comparison logic
        │       ├── security/         # Encryption features
        │       │   ├── __init__.py
        │       │   └── encryption.py
        │       ├── utils/            # Helper functions
        │       │   ├── __init__.py
        │       │   └── concurrency.py # Locking mechanisms
        │       ├── store.py            # Main file
        │       └── config.py         # Configuration handling (defaults, env vars)
        ├── tests/                    # Test suite
        │   ├── __init__.py
        │   ├── conftest.py         # Pytest fixtures
        │   ├── core/
        │   ├── indexing/
        │   ├── vectorization/
        │   ├── search/
        │   ├── security/
        │   └── fixtures/             # Sample test files (txt, pdf, etc.)
        ├── examples/                 # Usage examples scripts
        ├── docs/                     # Documentation (Sphinx source)
        ├── pyproject.toml            # Build config, dependencies
        ├── README.md
        ├── LICENSE
        └── .gitignore
        ```
    3.  **Initial Database Schema Design (`core/db.py`):** (Remains the same)
        *   `documents`: `doc_id (PK)`, `file_path (TEXT UNIQUE)`, `file_hash (TEXT)`, `full_text (BLOB/TEXT)`, `metadata (JSON)`, `added_timestamp (DATETIME)`
        *   `vectorization_methods`: `method_id (PK)`, `method_name (TEXT UNIQUE)`, `method_type (TEXT)`, `vector_dim (INTEGER)`, `params (JSON)`
        *   `chunks`: `chunk_id (PK)`, `doc_id (FK)`, `chunk_text (TEXT)`, `start_pos (INT)`, `end_pos (INT)`, `chunk_seq (INT)`, `tags (JSON, nullable)`, `is_encrypted (BOOL)`, `encryption_metadata (BLOB, nullable)`
        *   `vectors`: `vector_id (PK)`, `chunk_id (FK)`, `method_id (FK)`, `vector_data (BLOB)`
        *   Define functions for initializing the database and tables.
    4.  **Configuration (`config.py`):** Plan for handling API keys, default settings.
    5.  **Logging Setup:**
        *   No specific setup needed in `__init__.py` for basic console logging with `ascii_colors`. The library is ready to use via static methods (`ASCIIColors.info`, etc.).
        *   Plan to use `ascii_colors` static methods throughout the library code.
    6.  **Basic Testing Setup (`tests/`):**
        *   Set up `pytest`.
        *   Prepare to use `pytest`'s `capsys` or `capfd` fixtures to capture console output for testing log messages, as `caplog` is specific to standard `logging`.

---

**Phase 1: Core Indexing and Vectorization (Single Method)**

*   **Objective:** Implement document ingestion, chunking, storage, and vectorization using *one* initial method (e.g., Sentence Transformers), using `ascii_colors` for logging.
*   **Tasks:**
    1.  **Document Parsing (`indexing/parser.py`):**
        *   Implement parsing for TXT files.
        *   Add dependencies: `[parsing]` extra in `pyproject.toml`.
        *   Use `ASCIIColors.debug()` or `ASCIIColors.info()` for parsing steps (e.g., `ASCIIColors.debug(f"Parsing file: {file_path}")`).
    2.  **Chunking (`indexing/chunking.py`):**
        *   Implement a configurable chunking strategy.
        *   Ensure accurate start/end position tracking.
        *   Log chunking process details using `ASCIIColors` levels (e.g., `ASCIIColors.debug(f"Chunked text into {num_chunks} parts")`).
    3.  **Vectorization Interface (`vectorization/base.py`, `vectorization/methods/`):**
        *   Define `BaseVectorizer` abstract class.
        *   Implement `SentenceTransformerVectorizer`. Add `sentence-transformers` as optional dependency (`[sentence-transformers]`).
        *   Add `VectorizationManager` (`vectorization/manager.py`). Use `ASCIIColors.info` or `ASCIIColors.debug` for vectorizer loading messages.
    4.  **Indexing Workflow (`safe_store` class method `add_document`):**
        *   Create the main `safe_store` class (`safe_store/store.py` or similar).
        *   Implement `add_document(...)`.
        *   Use `ASCIIColors.info()`, `ASCIIColors.debug()`, `ASCIIColors.warning()`, `ASCIIColors.error()` to report progress, issues, and details (e.g., `ASCIIColors.info(f"Starting indexing for file: {filename}")`, `ASCIIColors.debug("Parsed document content.")`, `ASCIIColors.info(f"Vectorizing {num_chunks} chunks using '{vectorizer_name}'...")`, `ASCIIColors.warning(f"Document {filename} already exists, skipping/updating.")`).
        *   Parse, store `full_text`, chunk, get vectorizer, register method in DB, vectorize chunks, store chunks and vectors. Use DB transactions.
    5.  **Core DB Operations (`core/db.py`):** Implement CRUD functions. Potentially log errors using `ASCIIColors.error` or slow operations using `ASCIIColors.debug`.
    6.  **Testing:** Unit tests for parsing, chunking, vectorizer wrapper. Integration test `add_document`, verifying DB state. Use `capsys` fixture in pytest to check for expected console log messages.

---

**Phase 2: Querying and Multiple Vectorization Methods**

*   **Objective:** Implement core querying and support adding/using multiple vectorization methods, logging with `ascii_colors`.
*   **Tasks:**
    1.  **Similarity Search (`search/similarity.py`):** Implement efficient cosine similarity using `numpy`.
    2.  **Query Workflow (`safe_store` method `query`):**
        *   Implement `query(self, query_text: str, vectorizer_name: str, top_k: int = 5) -> list[dict]`.
        *   Use `ASCIIColors` to log query steps (e.g., `ASCIIColors.info("Received query.")`, `ASCIIColors.debug(f"Vectorizing query with method '{vectorizer_name}'")`, `ASCIIColors.debug(f"Loading {top_k} candidate vectors from DB")`, `ASCIIColors.debug("Calculating similarities...")`, `ASCIIColors.info(f"Found {len(results)} relevant chunks.")`).
        *   Get vectorizer, vectorize query, load vectors, calculate similarities, retrieve chunk/doc details, return results.
    3.  **Multiple Vectorizers:**
        *   Implement wrappers for TF-IDF, OpenAI, Ollama in `vectorization/methods/`. Add dependencies (`[tfidf]`, `[openai]`, `[ollama]`).
        *   Update `VectorizationManager`.
        *   Modify `add_document` to handle `vectorizer_name`.
        *   Implement `add_vectorization(self, vectorizer_name: str, ...)`: Log start/end and progress using `ASCIIColors` (e.g., `ASCIIColors.info(f"Adding vectorization method '{vectorizer_name}' to all documents.")`, `ASCIIColors.debug(f"Processing document ID {doc_id} for new vectors...")`).
        *   Implement `remove_vectorization(self, vectorizer_name: str)`: Log the removal action (`ASCIIColors.warning(f"Removing vectorization method '{vectorizer_name}' and associated vectors...")`).
    4.  **Tag Extraction (Optional):** Implement optional tag extraction during chunking, store in `chunks.tags`. Log if tagging is enabled/disabled (`ASCIIColors.debug`).
    5.  **Testing:** Unit test similarity, new vectorizer wrappers (mock APIs). Integration test `query` with different vectorizers, `add_vectorization`, `remove_vectorization`. Verify console output using `capsys`.

---

**Phase 3: Advanced Features & Robustness**

*   **Objective:** Add more parsers, encryption, re-indexing, and concurrency handling, with `ascii_colors` logging.
*   **Tasks:**
    1.  **Expanded Document Parsing (`indexing/parser.py`):** Add PDF, DOCX, HTML support. Add dependencies (`[pdf]`, `[docx]`, `[html]`). Refactor parser selection logic. Log which parser is being used (`ASCIIColors.debug`).
    2.  **Encryption (`security/encryption.py`, `safe_store` methods):**
        *   Add `cryptography` dependency (`[encryption]`).
        *   Implement symmetric encryption.
        *   Modify `add_document` and `query` to handle optional encryption/decryption. Log clearly when encryption is active/inactive (`ASCIIColors.info("Encryption enabled.")` or `ASCIIColors.debug("Encrypting chunk data...")`). Log errors using `ASCIIColors.error`.
        *   Update `chunks` table schema.
        *   **Crucially document:** User manages the key.
    3.  **Re-indexing (`safe_store` method `reindex`):**
        *   Implement `reindex(self, new_chunk_size: int = None, ...)`:
        *   Log start/end (`ASCIIColors.info("Starting re-indexing process...")`) and parameters (`ASCIIColors.debug(f"Using new chunk size: {new_chunk_size}")`).
        *   Log progress per document (`ASCIIColors.debug(f"Re-indexing document ID {doc_id}...")`).
        *   Consider using `ASCIIColors.execute_with_animation` for long-running vectorization steps within re-indexing for better user feedback if run interactively (though this might complicate testing/non-interactive use). Stick to standard logging messages primarily.
        *   Retrieve `full_text`, re-chunk, delete old, store new, re-vectorize, store new vectors.
    4.  **Concurrency Handling (`utils/concurrency.py`, `core/db.py`):**
        *   Enable SQLite WAL mode. Log the mode setting (`ASCIIColors.debug("Setting journal mode to WAL.")`).
        *   Implement file-based locking (`filelock` library). Log lock acquisition/release attempts/status at `DEBUG` level (`ASCIIColors.debug("Attempting to acquire write lock...")`, `ASCIIColors.debug("Write lock acquired/released.")`). Log lock timeouts or errors using `ASCIIColors.warning` or `ASCIIColors.error`.
        *   Document concurrency strategy.
    5.  **Testing:** Test new parsers. Test encryption/decryption. Test `reindex`. Test concurrency scenarios. Check console output for expected messages via `capsys`.

---

**Phase 4: Polish, Packaging, and Distribution**

*   **Objective:** Finalize library for release: documentation, examples, packaging, final testing.
*   **Tasks:**
    1.  **API Refinement:** Clarity, consistency, type hints.
    2.  **Error Handling:** Use custom exceptions (`exceptions.py`) and log errors using `ASCIIColors.error`, potentially with `exc_info=True` or `trace_exception`.
    3.  **Documentation (`docs/`):**
        *   Set up Sphinx.
        *   Write comprehensive docs: Installation, Quick Start, API Reference, Tutorials.
        *   **Add a dedicated "Logging / Console Output" section:** Explain that safe_store uses `ascii_colors` internally, providing colorful console output by default (INFO level and above). Show users how *they* can configure `ascii_colors` in *their application* to:
            *   Change the global level: `ASCIIColors.set_log_level(LogLevel.DEBUG)`
            *   Add file logging: `ASCIIColors.add_handler(FileHandler(...))`
            *   Use JSON logging: `ASCIIColors.add_handler(FileHandler(..., formatter=JSONFormatter(...)))`
            *   Customize formats: `handler.set_formatter(Formatter(...))`
            *   Disable default console output if needed: `ASCIIColors.clear_handlers()` before adding their own.
        *   Explain encryption key management.
    4.  **Examples (`examples/`):** Create clear example scripts, including one demonstrating custom `ascii_colors` configuration (e.g., logging to a file).
    5.  **README (`README.md`):** Update overview, features, install, basic usage, link to docs. Mention the use of `ascii_colors` for logging.
    6.  **Packaging (`pyproject.toml`):** Finalize dependencies/extras (`[all]`). Configure build settings. Include `LICENSE`. Build `sdist` and `wheel`.
    7.  **Final Testing:** Full test suite (`pytest`). Manual testing via examples. Performance benchmarks (optional). Test installation from built artifacts. Verify key log messages appear correctly on the console during tests using `capsys`.
    8.  **PyPI Release:** TestPyPI upload/install, then final PyPI upload.

---

**Cross-Cutting Concerns (Throughout all phases):**

*   **Logging:** Use `ascii_colors` static methods (`ASCIIColors.debug`, `.info`, `.warning`, `.error`) throughout the library code. Use appropriate levels. Document clearly how end-users can configure `ascii_colors` globally in their application to manage output level, format, and destinations (console, file, etc.).
*   **Code Quality:** Use `black`, `ruff`/`flake8`, `mypy`. Integrate into CI/CD if possible.
*   **Testing:** Write tests concurrently. Aim for high coverage. Use mocking (`unittest.mock`), pytest fixtures (`tmp_path`). Use `capsys` or `capfd` to capture and assert console output generated by `ascii_colors`.
*   **Dependencies:** Manage via `pyproject.toml`. Keep updated. Use optional extras. Ensure `ascii_colors` is listed as a core dependency.
