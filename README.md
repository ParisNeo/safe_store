# SafeStore: Simple, Concurrent SQLite Vector Store for Local RAG

[![PyPI version](https://img.shields.io/pypi/v/safestore.svg)](https://pypi.org/project/safestore/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safestore.svg)](https://pypi.org/project/safestore/)
[![PyPI license](https://img.shields.io/pypi/l/safestore.svg)](https://github.com/ParisNeo/safestore/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safestore.svg)](https://pypi.org/project/safestore/)
[![Tests](https://github.com/ParisNeo/safestore/actions/workflows/test.yml/badge.svg)](https://github.com/ParisNeo/safestore/actions/workflows/test.yml) <!-- Add CI Badge later -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**SafeStore** is a Python library providing a lightweight, file-based vector database using **SQLite**. It's designed for simplicity and efficiency, making it ideal for integrating into **local Retrieval-Augmented Generation (RAG)** pipelines.

Store, manage, and query your document embeddings locally with features like automatic change detection, support for multiple vectorization methods, safe concurrent access, and various document parsers.

---

## ✨ Why SafeStore?

*   **🎯 RAG Focused:** Built with local RAG use cases as a primary goal.
*   **🚀 Simple & Lightweight:** Uses a single SQLite file – no heavy dependencies or external database servers needed. Easy to deploy and manage.
*   **🏠 Local First:** Keep your embeddings and document text entirely on your local machine or network share.
*   **🤝 Concurrent Safe:** Uses file-based locking (`filelock`) to safely handle database writes from **multiple processes**, preventing data corruption. Read operations are designed to be concurrent using SQLite's WAL mode.
*   **🧠 Multiple Vectorizers:** Index documents using different embedding models (e.g., Sentence Transformers, TF-IDF) side-by-side and query using the method you choose.
*   **📄 Document Parsing:** Built-in parsers for `.txt`, `.pdf`, `.docx`, and `.html` files (requires optional `[parsing]` dependencies).
*   **🔍 Efficient Querying:** Find relevant document chunks based on cosine similarity to your query text.
*   **🔄 Change Aware:** Automatically detects changes in source files (via hashing) and efficiently re-indexes only modified documents.
*   **⚙️ Flexible:** Configurable text chunking (`chunk_size`, `chunk_overlap`). Add or remove vectorization methods as needed.
*   **🗣️ Informative Logging:** Uses [`ascii_colors`](https://github.com/ParisNeo/ascii_colors) for clear, leveled, and colorful console feedback by default. Easily configurable for different levels or file output.
*   **🔒 Secure (Future):** Optional encryption at rest for stored text is planned.

---

## ⚠️ Status: Beta

SafeStore is currently in Beta. The core API is stabilizing, but breaking changes are still possible before a `2.0.0` release. Feedback and contributions are welcome!

---

## ⚙️ Features

*   **Storage:** All data (documents, chunks, vectors, metadata) stored in a single SQLite file (`.db`).
*   **Concurrency:** Process-safe writes using `.db.lock` file (`filelock`). Concurrent reads enabled by SQLite's WAL mode. Configurable lock timeout.
*   **Indexing (`add_document`):**
    *   Parses `.txt`, `.pdf`, `.docx`, `.html`/`.htm` files (requires `safestore[parsing]`).
    *   Stores full original text for potential future re-indexing.
    *   Configurable character-based chunking with overlap.
    *   Stores chunk position (start/end characters).
    *   Calculates file hash (SHA256) for change detection.
    *   Automatically re-indexes if file content changes.
    *   Skips processing if file is unchanged and vectors exist for the specified method.
    *   Associates optional JSON metadata with documents.
*   **Vectorization:**
    *   Supports multiple vectorization methods simultaneously.
    *   **Sentence Transformers:** Integrates `sentence-transformers` models (e.g., `st:all-MiniLM-L6-v2`). Requires `safestore[sentence-transformers]`.
    *   **TF-IDF:** Integrates `scikit-learn`'s `TfidfVectorizer` (e.g., `tfidf:my_method`). Requires `safestore[tfidf]`. Handles fitting and stores vocabulary/IDF weights in the database.
    *   Stores method details (name, type, dimension, data type, parameters) in DB.
*   **Querying (`query`):**
    *   Find `top_k` chunks based on cosine similarity to a query text.
    *   Specify which `vectorizer_name` to use for embedding the query and retrieving vectors.
    *   Returns ranked results including chunk text, similarity score, source document path, position, and metadata.
*   **Management Methods:**
    *   **`add_vectorization`:** Adds embeddings for a *new* method to existing documents without re-parsing/re-chunking. Fits TF-IDF if needed (on all target docs or globally).
    *   **`remove_vectorization`:** Deletes a vectorization method and all its associated vectors from the database and cache.
    *   **`list_documents`:** Returns a list of stored documents and their metadata.
    *   **`list_vectorization_methods`:** Returns details of registered vectorization methods.
*   **Logging:** Rich console logging via `ascii_colors`. Default level is `INFO`. Configurable via `SafeStore(log_level=...)` or globally using `ASCIIColors` static methods (see [Logging](#logging) section).

---

## 🚀 Installation

```bash
pip install safestore
```

Install optional dependencies based on the features you need:

```bash
# For Sentence Transformers embedding models (recommended default)
pip install safestore[sentence-transformers]

# For TF-IDF vectorization (requires scikit-learn)
pip install safestore[tfidf]

# For parsing PDF, DOCX, and HTML files
pip install safestore[parsing] # Includes pypdf, python-docx, beautifulsoup4, lxml

# To install everything (all vectorizers, all parsers):
pip install safestore[all]

# Or install specific combinations:
pip install safestore[sentence-transformers,parsing]
```

---

## 🏁 Quick Start

```python
import safestore
from pathlib import Path
import time # For demonstrating concurrency

# --- 1. Prepare Sample Documents ---
doc_dir = Path("my_docs")
doc_dir.mkdir(exist_ok=True)
doc1_path = doc_dir / "doc1.txt"
doc1_path.write_text("SafeStore makes local vector storage simple and efficient.", encoding='utf-8')
doc2_path = doc_dir / "doc2.html"
doc2_path.write_text("<html><body><p>HTML content can also be indexed.</p></body></html>", encoding='utf-8')

print(f"Created sample files in: {doc_dir.resolve()}")

# --- 2. Initialize SafeStore ---
# Use DEBUG level for more verbose output, adjust lock timeout if needed
store = safestore.SafeStore(
    "my_vector_store.db",
    log_level=safestore.LogLevel.DEBUG,
    lock_timeout=10 # Wait up to 10s for write lock
)

# Best practice: Use SafeStore as a context manager
try:
    with store:
        # --- 3. Add Documents (acquires write lock) ---
        print("\n--- Indexing Documents ---")
        # Requires safestore[sentence-transformers]
        store.add_document(doc1_path, vectorizer_name="st:all-MiniLM-L6-v2", chunk_size=50, chunk_overlap=10)

        # Requires safestore[parsing] for HTML
        store.add_document(doc2_path, vectorizer_name="st:all-MiniLM-L6-v2")

        # Add TF-IDF vectors as well (requires safestore[tfidf])
        # This will fit TF-IDF on all documents
        print("\n--- Adding TF-IDF Vectorization ---")
        store.add_vectorization("tfidf:my_analysis")

        # --- 4. Query (read operation, concurrent with WAL) ---
        print("\n--- Querying using Sentence Transformer ---")
        query_st = "simple storage"
        results_st = store.query(query_st, vectorizer_name="st:all-MiniLM-L6-v2", top_k=2)
        for i, res in enumerate(results_st):
            print(f"ST Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")

        print("\n--- Querying using TF-IDF ---")
        query_tfidf = "html index"
        results_tfidf = store.query(query_tfidf, vectorizer_name="tfidf:my_analysis", top_k=2)
        for i, res in enumerate(results_tfidf):
            print(f"TFIDF Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")

        # --- 5. List Methods ---
        print("\n--- Listing Vectorization Methods ---")
        methods = store.list_vectorization_methods()
        for method in methods:
            print(f"- ID: {method['method_id']}, Name: {method['method_name']}, Type: {method['method_type']}, Dim: {method['vector_dim']}")

except safestore.ConfigurationError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Please install the required extras (e.g., pip install safestore[all])")
except safestore.ConcurrencyError as e:
    print(f"\n[ERROR] Lock timeout or concurrency issue: {e}")
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")
finally:
    # Connection is closed automatically by the 'with' statement exit
    print("\n--- Store context closed ---")
    # Cleanup (optional)
    # import shutil
    # shutil.rmtree(doc_dir)
    # Path("my_vector_store.db").unlink(missing_ok=True)
    # Path("my_vector_store.db.lock").unlink(missing_ok=True)
    # Path("my_vector_store.db-wal").unlink(missing_ok=True) # WAL file
    # Path("my_vector_store.db-shm").unlink(missing_ok=True) # Shared memory file

print("\nCheck 'my_vector_store.db' and console logs.")
```

*(See `examples/` directory for this and other usage examples.)*

---

## 💡 Concurrency

SafeStore uses `filelock` to provide process-safe write operations (`add_document`, `add_vectorization`, `remove_vectorization`). When one process performs a write, other processes attempting a write will wait up to the configured `lock_timeout`.

Read operations (`query`, `list_*`) are designed to be concurrent with writes thanks to SQLite's WAL (Write-Ahead Logging) mode, which is enabled by default. Multiple processes can typically read the database simultaneously, even while another process is writing.

---

## 🪵 Logging

SafeStore uses the [`ascii_colors`](https://github.com/ParisNeo/ascii_colors) library for flexible and colorful console logging.

*   **Default Level:** `INFO`. Only INFO, SUCCESS, WARNING, ERROR, CRITICAL messages are shown.
*   **Change Level:** Initialize with `SafeStore(log_level=safestore.LogLevel.DEBUG)` or `SafeStore(log_level=safestore.LogLevel.WARNING)` etc.
*   **Global Configuration:** You can configure `ascii_colors` globally in your application *before* initializing SafeStore:
    ```python
    import safestore
    from ascii_colors import ASCIIColors, LogLevel, FileHandler, Formatter

    # Set global log level (affects SafeStore and other uses of ascii_colors)
    ASCIIColors.set_log_level(LogLevel.DEBUG)

    # Add logging to a file
    file_handler = FileHandler("safestore_app.log", encoding='utf-8')
    file_handler.set_formatter(Formatter("%(asctime)s - %(levelname)s - %(message)s")) # Basic format
    ASCIIColors.add_handler(file_handler)

    # Optional: Remove default console handler if you only want file logging
    # ASCIIColors.remove_handler(ASCIIColors.get_default_handler())

    # Now initialize SafeStore - it will use the global settings
    store = safestore.SafeStore("my_store.db")
    # ... use store ...
    ```

---

## 🔮 Future Work (Planned Features)

*   **Encryption:** Optional encryption at rest for `full_text` and `chunk_text`.
*   **Re-indexing:** `reindex()` method to re-process documents with new chunking or other parameters without needing the original file.
*   **More Vectorizers:** Integrations for OpenAI, Cohere, Ollama embeddings.
*   **Metadata Filtering:** Allow filtering query results based on document metadata (e.g., `query(..., metadata_filter={'year': 2023})`).
*   **Tagging:** Optional automatic tag extraction from chunks during indexing.
*   **Performance:** Explore optimizations like ANN indexing (e.g., via Faiss or HNSWlib integration) for very large datasets, potentially as an optional backend.
*   **Async API:** Consider adding an asynchronous interface using `aiosqlite`.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ParisNeo/safestore).

See `CONTRIBUTING.md` (to be added) for more detailed guidelines.

---

## 📜 License

SafeStore is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.