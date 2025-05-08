# safe_store: Simple, Concurrent SQLite Vector Store for Local RAG

[![PyPI version](https://img.shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img.shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe_store.svg)](https://pypi.org/project/safe_store/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml/badge.svg)](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml) <!-- Add CI badge if applicable -->

**safe_store** is a Python library providing a lightweight, file-based vector database using **SQLite**. It's designed for simplicity and efficiency, making it ideal for integrating into **local Retrieval-Augmented Generation (RAG)** pipelines.

Store, manage, and query your document embeddings locally with features like automatic change detection, support for multiple vectorization methods, safe concurrent access, various document parsers, and **optional encryption at rest**.

---

## ✨ Why safe_store?

*   **🎯 RAG Focused:** Built with local RAG use cases as a primary goal.
*   **🚀 Simple & Lightweight:** Uses a single SQLite file – no heavy dependencies or external database servers needed. Easy to deploy and manage.
*   **🏠 Local First:** Keep your embeddings and document text entirely on your local machine or network share.
*   **🤝 Concurrent Safe:** Uses file-based locking (`filelock`) to safely handle database writes from **multiple processes**, preventing data corruption. Read operations are designed to be concurrent using SQLite's WAL mode.
*   **🧠 Multiple Vectorizers:** Index documents using different embedding models (e.g., Sentence Transformers, TF-IDF) side-by-side and query using the method you choose.
*   **📄 Document Parsing:** Built-in parsers for `.txt`, `.pdf`, `.docx`, and `.html` files (requires optional `[parsing]` dependencies).
*   **🔒 Optional Encryption:** Encrypts document chunk text at rest using AES-128 (via `cryptography` library) for enhanced security. Requires `safe_store[encryption]`.
*   **🔍 Efficient Querying:** Find relevant document chunks based on cosine similarity to your query text.
*   **🔄 Change Aware:** Automatically detects changes in source files (via hashing) and efficiently re-indexes only modified documents.
*   **⚙️ Flexible:** Configurable text chunking (`chunk_size`, `chunk_overlap`). Add or remove vectorization methods as needed.
*   **🗣️ Informative Logging:** Uses [`ascii_colors`](https://github.com/ParisNeo/ascii_colors) for clear, leveled, and colorful console feedback by default. Easily configurable for different levels or file output.

---

## ⚠️ Status: Beta

safe_store is currently in Beta. The core API is stabilizing, but breaking changes are still possible before a `2.0.0` release. Feedback and contributions are welcome!

---

## ⚙️ Features

*   **Storage:** All data (documents, chunks, vectors, metadata) stored in a single SQLite file (`.db`).
*   **Concurrency:** Process-safe writes using `.db.lock` file (`filelock`). Concurrent reads enabled by SQLite's WAL mode. Configurable lock timeout.
*   **Indexing (`add_document`):**
    *   Parses `.txt`, `.pdf`, `.docx`, `.html`/`.htm` files (requires `safe_store[parsing]`).
    *   Stores full original text for potential future re-indexing.
    *   Configurable character-based chunking with overlap.
    *   Stores chunk position (start/end characters).
    *   Calculates file hash (SHA256) for change detection.
    *   Automatically re-indexes if file content changes.
    *   Skips processing if file is unchanged and vectors exist for the specified method.
    *   Associates optional JSON metadata with documents.
*   **Encryption (Optional):**
    *   Encrypts `chunk_text` at rest using Fernet (AES-128-CBC + HMAC) via the `cryptography` library. Requires `safe_store[encryption]`.
    *   Enabled by providing an `encryption_key` (password) during `safe_store` initialization.
    *   Automatic decryption during queries if the correct key is provided.
    *   **Key Management is the user's responsibility.**
*   **Vectorization:**
    *   Supports multiple vectorization methods simultaneously.
    *   **Sentence Transformers:** Integrates `sentence-transformers` models (e.g., `st:all-MiniLM-L6-v2`). Requires `safe_store[sentence-transformers]`.
    *   **TF-IDF:** Integrates `scikit-learn`'s `TfidfVectorizer` (e.g., `tfidf:my_method`). Requires `safe_store[tfidf]`. Handles fitting and stores vocabulary/IDF weights in the database.
    *   Stores method details (name, type, dimension, data type, parameters) in DB.
*   **Querying (`query`):**
    *   Find `top_k` chunks based on cosine similarity to a query text.
    *   Specify which `vectorizer_name` to use for embedding the query and retrieving vectors.
    *   Returns ranked results including chunk text (decrypted if applicable), similarity score, source document path, position, and metadata.
*   **Management Methods:**
    *   **`add_vectorization`:** Adds embeddings for a *new* method to existing documents without re-parsing/re-chunking. Fits TF-IDF if needed (decrypting text first if store is encrypted).
    *   **`remove_vectorization`:** Deletes a vectorization method and all its associated vectors from the database and cache.
    *   **`list_documents`:** Returns a list of stored documents and their metadata.
    *   **`list_vectorization_methods`:** Returns details of registered vectorization methods.
*   **Logging:** Rich console logging via `ascii_colors`. Default level is `INFO`. Configurable via `SafeStore(log_level=...)` or globally using `ASCIIColors` static methods (see [Logging](#-logging) section).

---

## 🚀 Installation

```bash
pip install safe_store
```

Install optional dependencies based on the features you need:

```bash
# For Sentence Transformers embedding models (recommended default)
pip install safe_store[sentence-transformers]

# For TF-IDF vectorization (requires scikit-learn)
pip install safe_store[tfidf]

# For parsing PDF, DOCX, and HTML files
pip install safe_store[parsing] # Includes pypdf, python-docx, beautifulsoup4, lxml

# For encrypting chunk text at rest (requires cryptography)
pip install safe_store[encryption]

# To install everything (all vectorizers, all parsers, encryption):
pip install safe_store[all]

# Or install specific combinations:
pip install safe_store[sentence-transformers,parsing,encryption]
```

---

## 🏁 Quick Start

```python
import safe_store
from pathlib import Path
import time # For demonstrating concurrency

# --- 1. Prepare Sample Documents ---
doc_dir = Path("my_docs")
doc_dir.mkdir(exist_ok=True)
doc1_path = doc_dir / "doc1.txt"
doc1_path.write_text("safe_store makes local vector storage simple and efficient.", encoding='utf-8')
doc2_path = doc_dir / "doc2.html"
doc2_path.write_text("<html><body><p>HTML content can also be indexed.</p></body></html>", encoding='utf-8')

print(f"Created sample files in: {doc_dir.resolve()}")

# --- 2. Initialize safe_store ---
# Provide a secure key to enable encryption. Omit for no encryption.
# !! MANAGE YOUR KEY SECURELY IN REAL APPLICATIONS !!
encryption_password = "your-secret-password" # Or None

store = safe_store.SafeStore(
    "my_vector_store.db",
    log_level=safe_store.LogLevel.INFO, # Use INFO for less noise in example
    lock_timeout=10,
    encryption_key=encryption_password # Provide key here
)

# Best practice: Use safe_store as a context manager
try:
    with store:
        # --- 3. Add Documents ---
        # Chunk text will be encrypted if encryption_key was provided
        print("\n--- Indexing Documents ---")
        store.add_document(doc1_path, vectorizer_name="st:all-MiniLM-L6-v2", chunk_size=50)
        store.add_document(doc2_path, vectorizer_name="st:all-MiniLM-L6-v2")

        # --- 4. Query ---
        # Results will be automatically decrypted if the store has the key
        print("\n--- Querying using Sentence Transformer ---")
        query = "simple storage"
        results = store.query(query, vectorizer_name="st:all-MiniLM-L6-v2", top_k=1)
        for i, res in enumerate(results):
            print(f"Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
            # Verify decryption occurred if key was provided
            if encryption_password:
                assert "[Encrypted" not in res['chunk_text']

except safe_store.ConfigurationError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Please install the required extras (e.g., pip install safe_store[all])")
except safe_store.ConcurrencyError as e:
    print(f"\n[ERROR] Lock timeout or concurrency issue: {e}")
except safe_store.EncryptionError as e:
    print(f"\n[ERROR] Encryption/Decryption issue: {e}")
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
```

*(See `examples/` directory for this and other usage examples, including `encryption_usage.py` and `custom_logging.py`.)*

---

## 💡 Concurrency

safe_store uses `filelock` to provide process-safe write operations (`add_document`, `add_vectorization`, `remove_vectorization`). When one process performs a write, other processes attempting a write will wait up to the configured `lock_timeout`.

Read operations (`query`, `list_*`) are designed to be concurrent with writes thanks to SQLite's WAL (Write-Ahead Logging) mode, which is enabled by default. Multiple processes can typically read the database simultaneously, even while another process is writing.

---

## 🔒 Encryption

safe_store can optionally encrypt the text content of document chunks stored in the database using **Fernet** (AES-128-CBC + HMAC) from the `cryptography` library.

*   **Enable:** Provide a password string via the `encryption_key` parameter during `safe_store` initialization. Requires `safe_store[encryption]` to be installed.
*   **Mechanism:** A strong encryption key is derived from your password using PBKDF2. Chunk text is encrypted before being saved to the database.
*   **Decryption:** When querying, if the `safe_store` instance has the correct `encryption_key`, chunk text is automatically decrypted before being returned. If the key is missing or incorrect, placeholder text (e.g., `[Encrypted - Key Unavailable]`) is returned instead.
*   **Target:** Only chunk text is encrypted. Document paths, metadata, vectors, etc., remain unencrypted.
*   **Key Management:** **You are responsible for securely managing the `encryption_key`. Lost keys mean lost data.** Avoid hardcoding keys; use environment variables or secrets management tools.
*   **Security Note:** ``safe_store`` currently uses a fixed internal salt for key derivation for simplicity. See the full documentation section on Encryption for details and security considerations.

---

## 🪵 Logging

safe_store uses the [`ascii_colors`](https://github.com/ParisNeo/ascii_colors) library for flexible and colorful console logging.

*   **Default Level:** `INFO`. Only INFO, SUCCESS, WARNING, ERROR, CRITICAL messages are shown.
*   **Change Level:** Initialize with `SafeStore(log_level=safe_store.LogLevel.DEBUG)` or `SafeStore(log_level=safe_store.LogLevel.WARNING)` etc.
*   **Global Configuration:** You can configure `ascii_colors` globally in your application *before* initializing safe_store to control output destinations (console, file), formats (text, JSON), and levels:
    ```python
    import safe_store
    from ascii_colors import ASCIIColors, LogLevel, FileHandler, Formatter

    # Set global log level (affects safe_store and other uses of ascii_colors)
    ASCIIColors.set_log_level(LogLevel.DEBUG)

    # Add logging to a file with a specific format
    file_handler = FileHandler("safe_store_app.log", encoding='utf-8')
    file_handler.set_formatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ASCIIColors.add_handler(file_handler)

    # Optional: Remove default console handler if you only want file logging
    # default_console_handler = ASCIIColors.get_default_handler()
    # if default_console_handler: ASCIIColors.remove_handler(default_console_handler)

    # Now initialize safe_store - it will use the global settings
    store = safe_store.SafeStore("my_store.db")
    # ... use store ...
    ```
    *(See `examples/custom_logging.py` and the full documentation for more.)*

---

## 🔮 Future Work (Planned Features)

*   **Re-indexing:** `reindex()` method to re-process documents with new chunking or other parameters without needing the original file.
*   **More Vectorizers:** Integrations for OpenAI, Cohere, Ollama embeddings.
*   **Metadata Filtering:** Allow filtering query results based on document metadata (e.g., `query(..., metadata_filter={'year': 2023})`).
*   **Performance:** Explore optimizations like ANN indexing (e.g., via Faiss or HNSWlib integration) for very large datasets, potentially as an optional backend.
*   **Async API:** Consider adding an asynchronous interface using `aiosqlite`.
*   **Encryption:** Consider adding options for unique salts per database or per chunk for enhanced security, potentially sacrificing some simplicity.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ParisNeo/safe_store).

See `CONTRIBUTING.md` (to be added) for more detailed guidelines.

---

## 📜 License

safe_store is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.