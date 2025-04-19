# SafeStore: Simple SQLite Vector Store for RAG

[![PyPI version](https://img.shields.io/pypi/v/safe-store.svg)](https://pypi.org/project/safe-store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe-store.svg)](https://pypi.org/project/safe-store/)
[![PyPI license](https://img.shields.io/pypi/l/safe-store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe-store.svg)](https://pypi.org/project/safe-store/)


SafeStore is a Python utility library providing a lightweight, efficient, and file-based vector database using **SQLite**. It's specifically designed and optimized for easy integration into **Retrieval-Augmented Generation (RAG)** pipelines for Large Language Models (LLMs).

Need a simple way to index your documents locally, manage different vector embeddings, find relevant chunks, and handle **concurrent access** safely? SafeStore is your answer!

---

## ✨ Why SafeStore?

*   **🎯 RAG Focused:** Built from the ground up with RAG use cases in mind.
*   **🚀 Simple & Lightweight:** Uses SQLite – no heavy dependencies or external database servers needed. Just a single file!
*   **🏠 Local First:** Keep your data entirely on your local machine or network share.
*   **🧠 Multiple Vectorizers:** Supports different embedding models (Sentence Transformers, TF-IDF currently) side-by-side for the same documents.
*   **🔍 Querying:** Find relevant document chunks based on semantic similarity using your chosen vectorization method.
*   **🔄 Change Aware:** Automatically detects changes in source files (via hashing) and handles re-indexing efficiently.
*   **✍️ Concurrent Safe:** Uses file-based locking (`filelock`) to safely handle writes from multiple processes.
*   ** гибкий (Flexible):** Configurable chunking, add/remove vectorization methods as needed.
*   **🔒 Secure (Planned):** Optional data encryption at rest is planned.
*   **🗣️ Informative:** Uses `ascii_colors` for clear, leveled console feedback.

---

## ⚠️ Breaking Changes in v1.2.0?

*   No breaking API changes introduced in v1.2.0.
*   A `.db.lock` file will now be created alongside the `.db` file to manage concurrency. Ensure your application has write permissions in the database directory.

---

## ⚙️ Features

*   **SQLite Backend:** Stores all data in a single file.
*   **Concurrency Control:** Uses `filelock` for safe multi-process write access. Configurable lock timeout.
*   **Document Ingestion (`add_document`):**
    *   Process files (currently `.txt`, framework for `.pdf`, `.docx`, `.html` added). Requires `safestore[parsing]`.
    *   Stores full original text.
    *   Configurable chunking (`chunk_size`, `chunk_overlap`).
    *   Stores start/end character positions.
*   **Multiple Vectorization Methods:**
    *   **Sentence Transformers:** Integrates `sentence-transformers` (`st:model-name`). Requires `safestore[sentence-transformers]`.
    *   **TF-IDF:** Integrates `scikit-learn` (`tfidf:name`). Requires `safestore[tfidf]`. Handles fitting and stores state.
    *   Stores method details (name, type, dimension, parameters) in the DB.
*   **Querying (`query`):**
    *   Find `top_k` chunks similar to a query text.
    *   Specify which `vectorizer_name` to use for the search.
    *   Returns ranked results with text, score, source document info.
*   **Vectorizer Management:**
    *   **`add_vectorization`:** Add embeddings for a *new* method to existing documents without re-parsing. Fits TF-IDF if needed.
    *   **`remove_vectorization`:** Delete a vectorization method and all its associated vectors.
*   **Change Detection:** Uses file hashes, skips unchanged files, re-indexes changed files.
*   **Logging:** Uses `ascii_colors` for console output.

---

## 🚀 Installation

```bash
pip install safe-store
```

Install optional dependencies as needed:

```bash
# For Sentence Transformers (recommended default)
pip install safe-store[sentence-transformers]

# For TF-IDF (requires scikit-learn)
pip install safe-store[tfidf]

# For parsing PDF, DOCX, HTML files
pip install safe-store[parsing]

# For upcoming encryption features
pip install safe-store[encryption]

# To install everything:
pip install safe-store[all]
# Or combine: pip install safe-store[sentence-transformers,parsing]
```

---

## 🏁 Quick Start

```python
from pathlib import Path
from safestore import SafeStore, LogLevel

# --- 1. Prepare documents ---
doc1_content = "First document content."
doc1_path = Path("my_document1.txt")
doc1_path.write_text(doc1_content, encoding='utf-8')

# --- 2. Initialize SafeStore ---
# The store manages a .db file and a .db.lock file
# Use a timeout if multiple processes might access the db heavily
store = SafeStore("my_vector_store.db", log_level=LogLevel.INFO, lock_timeout=60)

# Use 'with' for automatic connection closing and lock handling context
with store:
    # --- 3. Add documents (acquires write lock) ---
    store.add_document(doc1_path)

    # --- 4. Query (doesn't require write lock with WAL) ---
    print("\n--- Querying ---")
    query = "document"
    results = store.query(query, top_k=1)
    for res in results:
        print(f"Result: Score={res['similarity']:.4f}, Text='{res['chunk_text'][:80]}...'")

# Connection is automatically closed here

# --- Cleanup (optional) ---
# doc1_path.unlink()
# Path("my_vector_store.db").unlink()
# Path("my_vector_store.db.lock").unlink() # Remove lock file too

print("\nCheck 'my_vector_store.db' and console logs.")
```

---

## 🔮 Future Work (Planned Features)

*   **Parser Implementation:** Finish PDF, DOCX, HTML parsing logic.
*   **More Vectorizers:** OpenAI Embeddings (`[openai]`), Ollama models (`[ollama]`).
*   **Re-indexing:** Allow manual re-chunking/re-vectorizing (`reindex()` method).
*   **Encryption:** Add optional encryption for content stored in the database (`[encryption]` extra).
*   **Metadata Filtering:** Allow filtering query results based on document metadata.
*   **Tagging:** Optional automatic tag extraction from chunks.
*   **Performance:** Optimize vector loading/search (e.g., approximate nearest neighbors).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ParisNeo/safe_store). (A formal CONTRIBUTING.md guide will be added later).

---

## 📜 License

SafeStore is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.