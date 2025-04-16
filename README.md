# SafeStore: Simple SQLite Vector Store for RAG

[![PyPI version](https://img.shields.io/pypi/v/safe-store.svg)](https://pypi.org/project/safe-store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe-store.svg)](https://pypi.org/project/safe-store/)
[![PyPI license](https://img.shields.io/pypi/l/safe-store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe-store.svg)](https://pypi.org/project/safe-store/)

SafeStore is a Python utility library providing a lightweight, efficient, and file-based vector database using **SQLite**. It's specifically designed and optimized for easy integration into **Retrieval-Augmented Generation (RAG)** pipelines for Large Language Models (LLMs).

Need a simple way to index your documents locally and find relevant chunks without setting up complex databases or relying on external services? SafeStore is your answer!

---

## ✨ Why SafeStore?

*   **🎯 RAG Focused:** Built from the ground up with RAG use cases in mind.
*   **🚀 Simple & Lightweight:** Uses SQLite – no heavy dependencies or external database servers needed. Just a single file!
*   **🏠 Local First:** Keep your data entirely on your local machine or network share.
*   **🧠 Memory Efficient:** Loads vectors into memory only when needed for querying (feature planned for Phase 2), reducing constant overhead. Optimized NumPy comparisons.
*   **🔄 Change Aware:** Automatically detects changes in source files (via hashing) and handles re-indexing.
*   ** гибкий (Flexible):** Supports configurable chunking and planned support for multiple vectorization methods.
*   **🔒 Secure (Future):** Optional data encryption at rest is planned.
*   **🗣️ Informative:** Uses `ascii_colors` for clear, leveled console feedback.

---

## ⚠️ Important Version Note (v1.0.0)

**This version (1.0.0) is a complete rewrite of any previous iterations of `safe-store`. It introduces a new API and database schema and is NOT backward compatible.** If you were using an older (pre-release/development) version, you will need to re-index your documents.

---

## ⚙️ Features (v1.0.0 - Phase 1)

*   **SQLite Backend:** Stores all data (documents, chunks, vectors) in a single SQLite file.
*   **Document Ingestion:**
    *   `add_document()` method to process files.
    *   Initial support for `.txt` files.
    *   Stores full original text to allow future re-chunking.
*   **Configurable Chunking:**
    *   Splits documents into text chunks based on character `chunk_size` and `chunk_overlap`.
    *   Stores start/end character positions for each chunk.
*   **Sentence Transformer Vectorization:**
    *   Integrates `sentence-transformers` for creating embeddings (Default: `all-MiniLM-L6-v2`).
    *   Stores vectorization method details in the database.
*   **Change Detection:**
    *   Uses file hashes to detect if a source document has changed.
    *   Skips indexing if file is unchanged and vectors exist for the current method.
    *   Automatically deletes old data and re-indexes if a file has changed.
*   **Logging:** Uses `ascii_colors` for console output (INFO level by default).

---

## 🚀 Installation

```bash
pip install safe-store
```

You might also need the vectorizer dependency (only Sentence Transformers supported initially):

```bash
pip install safe-store[sentence-transformers]
```

---

## 🏁 Quick Start

```python
from pathlib import Path
from safestore import SafeStore, LogLevel

# --- 1. Prepare a document ---
doc_content = """
This is the first sentence for SafeStore.
It helps demonstrate the basic indexing process.
SafeStore uses SQLite and is designed for RAG.
This is the final sentence.
"""
doc_path = Path("my_document.txt")
doc_path.write_text(doc_content, encoding='utf-8')

# --- 2. Initialize SafeStore ---
# Creates 'my_vector_store.db' if it doesn't exist
# Use log_level=LogLevel.DEBUG to see more detailed output
# Use 'with' statement for automatic connection closing
with SafeStore("my_vector_store.db", log_level=LogLevel.INFO) as store:

    # --- 3. Add the document ---
    # This will parse, chunk, vectorize (using default model), and store.
    store.add_document(
        file_path=doc_path,
        chunk_size=100,  # Optional: default is 1000
        chunk_overlap=20   # Optional: default is 150
    )

    # --- 4. Add it again (optional) ---
    # SafeStore will detect it's unchanged and skip re-processing (if vectors exist)
    print("\nAdding the same document again:")
    store.add_document(doc_path)

# --- 5. Modify the document and add again (optional) ---
doc_path.write_text("Completely new content!", encoding='utf-8')
print("\nAdding modified document:")
with SafeStore("my_vector_store.db") as store:
     store.add_document(doc_path) # Will detect change and re-index

# --- Cleanup (optional) ---
# doc_path.unlink()
# Path("my_vector_store.db").unlink()

print("\nCheck 'my_vector_store.db' file and console logs.")
```

You will see logs in your console indicating the steps being performed (parsing, chunking, vectorizing, storing, skipping, re-indexing). The data is now stored in `my_vector_store.db`.

---

## 🔮 Future Work (Planned Features)

*   **Querying:** Implement `query()` method to find relevant chunks based on vector similarity.
*   **More Parsers:** Add support for PDF, DOCX, HTML, etc.
*   **Multiple Vectorizers:** Add support for TF-IDF, OpenAI Embeddings (API key needed), Ollama models, etc.
*   **Vectorizer Management:** Add/remove specific vectorizations from the store.
*   **Re-indexing:** Allow manual re-chunking/re-vectorizing without changing the source file.
*   **Encryption:** Add optional encryption for content stored in the database.
*   **Concurrency:** Improve handling for multiple processes accessing the same database file.
*   **Metadata Filtering:** Allow filtering query results based on document metadata.
*   **Tagging:** Optional automatic tag extraction from chunks.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ParisNeo/safe_store). (A formal CONTRIBUTING.md guide will be added later).

---

## 📜 License

SafeStore is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.