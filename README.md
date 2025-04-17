# SafeStore: Simple SQLite Vector Store for RAG

[![PyPI version](https://img.shields.io/pypi/v/safe-store.svg)](https://pypi.org/project/safe-store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe-store.svg)](https://pypi.org/project/safe-store/)
[![PyPI license](https://img.shields.io/pypi/l/safe-store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe-store.svg)](https://pypi.org/project/safe-store/)

SafeStore is a Python utility library providing a lightweight, efficient, and file-based vector database using **SQLite**. It's specifically designed and optimized for easy integration into **Retrieval-Augmented Generation (RAG)** pipelines for Large Language Models (LLMs).

Need a simple way to index your documents locally, manage different vector embeddings, and find relevant chunks without setting up complex databases or relying on external services? SafeStore is your answer!

---

## ✨ Why SafeStore?

*   **🎯 RAG Focused:** Built from the ground up with RAG use cases in mind.
*   **🚀 Simple & Lightweight:** Uses SQLite – no heavy dependencies or external database servers needed. Just a single file!
*   **🏠 Local First:** Keep your data entirely on your local machine or network share.
*   **🧠 Multiple Vectorizers:** Supports different embedding models (Sentence Transformers, TF-IDF currently) side-by-side for the same documents.
*   **🔍 Querying:** Find relevant document chunks based on semantic similarity using your chosen vectorization method.
*   **🔄 Change Aware:** Automatically detects changes in source files (via hashing) and handles re-indexing efficiently.
*   ** гибкий (Flexible):** Configurable chunking, add/remove vectorization methods as needed.
*   **🔒 Secure (Future):** Optional data encryption at rest is planned.
*   **🗣️ Informative:** Uses `ascii_colors` for clear, leveled console feedback.

---

## ⚠️ Breaking Changes in v1.1.0?

While v1.1.0 adds features, the core `add_document` API and database schema from v1.0.0 remain compatible. Existing databases created with v1.0.0 should work with v1.1.0. However, the internal handling of TF-IDF state is new.

---

## ⚙️ Features

*   **SQLite Backend:** Stores all data (documents, chunks, vectors) in a single SQLite file.
*   **Document Ingestion (`add_document`):**
    *   Process files (currently `.txt`).
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
*   **Change Detection:**
    *   Uses file hashes to detect content changes.
    *   Skips indexing if file is unchanged *and* vectors for the specified method exist.
    *   Automatically deletes old data and re-indexes if a file has changed or `force_reindex=True`.
*   **Logging:** Uses `ascii_colors` for console output (INFO level by default).

---

## 🚀 Installation

```bash
pip install safe-store
```

Install optional dependencies for the vectorizers you need:

```bash
# For Sentence Transformers (recommended default)
pip install safe-store[sentence-transformers]

# For TF-IDF (requires scikit-learn)
pip install safe-store[tfidf]

# To install both:
pip install safe-store[all-vectorizers]
# Or: pip install safe-store[sentence-transformers,tfidf]
```

---

## 🏁 Quick Start

```python
from pathlib import Path
from safestore import SafeStore, LogLevel

# --- 1. Prepare documents ---
doc1_content = """
This is the first sentence for SafeStore.
It helps demonstrate the basic indexing process.
SafeStore uses SQLite and is designed for RAG.
This is the final sentence.
"""
doc1_path = Path("my_document1.txt")
doc1_path.write_text(doc1_content, encoding='utf-8')

doc2_content = "A second document, shorter and simpler."
doc2_path = Path("my_document2.txt")
doc2_path.write_text(doc2_content, encoding='utf-8')

# --- 2. Initialize SafeStore ---
# Use 'with' for automatic connection closing
with SafeStore("my_vector_store.db", log_level=LogLevel.INFO) as store:

    # --- 3. Add documents with default vectorizer ---
    store.add_document(doc1_path, chunk_size=100, chunk_overlap=20)
    store.add_document(doc2_path) # Uses default chunk size

    # --- 4. Add a different vectorization (TF-IDF) ---
    # Requires: pip install safe-store[tfidf]
    try:
        print("\n--- Adding TF-IDF Vectors ---")
        # This will fit TF-IDF on all documents currently in the store
        store.add_vectorization("tfidf:my_tfidf")
    except ImportError:
        print("Skipping TF-IDF: scikit-learn not installed (pip install safe-store[tfidf])")
    except Exception as e:
        print(f"An error occurred adding TF-IDF: {e}")


    # --- 5. Query using the default vectorizer ---
    print("\n--- Querying with Default (Sentence Transformer) ---")
    query = "What is SafeStore used for?"
    results_st = store.query(query, top_k=2) # Uses default vectorizer implicitly
    for i, res in enumerate(results_st):
        print(f"Result {i+1} (ST): Score={res['similarity']:.4f}, Doc='{Path(res['file_path']).name}'")
        print(f"  Text: '{res['chunk_text'][:80]}...'")


    # --- 6. Query using the TF-IDF vectorizer ---
    # Requires TF-IDF to have been added successfully
    print("\n--- Querying with TF-IDF ---")
    try:
        results_tfidf = store.query(query, vectorizer_name="tfidf:my_tfidf", top_k=2)
        if results_tfidf: # Check if TF-IDF was added and returned results
            for i, res in enumerate(results_tfidf):
                print(f"Result {i+1} (TF-IDF): Score={res['similarity']:.4f}, Doc='{Path(res['file_path']).name}'")
                print(f"  Text: '{res['chunk_text'][:80]}...'")
        else:
            print("TF-IDF query returned no results (was it added successfully?).")
    except ValueError as e: # Catch if method doesn't exist
         print(f"Could not query with TF-IDF: {e}")
    except Exception as e: # Catch other potential errors
         print(f"An error occurred querying with TF-IDF: {e}")

# --- Cleanup (optional) ---
# doc1_path.unlink()
# doc2_path.unlink()
# Path("my_vector_store.db").unlink()

print("\nCheck 'my_vector_store.db' file and console logs.")
```

---

## 🔮 Future Work (Planned Features)

*   **More Parsers:** Add support for PDF, DOCX, HTML, etc. (`[parsing]` extra).
*   **More Vectorizers:** OpenAI Embeddings (`[openai]`), Ollama models (`[ollama]`).
*   **Re-indexing:** Allow manual re-chunking/re-vectorizing without changing the source file (`reindex()` method).
*   **Encryption:** Add optional encryption for content stored in the database (`[encryption]` extra).
*   **Concurrency:** Improve handling for multiple processes accessing the same database file (using `filelock`).
*   **Metadata Filtering:** Allow filtering query results based on document metadata.
*   **Tagging:** Optional automatic tag extraction from chunks.
*   **Performance:** Optimize vector loading/search (e.g., approximate nearest neighbors).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/ParisNeo/safe_store). (A formal CONTRIBUTING.md guide will be added later).

---

## 📜 License

SafeStore is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.