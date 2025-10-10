# safe_store: Simple, Concurrent SQLite Vector Store & Graph Database for Local RAG

[![PyPI version](https://img.shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img.shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe_store.svg)](https://pypi.org/project/safe_store/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml/badge.svg)](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml)

**safe_store** is a Python library providing a lightweight, file-based **vector database AND graph database** using a single **SQLite** file. It's designed for simplicity and efficiency, making it ideal for integrating into **local Retrieval-Augmented Generation (RAG)** pipelines and knowledge graph applications.

Store, manage, and query your document embeddings and extracted graph data locally with features like automatic change detection, multiple vectorization methods, safe concurrent access, document parsers, optional encryption, and LLM-powered graph construction & querying. The library also includes an experimental **Web User Interface (WebUI)** for interacting with the graph database.

---

## ✨ Why safe_store?

*   **🎯 RAG & Knowledge Graph Focused:** Built for local RAG and knowledge graph use cases.
*   **🚀 Simple & Lightweight:** Uses a single SQLite file – no heavy dependencies or external servers.
*   **🏠 Local First:** Keep your embeddings, document text, and graph data entirely on your local machine.
*   **🤝 Concurrent Safe:** Handles database writes from multiple processes safely using file-based locking.
*   **💡 Dual Capabilities:**
    *   **Vector Store (`SafeStore` class):** Index documents, generate embeddings (Sentence Transformers, TF-IDF, OpenAI, Cohere, Ollama), and perform semantic similarity search.
    *   **Graph Store (`GraphStore` class):** Extract entities and relationships from text chunks using an LLM, build a persistent knowledge graph, and query it using natural language or direct graph operations.
*   **🌐 Web User Interface (WebUI):** An experimental interface to upload documents, trigger graph building, and visualize the resulting knowledge graph.
*   **🔗 Integrated Backend:** Both vector and graph data reside in the same SQLite database, allowing for potential future synergies.
*   **🤖 LLM-Powered Graph:**
    *   Uses a flexible callback mechanism to integrate with your choice of LLM (e.g., via `lollms-client`) for extracting graph structures from text.
    *   Internal prompt templates guide the LLM for consistent graph extraction and query parsing.
    *   Supports natural language querying of the graph, also powered by an LLM callback.
*   **📄 Document Parsing:** Built-in parsers for `.txt`, `.pdf`, `.docx`, `.html`, and many common text-based formats (requires `[parsing]` extra for rich formats).
*   **🔒 Optional Encryption:** Encrypts document chunk text at rest (AES-128) for enhanced security (requires `[encryption]` extra).
*   **🔄 Change Aware (Vector Store):** Automatically detects file changes for efficient re-indexing of vectors.
*   **🗣️ Informative Logging:** Clear, leveled, and colorful console feedback via `ascii_colors`.

---

## ⚠️ Status: Beta

safe_store is currently in Beta. The core API for both `SafeStore` (vectors) and `GraphStore` (graphs) is stabilizing, but breaking changes are still possible. The WebUI is experimental. Feedback and contributions are welcome!

---

## ⚙️ Features

### Core
*   **Storage:** All data (documents, chunks, vectors, graph nodes, relationships, metadata) in a single SQLite file.
*   **Concurrency:** Process-safe writes using file-based locking (`filelock`). SQLite's WAL mode for concurrent reads.

### `SafeStore` (Vector Database)
*   **Indexing (`add_document`):**
    *   Parses various file types (see [Installation](#-installation) for extras).
    *   Stores full text, performs configurable chunking.
    *   File hashing (SHA256) for change detection and efficient re-indexing.
    *   Optional JSON metadata per document.
*   **Encryption (Optional):** Encrypts `chunk_text` at rest (AES-128-CBC via `cryptography`).
*   **Vectorization:**
    *   Uses a clean `vectorizer_name` (e.g., "st", "tfidf", "openai") and a `vectorizer_config` dictionary for parameters.
    *   Supports multiple vectorization methods: Sentence Transformers, TF-IDF, OpenAI, Cohere, Ollama.
    *   Manages vectorizer state (e.g., fitted TF-IDF models).
*   **Querying (`query`):** Cosine similarity search for `top_k` relevant chunks.
*   **Management:** `add_vectorization`, `remove_vectorization`, `list_documents`, `list_vectorization_methods`.

### `GraphStore` (Graph Database)
*   **Schema:** Dedicated tables for `graph_nodes`, `graph_relationships`, and `node_chunk_links` within the same SQLite DB.
*   **LLM Integration for Graph Building:**
    *   Uses a user-provided `llm_executor_callback` (takes prompt string, returns LLM response string).
    *   Internal prompts for extracting nodes and relationships from text chunks.
*   **Graph Building Methods:** `process_chunk_for_graph`, `build_graph_for_document`, `build_graph_for_all_documents`.
*   **LLM Integration for Graph Querying (`query_graph`):**
    *   Uses `llm_executor_callback` to translate Natural Language Queries into structured graph queries.
*   **Direct Graph Read Methods:** `get_node_details`, `get_nodes_by_label`, `find_neighbors`, etc.
*   **Encryption Awareness:** Decrypts chunk text for LLM processing if `encryption_key` is provided.

---

## 🚀 Installation

```bash
pip install safe-store
```

Install optional dependencies based on the features you need:

```bash
# For Sentence Transformers embedding models
pip install safe-store[sentence-transformers]

# For OpenAI embedding models (requires OpenAI API key)
pip install safe-store[openai]

# To install everything (all vectorizers, parsers, encryption, webui):
pip install safe-store[all] 
```
**Note:** For API-based vectorizers like "openai", you need to provide API keys in the `vectorizer_config` or ensure environment variables are set.

---

## 🏁 Quick Start (Python Library)

This example shows basic `SafeStore` usage followed by `GraphStore` graph building and querying.

```python
import safe_store
from safe_store import GraphStore, LogLevel
from lollms_client import LollmsClient
from pathlib import Path
import shutil
from typing import Optional

# --- 0. Configuration & LLM Setup ---
DB_FILE = "quickstart_store.db"
DOC_DIR = Path("temp_docs_qs")

# Cleanup previous run (optional)
if DOC_DIR.exists(): shutil.rmtree(DOC_DIR)
DOC_DIR.mkdir(exist_ok=True, parents=True)
Path(DB_FILE).unlink(missing_ok=True)

# Vectorizer configuration
vectorizer_name = "st"
vectorizer_config = {"model": "all-MiniLM-L6-v2"}

# LollmsClient setup (replace with your actual LLM server config)
LC_CLIENT: Optional[LollmsClient] = None
def init_llm():
    global LC_CLIENT
    try:
        LC_CLIENT = LollmsClient() # Assumes default connection
        print("LLM Client Initialized for GraphStore.")
        return True
    except Exception as e:
        print(f"LLM Client init failed: {e}. Graph features will not work.")
        return False

# LLM Executor Callback for GraphStore
def llm_executor(prompt: str) -> str:
    if not LC_CLIENT: raise ConnectionError("LLM Client not ready.")
    response = LC_CLIENT.generate_text(prompt, max_size=4096)
    return response if response else ""

if not init_llm():
    print("Skipping GraphStore parts of Quick Start as LLM is not available.")

# --- 1. Prepare Sample Document ---
doc1_path = DOC_DIR / "ceo_info.txt"
doc1_content = "Dr. Aris Thorne is the CEO of QuantumLeap AI, a company focusing on advanced AI research. QuantumLeap AI is based in Geneva."
doc1_path.write_text(doc1_content)

# --- 2. Use SafeStore for Vector Indexing ---
print("\n--- SafeStore Operations ---")
store = safe_store.SafeStore(DB_FILE, log_level=LogLevel.INFO)
doc_id_1 = -1
with store:
    store.add_document(
        doc1_path,
        vectorizer_name=vectorizer_name,
        vectorizer_config=vectorizer_config,
        chunk_size=100
    )
    docs = store.list_documents()
    if docs: doc_id_1 = docs['doc_id']
    print(f"Document '{doc1_path.name}' (ID: {doc_id_1}) indexed.")
    
    query_results = store.query(
        "AI research in Geneva",
        top_k=1,
        vectorizer_name=vectorizer_name,
        vectorizer_config=vectorizer_config
    )
    if query_results:
        print(f"SafeStore query result: {query_results['chunk_text'][:100]}...")

if LC_CLIENT and doc_id_1 != -1:
    # --- 3. Use GraphStore to Build & Query Knowledge Graph ---
    print("\n--- GraphStore Operations ---")
    graph_store = GraphStore(
        db_path=DB_FILE,
        llm_executor_callback=llm_executor,
        log_level=LogLevel.INFO
    )
    with graph_store:
        print(f"Building graph for document ID: {doc_id_1}...")
        graph_store.build_graph_for_document(doc_id_1)
        print("Graph building complete.")

        nl_query = "Who is the CEO of QuantumLeap AI and where is it based?"
        print(f"\nGraphStore NLQ: \"{nl_query}\"")
        
        graph_data = graph_store.query_graph(nl_query, output_mode="graph_only")
        if graph_data.get('nodes'): 
            print(f"  Sample Node from NLQ result: {graph_data['nodes']['properties']}")

```

*(See `examples/` directory for more detailed usage.)*

---

## 🖥️ Web User Interface (Experimental)

`safe_store` includes an experimental Web User Interface (WebUI) that allows you to upload documents, trigger graph building, and visualize the knowledge graph.

### Launching the WebUI

```bash
safestore-webui
```
This starts the UI, accessible at `http://0.0.0.0:8000`.

### Configuring the WebUI

The WebUI's behavior is controlled by `config.toml`. A default is created in `.../site-packages/safe_store/webui/` on the first run.

**Key `config.toml` settings:**
*   **`[lollms]`**: Configure your LLM connection for graph extraction.
*   **`[safestore]`**: Define the database and document directories for the WebUI.
    *   `default_vectorizer_name`: The name of the vectorizer (e.g., "st").
    *   `default_vectorizer_config`: A TOML inline table or multiline table with the vectorizer's config. Example: `default_vectorizer_config = '{model = "all-MiniLM-L6-v2"}'`

---

## 💡 Key Concepts

### `SafeStore`
*   Manages vector embeddings for semantic search.
*   Uses a `vectorizer_name` and `vectorizer_config` to specify how text is converted to vectors.

### `GraphStore`
*   Builds and queries a knowledge graph from text data.
*   **LLM Executor Callback:** You provide a function `(prompt_string: str) -> llm_response_string` that `GraphStore` uses to communicate with your chosen LLM.

---
## 🪵 Logging & Concurrency

*   **Logging:** Uses [`ascii_colors`](https://github.com/ParisNeo/ascii_colors). Configurable via `SafeStore(log_level=...)`.
*   **Concurrency:** `filelock` ensures process-safe writes to the shared SQLite DB.

---

## 🤝 Contributing & License

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/ParisNeo/safe_store).
Licensed under Apache 2.0.