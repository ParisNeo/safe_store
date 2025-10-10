# safe_store: Simple, Concurrent SQLite Vector Store & Graph Database for Local RAG

[![PyPI version](https://img.shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img.shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe_store.svg)](https://pypi.org/project/safe_store/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml/badge.svg)](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml)

**safe_store** is a Python library providing a lightweight, file-based **vector database AND graph database** using a single **SQLite** file. It's designed for simplicity, robustness, and efficiency, making it ideal for integrating into **local Retrieval-Augmented Generation (RAG)** pipelines and knowledge graph applications.

Each `SafeStore` database is tied to a **single, consistent vector space**, ensuring that all documents are vectorized and queried using the same model. This robust design simplifies the API and prevents errors common in multi-vectorizer systems.

---

## ✨ Why safe_store?

*   **🎯 RAG & Knowledge Graph Focused:** Built for local RAG and knowledge graph use cases.
*   **🚀 Simple & Robust API:** One database, one vectorizer. This core design principle makes the API intuitive and less error-prone.
*   **🏠 Local First:** Keep your embeddings, document text, and graph data entirely on your local machine.
*   **🤝 Concurrent Safe:** Handles database writes from multiple processes safely using file-based locking.
*   **💡 Dual Capabilities:**
    *   **Vector Store (`SafeStore` class):** Index documents and perform semantic similarity search within a single, consistent vector space. Supports Sentence Transformers, TF-IDF, OpenAI, Cohere, and Ollama.
    *   **Graph Store (`GraphStore` class):** Extract entities and relationships from text using an LLM, build a persistent knowledge graph, and query it using natural language.
*   **🌐 Web User Interface (WebUI):** An experimental interface to upload documents, trigger graph building, and visualize the resulting knowledge graph.
*   **🔗 Integrated Backend:** Both vector and graph data reside in the same SQLite database.
*   **🤖 LLM-Powered Graph:** Uses a flexible callback mechanism to integrate with your choice of LLM for extracting graph structures from text and for natural language querying.
*   **📄 Document Parsing:** Built-in parsers for `.txt`, `.pdf`, `.docx`, `.html`, and many common text-based formats.
*   **🔒 Optional Encryption:** Encrypts document chunk text at rest (AES-128) for enhanced security.
*   **🔄 Change Aware:** Automatically detects file changes for efficient re-indexing of vectors.

---

## ⚠️ Status: Beta

safe_store is currently in Beta. The core API for both `SafeStore` (vectors) and `GraphStore` (graphs) is stabilizing, but breaking changes are still possible. The WebUI is experimental. Feedback and contributions are welcome!

---

## ⚙️ Features

### Core
*   **Storage:** All data (documents, chunks, vectors, graph nodes, relationships, metadata) in a single SQLite file.
*   **Concurrency:** Process-safe writes using file-based locking (`filelock`).

### `SafeStore` (Vector Database)
*   **Single Vector Space:** Each database is configured with **one** vectorizer upon initialization, ensuring all vectors are comparable.
*   **Indexing (`add_document`, `add_text`):**
    *   Parses various file types.
    *   Stores full text and performs configurable chunking.
    *   File hashing (SHA256) for change detection and efficient re-indexing.
*   **Encryption (Optional):** Encrypts `chunk_text` at rest.
*   **Vectorization:**
    *   The vectorizer is defined when creating the `SafeStore` instance using a `vectorizer_name` (e.g., "st", "openai") and a `vectorizer_config` dictionary.
    *   Supports multiple vectorizer types, including Sentence Transformers, TF-IDF, OpenAI, Cohere, and Ollama.
*   **Querying (`query`):** Performs cosine similarity search for `top_k` relevant chunks using the instance's configured vectorizer.
*   **Management:** `list_documents`, `list_vectorization_methods`, `delete_document_by_path`.

### `GraphStore` (Graph Database)
*   **Schema:** Dedicated tables for `graph_nodes`, `graph_relationships`, and `node_chunk_links` within the same SQLite DB.
*   **LLM Integration for Graph Building:**
    *   Uses a user-provided `llm_executor_callback` (takes prompt string, returns LLM response string).
    *   Internal prompts for extracting nodes and relationships from text chunks.
    *   Handles de-duplication and property updates of nodes.
    *   Links graph nodes to source text chunks.
*   **Graph Building Methods:** `process_chunk_for_graph`, `build_graph_for_document`, `build_graph_for_all_documents`.
*   **LLM Integration for Graph Querying (`query_graph`):**
    *   Uses `llm_executor_callback` with an internal query-parsing prompt to translate Natural Language Query (NLQ) into structured graph query parameters.
    *   Performs graph traversal and returns results in various modes (`"graph_only"`, `"chunks_summary"`, `"full"`).
*   **Direct Graph Read Methods:** `get_node_details`, `get_nodes_by_label`, `get_relationships`, `find_neighbors`, `get_chunks_for_node`.
*   **Encryption Awareness:** Decrypts chunk text for LLM processing if `encryption_key` is provided.

---

## 🚀 Installation

```bash
pip install safe-store
```

Install optional dependencies for the vectorizers or features you need:

```bash
# For Sentence Transformers embedding models
pip install safe-store[sentence-transformers]

# For OpenAI embedding models (requires OpenAI API key)
pip install safe-store[openai]

# For Ollama embedding models (requires Ollama server running)
pip install safe-store[ollama]

# For parsing PDF, DOCX, etc.
pip install safe-store[parsing]

# For encryption features
pip install safe-store[encryption]

# For the Web User Interface
pip install safe-store[webui]

# To install everything:
pip install safe-store[all] 
```

---

## 🏁 Quick Start (Python Library)

This example shows `SafeStore` usage followed by `GraphStore` graph building and querying.

```python
import safe_store
from safe_store import GraphStore, LogLevel
from lollms_client import LollmsClient # Example LLM client
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
Path(f"{DB_FILE}.lock").unlink(missing_ok=True)

# LollmsClient setup (replace with your actual LLM server config)
LC_CLIENT: Optional[LollmsClient] = None
def init_llm():
    global LC_CLIENT
    try:
        LC_CLIENT = LollmsClient() # Assumes default connection to a running Lollms-compatible server
        print("LLM Client Initialized for GraphStore.")
        return True
    except Exception as e:
        print(f"LLM Client init failed: {e}. Graph features needing LLM will not work.")
        return False

# LLM Executor Callback for GraphStore
def llm_executor(prompt_to_llm: str) -> str:
    if not LC_CLIENT: raise ConnectionError("LLM Client not ready for executor callback.")
    response = LC_CLIENT.generate_code(prompt_to_llm, language="json", temperature=0.1, max_size=4096)
    return response if response else ""

if not init_llm():
    print("Skipping GraphStore parts of Quick Start as LLM is not available.")

# --- 1. Prepare Sample Document ---
doc1_path = DOC_DIR / "ceo_info.txt"
doc1_content = "Dr. Aris Thorne is the CEO of QuantumLeap AI, a company focusing on advanced AI research. QuantumLeap AI is based in Geneva."
doc1_path.write_text(doc1_content)

# --- 2. Use SafeStore for Vector Indexing ---
print("\n--- SafeStore Operations ---")
# Initialize the store with a single, dedicated vectorizer
store = safe_store.SafeStore(
    db_path=DB_FILE,
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    log_level=LogLevel.INFO
)

doc_id_1 = -1
with store:
    # Add document - no need to specify vectorizer here
    store.add_document(doc1_path, chunk_size=100)
    docs = store.list_documents()
    if docs: doc_id_1 = docs['doc_id']
    print(f"Document '{doc1_path.name}' (ID: {doc_id_1}) indexed by SafeStore.")
    
    # Query using the instance's configured vectorizer
    query_results = store.query("AI research in Geneva", top_k=1)
    if query_results:
        print(f"SafeStore query result for 'AI research in Geneva': {query_results['chunk_text'][:100]}...")

if LC_CLIENT and doc_id_1 != -1: # Proceed with GraphStore only if LLM and doc are ready
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
        print("Graph building for document complete.")

        aris_nodes = graph_store.get_nodes_by_label("Person", limit=5)
        print(f"\nFound Person nodes: {[n.get('properties',{}).get('name') for n in aris_nodes if n.get('properties')]}")

        nl_query = "Who is the CEO of QuantumLeap AI and where is it based?"
        print(f"\nGraphStore NLQ: \"{nl_query}\"")
        
        graph_data = graph_store.query_graph(nl_query, output_mode="graph_only")
        print("\nQuery Result (graph_only):")
        if graph_data.get('nodes'): 
            print(f"  Sample Node: {graph_data['nodes']['label']} - {graph_data['nodes']['properties']}")
```

*(See `examples/` directory for more detailed usage.)*

---

## 🖥️ Web User Interface (Experimental)

`safe_store` includes an experimental Web User Interface (WebUI) that allows you to:

*   **Upload documents:** Add new documents to your `SafeStore` instance.
*   **Trigger graph building:** The WebUI automatically initiates graph extraction for new documents.
*   **Visualize the knowledge graph:** View extracted nodes and relationships in an interactive graph.
*   **Inspect node/edge details:** Click on elements in the graph to see their properties.

### Prerequisites for WebUI

1.  **Install WebUI dependencies:**
    ```bash
    pip install safe-store[webui]
    ```
2.  **LLM Server:** The WebUI relies on an LLM for graph extraction. You need a running LLM server that `lollms-client` can connect to, such as Ollama.

### Launching the WebUI

Once `safe_store` is installed with the `[webui]` extra, you can launch the WebUI:

```bash
safestore-webui
```

This will start a server, by default accessible at `http://0.0.0.0:8000`.

### Configuring the WebUI

The WebUI's behavior is controlled by the `config.toml` file. A default is created in `.../site-packages/safe_store/webui/` on the first run.

**Default `config.toml` structure:**

```toml
[lollms]
binding_name = "ollama"
host_address = "http://localhost:11434"
model_name = "mistral:latest"

[safestore]
db_file = "webui_store.db"
doc_dir = "webui_safestore_docs"
# Define the single vectorizer for the WebUI's database instance
vectorizer_name = "st"
# Use a TOML inline table for the config dictionary
vectorizer_config = '{model = "all-MiniLM-L6-v2"}' 
chunk_size = 250
chunk_overlap = 40

[webui]
host = "0.0.0.0"
port = 8000
log_level = "INFO"
```

**Using the WebUI:**

1.  **Launch:** Run `safestore-webui` in your terminal.
2.  **Access:** Open `http://localhost:8000` in your web browser.
3.  **Upload Document:** Use the form to upload a file (`.txt`, `.pdf`, `.docx`, etc.).
4.  **Processing:** The backend will save the file, add it to `SafeStore` (creating vector embeddings using the configured vectorizer), and then trigger `GraphStore` to build graph elements using the configured LLM.
5.  **View Graph:** The graph visualization should update automatically.

---

## 💡 Key Concepts

### `SafeStore`
*   **One Store, One Vectorizer**: When you create a `SafeStore` instance, you define the vectorization method for that entire database. This ensures all your data lives in a consistent, comparable vector space. If you need to use a different vectorization model, you create a new database file.
*   **Simple API**: Methods like `add_document` and `query` are clean and simple, as they automatically use the vectorizer configured on the store instance.

### `GraphStore`
*   Builds and queries a knowledge graph from the text data in an existing `SafeStore` database.
*   **LLM Executor Callback:** This is the core of its flexibility. You provide a simple Python function `(prompt_string: str) -> llm_response_string`. `GraphStore` uses this function to send its internally crafted prompts (for graph extraction or query parsing) to any LLM you have access to.
*   **Internal Prompts:** `GraphStore` contains default prompt templates optimized for extracting graph data and for parsing natural language queries.
*   **Graph Querying (`query_graph`):** Translates natural language questions like "Who is the CEO of Company X?" into graph traversals, returning subgraphs or linked text chunks.

---
## 🪵 Logging & Concurrency

*   **Logging:** Uses [`ascii_colors`](https://github.com/ParisNeo/ascii_colors). Configurable via `SafeStore(log_level=...)` or `GraphStore(log_level=...)`.
*   **Concurrency:** `filelock` ensures process-safe writes for both `SafeStore` and `GraphStore` operations on the shared SQLite DB.

---

## 🔮 Future Work

*   **WebUI Enhancements:** Natural language query input, graph editing, metadata filtering.
*   **Advanced Graph Traversal:** More complex pathfinding in `query_graph`.
*   **Hybrid Search:** Combining vector similarity search with graph query results.
*   **Async API.**

---

## 🤝 Contributing & License

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/ParisNeo/safe_store).
Licensed under Apache 2.0. See [LICENSE](LICENSE).