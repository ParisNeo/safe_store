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
    *   Supports multiple methods (Sentence Transformers `st:`, TF-IDF `tfidf:`, OpenAI `openai:`, Cohere `cohere:`, Ollama `ollama:`).
    *   Manages vectorizer state (e.g., fitted TF-IDF models).
*   **Querying (`query`):** Cosine similarity search for `top_k` relevant chunks.
*   **Management:** `add_vectorization`, `remove_vectorization`, `list_documents`, `list_vectorization_methods`.

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
    *   Performs graph traversal.
    *   Output Modes: `"graph_only"`, `"chunks_summary"`, `"full"`.
*   **Direct Graph Read Methods:** `get_node_details`, `get_nodes_by_label`, `get_relationships`, `find_neighbors`, `get_chunks_for_node`.
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

# For TF-IDF vectorization
pip install safe-store[tfidf]

# For Ollama embedding models (requires Ollama server running)
pip install safe-store[ollama] # Installs 'ollama' library

# For OpenAI embedding models (requires OpenAI API key)
pip install safe-store[openai] # Installs 'openai' library

# For Cohere embedding models (requires Cohere API key)
pip install safe-store[cohere] # Installs 'cohere' library

# For parsing PDF, DOCX, HTML files
pip install safe-store[parsing]

# For encrypting chunk text at rest
pip install safe-store[encryption]

# For the Web User Interface and its LLM client (lollms-client)
pip install safe-store[webui]

# To install everything (all vectorizers, parsers, encryption, webui):
pip install safe-store[all] 
```
**Note:** For `openai:`, `cohere:`, and `ollama:` vectorizers, you need to provide API keys or ensure the respective services are accessible as per their documentation. The WebUI uses `lollms-client`, so ensure your LLM service (e.g., Ollama) is running and configured in `webui/config.toml`.

---

## 🏁 Quick Start (Python Library)

This example shows basic `SafeStore` usage followed by `GraphStore` graph building and querying.

```python
import safe_store
from safe_store import GraphStore, LogLevel
from lollms_client import LollmsClient # Example LLM client
from pathlib import Path
import shutil # For cleanup

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
        # Example: Using Ollama with mistral
        LC_CLIENT = LollmsClient(binding_name="ollama", model_name="mistral:latest", host_address="http://localhost:11434")
        if not hasattr(LC_CLIENT, 'binding') or LC_CLIENT.binding is None: # Basic check
             raise ConnectionError("LLM client binding not loaded.")
        print("LLM Client Initialized for GraphStore.")
        return True
    except Exception as e:
        print(f"LLM Client init failed: {e}. Graph features needing LLM will not work.")
        return False

# LLM Executor Callback for GraphStore
def llm_executor(prompt_to_llm: str) -> str:
    if not LC_CLIENT: raise ConnectionError("LLM Client not ready for executor callback.")
    # generate_code expects LLM to output markdown ```json ... ```
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
store = safe_store.SafeStore(DB_FILE, log_level=LogLevel.INFO)
doc_id_1 = -1
with store:
    store.add_document(doc1_path, vectorizer_name="st:all-MiniLM-L6-v2", chunk_size=100)
    docs = store.list_documents()
    if docs: doc_id_1 = docs[0]['doc_id']
    print(f"Document '{doc1_path.name}' (ID: {doc_id_1}) indexed by SafeStore.")
    
    query_results = store.query("AI research in Geneva", top_k=1)
    if query_results:
        print(f"SafeStore query result for 'AI research in Geneva': {query_results[0]['chunk_text'][:100]}...")

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
            print(f"  Sample Node: {graph_data['nodes'][0]['label']} - {graph_data['nodes'][0]['properties']}")

# Optional: Cleanup after example
# shutil.rmtree(DOC_DIR, ignore_errors=True)
# Path(DB_FILE).unlink(missing_ok=True)
# Path(f"{DB_FILE}.lock").unlink(missing_ok=True)
```

*(See `examples/` directory for more detailed usage, including `graph_usage.py`.)*

---

## 🖥️ Web User Interface (Experimental)

`safe_store` includes an experimental Web User Interface (WebUI) that allows you to:

*   **Upload documents:** Add new documents to your `SafeStore` instance.
*   **Trigger graph building:** After uploading, the WebUI automatically initiates graph extraction for the new document using `GraphStore`.
*   **Visualize the knowledge graph:** View the extracted nodes and relationships in an interactive graph visualization.
*   **Inspect node/edge details:** Click on elements in the graph to see their properties.

### Prerequisites for WebUI

1.  **Install WebUI dependencies:**
    ```bash
    pip install safe-store[webui]
    # This installs uvicorn, fastapi, python-multipart, toml, and lollms-client.
    ```
2.  **LLM Server:** The WebUI (and `GraphStore`) relies on an LLM for graph extraction. You need a running LLM server that `lollms-client` can connect to. Ollama is a common choice.
    *   Ensure Ollama (or your chosen backend for `lollms-client`) is running.
    *   Ensure the model specified in the WebUI's configuration is pulled/available (e.g., `ollama pull mistral:latest`).

### Launching the WebUI

Once `safe_store` is installed with the `[webui]` extra, you can launch the WebUI using the command-line:

```bash
safestore-webui
```

This command will:
1.  Look for a `config.toml` file in the `safe_store/webui/` directory (within your Python environment's `site-packages` where `safe_store` is installed). If not found, it will create a default `config.toml` there.
2.  Start a Uvicorn server. By default, it will be accessible at `http://0.0.0.0:8000`.

### Configuring the WebUI

The WebUI's behavior is controlled by the `config.toml` file. When first launched via `safestore-webui`, if `config.toml` doesn't exist in the expected location (`.../site-packages/safe_store/webui/config.toml`), a default one is created. You can then edit this file.

**Default `config.toml` structure (located in `.../site-packages/safe_store/webui/` after first run):**

```toml
[lollms]
binding_name = "ollama" # Examples: "ollama", "lollms", "openai"
host_address = "http://localhost:11434" # e.g., "http://localhost:9600" for lollms, null for openai
model_name = "mistral:latest" # e.g., "mistral:latest", "gpt-4", specific model path for lollms
# service_key = null # Only if needed, e.g. for OpenAI if not using env var

[safestore]
db_file = "webui_store.db" # Path to the SQLite DB file used by the WebUI
doc_dir = "webui_safestore_docs" # Directory where uploaded files are temporarily copied for SafeStore processing
default_vectorizer = "st:all-MiniLM-L6-v2"
chunk_size = 250
chunk_overlap = 40

[graphstore]
# graph_extraction_prompt_template_file = null # Path to custom extraction prompt (optional)
# query_parsing_prompt_template_file = null # Path to custom query parsing prompt (optional)

[webui]
host = "0.0.0.0"
port = 8000
temp_upload_dir = "temp_uploaded_files_webui" # Initial upload destination before copying to doc_dir
log_level = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Key `config.toml` settings for WebUI:**

*   **`[lollms]` section:** Configure how `lollms-client` connects to your LLM.
    *   `binding_name`: The `lollms-client` binding (e.g., "ollama").
    *   `host_address`: URL of your LLM server (e.g., Ollama's default `http://localhost:11434`).
    *   `model_name`: The specific model for graph extraction (e.g., `mistral:latest`).
*   **`[safestore]` section:**
    *   `db_file`: The SQLite database file the WebUI will use. It will be created if it doesn't exist.
    *   `doc_dir`: A directory where files uploaded via the WebUI are placed for `SafeStore` to process.
*   **`[webui]` section:**
    *   `host`, `port`: Network interface and port for the WebUI.
    *   `temp_upload_dir`: A temporary staging area for uploads.
    *   `log_level`: Logging level for the WebUI backend console output.

**Using the WebUI:**

1.  **Launch:** Run `safestore-webui` in your terminal.
2.  **Access:** Open your web browser and go to `http://localhost:8000` (or the configured host/port).
3.  **Upload Document:**
    *   Use the "Upload Document" section.
    *   Select a file (`.txt`, `.pdf`, `.docx`, `.html` are supported if parsing extras are installed).
    *   Click "Upload & Process".
4.  **Processing:**
    *   The backend will save the file, add it to `SafeStore` (creating vector embeddings using the `default_vectorizer` from `config.toml`), and then trigger `GraphStore` to build graph elements from the document's text chunks using the configured LLM.
    *   Status messages will appear below the upload form.
5.  **View Graph:**
    *   The graph visualization should update automatically after processing (or on page load if data already exists).
    *   You can pan, zoom, and drag nodes.
6.  **Inspect Details:**
    *   Click on a node or an edge in the graph.
    *   Its details (label, properties) will appear in the "Selection Details" panel on the sidebar.

**Notes on WebUI:**
*   The WebUI currently uses the `default_vectorizer` specified in `config.toml` when `SafeStore` indexes uploaded documents.
*   Graph extraction and natural language querying in the WebUI (if implemented later) will use the LLM configured in the `[lollms]` section.
*   The graph visualization might become slow with very large graphs. The WebUI currently fetches a limited number of nodes/edges for display.
*   This WebUI is experimental and primarily for demonstration and basic interaction.

---

## 💡 Key Concepts

### `SafeStore`
*   Manages vector embeddings for semantic search.
*   Focuses on indexing documents, chunking, vectorizing text, and similarity queries.

### `GraphStore`
*   Builds and queries a knowledge graph from text data.
*   **LLM Executor Callback:** You provide a simple function `(prompt_string: str) -> llm_response_string`. `GraphStore` uses this to send its internally crafted prompts (for graph extraction or query parsing) to your chosen LLM.
*   **Internal Prompts:** `GraphStore` contains default prompt templates optimized for extracting graph data and for parsing natural language queries.
*   **Graph Querying (`query_graph`):** Translates natural language to graph traversals, returning subgraphs or linked text chunks.

---
## 🪵 Logging & Concurrency

*   **Logging:** Uses [`ascii_colors`](https://github.com/ParisNeo/ascii_colors). Configurable via `SafeStore(log_level=...)` or `GraphStore(log_level=...)`, or globally. The WebUI also uses `ascii_colors`, configured by `webui.log_level` in `config.toml`.
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