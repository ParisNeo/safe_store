# safe_store: Simple, Concurrent SQLite Vector Store & Graph Database for Local RAG

[![PyPI version](https://img.shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img.shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/safe_store.svg)](https://pypi.org/project/safe_store/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml/badge.svg)](https://github.com/ParisNeo/safe_store/actions/workflows/test.yml)

**safe_store** is a Python library providing a lightweight, file-based **vector database AND graph database** using a single **SQLite** file. It's designed for simplicity and efficiency, making it ideal for integrating into **local Retrieval-Augmented Generation (RAG)** pipelines and knowledge graph applications.

Store, manage, and query your document embeddings and extracted graph data locally with features like automatic change detection, multiple vectorization methods, safe concurrent access, document parsers, optional encryption, and LLM-powered graph construction & querying.

---

## ✨ Why safe_store?

*   **🎯 RAG & Knowledge Graph Focused:** Built for local RAG and knowledge graph use cases.
*   **🚀 Simple & Lightweight:** Uses a single SQLite file – no heavy dependencies or external servers.
*   **🏠 Local First:** Keep your embeddings, document text, and graph data entirely on your local machine.
*   **🤝 Concurrent Safe:** Handles database writes from multiple processes safely using file-based locking.
*   **🧠 Dual Capabilities:**
    *   **Vector Store (`SafeStore` class):** Index documents, generate embeddings (Sentence Transformers, TF-IDF), and perform semantic similarity search.
    *   **Graph Store (`GraphStore` class):** Extract entities and relationships from text chunks using an LLM, build a persistent knowledge graph, and query it using natural language or direct graph operations.
*   **🔗 Integrated Backend:** Both vector and graph data reside in the same SQLite database, allowing for potential future synergies.
*   **🤖 LLM-Powered Graph:**
    *   Uses a flexible callback mechanism to integrate with your choice of LLM (e.g., via `lollms-client`) for extracting graph structures from text.
    *   Internal prompt templates guide the LLM for consistent graph extraction and query parsing.
    *   Supports natural language querying of the graph, also powered by an LLM callback.
*   **📄 Document Parsing:** Built-in parsers for `.txt`, `.pdf`, `.docx`, and `.html` (requires `[parsing]` extra).
*   **🔒 Optional Encryption:** Encrypts document chunk text at rest (AES-128) for enhanced security (requires `[encryption]` extra).
*   **🔄 Change Aware (Vector Store):** Automatically detects file changes for efficient re-indexing of vectors.
*   **🗣️ Informative Logging:** Clear, leveled, and colorful console feedback via `ascii_colors`.

---

## ⚠️ Status: Beta

safe_store is currently in Beta. The core API for both `SafeStore` (vectors) and `GraphStore` (graphs) is stabilizing, but breaking changes are still possible before a `2.0.0` release. Feedback and contributions are welcome!

---

## ⚙️ Features

### Core
*   **Storage:** All data (documents, chunks, vectors, graph nodes, relationships, metadata) in a single SQLite file.
*   **Concurrency:** Process-safe writes using file-based locking (`filelock`). SQLite's WAL mode for concurrent reads.

### `SafeStore` (Vector Database)
*   **Indexing (`add_document`):**
    *   Parses `.txt`, `.pdf`, `.docx`, `.html`/`.htm` (requires `safe_store[parsing]`).
    *   Stores full text, performs configurable chunking.
    *   File hashing (SHA256) for change detection and efficient re-indexing.
    *   Optional JSON metadata per document.
*   **Encryption (Optional):** Encrypts `chunk_text` at rest (AES-128-CBC via `cryptography`).
*   **Vectorization:**
    *   Supports multiple methods (Sentence Transformers `st:`, TF-IDF `tfidf:`).
    *   Manages vectorizer state (e.g., fitted TF-IDF models).
*   **Querying (`query`):** Cosine similarity search for `top_k` relevant chunks.
*   **Management:** `add_vectorization`, `remove_vectorization`, `list_documents`, `list_vectorization_methods`.

### `GraphStore` (Graph Database) - New!
*   **Schema:** Dedicated tables for `graph_nodes`, `graph_relationships`, and `node_chunk_links` within the same SQLite DB.
*   **LLM Integration for Graph Building:**
    *   Uses a user-provided `llm_executor_callback` (which takes a full prompt string and returns the LLM's raw string response).
    *   `GraphStore` internally manages and provides optimized prompts to this callback for:
        *   Extracting nodes (label, properties, `unique_id_key`) and relationships (source, target, type, properties) from text chunks.
    *   Handles de-duplication of nodes based on a generated `unique_signature`.
    *   Updates properties of existing nodes if new information is extracted for the same entity.
    *   Links extracted graph nodes back to their source text chunks.
*   **Graph Building Methods:**
    *   `process_chunk_for_graph(chunk_id)`: Processes a single chunk.
    *   `build_graph_for_document(doc_id)`: Processes all chunks of a document.
    *   `build_graph_for_all_documents()`: Processes all unprocessed chunks in the store.
*   **LLM Integration for Graph Querying:**
    *   `query_graph(natural_language_query, output_mode, ...)`:
        *   Uses the `llm_executor_callback` with an internal query-parsing prompt to translate NLQ into a structured graph query (identifying seed nodes, target relationships, etc.).
        *   Performs graph traversal based on the parsed query.
    *   **Output Modes for `query_graph`:**
        *   `"graph_only"`: Returns the subgraph (nodes and relationships).
        *   `"chunks_summary"`: Returns text chunks linked to the subgraph, similar to `SafeStore.query`.
        *   `"full"`: Returns both graph data and linked chunk summaries.
*   **Direct Graph Read Methods:**
    *   `get_node_details(node_id)`
    *   `get_nodes_by_label(label)`
    *   `get_relationships(node_id, type, direction)`
    *   `find_neighbors(node_id, type, direction)`
    *   `get_chunks_for_node(node_id)`: Retrieves text chunks that contributed to a specific node.
*   **Encryption Awareness:** If `GraphStore` is initialized with an `encryption_key` (matching `SafeStore`), it will decrypt chunk text before sending it to the LLM for graph extraction.

---

## 🚀 Installation

```bash
pip install safe_store
```

Install optional dependencies:

```bash
# For Sentence Transformers embedding models
pip install safe_store[sentence-transformers]

# For TF-IDF vectorization
pip install safe_store[tfidf]

# For parsing PDF, DOCX, HTML files
pip install safe_store[parsing]

# For encrypting chunk text at rest
pip install safe_store[encryption]

# To install everything (all vectorizers, parsers, encryption):
pip install safe_store[all] 
# Note: [all] now implicitly includes dependencies for graph features if any were specific.
# lollms-client or other LLM libraries are NOT included by default; install them separately.
```

To use the `GraphStore` features with an LLM, you'll also need an LLM client library like `lollms-client`:
```bash
pip install lollms-client
# And any specific bindings for lollms-client, e.g., pip install ollama
```

---

## 🏁 Quick Start

This example shows basic `SafeStore` usage followed by `GraphStore` graph building and querying.

```python
import safe_store
from safe_store import GraphStore, LogLevel # SafeStore is also in safe_store module
from lollms_client import LollmsClient # For the LLM callback
from pathlib import Path
import json # For pretty printing results

# --- 0. Configuration & LLM Setup ---
DB_FILE = "quickstart_store.db"
DOC_DIR = Path("temp_docs_qs")
DOC_DIR.mkdir(exist_ok=True, parents=True)
Path(DB_FILE).unlink(missing_ok=True) # Clean start

# LollmsClient setup (replace with your actual LLM server config)
LC_CLIENT: Optional[LollmsClient] = None
def init_llm():
    global LC_CLIENT
    try:
        LC_CLIENT = LollmsClient(binding_name="ollama", model_name="mistral:latest") # Example
        if not LC_CLIENT.ping(): raise ConnectionError("LLM server not reachable")
        print("LLM Client Initialized.")
        return True
    except Exception as e:
        print(f"LLM Client init failed: {e}. Graph features needing LLM will not work.")
        return False

# LLM Executor Callback for GraphStore
def llm_executor(prompt_to_llm: str) -> str:
    if not LC_CLIENT: raise ConnectionError("LLM Client not ready for executor callback.")
    # generate_code expects LLM to output markdown ```json ... ```
    # GraphStore's internal prompts already ask for this.
    response = LC_CLIENT.generate_code(prompt_to_llm, language="json", temperature=0.1, max_size=3000)
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
        llm_executor_callback=llm_executor, # Pass the executor
        log_level=LogLevel.INFO
    )
    with graph_store:
        print(f"Building graph for document ID: {doc_id_1}...")
        graph_store.build_graph_for_document(doc_id_1)
        print("Graph building for document complete.")

        # Demonstrate a direct graph read
        aris_nodes = graph_store.get_nodes_by_label("Person", limit=5)
        print(f"\nFound Person nodes: {[n.get('properties',{}).get('name') for n in aris_nodes if n.get('properties')]}")

        # Demonstrate Natural Language Querying of the Graph
        nl_query = "Who is the CEO of QuantumLeap AI and where is it based?"
        print(f"\nGraphStore NLQ: \"{nl_query}\"")
        
        # Mode 1: Graph Only
        graph_data = graph_store.query_graph(nl_query, output_mode="graph_only")
        print("\nQuery Result (graph_only):")
        print(f"  Nodes: {len(graph_data.get('nodes',[]))}, Relationships: {len(graph_data.get('relationships',[]))}")
        if graph_data.get('nodes'): 
            print(f"  Sample Node: {graph_data['nodes'][0]['label']} - {graph_data['nodes'][0]['properties']}")

        # Mode 2: Chunks Summary
        chunk_summary = graph_store.query_graph(nl_query, output_mode="chunks_summary")
        print("\nQuery Result (chunks_summary):")
        for i, chunk in enumerate(chunk_summary[:2]): # Show first 2 chunks
            print(f"  Chunk {i+1} (ID {chunk['chunk_id']}): {chunk['chunk_text'][:80]}...")
            print(f"    Linked to: {chunk.get('linked_graph_nodes')}")

# Cleanup (optional)
# import shutil
# shutil.rmtree(DOC_DIR, ignore_errors=True)
# Path(DB_FILE).unlink(missing_ok=True)
```

*(See `examples/` directory for more detailed usage, including `graph_usage.py`.)*

---

## 💡 Key Concepts

### `SafeStore`
*   Manages vector embeddings for semantic search.
*   Focuses on indexing documents, chunking, vectorizing text, and similarity queries.

### `GraphStore`
*   Builds and queries a knowledge graph from text data.
*   **LLM Executor Callback:** You provide a simple function `(prompt_string: str) -> llm_response_string`. `GraphStore` uses this to send its internally crafted prompts (for graph extraction or query parsing) to your chosen LLM.
*   **Internal Prompts:** `GraphStore` contains default prompt templates optimized for extracting graph data (nodes with labels, properties, unique identifiers; relationships with types, properties) and for parsing natural language queries into structured graph search parameters. These prompts instruct the LLM to return JSON wrapped in markdown code blocks, which `lollms-client`'s `generate_code` method can then parse.
*   **Graph Querying (`query_graph`):**
    1.  Takes your natural language question.
    2.  Uses the LLM executor with its internal query-parsing prompt to understand your question (e.g., identify "Person" named "Alice" as a starting point).
    3.  Traverses its stored graph based on this understanding (e.g., find "Alice", then find companies she "WORKS_AT").
    4.  Returns results in one of three modes:
        *   `"graph_only"`: The nodes and relationships found.
        *   `"chunks_summary"`: The original text chunks that are linked to the found graph elements.
        *   `"full"`: Both the graph data and the linked text chunks.

---
## 🪵 Logging & Concurrency

*   **Logging:** Uses [`ascii_colors`](https://github.com/ParisNeo/ascii_colors). Configurable via `SafeStore(log_level=...)` or `GraphStore(log_level=...)`, or globally.
*   **Concurrency:** `filelock` ensures process-safe writes for both `SafeStore` and `GraphStore` operations on the shared SQLite DB.

---

## 🔮 Future Work

*   **Advanced Graph Traversal:** More complex pathfinding, weighted relationships in `query_graph`.
*   **Hybrid Search:** Combining vector similarity search with graph query results.
*   **Graph Curation API:** More methods in `GraphStore` for direct node/relationship updates, merges, and deletions.
*   **More Vectorizers/Embedders.**
*   **Async API.**

---

## 🤝 Contributing & License

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/ParisNeo/safe_store).
Licensed under Apache 2.0. See [LICENSE](LICENSE).