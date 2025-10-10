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
*   **🔧 Extensible:** Supports custom vectorizers through a dedicated folder path.
*   **⚙️ Advanced Indexing:** Configurable chunking strategies ('token' or 'character'), context expansion for stored chunks, and pluggable text cleaning.
*   **📄 Document Parsing:** Built-in parsers for `.txt`, `.pdf`, `.docx`, `.html`, and many common text-based formats.
*   **🔒 Optional Encryption:** Encrypts document chunk text at rest (AES-128) for enhanced security.
*   **🔄 Change Aware:** Automatically detects file changes for efficient re-indexing of vectors.

---

## ⚙️ Features

### `SafeStore` (Vector Database)
*   **Single Vector Space:** Each database is configured with **one** vectorizer upon initialization, ensuring all vectors are comparable.
*   **Advanced Indexing Pipeline:**
    *   **Text Cleaning:** Pluggable function to clean text before processing (e.g., remove artifacts).
    *   **Chunking Strategy:** Chunk text by `'character'` count or by `'token'` count using a model's tokenizer.
    *   **Context Expansion:** Store chunks with extra context (`expand_before`, `expand_after`) around the vectorized portion to improve RAG quality.
*   **Vectorization:**
    *   The vectorizer is defined when creating the `SafeStore` instance using a `vectorizer_name` (e.g., "st", "openai") and a `vectorizer_config` dictionary.
    *   Supports Sentence Transformers, TF-IDF, OpenAI, Cohere, and Ollama.
*   **Extensibility:** Provide a path to a folder with your own vectorizer implementations via the `custom_vectorizers_path` parameter.
*   **Querying (`query`):** Performs cosine similarity search using the instance's configured vectorizer.
*   **Model Discovery:** A class method `SafeStore.list_available_models()` to dynamically find available models for vectorizers like Ollama.

### `GraphStore` (Graph Database)
*   **Schema:** Dedicated tables for `graph_nodes`, `graph_relationships`, and `node_chunk_links` within the same SQLite DB.
*   **LLM Integration:** Uses a flexible `llm_executor_callback` to integrate with your choice of LLM (e.g., via `lollms-client`) for graph extraction and natural language querying.
*   **Graph Building Methods:** `process_chunk_for_graph`, `build_graph_for_document`, `build_graph_for_all_documents`.
*   **Natural Language Querying (`query_graph`):** Translates questions into graph traversals and returns results in various modes (`"graph_only"`, `"chunks_summary"`, `"full"`).
*   **Encryption Awareness:** Decrypts chunk text for LLM processing if the `encryption_key` is provided.

---

## 🚀 Installation

```bash
pip install safe-store
```
Install optional dependencies for the vectorizers or features you need:
```bash
# For Sentence Transformers (provides a tokenizer)
pip install safe-store[sentence-transformers]

# For API-based vectorizers (OpenAI, Ollama, Cohere)
pip install safe-store[openai]
pip install safe-store[ollama]
pip install safe-store[cohere]

# For providing a custom tokenizer like tiktoken
pip install tiktoken

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

This example shows `SafeStore` usage followed by `GraphStore` graph building and querying, reflecting the new, simplified API.

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
        LC_CLIENT = LollmsClient()
        print("LLM Client Initialized for GraphStore.")
        return True
    except Exception as e:
        print(f"LLM Client init failed: {e}. Graph features will not work.")
        return False

# LLM Executor Callback for GraphStore
def llm_executor(prompt_to_llm: str) -> str:
    if not LC_CLIENT: raise ConnectionError("LLM Client not ready.")
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
# Initialize the store with a single, dedicated vectorizer and chunking strategy
store = safe_store.SafeStore(
    db_path=DB_FILE,
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    chunking_strategy='token', # This works because 'st' provides its own tokenizer
    chunk_size=30,
    log_level=LogLevel.INFO
)

doc_id_1 = -1
with store:
    # Add document - no need to specify vectorizer or chunking here
    store.add_document(doc1_path)
    docs = store.list_documents()
    if docs: doc_id_1 = docs['doc_id']
    print(f"Document '{doc1_path.name}' (ID: {doc_id_1}) indexed by SafeStore.")
    
    # Query using the instance's configured vectorizer
    query_results = store.query("AI research in Geneva", top_k=1)
    if query_results:
        print(f"SafeStore query result for 'AI research in Geneva': {query_results['chunk_text'][:100]}...")

if LC_CLIENT and doc_id_1 != -1:
    # --- 3. Use GraphStore to Build & Query Knowledge Graph ---
    print("\n--- GraphStore Operations ---")
    graph_store = GraphStore(
        db_path=DB_FILE, # Use the same database
        llm_executor_callback=llm_executor,
        log_level=LogLevel.INFO
    )
    with graph_store:
        print(f"Building graph for document ID: {doc_id_1}...")
        graph_store.build_graph_for_document(doc_id_1)
        print("Graph building for document complete.")

        nl_query = "Who is the CEO of QuantumLeap AI and where is it based?"
        print(f"\nGraphStore NLQ: \"{nl_query}\"")
        
        graph_data = graph_store.query_graph(nl_query, output_mode="graph_only")
        if graph_data.get('nodes'): 
            print(f"  Sample Node from NLQ result: {graph_data['nodes']['properties']}")
```

---

## 🖥️ Web User Interface (Experimental)

`safe_store` includes an experimental WebUI to upload documents and visualize the extracted knowledge graph.

### Launching the WebUI
```bash
safestore-webui
```
By default, it is accessible at `http://0.0.0.0:8000`.

### Configuring the WebUI
The UI's behavior is controlled by `config.toml`, created in `.../site-packages/safe_store/webui/` on first run.

**Key `config.toml` settings for the `[safestore]` section:**
```toml
[safestore]
db_file = "webui_store.db"
doc_dir = "webui_safestore_docs"

# Define the single vectorizer for the WebUI's database instance
vectorizer_name = "st"
# Use a TOML inline table for the config dictionary
vectorizer_config = '{model = "all-MiniLM-L6-v2"}' 

# Define chunking strategy
chunk_size = 250
chunk_overlap = 40
chunking_strategy = "token" # 'st' provides a tokenizer, so this works
```

---

## 💡 Key Concepts

### `SafeStore`
*   **One Store, One Configuration**: When you create a `SafeStore` instance, you define the entire indexing pipeline for that database: the vectorizer, the chunking strategy, text cleaning, etc. This ensures all data is processed consistently.
*   **Smart Token Chunking**: For remote vectorizers like Ollama or OpenAI that don't have a client-side tokenizer, you can still perform token-based chunking by providing a proxy tokenizer (e.g., `tiktoken`) via the `custom_tokenizer` parameter.
*   **Context-Aware Chunks**: The `expand_before` and `expand_after` parameters allow you to store more context around each vectorized chunk, which is ideal for high-quality RAG.

### `GraphStore`
*   Builds upon a `SafeStore` database to create a knowledge graph.
*   **LLM Executor Callback**: You provide a simple function `(prompt: str) -> response: str`. `GraphStore` uses this to send prompts to your chosen LLM for graph extraction and querying.

---

## 🤝 Contributing & License

Contributions are welcome! Please open an issue or PR on [GitHub](https://github.com/ParisNeo/safe_store).
Licensed under Apache 2.0. See [LICENSE](LICENSE).