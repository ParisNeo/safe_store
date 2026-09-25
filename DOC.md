# safe_store Documentation
*(Version: 3.6.1)*

## Table of Contents

1. [Introduction](#1-introduction)
   * [What is safe_store?](#what-is-safe_store)
   * [Key Features & Architecture](#key-features--architecture)
2. [Installation & Dependencies](#2-installation--dependencies)
3. [Quick Start](#3-quick-start)
4. [Database Diagnostics & Inspection (`store.info()`)](#4-database-diagnostics--inspection)
5. [SafeStore Studio (Desktop & Web GUI)](#5-safestore-studio-desktop--web-gui)
5. [Vectorization Backends](#5-vectorization-backends)
6. [The 8 RAG Chunking Strategies](#6-the-8-rag-chunking-strategies)
7. [Search & Retrieval Modes](#7-search--retrieval-modes)
   * [7.1. Dense Vector Search (`query`)](#71-dense-vector-search)
   * [7.2. Sparse Lexical Search (`BM25Retriever`)](#72-sparse-lexical-search)
   * [7.3. Tri-Modal Hybrid Retrieval (`hybrid_query`)](#73-tri-modal-hybrid-retrieval)
   * [7.4. Universal 0–100 Relevance Grading & Thresholding](#74-universal-0100-relevance-grading--thresholding)
   * [7.5. Full Document & Context Window Retrieval](#75-full-document--context-window-retrieval)
   * [7.6. Overlapping Chunk Reconstruction & Chronological Fusion](#76-overlapping-chunk-reconstruction--chronological-fusion)
8. [Knowledge Graph & W3C SPARQL 1.1 Engine](#8-knowledge-graph--w3c-sparql-11-engine)
   * [8.1. Automatic Graph Extraction & Dynamic Extraction Prompt](#81-automatic-graph-extraction)
   * [8.2. W3C SPARQL 1.1 Query Engine (`query_sparql`)](#82-w3c-sparql-11-query-engine)
   * [8.3. SPARQL 1.1 Update Engine (`execute_sparql_update`)](#83-sparql-11-update-engine)
   * [8.4. TBox Ontology Management (`TBoxManager`)](#84-tbox-ontology-management)
   * [8.5. Declarative Tabular-to-Graph Mapping (`TabularMapper`)](#85-declarative-tabular-to-graph-mapping)
   * [8.6. Tri-Modal Unified Graph Search (`query_graph_hybrid`)](#86-tri-modal-unified-graph-search)
   * [8.7. Custom LLM Generator Callable Specification](#87-custom-llm-generator-callable-specification)
9. [LLM Cognitive Memory & Tool Calling](#9-llm-cognitive-memory--tool-calling)
   * [9.1. Episodic Memory Logging](#91-episodic-memory-logging)
   * [9.2. Associative Pathways & Chunk Grounding](#92-associative-pathways--chunk-grounding)
   * [9.3. Function Calling Tool Dispatcher](#93-function-calling-tool-dispatcher)
10. [Semantic Datalake & Point Cloud Engine](#10-semantic-datalake--point-cloud-engine)
11. [Document Clustering & Thematic Grouping](#11-document-clustering--thematic-grouping)
12. [Zero-Leakage Local Encryption (Fernet AES-128/HMAC)](#12-zero-leakage-local-encryption)
12. [Database Portability, Re-Vectorization, Export & Import](#12-database-portability--re-vectorization)
13. [API Reference Summary](#13-api-reference-summary)
14. [License](#14-license)

---

## 1. Introduction

### What is safe_store?
**`safe_store`** is an ultra-fast, local, and sovereign knowledge engine for Python. It stores unstructured documents (PDF, DOCX, HTML, Markdown, Text, Code) and structured datasets (CSV, XLSX, SQLite) inside a single SQLite database (`.db`), unifying:
- **Dense Vector Search**: Powered by Sentence-Transformers, Ollama, OpenAI, Cohere, Lollms, TF-IDF, or Grepper.
- **Sparse BM25 Lexical Search**: Native SQLite FTS5 index for exact keywords, error codes, and identifiers.
- **W3C SPARQL 1.1 Knowledge Graph**: Full `SELECT`, `ASK`, `CONSTRUCT`, `DESCRIBE`, and `INSERT/DELETE DATA` updates.
- **LLM Cognitive Memory**: Episodic memory recording, associative traversal, and chunk-grounded evidence linking.
- **Tri-Modal Reciprocal Rank Fusion (RRF)**: Combining vector, lexical, and graph signals with calibrated 0–100 relevance grades.
- **Semantic Datalake Point Cloud**: 2D/3D PCA & t-SNE projections with instant SQLite caching and interactive HTML exports.
- **Zero-Leakage Encryption**: Transparent AES-128-CBC + HMAC-SHA256 authenticated encryption at rest.

---

## 2. Installation & Dependencies

```bash
# Core package
pip install safe_store

# Optional extras
pip install safe_store[sentence-transformers] # Local Hugging Face models
pip install safe_store[openai]                # OpenAI & Lollms API
pip install safe_store[ollama]                # Local Ollama client
pip install safe_store[cohere]                # Cohere API
pip install safe_store[parsing]               # PDF, DOCX, HTML parser dependencies
pip install safe_store[encryption]            # Cryptography & Fernet encryption
pip install safe_store[all]                   # Everything
```

---

## 3. Quick Start

```python
import safe_store

# 1. Initialize SafeStore with persistent configuration
store = safe_store.SafeStore(
    db_path="knowledge.db",
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    chunk_size=128,
    chunk_overlap=16,
    chunking_strategy="token"
)

with store:
    # 2. Add unstructured documents
    store.add_text(
        unique_id="service_manual",
        text="Telemetry controller emitted error code ERR-9042 during initialization. Replace memory buffer.",
        metadata={"service": "Telemetry", "severity": "High"}
    )

    # 3. Hybrid search combining dense embeddings + BM25 lexical matches
    results = store.hybrid_query(
        query_text="troubleshooting memory failure ERR-9042",
        top_k=3,
        min_relevance_percent=30.0 # Standard 0-100 threshold filter
    )

    for r in results:
        print(f"[{r['file_path']}] Grade: {r['relevance_score']:.1f}% | Text: {r['chunk_text']}")
```

---

## 4. Database Diagnostics & Inspection
```python
from safe_store import SafeStore

store = SafeStore("knowledge.db")
store.info()
```

---

## 5. SafeStore Studio (Desktop & Web GUI)

Launch the interactive studio from the command line:

```bash
# Launch Projects Cards Hub (landing page with full CRUD for all stores)
safe-store-studio

# Or open a specific database directly in the Studio Workspace
safe-store-studio database.db

# Browser mode
safe-store-studio --browser --port 8080
```

### Studio Capabilities & Layout:
1. **Projects Hub & Store CRUD**:
   - Visual responsive cards for every `.db` database in your working directory and `projects/` folder.
   - **Create Store**: Create fresh stores with your choice of vectorizer (`st`, `tfidf`, `ollama`, `openai`, `cohere`, `grepper`), chunking strategy, and optional encryption password.
   - **Edit Store**: Rename databases and edit descriptions.
   - **Delete Store**: Permanently remove databases and their lock/WAL artifacts with a confirmation dialog.
2. **Deep-Dive Store Workspace (The 5 Tabs)**:
   - **Files & Documents**: Ingest documents (`.pdf`, `.docx`, `.md`, `.txt`, `.csv`), review chunks, and browse full reconstructed text.
   - **Semantic Datalake**: Interactive 2D/3D UMAP/PCA/t-SNE point-cloud explorer with center-of-gravity markers and chunk inspection.
   - **Knowledge Graph Studio**: 3-panel visual network workspace with interactive physics canvas (`vis-network`), manual node/edge creation, fast full-document extraction, and SPARQL 1.1 query highlighting!
   - **RAG Search Studio**: Compare dense, lexical BM25, and hybrid queries with real-time relevance threshold sliders and contiguous chunk reconstruction.
   - **Database Diagnostics**: Instant introspection of models, schemas, and topology counts.
   - **`← All Stores`**: Navigate back to the Projects cards grid anytime.

Inspect any store instance or `.db` file programmatically in one call with `store.info()` or `store.get_database_info()`:

```python
from safe_store import SafeStore

store = SafeStore("knowledge.db")

# 1. Print formatted diagnostic panel to console
store.info()

# 2. Retrieve structured dictionary
diag = store.get_database_info()
print(f"Total Documents: {diag['documents']['total_documents']}")
print(f"Total Chunks:    {diag['documents']['total_chunks']}")
print(f"Knowledge Graph: {diag['knowledge_graph']['total_nodes']} nodes, {diag['knowledge_graph']['total_relationships']} edges")
```

---

## 5. Vectorization Backends

Configure any vectorizer at creation time:
- `"st"`: Sentence-Transformers (e.g. `{"model": "all-MiniLM-L6-v2"}`). Supports `use_shared_server=True` for multi-user/multi-process server deployments with dynamic micro-batching and persistent resource sharing (see [Shared Model Server Blueprint](docs/shared_model_server_architecture.md)).
- `"ollama"`: Ollama local daemon (e.g. `{"model": "nomic-embed-text", "host": "http://localhost:11434"}`)
- `"openai"`: OpenAI API (e.g. `{"model": "text-embedding-3-small"}`)
- `"cohere"`: Cohere API (e.g. `{"model": "embed-english-v3.0"}`)
- `"lollms"`: Any OpenAI-compatible endpoint (e.g. `{"model": "nomic-embed-text", "base_url": "http://localhost:9600"}`)
- `"tfidf"`: Data-dependent Term Frequency - Inverse Document Frequency.
- `"grepper"`: Lightweight inverted index with markdown section tree extraction.

---

## 6. The 8 RAG Chunking Strategies

Specify `chunking_strategy` when initializing `SafeStore`:
1. `'token'`: (Default) Token-based sliding window preserving line breaks.
2. `'recursive'`: Splits text across paragraphs -> headers -> code -> sentences -> words.
3. `'structure'` / `'markdown'`: Extracts Markdown `# H1 > ## H2` header lineage breadcrumbs.
4. `'semantic'`: Embeds sentences and cuts at cosine similarity valleys (topic shifts).
5. `'contextual'`: Prepends full-document situating context (Anthropic pattern).
6. `'late'`: Full-document contextual token embedding with mean pooling (Jina AI pattern).
7. `'paragraph'`: Groups natural double-newline paragraphs up to chunk size.
8. `'character'`: Fixed character sliding window.

---

## 7. Search & Retrieval Modes

### 7.1. Dense Vector Search
```python
results = store.query("neural network architecture", top_k=3, min_relevance_percent=40.0)
```

### 7.2. Sparse Lexical Search
```python
from safe_store import BM25Retriever
bm25 = BM25Retriever(store.conn)
results = bm25.search("ERR-9042", top_k=3, min_relevance_percent=20.0)
```

### 7.3. Tri-Modal Hybrid Retrieval
```python
fused = store.hybrid_query(
    query_text="database connection pool leak",
    top_k=5,
    dense_weight=0.5,
    bm25_weight=0.5,
    min_relevance_percent=35.0
)
```

### 7.4. Universal 0–100 Relevance Grading & Thresholding
Every query returns a calibrated `relevance_score` and `similarity_percent` from `0.0` to `100.0`. Queries with results below `min_relevance_percent` return a clean empty list `[]` to prevent LLM context pollution.

### 7.5. Full Document & Context Window Retrieval
```python
# Aggregate chunk matches to return ranked full documents
full_docs = store.query_full_documents("write ahead log durability", top_k_docs=1, min_relevance_percent=45.0)

# Expand matching chunks with surrounding neighborhood context
windows = store.query_document_content_window("leader election", top_k_hits=1, window_before=1, window_after=1)

# Paginate through document chunks
page_view = store.get_document_content_paginated("doc_id_or_path", page=1, page_size=5)
```

### 7.6. Overlapping Chunk Reconstruction & Chronological Fusion
Eliminate chunk fragmentation and duplicate text at boundary seams:
```python
# 1. Enable directly in dense or hybrid queries
reconstructed = store.query(
    "supervisor daemon telemetry metrics",
    top_k=5,
    reconstruct_overlapping_chunks=True, # Seamless chronological fusion
    add_metadata=True                   # Single unified metadata header per doc
)

# 2. Or post-process any query result set
fused = store.reconstruct_overlapping_chunks(raw_results, add_metadata=True)
```

### 7.6. Overlapping Chunk Reconstruction & Chronological Fusion
Eliminate chunk fragmentation and duplicate text at boundary seams:
```python
# 1. Enable directly in dense or hybrid queries
reconstructed = store.query(
    "supervisor daemon telemetry metrics",
    top_k=5,
    reconstruct_overlapping_chunks=True, # Seamless chronological fusion
    add_metadata=True                   # Single unified metadata header per doc
)

# 2. Or post-process any query result set
fused = store.reconstruct_overlapping_chunks(raw_results, add_metadata=True)
```

---

## 8. Knowledge Graph & W3C SPARQL 1.1 Engine

### 8.1. Automatic Graph Extraction (High-Context Fast Modes)
Modern LLMs have context windows of 32k to 128k+ tokens. Rather than making dozens of slow, sequential calls chunk-by-chunk, `GraphStore` supports three high-speed extraction modes:

```python
# Mode A: 'document' (Default & Fastest - 1 LLM call per document)
# Executes up to 20x faster and captures relationships spanning across multiple sections.
stats = graph.build_graph_for_all_documents(
    mode='document',
    guidance="Focus on software microservices, APIs, and team owners."
)

# Mode B: 'batch_chunks' (Balanced - groups N chunks per call)
stats = graph.build_graph_for_all_documents(
    mode='batch_chunks',
    chunks_per_batch=10 # Slices of 10 chunks per LLM call
)

# Mode C: 'chunk' (Granular - processes each chunk in isolation)
stats = graph.build_graph_for_all_documents(mode='chunk')
```

- **With Ontology**: Constrains extraction strictly to defined TBox classes and properties.
- **Without Ontology**: Dynamically extracts rich open-ended concepts, entities, attributes, and relationships.
- **Evidence Provenance**: All nodes extracted in full document or batched modes are automatically linked to their respective chunk IDs in SQLite for grounded verification.

### 8.2. W3C SPARQL 1.1 Query Engine & AI SPARQL Generator
Execute standard `SELECT`, `ASK`, `CONSTRUCT`, and `DESCRIBE` queries across your knowledge graph:

```python
from safe_store import GraphStore

graph = GraphStore(store=store)

# 1. AI-Powered SPARQL Generation (Natural Language to SPARQL 1.1):
# Grounded in your database's actual entity classes, attributes, and relationships!
sparql_query = graph.generate_sparql("Find all tools that depend on other modules and who created them")
print("Generated SPARQL:\n", sparql_query)

# 2. Execute SPARQL Query:
results = graph.query_sparql(sparql_query)
for b in results["results"]["bindings"]:
    print(b)
```

In **SafeStore Studio**:
The Knowledge Graph right sidebar features a built-in **AI SPARQL Generator (LOLLMS)** bar. Type your question in natural language, press **Enter** or click **Generate SPARQL**, and the query will be written into the editor, ready to execute and highlight matching nodes on the interactive canvas.

### 8.3. SPARQL 1.1 Update Engine
Reorganize knowledge graphs using standard SPARQL 1.1 updates:
```python
graph.execute_sparql_update("""
PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
INSERT DATA {
    ex:Alice a ont:Architect ;
             ont:name "Alice Smith" ;
             ont:leadsProject ex:ProjectPhoenix .
}
""")
```

### 8.4. TBox Ontology Management
```python
from safe_store import TBoxManager

tbox = TBoxManager()
tbox.load_ontology("domain.ttl", format="turtle")
classes = tbox.get_classes()
subclasses = tbox.get_subclasses("http://example.org/ontology/Agent")
```

### 8.5. Declarative Tabular-to-Graph Mapping (Zero-LLM)
Map structured CSV, XLSX, and SQLite files into grounded ABox graphs in milliseconds:
```python
from safe_store import TabularMapper

mapper = TabularMapper(store=store, tbox=tbox)
mapper.map_csv("inventory.csv", mapping_rules={
    "entity_mappings": [
        {
            "class": "http://example.org/ontology/Product",
            "subject_template": "http://example.org/product/{sku}",
            "properties": {"product_name": "http://example.org/ontology/hasName"}
        }
    ],
    "relationship_mappings": [
        {
            "predicate": "http://example.org/ontology/suppliedBy",
            "source_template": "http://example.org/product/{sku}",
            "target_template": "http://example.org/supplier/{supplier_id}"
        }
    ]
})
```

### 8.6. Tri-Modal Unified Graph Search
```python
response = graph.query_graph_hybrid(
    query_text="What microservices depend on AuthEngine?",
    top_k=5,
    dense_weight=0.4,
    bm25_weight=0.3,
    graph_weight=0.3
)
```

### 8.7. Custom LLM Generator Callable Specification

SafeStore allows the calling application to inject **any** tool or LLM provider using a flexible, standardized callable. This completely decouples SafeStore from any specific client library.

#### 1. Standard Protocol Signature
```python
def my_llm_generator(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_mode: bool = False,
    **kwargs: Any
) -> str:
    """
    Args:
        prompt: Main instruction or text content.
        system_prompt: Role or task description (e.g. JSON schema instructions).
        json_mode: True when the engine expects well-formed JSON output.
        **kwargs: Additional generation options (e.g. temperature, max_tokens).
    Returns:
        Generated text or JSON code string.
    """
    ...
```

#### 2. Plug in OpenAI, Ollama, or Anthropic
```python
from openai import OpenAI
from safe_store import SafeStore

client = OpenAI()

def openai_generator(prompt: str, system_prompt: str = None, json_mode: bool = False, **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"} if json_mode else None,
        temperature=0.1
    )
    return response.choices[0].message.content

# Pass directly to SafeStore
store = SafeStore("enterprise.db", llm_generator=openai_generator)
graph = store.graph  # Automatically inherits the custom generator
```

#### 3. Single-Argument Lambda (Fastest 1-Liner)
If your generator only accepts `(prompt)`, SafeStore automatically adapts it and prepends the system prompt:
```python
store = SafeStore(
    "knowledge.db",
    llm_generator=lambda prompt: my_local_model.generate(prompt)
)
```

#### 4. Change Generator at Runtime
```python
store.set_llm_generator(new_generator)
```

---

## 9. LLM Cognitive Memory & Tool Calling

Empower autonomous agents with episodic event recording, associative recall, and chunk grounding:

```python
# 1. Record an episodic event linked to chunk evidence
episode_id = graph.memory.record_episode(
    title="Ledger Protocol Architecture Review",
    description="Alice presented the decentralized consensus specification.",
    participants=["Alice Smith"],
    outcome="Approved",
    source_chunk_ids=[1]
)

# 2. Associative recall
memory_view = graph.memory.recall_associative("Alice Smith", max_hops=2)

# 3. Standard tool schemas for OpenAI, Anthropic, Ollama, Lollms
tools = graph.get_tool_definitions()
# Dispatch LLM tool calls
result = graph.dispatch_tool("recall_associative_memory", {"concept": "Alice Smith"})
```

---

## 10. Semantic Datalake & Point Cloud Engine

Visualize multi-dimensional embeddings as 2D/3D point clouds using state-of-the-art **UMAP** (Uniform Manifold Approximation and Projection) with native cosine distance metric and persistent SQLite caching:
```python
# 2D UMAP point cloud (State of the Art)
points = store.get_datalake_view(method='umap', n_components=2, use_cache=True)

# 3D UMAP point cloud
points_3d = store.get_datalake_view(method='umap', n_components=3, use_cache=True)

# Export standalone interactive HTML visualizer
store.export_datalake_html(output_file="datalake.html", method='umap', n_components=2)
```

---

## 11. Document Clustering & Thematic Grouping

SafeStore provides semantic document clustering and automated theme generation:
- Calculates normalized document-level vector centroids from all constituent chunk embeddings.
- Groups documents using **K-Means** or **Agglomerative Hierarchical Clustering**, with automatic optimal cluster estimation ($k$-selection).
- Synthesizes descriptive thematic titles, descriptions, and topic tags using the configured LLM callable or deterministic c-TF-IDF keyword extraction.
- Persists clusters in SQLite for instant cache hits, automatically invalidating whenever documents are added or removed.

### 11.1. Programmatic Clustering
```python
from safe_store import SafeStore

store = SafeStore("enterprise.db")

# 1. Cluster all documents (auto-estimating optimal k)
clusters = store.cluster_documents(
    n_clusters='auto',       # Or specify an integer e.g. 4
    method='kmeans',         # 'kmeans' or 'agglomerative'
    generate_themes=True,    # Generate titles, summaries, and tags
    save_to_store=True       # Cache results in SQLite
)

for c in clusters:
    print(f"Theme #{c['cluster_id'] + 1}: {c['theme_title']}")
    print(f"Description: {c['theme_description']}")
    print(f"Topics: {', '.join(c['key_topics'])}")
    print(f"Documents ({c['document_count']}): {[d['document_title'] for d in c['documents']]}\n")

# 2. Retrieve cached clusters instantly
cached = store.get_document_clusters(use_cache=True)
```

### 11.2. SafeStore Studio Interactive Clustering Workspace
SafeStore Studio features a dedicated **Clusters & Themes** tab:
- Configure cluster counts ($k=0$ for auto-estimation) and algorithm (`K-Means` or `Agglomerative`).
- Click **Cluster Documents** to inspect responsive theme cards with color bands, topic tags, and member document tables.
- Click **Query in Search Studio** on any theme card to immediately search the corpus for related passages.

---

## 12. Zero-Leakage Local Encryption

Supply an `encryption_key` when opening `SafeStore` to enable authenticated Fernet (AES-128-CBC + HMAC-SHA256) encryption for all chunk texts and metadata blobs:
```python
store = safe_store.SafeStore("secure.db", encryption_key="my-secret-passphrase")
```

---

## 12. Database Portability & Re-Vectorization

```python
# In-place re-vectorization using a new embedding model
store.revectorize_database("openai", {"model": "text-embedding-3-small"})

# Portable JSON backup & restore
store.export_database("backup.json", decrypt=False)
restored = safe_store.SafeStore.import_database("backup.json", "restored.db", decryption_key="my-key")
```

---

## 13. API Reference Summary

| Feature / Command | Purpose & Return Value |
| :--- | :--- |
| **`store.info()`** / **`store.get_database_info()`** | Returns/prints comprehensive diagnostics: vectorizer info, per-document chunk counts, ontology schemas, and graph topology counts. |
| **`SafeStore(db_path, ...)`** | Main SQLite vector, lexical, and hybrid database handle. |
| **`store.cluster_documents(...)`** | Clusters documents by semantic centroid and synthesizes thematic titles, descriptions, and topic tags. |
| **`store.get_document_clusters(...)`** | Retrieves cached or freshly computed document clusters and thematic groups. |
| `store.unload_vectorizer()` | Forces immediate unloading of local model weights and purges GPU VRAM. |
| `SafeStore.shutdown_shared_vectorizer(port)` | Special command to shut down the persistent shared model server daemon. |
| `store.query(...)` | Dense vector similarity search with 0–100 relevance score (supports `reconstruct_overlapping_chunks=True`). |
| `store.reconstruct_overlapping_chunks(...)` | Reconstructs and chronologically fuses overlapping and non-contiguous chunks with single metadata header. |
| `store.hybrid_query(...)` | Tri-Modal Reciprocal Rank Fusion (Dense + BM25). |
| `store.query_full_documents(...)` | Full document retrieval aggregated from chunk hits. |
| `store.query_document_content_window(...)` | Retrieves matching chunks with surrounding context window. |
| `store.get_document_content_paginated(...)` | Page-by-page chunk inspection. |
| `GraphStore(store, ...)` | Knowledge graph management with W3C SPARQL 1.1 query/update support. |
| **`graph.get_graph_info()`** | Returns knowledge graph diagnostics: total nodes, total edges, label breakdown, and chunk provenance link counts. |
| **`graph.build_graph_for_all_documents()`** | Automatically extracts graph nodes and relationships across all documents with live per-chunk progress reporting. |
| `graph.query_sparql(...)` | Executes SPARQL 1.1 `SELECT`, `ASK`, `CONSTRUCT`, `DESCRIBE`. |
| `graph.execute_sparql_update(...)` | Executes SPARQL 1.1 `INSERT DATA`, `DELETE DATA`, `DELETE WHERE`. |
| `graph.memory` (`CognitiveMemoryStore`) | Episodic logging, associative recall, and chunk grounding. |
| `TBoxManager` | OWL/RDFS ontology schema management. |
| `TabularMapper` | Declarative mapping for CSV, XLSX, and SQLite tables. |

---

## 14. License

Licensed under the [Apache 2.0 License](LICENSE).