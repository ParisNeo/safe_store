# safe_store Documentation
*(Version: 3.6.1)*

## Table of Contents

1. [Introduction](#1-introduction)
   * [What is safe_store?](#what-is-safe_store)
   * [Key Features & Architecture](#key-features--architecture)
2. [Installation & Dependencies](#2-installation--dependencies)
3. [Quick Start](#3-quick-start)
4. [Database Diagnostics & Inspection (`store.info()`)](#4-database-diagnostics--inspection)
5. [Vectorization Backends](#5-vectorization-backends)
6. [The 8 RAG Chunking Strategies](#6-the-8-rag-chunking-strategies)
7. [Search & Retrieval Modes](#7-search--retrieval-modes)
   * [7.1. Dense Vector Search (`query`)](#71-dense-vector-search)
   * [7.2. Sparse Lexical Search (`BM25Retriever`)](#72-sparse-lexical-search)
   * [7.3. Tri-Modal Hybrid Retrieval (`hybrid_query`)](#73-tri-modal-hybrid-retrieval)
   * [7.4. Universal 0–100 Relevance Grading & Thresholding](#74-universal-0100-relevance-grading--thresholding)
   * [7.5. Full Document & Context Window Retrieval](#75-full-document--context-window-retrieval)
8. [Knowledge Graph & W3C SPARQL 1.1 Engine](#8-knowledge-graph--w3c-sparql-11-engine)
   * [8.1. Automatic Graph Extraction & Dynamic Extraction Prompt](#81-automatic-graph-extraction)
   * [8.2. W3C SPARQL 1.1 Query Engine (`query_sparql`)](#82-w3c-sparql-11-query-engine)
   * [8.3. SPARQL 1.1 Update Engine (`execute_sparql_update`)](#83-sparql-11-update-engine)
   * [8.4. TBox Ontology Management (`TBoxManager`)](#84-tbox-ontology-management)
   * [8.5. Declarative Tabular-to-Graph Mapping (`TabularMapper`)](#85-declarative-tabular-to-graph-mapping)
   * [8.6. Tri-Modal Unified Graph Search (`query_graph_hybrid`)](#86-tri-modal-unified-graph-search)
9. [LLM Cognitive Memory & Tool Calling](#9-llm-cognitive-memory--tool-calling)
   * [9.1. Episodic Memory Logging](#91-episodic-memory-logging)
   * [9.2. Associative Pathways & Chunk Grounding](#92-associative-pathways--chunk-grounding)
   * [9.3. Function Calling Tool Dispatcher](#93-function-calling-tool-dispatcher)
10. [Semantic Datalake & Point Cloud Engine](#10-semantic-datalake--point-cloud-engine)
11. [Zero-Leakage Local Encryption (Fernet AES-128/HMAC)](#11-zero-leakage-local-encryption)
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

Inspect any store instance or `.db` file in one call with `store.info()` or `store.get_database_info()`:

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
- `"st"`: Sentence-Transformers (e.g. `{"model": "all-MiniLM-L6-v2"}`)
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

---

## 8. Knowledge Graph & W3C SPARQL 1.1 Engine

### 8.1. Automatic Graph Extraction
`GraphStore.build_graph_for_all_documents()` dynamically parses document chunks using an LLM callback:
- **With Ontology**: Constrains extraction strictly to defined TBox classes and properties.
- **Without Ontology**: Dynamically extracts key concepts, entities, attributes, and relationships.

### 8.2. W3C SPARQL 1.1 Query Engine
Execute standard `SELECT`, `ASK`, `CONSTRUCT`, and `DESCRIBE` queries:
```python
from safe_store import GraphStore

graph = GraphStore(store=store)
results = graph.query_sparql("""
PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?personName ?companyName WHERE {
    ?person a ont:Person ;
            ont:name ?personName ;
            ont:worksFor ?company .
    ?company ont:name ?companyName .
}
""")
```

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

Visualize multi-dimensional embeddings as 2D/3D point clouds with persistent SQLite caching:
```python
# 2D PCA point cloud
points = store.get_datalake_view(method='pca', n_components=2, use_cache=True)

# Export standalone interactive HTML visualizer
store.export_datalake_html(output_file="datalake.html", method='pca', n_components=2)
```

---

## 11. Zero-Leakage Local Encryption

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
| `store.query(...)` | Dense vector similarity search with 0–100 relevance score. |
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