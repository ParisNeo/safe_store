# safe_store: The Local Multi-Modal Vector, Graph & Semantic Engine

[![PyPI version](https://img.shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img.shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)

**`safe_store` is an ultra-fast, local, and sovereign knowledge engine for Python.** It transforms unstructured documents (PDF, DOCX, HTML, Markdown, Code) and structured datasets (CSV, Excel XLSX, SQLite) into an interconnected, queryable knowledge base combining:

1. 🧠 **Dense Semantic Vector Search**: Embeddings powered by Sentence-Transformers, Ollama, OpenAI, Cohere, Lollms, or TF-IDF.
2. ⚡ **Sparse Lexical Search (BM25)**: Native SQLite FTS5 full-text indexing for exact technical identifiers, part numbers, and error codes.
3. 🕸️ **Knowledge Graph & W3C SPARQL 1.1 Engine**: Native TBox/ABox ontology management, declarative tabular mapping, and full SPARQL (`SELECT`, `ASK`, `CONSTRUCT`, `DESCRIBE`).
4. 🔀 **Tri-Modal Reciprocal Rank Fusion (RRF)**: Merges dense similarity, lexical BM25, and symbolic graph traversals into unified, context-rich results.
5. 🔐 **Zero-Leakage Local Encryption**: End-to-end AES-128/HMAC (Fernet) encryption at rest inside a single, portable `.db` file.

---

## 📦 Installation

```bash
pip install safe_store
```

---

## 🌟 Core Architecture & Pillars

```
                               ┌────────────────────────────────────────┐
                               │           User Natural Query           │
                               └──────────────────┬─────────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
    ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
    │  Dense Vector Search  │         │   Sparse BM25 Search  │         │  Symbolic Graph Query │
    │  (Semantic Context)   │         │ (Exact IDs/SKUs/Names)│         │ (TBox/ABox/SPARQL/Hop)│
    └───────────┬───────────┘         └───────────┬───────────┘         └───────────┬───────────┘
                │                                 │                                 │
                │        [Candidate Set 1]        │        [Candidate Set 2]        │ [Candidate Set 3]
                └─────────────────────────────────┼─────────────────────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  Reciprocal Rank Fusion (RRF / WCS) │
                               │  Score = Σ (w_i / (k + rank_i))     │
                               └──────────────────┬──────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │  Enriched Context + Provenance Lineage│
                               └──────────────────┬──────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │       LLM Response Generation       │
                               └─────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Tri-Modal Hybrid Retrieval (Dense Vectors + BM25 Lexical + RRF)

Combining dense embeddings with sparse BM25 guarantees precision for both fuzzy conceptual questions and exact code/identifier queries.

```python
import safe_store

store = safe_store.SafeStore(
    db_path="hybrid_kb.db",
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    chunk_size=128,
    chunk_overlap=16
)

with store:
    # Index unstructured technical documents
    store.add_text(
        unique_id="incident_001",
        text="Production node crashed due to OOMKilled condition in supervisor daemon. "
             "Error code ERR-4091 was emitted by telemetry controller.",
        metadata={"service": "Telemetry", "severity": "Critical"}
    )
    store.add_text(
        unique_id="manual_001",
        text="Troubleshooting Guide: When encountering error code ERR-4091, replace the "
             "memory buffer chip and execute supervisor restart.",
        metadata={"doc_type": "Runbook"}
    )

    # Hybrid Query: Fuses Dense Semantic Similarity with BM25 Sparse Lexical Score via RRF
    results = store.hybrid_query(
        query_text="troubleshooting memory failure ERR-4091",
        top_k=2,
        dense_weight=0.5,
        bm25_weight=0.5,
        rrf_k=60
    )

    for r in results:
        print(f"[{r['file_path']}] (Fused Score: {r['fused_score']:.4f})")
        print(f"Content: {r['chunk_text']}\n")
```

---

### 2. W3C SPARQL 1.1 Knowledge Graph Engine

`safe_store` provides a full, standards-compliant SPARQL 1.1 engine supporting `SELECT`, `ASK`, `CONSTRUCT`, and `DESCRIBE` queries across multi-hop relational graphs.

```python
from safe_store import SafeStore, GraphStore

store = SafeStore(db_path="enterprise_kg.db", vectorizer_name="st")
graph = GraphStore(store=store)

# Create Graph Entities and Relationships
alice_id = graph.add_node("Person", {"name": "Alice Smith", "role": "Lead Architect"})
bob_id = graph.add_node("Person", {"name": "Bob Jones", "role": "Data Scientist"})
acme_id = graph.add_node("Company", {"name": "Acme Robotics", "industry": "AI"})
paris_id = graph.add_node("City", {"name": "Paris", "country": "France"})

graph.add_relationship(alice_id, acme_id, "worksFor", {"since": 2021})
graph.add_relationship(bob_id, acme_id, "worksFor", {"since": 2023})
graph.add_relationship(acme_id, paris_id, "locatedIn")
graph.add_relationship(alice_id, bob_id, "collaboratesWith")

# 1. SPARQL SELECT: Multi-Hop Relational Traversal
sparql_select = """
PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?personName ?cityName WHERE {
    ?person ont:worksFor ?company ;
            ont:hasName ?personName .
    ?company ont:locatedIn ?city .
    ?city ont:hasName ?cityName .
}
"""
results = graph.query_sparql(sparql_select)
for b in results["results"]["bindings"]:
    print(f"Person: {b['personName']['value']} works in City: {b['cityName']['value']}")

# 2. SPARQL ASK: Boolean Verification
sparql_ask = """
PREFIX ont: <http://example.org/ontology/>
ASK {
    ?person ont:worksFor ?company .
    ?company ont:hasName "Acme Robotics" .
}
"""
is_valid = graph.query_sparql(sparql_ask)
print(f"Acme Robotics employs personnel: {is_valid['boolean']}")

# 3. SPARQL CONSTRUCT: Subgraph Transformation
sparql_construct = """
PREFIX ont: <http://example.org/ontology/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
CONSTRUCT {
    ?person foaf:workplaceHomepage ?company .
}
WHERE {
    ?person ont:worksFor ?company .
}
"""
subgraph = graph.query_sparql(sparql_construct)
for triple in subgraph["triples"]:
    print(f"Constructed: {triple['subject']['value']} -> {triple['predicate']['value']} -> {triple['object']['value']}")
```

---

### 3. TBox (Ontology) & Declarative Tabular-to-Graph Mapping (CSV / XLSX / SQLite)

Convert structured business tables directly into grounded RDF knowledge graphs matching an explicit RDFS/OWL ontology (TBox).

```python
from safe_store import SafeStore, TBoxManager, TabularMapper

store = SafeStore(db_path="supply_chain.db")

# 1. Load TBox Ontology (Turtle format)
tbox = TBoxManager()
tbox.load_ontology("""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/ontology/> .

ex:Product a owl:Class .
ex:Supplier a owl:Class .

ex:suppliedBy a owl:ObjectProperty ;
    rdfs:domain ex:Product ;
    rdfs:range ex:Supplier .

ex:hasPrice a owl:DatatypeProperty ;
    rdfs:domain ex:Product .
""", format="turtle")

# 2. Declarative Mapping Configuration
mapping_rules = {
    "entity_mappings": [
        {
            "class": "http://example.org/ontology/Product",
            "subject_template": "http://example.org/product/{sku}",
            "properties": {
                "product_name": "http://example.org/ontology/hasName",
                "unit_price": "http://example.org/ontology/hasPrice"
            }
        },
        {
            "class": "http://example.org/ontology/Supplier",
            "subject_template": "http://example.org/supplier/{supplier_id}",
            "properties": {
                "supplier_name": "http://example.org/ontology/hasName"
            }
        }
    ],
    "relationship_mappings": [
        {
            "predicate": "http://example.org/ontology/suppliedBy",
            "source_template": "http://example.org/product/{sku}",
            "target_template": "http://example.org/supplier/{supplier_id}"
        }
    ]
}

# 3. Ingest CSV or Excel Sheet directly into ABox Graph
mapper = TabularMapper(store=store, tbox=tbox)
summary = mapper.map_csv("inventory.csv", mapping_rules=mapping_rules)
# Alternatively: mapper.map_excel("inventory.xlsx", mapping_rules=mapping_rules, sheet_name="Q3_Stock")
# Alternatively: mapper.map_sqlite_table("legacy.db", "products", mapping_rules=mapping_rules)

print(f"Mapped {summary['records_processed']} records into {summary['triples_generated']} RDF triples.")
```

---

### 4. Tri-Modal Unified Graph Retrieval (`query_graph_hybrid`)

Execute multi-channel queries combining Graph Subgraph Exploration, Dense Vectors, and Sparse BM25 Lexical search in a single call.

```python
from safe_store import SafeStore, GraphStore

store = SafeStore(db_path="enterprise_kb.db", vectorizer_name="st")
graph = GraphStore(store=store)

# Unified retrieval: discovers related subgraph entities + BM25 hits + semantic vector chunks
response = graph.query_graph_hybrid(
    query_text="What microservices depend on AuthEngine and what database tables do they use?",
    top_k=5,
    dense_weight=0.4,
    bm25_weight=0.3,
    graph_weight=0.3
)

print(f"Retrieved {len(response['ranked_chunks'])} fused context chunks.")
print(f"Identified Subgraph Nodes: {len(response['subgraph']['nodes'])}")
print(f"Identified Subgraph Edges: {len(response['subgraph']['relationships'])}")
```

---
## 🔐 Zero-Leakage Encryption at Rest

`safe_store` provides transparent, chunk-level authenticated encryption using **Fernet** (AES-128-CBC with HMAC-SHA256). User-supplied passwords are hardened via **PBKDF2-HMAC-SHA256** (600,000 iterations) before key derivation.

### What Is Protected

| Data | Encrypted? | Notes |
|------|------------|-------|
| Chunk text | ✅ Yes | Decrypted transparently during `query()` |
| Document metadata | ✅ Yes | JSON blob is encrypted at rest |
| Document full_text | ✅ Yes | Stored in `documents` table |
| Vector embeddings | ❌ No | Required for similarity search |
| Graph nodes/edges | ❌ No | Structural knowledge graph data |
| File paths / timestamps | ❌ No | Operational metadata |

### Basic Usage

```python
import safe_store

# 1. Create an encrypted store
store = safe_store.SafeStore(
    db_path="classified.db",
    encryption_key="my-super-secure-passphrase",
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"}
)

with store:
    # Document and metadata are encrypted before hitting SQLite
    store.add_document("confidential_contract.pdf", metadata={"classification": "Top Secret"})
    
    # Query decrypts chunks transparently in memory
    results = store.query("liability clauses", top_k=2)
    print(results[0]["chunk_text"])
```

### Opening Without a Key (Graceful Degradation)

If the database is opened without providing the encryption key, queries still function but return encrypted placeholders instead of plaintext. This prevents accidental crashes while signalling that the data is protected.

```python
# Re-open the same database WITHOUT the key
unauth_store = safe_store.SafeStore("classified.db", encryption_key=None)

with unauth_store:
    res = unauth_store.query("liability clauses", top_k=1)
    print(res[0]["chunk_text"])
    # >>> "[Encrypted Chunk - Key Unavailable]"
```

### Wrong Key Detection

Supplying an incorrect key is detected immediately during decryption (via Fernet's HMAC verification). The library distinguishes between "no key provided" and "wrong key provided":

```python
# Re-open with an INCORRECT key
wrong_store = safe_store.SafeStore(
    "classified.db",
    encryption_key="this-is-definitely-wrong"
)

with wrong_store:
    res = wrong_store.query("liability clauses", top_k=1)
    print(res[0]["chunk_text"])
    # >>> "[Encrypted Chunk - Decryption Failed]"
```

### Verifying Encryption Programmatically

You can inspect the database directly to confirm that encryption flags are set correctly on every chunk and document:

```python
import sqlite3

store = safe_store.SafeStore(
    "audit.db",
    encryption_key="audit-key",
    vectorizer_name="st"
)

with store:
    store.add_text("sensitive_unique_42", "Payload data here.", metadata={"owner": "Alice"})

# Verify raw DB state
conn = sqlite3.connect("audit.db")
cursor = conn.cursor()
cursor.execute("SELECT is_encrypted FROM chunks WHERE doc_id = 1")
flags = cursor.fetchall()
assert all(flag[0] == 1 for flag in flags), "Not all chunks are encrypted!"
conn.close()
```

### Metadata Encryption

When encryption is enabled, the metadata dictionary is also encrypted as a single JSON blob. This is transparent during queries:

```python
with store:
    store.add_text(
        unique_id="report_001",
        text="Q3 Financial Analysis...",
        metadata={"department": "Finance", "clearance": "Restricted"}
    )
    
    # The metadata is decrypted and prepended as context in query results
    results = store.query("Q3 analysis", top_k=1)
    print(results[0]["document_metadata"])
    # >>> {'department': 'Finance', 'clearance': 'Restricted'}
```

### Security Considerations

- **Fixed Salt**: This implementation uses a fixed salt for PBKDF2 derivation. This means the same password always yields the same key, which is a deliberate trade-off for portability (a single `.db` file can be moved between machines without external salt storage). For higher security requirements, consider wrapping the database file with OS-level full-disk encryption.
- **Vectors Remain Plaintext**: Vector embeddings are stored as raw `BLOB`s to allow cosine-similarity search without decrypting the entire dataset. If your threat model requires vectors to be secret, encrypt the underlying filesystem.
- **Memory Safety**: Decryption occurs in-memory during `query()`. Plaintext chunks exist only for the duration of the result formatting and are not cached outside of the SQLite connection scope.

### Complete Example: Encrypted Document Lifecycle

```python
import safe_store
from pathlib import Path
import shutil

DB_FILE = "encrypted_lifecycle.db"
KEY = "correct-horse-battery-staple"

# Cleanup from previous runs
for p in [DB_FILE, f"{DB_FILE}.lock", f"{DB_FILE}-wal", f"{DB_FILE}-shm"]:
    Path(p).unlink(missing_ok=True)

# Phase 1: Write encrypted data
writer = safe_store.SafeStore(
    db_path=DB_FILE,
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    encryption_key=KEY
)

doc = Path("secret_notes.txt")
doc.write_text("Project Phoenix launch is Q4. Key personnel: Alice, Bob.")

with writer:
    writer.add_document(doc, metadata={"sensitivity": "high"})
    print("Document encrypted and stored.")

# Phase 2: Read with correct key
reader = safe_store.SafeStore(DB_FILE, encryption_key=KEY)
with reader:
    results = reader.query("Project Phoenix", top_k=1)
    assert "Project Phoenix" in results[0]["chunk_text"]
    print("Decryption successful with correct key.")

# Phase 3: Read without key (placeholder)
no_key = safe_store.SafeStore(DB_FILE, encryption_key=None)
with no_key:
    res = no_key.query("Project Phoenix", top_k=1)
    assert res[0]["chunk_text"] == "[Encrypted Chunk - Key Unavailable]"
    print("Confirmed: no key returns placeholder.")

# Phase 4: Read with wrong key (tamper detection)
bad_key = safe_store.SafeStore(DB_FILE, encryption_key="wrong-key")
with bad_key:
    res = bad_key.query("Project Phoenix", top_k=1)
    assert res[0]["chunk_text"] == "[Encrypted Chunk - Decryption Failed]"
    print("Confirmed: wrong key is rejected via HMAC.")

# Cleanup
doc.unlink(missing_ok=True)
for p in [DB_FILE, f"{DB_FILE}.lock", f"{DB_FILE}-wal", f"{DB_FILE}-shm"]:
    Path(p).unlink(missing_ok=True)
print("Encrypted lifecycle demo complete.")
```
---

## 🎯 Supported Vectorization Backends

| Backend | Identifier | Typical Model / Target | Local / Remote |
| :--- | :--- | :--- | :--- |
| **Sentence-Transformers** | `"st"` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` | Local (PyTorch / HuggingFace) |
| **Ollama** | `"ollama"` | `nomic-embed-text`, `qwen3-embedding` | Local (Ollama Server) |
| **OpenAI** | `"openai"` | `text-embedding-3-small`, `text-embedding-3-large` | Remote API |
| **Cohere** | `"cohere"` | `embed-english-v3.0`, `embed-multilingual-v3.0` | Remote API |
| **Lollms** | `"lollms"` | Any OpenAI-compatible local/remote endpoint | Local / Remote |
| **TF-IDF** | `"tfidf"` / `"tf_idf"` | Data-dependent sparse baseline | Local (Scikit-Learn) |
| **Grepper** | `"grepper"` | Lightweight inverted index with markdown trees | Local (Zero-ML) |

---

## 📑 Supported Document & File Formats

`safe_store` parses structured, unstructured, and source files out-of-the-box:

- **Unstructured Documents**: `.pdf`, `.docx`, `.pptx`, `.html`, `.htm`, `.txt`, `.md`, `.rst`, `.msg`, `.rtf`
- **Data & Tables**: `.csv`, `.tsv`, `.json`, `.xlsx`, `.xls`, `.xml`, `.sql`
- **Source Code**: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.c`, `.cpp`, `.h`, `.cs`, `.java`, `.go`, `.rs`, `.php`, `.rb`, `.swift`, `.kt`, `.sh`, `.ps1`, `.lua`, `.sql`

---

## 🔍 W3C SPARQL 1.1 Query Forms Cheat Sheet

`safe_store` natively executes all four standard W3C SPARQL 1.1 query forms across your knowledge graph:

| Query Form | Purpose | Return Type | Typical Use Case |
|:---|:---|:---|:---|
| **`SELECT`** | Tabular projections across graph patterns | `{"head": {"vars": [...]}, "results": {"bindings": [...]}}` | Relational multi-hop traversals, aggregations (`COUNT`, `GROUP BY`), and filtered lookups. |
| **`ASK`** | Boolean existence test | `{"boolean": True / False}` | Fast sanity checking and compliance verification without retrieving payloads. |
| **`CONSTRUCT`** | Subgraph transformation & inference | `{"triples": [{"subject": ..., "predicate": ..., "object": ...}]}` | Transforming schemas, creating direct shortcut edges, or exporting custom RDF subgraphs. |
| **`DESCRIBE`** | Resource neighborhood extraction | `{"triples": [...]}` | Pulling all known incoming and outgoing triples associated with an entity. |

---

## 📊 Performance Benchmarks

Typical benchmarks measured on consumer hardware (Intel i7 / 16GB RAM / SSD):

| Operation | Scale / Dataset | Elapsed Time | Mode |
| :--- | :--- | :--- | :--- |
| **Dense Vector Query** | 50,000 Chunks | ~15 ms | NumPy Cosine Dot Product |
| **BM25 Lexical Search** | 100,000 Chunks | ~4 ms | SQLite FTS5 (Porter Stemmed) |
| **W3C SPARQL Relational Join** | 20,000 Triples (2-hop) | ~8 ms | RDFLib + In-Memory Quad Index |
| **Tabular Mapping** | 10,000 CSV Rows | ~1.2 s | Batch Transactional Insertion |
| **Document Ingestion (ST)** | 1 MB Text (~300 pages) | ~3.5 s | Parsing + Token Chunking + Embedding |


---
### 6. The 8 RAG Chunking Strategies (Beyond the Basics)

Retrieval quality is decided at cut time. `safe_store` implements a complete suite of **8 distinct chunking strategies**:

```
 1. Fixed-Size [====][====][====]  -> Slices at fixed intervals (fast, baseline)
 2. Overlap    [====--]            -> Rescues broken sentences across boundaries
                  [--====--]
 3. Recursive  Document            -> Splits paragraphs -> sentences -> words
               ├── Para 1
               └── Para 2 -> S1, S2
 4. Semantic   ───📉───📉───       -> Cuts at cosine similarity valleys (topic shifts)
 5. Contextual [Prefix] + [Chunk]  -> Prepends full-document situating context (Anthropic)
 6. Structure  # H1 > ## H2        -> Injects section breadcrumb paths [H1 > H2]
 7. Late       Tokens ──[Transformer]──> Contextual Embeddings ──[Mean Pool]──> Vectors
 8. Graph      Entities & Relations-> Tri-Tier Multi-Hop Graph Traversal
```

| Strategy | Flag | Ideal For | Mechanics & Key Benefit |
|:---|:---|:---|:---|
| **Token Window** | `'token'` *(Default)* | Standard RAG | Slices by tokenizer limits (`tiktoken`/HF) with offset mapping preserving all `\n` line breaks. |
| **Recursive Tree** | `'recursive'` | General Docs & Code | Hierarchically splits by `\n\n` $\rightarrow$ `# Headers` $\rightarrow$ `\n` $\rightarrow$ sentences $\rightarrow$ words. Best all-around balance. |
| **Structure-Aware** | `'structure'` / `'markdown'` | Technical Manuals & Specs | Parses Markdown `# H1` $\rightarrow$ `## H2` $\rightarrow$ `### H3` stacks, attaching lineage breadcrumbs `[H1 > H2]`. |
| **Semantic Valley** | `'semantic'` | Long Essays & Narrative | Embeds sentences and cuts where adjacent cosine similarity drops below threshold (topic boundary). |
| **Contextual Retrieval**| `'contextual'` | Complex Knowledge Bases | Injects full-document situating summaries before storage (Anthropic pattern), eliminating ambiguous pronouns. |
| **Late Chunking** | `'late'` | Dense Technical Context | Passes the entire document through the transformer *first*, then mean-pools chunk token representations (Jina AI pattern). |
| **Paragraph** | `'paragraph'` | Articles & Prose | Groups double-newline paragraph blocks up to `chunk_size` without mid-thought cuts. |
| **Fixed Character** | `'character'` | Raw Log Streams | Fast character slicing with sliding window overlap. |

#### Strategy Implementation Examples

```python
from safe_store import SafeStore

# Strategy A: Structure-Aware Markdown with Breadcrumbs
store_md = SafeStore(
    "manual.db",
    vectorizer_name="st",
    chunk_size=200,
    chunking_strategy="structure" # Injects [Section: Architecture > Storage > WAL] into chunks
)

# Strategy B: Semantic Chunking (Topic Shift Detection)
store_sem = SafeStore(
    "research.db",
    vectorizer_name="st",
    chunk_size=300,
    chunking_strategy="semantic", # Splits at cosine similarity valleys
    chunking_kwargs={"similarity_threshold": 0.65}
)

# Strategy C: Contextual Retrieval (Anthropic Pattern)
def my_context_enricher(full_doc: str, chunk: str) -> str:
    # Optional LLM or heuristic summary
    return f"From document '{full_doc[:40]}...': Topic covers database storage engine."

store_ctx = SafeStore(
    "enterprise.db",
    vectorizer_name="st",
    chunk_size=256,
    chunking_strategy="contextual",
    context_enricher=my_context_enricher
)

# Strategy D: Context Expansion Windowing
store_exp = SafeStore(
    "logs.db",
    vectorizer_name="st",
    chunk_size=128,
    expand_before=30, # Injects 30 tokens of preceding context into LLM prompt
    expand_after=30   # Injects 30 tokens of succeeding context into LLM prompt
)
```
---

## 🗺️ Roadmap

- [x] SQLite-backed dense vector database with auto-configuration persistence
- [x] Multi-backend vectorizer hub (ST, Ollama, OpenAI, Cohere, Lollms, TF-IDF, Grepper)
- [x] W3C SPARQL 1.1 Engine (`SELECT`, `ASK`, `CONSTRUCT`, `DESCRIBE`)
- [x] TBox & ABox Ontology Management (OWL / RDFS introspection)
- [x] Declarative Tabular Mapping for CSV, XLSX, and SQLite tables
- [x] Tri-Modal Hybrid Retrieval Engine (BM25 FTS5 + Dense Vectors + RRF)
- [x] AES-128/HMAC Authenticated Encryption at Rest
- [ ] Multi-Modal Image Vector Database using SigLIP / CLIP embeddings
- [ ] Web-based Visual Knowledge Graph Studio & Inspector

---

## 🤝 Contributing & License

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/ParisNeo/safe_store).

Licensed under the [Apache 2.0 License](LICENSE).