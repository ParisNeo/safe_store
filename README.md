# safe_store: Transform Your Digital Chaos into a Queryable Knowledge Base

[![PyPI version](https://img.shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img.shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)

**`safe_store` is a Python library that turns your local folders of documents into a powerful, private, and intelligent knowledge base.** It achieves this by combining two powerful AI concepts into a single, seamless tool:

1.  **Deep Semantic Search:** It reads and *understands* the content of your files, allowing you to search by meaning and context, not just keywords.
2.  **AI-Powered Knowledge Graph:** It uses a Large Language Model (LLM) to automatically identify key entities (people, companies, concepts) and the relationships between them, building an interconnected web of your knowledge.

All of this happens entirely on your local machine, using a single, portable SQLite file. Your data never leaves your control.

---

## The Journey from Search to Understanding

`safe_store` is designed to grow with your needs. You can start with a simple, powerful RAG system in minutes, and then evolve it into a sophisticated knowledge engine.

### Level 1: Build a Powerful RAG System with Semantic Search
**The Foundation: Retrieval-Augmented Generation (RAG)**

RAG is the state-of-the-art technique for making Large Language Models (LLMs) answer questions about your private documents. The process is simple:
1.  **Retrieve:** Find the most relevant text chunks from your documents related to a user's query.
2.  **Augment:** Add those chunks as context to your prompt.
3.  **Generate:** Ask the LLM to generate an answer based *only* on the provided context.

`SafeStore` is the perfect tool for the "Retrieve" step. It uses vector embeddings to understand the *meaning* of your text, allowing you to find relevant passages even if they don't contain the exact keywords.

**Example: A Simple RAG Pipeline**
```python
import safe_store
import numpy as np

# Create stores with different vectorizers
semantic_store = safe_store.SafeStore(
    db_path="semantic.db",
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"}
)

keyword_store = safe_store.SafeStore(
    db_path="keyword.db", 
    vectorizer_name="tf_idf",
    vectorizer_config={"name": "my_tfidf"}
)

# Index same documents in both
with semantic_store, keyword_store:
    for doc_path in document_paths:
        semantic_store.add_document(doc_path)
        keyword_store.add_document(doc_path)

# Hybrid query: combine semantic and keyword search
def hybrid_query(query_text, alpha=0.7):
    """Combine semantic and keyword results with weighting."""
    semantic_results = semantic_store.query(query_text, top_k=10)
    keyword_results = keyword_store.query(query_text, top_k=10)
    
    # Normalize scores and combine
    combined = {}
    for r in semantic_results:
        combined[r['chunk_id']] = {'data': r, 'score': alpha * r['similarity']}
    for r in keyword_results:
        if r['chunk_id'] in combined:
            combined[r['chunk_id']]['score'] += (1-alpha) * r['similarity']
        else:
            combined[r['chunk_id']] = {'data': r, 'score': (1-alpha) * r['similarity']}
    
    # Return top results by combined score
    sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
    return [r['data'] for r in sorted_results[:5]]
```

### 2. Incremental Updates with Change Detection

Perfect for monitoring document folders and only re-indexing changed files:

```python
from pathlib import Path
import hashlib

class DocumentWatcher:
    def __init__(self, store, watch_dir):
        self.store = store
        self.watch_dir = Path(watch_dir)
        self.known_hashes = {}
        
    def scan_and_update(self):
        """Efficiently update only changed documents."""
        for file_path in self.watch_dir.rglob("*.*"):
            if file_path.suffix.lower() not in safe_store.SAFE_STORE_SUPPORTED_FILE_EXTENSIONS:
                continue
                
            current_hash = self._hash_file(file_path)
            
            # Check if file exists in store
            existing = self._get_doc_by_path(str(file_path))
            
            if existing:
                stored_hash = existing.get('file_hash')
                if stored_hash != current_hash:
                    print(f"Updating changed file: {file_path.name}")
                    self.store.add_document(file_path, force_reindex=True)
                else:
                    print(f"Skipping unchanged: {file_path.name}")
            else:
                print(f"Adding new file: {file_path.name}")
                self.store.add_document(file_path)
                
            self.known_hashes[str(file_path)] = current_hash
    
    def _hash_file(self, path):
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_doc_by_path(self, file_path):
        docs = self.store.list_documents()
        return next((d for d in docs if d['file_path'] == file_path), None)
```

### 3. Custom Chunking for Code Analysis

Specialized chunking strategies for code files preserve semantic structure:

```python
store = safe_store.SafeStore(
    db_path="codebase.db",
    vectorizer_name="st",
    chunking_strategy="recursive",  # Respects code structure
    chunk_size=512,
    chunk_overlap=50
)

# Process a Python project
for py_file in Path("src").rglob("*.py"):
    # The recursive strategy intelligently splits by:
    # - Module/class boundaries
    # - Function definitions  
    # - Logical blocks
    store.add_document(py_file)
    
# Query for specific functions or patterns
results = store.query(
    "function that handles authentication and validates JWT tokens",
    top_k=5
)
```

### 4. Privacy-First RAG with Local Models

Complete offline pipeline using local models:

```python
import safe_store

# Ollama for embeddings (completely local)
store = safe_store.SafeStore(
    db_path="private_kb.db",
    vectorizer_name="ollama",
    vectorizer_config={
        "model": "nomic-embed-text",
        "host": "http://localhost:11434"
    },
    encryption_key="user-provided-password"  # Optional: encrypt at rest
)

# Build knowledge base
with store:
    for doc in sensitive_documents:
        store.add_document(doc, metadata={"classification": "confidential"})
    
    # Query locally - no data leaves your machine
    results = store.query("quarterly financial projections", top_k=3)
    
    # Pass to local LLM (e.g., via Ollama)
    context = "\n\n".join(r['chunk_text'] for r in results)
    # ... send to local LLM for answer generation
```

### 5. Knowledge Graph Extraction from Technical Documentation

Build structured knowledge from unstructured docs:

```python
from safe_store import GraphStore

# Define domain-specific ontology
ontology = {
    "nodes": {
        "APIEndpoint": {
            "description": "A REST API endpoint",
            "properties": {"path": "string", "method": "string", "auth_required": "boolean"}
        },
        "DatabaseTable": {
            "description": "A database table", 
            "properties": {"name": "string", "primary_key": "string"}
        },
        "Microservice": {
            "description": "A microservice component",
            "properties": {"name": "string", "owner": "string", "repo_url": "string"}
        }
    },
    "relationships": {
        "DEPENDS_ON": {
            "description": "Service dependency",
            "source": "Microservice",
            "target": "Microservice"
        },
        "USES_TABLE": {
            "description": "API uses database table",
            "source": "APIEndpoint",
            "target": "DatabaseTable"
        }
    }
}

# Initialize graph store
graph = GraphStore(
    store=store,
    llm_executor_callback=my_local_llm,
    ontology=ontology
)

# Process architecture documentation
graph.build_graph_for_all_documents()

# Query the system architecture
result = graph.query_graph(
    "Which microservices depend on the User service and what database tables do they use?",
    output_mode="full"
)
```

### 6. Document Comparison and Deduplication

Identify similar documents across large collections:

```python
import numpy as np
from sklearn.cluster import DBSCAN

def find_duplicate_documents(store, similarity_threshold=0.95):
    """Identify potential duplicate or near-duplicate documents."""
    with store:
        # Export point cloud for all chunks
        points = store.export_point_cloud(output_format='dict')
        
        # Group by document
        doc_vectors = {}
        for p in points:
            doc = p['document_title']
            if doc not in doc_vectors:
                doc_vectors[doc] = []
            doc_vectors[doc].append([p['x'], p['y']])
        
        # Average chunk positions per document
        doc_centroids = {}
        for doc, vectors in doc_vectors.items():
            doc_centroids[doc] = np.mean(vectors, axis=0)
        
        # Cluster to find similar documents
        docs = list(doc_centroids.keys())
        centroids = np.array([doc_centroids[d] for d in docs])
        
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(centroids)
        
        # Return groups of similar documents
        groups = {}
        for doc, label in zip(docs, clustering.labels_):
            if label not in groups:
                groups[label] = []
            groups[label].append(doc)
        
        return [g for g in groups.values() if len(g) > 1]
```

---
## 🎓 Best Practices

### Performance Optimization

1. **Choose the Right Chunk Size**
   - Small chunks (256-512 tokens): Better for precise retrieval, higher storage cost
   - Large chunks (1024-2048 tokens): Better context, may include irrelevant text
   - For code: Use `recursive` strategy with smaller sizes
   - For prose: Use `token` strategy with moderate overlap

2. **Vectorizer Selection Guide**

| Use Case | Recommended Vectorizer | Why |
|----------|------------------------|-----|
| General English text | `st:all-MiniLM-L6-v2` | Fast, good quality, small |
| Multilingual content | `st:LaBSE` | Supports 100+ languages |
| Code search | `st:code-*` models | Understands syntax patterns |
| Legal/medical | Fine-tuned domain models | Domain-specific terminology |
| Very large scale | `tfidf` | No GPU required, fast indexing |
| API-based RAG | `openai:text-embedding-3-small` | Best quality, cost consideration |

3. **Database Maintenance**
   ```python
   # Regular cleanup of WAL files
   store = safe_store.SafeStore("my_store.db")
   with store:
       # Vacuum to reclaim space after deletions
       store.conn.execute("VACUUM")
       
       # Analyze for query optimization
       store.conn.execute("ANALYZE")
   ```

4. **Memory Management for Large Collections**
   ```python
   # Process documents in batches to control memory
   batch_size = 100
   for i in range(0, len(all_documents), batch_size):
       batch = all_documents[i:i+batch_size]
       with store:
           for doc in batch:
               store.add_document(doc)
       # Force garbage collection between batches
       import gc; gc.collect()
   ```

### Security Considerations

1. **Encryption at Rest**
   ```python
   # Use strong passwords for encryption
   import secrets
   encryption_key = secrets.token_hex(32)  # 256-bit key
   
   store = safe_store.SafeStore(
       db_path="encrypted.db",
       encryption_key=encryption_key
   )
   
   # Store key securely (e.g., environment variable, key management service)
   ```

2. **Access Control Pattern**
   ```python
   class SecureStore:
       def __init__(self, db_path, key_provider):
           self.key_provider = key_provider
           
       def get_store(self, user_id):
           key = self.key_provider.get_key(user_id)
           return safe_store.SafeStore(self.db_path, encryption_key=key)
   ```

---
## 🔧 Troubleshooting

### Common Issues

**"Database is locked" errors**
- Ensure only one SafeStore instance per database at a time
- Check for stale lock files (`.db.lock`) and remove if process crashed
- Increase `lock_timeout` if operations are slow: `SafeStore(..., lock_timeout=120)`

**"Vectorizer dimension mismatch"**
- You cannot change vectorizers on an existing database
- Create a new database or use `add_vectorization()` to add new vectorization methods

**Slow query performance**
- Enable WAL mode (automatic): Improves concurrent read/write
- Add indexes: `CREATE INDEX IF NOT EXISTS idx_vectors_method ON vectors(method_id)`
- Reduce `top_k` if fetching too many results

**Out of memory during indexing**
- Reduce batch size in `build_graph_for_all_documents(batch_size_chunks=10)`
- Use smaller chunks to reduce per-document memory footprint
- Process documents sequentially rather than in parallel

**Graph extraction quality issues**
- Refine your ontology: Be specific about expected properties
- Increase `llm_retries` for better JSON parsing success
- Use `guidance` parameter to steer LLM extraction
- Check chunk size: Too small may break context, too large may dilute focus

### Debugging Queries

```python
# Enable debug logging
import logging
logging.getLogger('safe_store').setLevel(logging.DEBUG)

# Inspect what's actually stored
with store:
    docs = store.list_documents()
    for doc in docs:
        print(f"Document: {doc['file_path']}")
        print(f"Metadata: {doc['metadata']}")
        print(f"Chunks: {store.conn.execute('SELECT COUNT(*) FROM chunks WHERE doc_id=?', (doc['doc_id'],)).fetchone()[0]}")
```

---
## 📊 Performance Benchmarks

Typical performance on a modern laptop (i7, 16GB RAM, SSD):

| Operation | Time | Notes |
|-----------|------|-------|
| Index 1MB text (ST) | ~2-5s | Includes chunking, vectorization, storage |
| Index 1MB text (TF-IDF) | ~0.5-1s | Faster, no neural model loading |
| Query (10k chunks) | ~50-100ms | Cosine similarity on pre-loaded vectors |
| Graph build (100 chunks) | ~30-60s | Depends on LLM latency |
| Export point cloud (50k vectors) | ~5-10s | PCA computation time |

*Benchmarks are approximate and vary based on hardware, document complexity, and network conditions for API-based vectorizers.*

---
## 🔗 Integration Examples

### LangChain Integration

```python
from langchain.schema import Document
from langchain.vectorstores import VectorStore

class SafeStoreLangChain(VectorStore):
    def __init__(self, safe_store_instance):
        self.store = safe_store_instance
        
    def add_texts(self, texts, metadatas=None):
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            self.store.add_text(f"langchain_doc_{i}", text, metadata=meta)
        return [f"langchain_doc_{i}" for i in range(len(texts))]
    
    def similarity_search(self, query, k=4):
        results = self.store.query(query, top_k=k)
        return [
            Document(page_content=r['chunk_text'], metadata=r.get('document_metadata', {}))
            for r in results
        ]
```

### LlamaIndex Integration

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Create a custom loader that uses safe_store
documents = SimpleDirectoryReader("docs").load_data()

# Build index using safe_store as the backend
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=SafeStoreLangChain(store)  # Using wrapper above
)

# Query
response = index.query("What are the main points?")
```

### FastAPI Web Service

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()
store = safe_store.SafeStore("api_store.db")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save uploaded file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Index in safe_store
    with store:
        result = store.add_document(temp_path)
    
    return {"status": "indexed", "chunks_added": result['num_chunks_added']}

@app.post("/query")
async def query_documents(request: QueryRequest):
    with store:
        results = store.query(request.question, top_k=request.top_k)
    return {"results": results}
```
### Pre-processing Chunks on the Fly with `chunk_processor`
For advanced RAG, you might need to transform the text of a chunk *before* it's vectorized and stored. The `chunk_processor` is a powerful hook that lets you do exactly that.

It's an optional callable that you can pass to `add_document` or `add_text`. The function receives the raw text of each chunk and the document's metadata, and it must return the string that you want to be stored and vectorized instead.

**Example: Prepending Metadata to Each Chunk**

```python
import safe_store

store = safe_store.SafeStore(db_path="processed_store.db")

def prepend_topic_processor(chunk_text: str, metadata: dict) -> str:
    """A processor that adds the 'topic' from metadata to the chunk text."""
    topic = metadata.get("topic", "general")
    return f"[Topic: {topic}] {chunk_text}"

with store:
    store.add_text(
        unique_id="processed_doc_1",
        text="This chunk is about quantum mechanics.",
        metadata={"topic": "Physics"},
        chunk_processor=prepend_topic_processor,
        force_reindex=True
    )

# When you query this, the stored text will be:
# "[Topic: Physics] This chunk is about quantum mechanics."
# This can make the vector more specific to the topic.
results = store.query("information related to physics", top_k=1)
if results:
    print(results['chunk_text'])

store.close()
```

**Example: Multi-Stage Processing Pipeline**

```python
import re

def create_advanced_processor():
    """Creates a processor that normalizes citations and adds context."""
    
    def processor(chunk_text: str, metadata: dict) -> str:
        # Step 1: Normalize citation formats (e.g., [1], [2] -> [REF-1], [REF-2])
        normalized = re.sub(r'\[(\d+)\]', r'[REF-\1]', chunk_text)
        
        # Step 2: Add document section context if available
        section = metadata.get("section", "unknown")
        if section != "unknown":
            normalized = f"[Section: {section}] {normalized}"
        
        # Step 3: Mark code blocks for special handling
        if metadata.get("contains_code", False):
            normalized = "[CODE] " + normalized
        
        return normalized
    
    return processor

store = safe_store.SafeStore(db_path="advanced_store.db")

with store:
    # Process a technical document with multiple transformations
    store.add_document(
        "technical_spec.md",
        metadata={"section": "API Reference", "contains_code": True},
        chunk_processor=create_advanced_processor()
    )
```

**Example: Conditional Processing Based on Content Type**

```python
def smart_chunk_processor(chunk_text: str, metadata: dict) -> str:
    """
    Applies different processing based on detected content type.
    """
    content_type = metadata.get("content_type", "text")
    
    if content_type == "legal":
        # Expand legal abbreviations
        replacements = {
            "Sec.": "Section",
            "Art.": "Article",
            "para.": "paragraph"
        }
        for short, full in replacements.items():
            chunk_text = chunk_text.replace(short, full)
        return "[LEGAL] " + chunk_text
    
    elif content_type == "medical":
        # Ensure medical terms are properly spaced
        chunk_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', chunk_text)
        return "[MEDICAL] " + chunk_text
    
    return chunk_text

store = safe_store.SafeStore(db_path="multi_domain_store.db")

with store:
    # Legal document
    store.add_document(
        "contract.pdf",
        metadata={"content_type": "legal"}
    )
    
    # Medical document
    store.add_document(
        "patient_notes.txt",
        metadata={"content_type": "medical"}
    )
```

This simple hook provides immense flexibility for customizing your data ingestion pipeline.

### Reconstructing Original Content
After indexing, you may need to retrieve the full, original text of a document as it was processed by `safe_store`. The `reconstruct_document_text` method does this by fetching and reassembling all of a document's stored chunks.

```python
# Assuming 'store' is an initialized SafeStore instance
# with "path/to/research_paper.txt" already added.
full_text = store.reconstruct_document_text("path/to/research_paper.txt")

if full_text:
    print("--- Reconstructed Text ---")
    print(full_text[:500] + "...")

# Note: If a chunk_overlap was used during indexing, the reconstructed text
# will contain these repeated, overlapping segments. This method provides a
# raw reassembly of the stored data.
```

**Use Case: Document Verification and Comparison**

```python
def verify_document_integrity(store, file_path: str) -> dict:
    """
    Compares the reconstructed document with the original to verify
    that all content was properly stored.
    """
    from pathlib import Path
    
    # Get original content
    original = Path(file_path).read_text(encoding='utf-8')
    
    # Get reconstructed content
    reconstructed = store.reconstruct_document_text(file_path)
    
    # Compare (ignoring whitespace differences from chunking)
    original_normalized = ' '.join(original.split())
    reconstructed_normalized = ' '.join(reconstructed.split())
    
    return {
        "file": file_path,
        "original_length": len(original),
        "reconstructed_length": len(reconstructed),
        "match": original_normalized == reconstructed_normalized,
        "difference_ratio": len(set(original_normalized.split()) ^ set(reconstructed_normalized.split())) / len(set(original_normalized.split()))
    }

with store:
    result = verify_document_integrity(store, "important_contract.pdf")
    if not result["match"]:
        print(f"Warning: Document reconstruction mismatch!")
        print(f"Difference ratio: {result['difference_ratio']:.2%}")
```
---
## 🗺️ Roadmap

- [x] Core vector storage with SQLite
- [x] Multiple vectorizer support (ST, TF-IDF, OpenAI, Ollama, Cohere)
- [x] Knowledge graph extraction and querying
- [x] SPARQL support for graph queries
- [x] Encryption at rest
- [ ] Vector quantization for memory efficiency
- [ ] Distributed/multi-instance synchronization
- [ ] Web-based management UI
- [ ] Hybrid search (dense + sparse vectors)
- [ ] Automatic ontology suggestion
- [ ] Incremental graph updates

---
## 🤝 Contributing & License

Contributions are highly welcome! Please open an issue to discuss a new feature or submit a pull request on [GitHub](https://github.com/ParisNeo/safe_store).

### Areas We Need Help With

- Additional vectorizer implementations
- Performance optimizations for large-scale deployments  
- More comprehensive test coverage
- Documentation translations
- Example integrations with popular frameworks

Licensed under Apache 2.0. See [LICENSE](LICENSE).

---
## 📚 Additional Resources

- [Full API Documentation](https://parisneo.github.io/safe_store/)
- [Example Projects](https://github.com/ParisNeo/safe_store/tree/main/examples)
- [Discord Community](https://discord.gg/safe_store)
- [Blog Posts & Tutorials](https://parisneo.github.io/blog/tag/safe_store)
