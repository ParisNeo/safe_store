# safe_store: Transform Your Digital Chaos into a Queryable Knowledge Base

[![PyPI version](https://img_shields.io/pypi/v/safe_store.svg)](https://pypi.org/project/safe_store/)
[![PyPI license](https://img_shields.io/pypi/l/safe_store.svg)](https://github.com/ParisNeo/safe_store/blob/main/LICENSE)
[![PyPI pyversions](https://img_shields.io/pypi/pyversions/safe_store.svg)](https://pypi.org/project/safe_store/)

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

# 1. Create a store. This will create a 'my_notes.db' file.
store = safe_store.SafeStore(db_path="my_notes.db", vectorizer_name="st")

# 2. Add your documents. It will scan the folder and process all supported files.
with store:
    store.add_document("path/to/my_notes_and_articles/")

# 3. Query the store to find context for your RAG prompt.
user_query = "What were the main arguments about AI consciousness in my research?"
context_chunks = store.query(user_query, top_k=3)

# 4. Build the prompt and send to your LLM.
context_text = "\n\n".join([chunk['chunk_text'] for chunk in context_chunks])
prompt = f"""
Based on the following context, please answer the user's question.
Do not use any external knowledge.

Context:
---
{context_text}
---

Question: {user_query}
"""

# result = my_llm_function(prompt) # Send to your LLM of choice
```
With just this, you have a powerful, private RAG system running on your local files.

### Level 2: Uncover Hidden Connections with a Knowledge Graph
**The Next Dimension: From Passages to a Web of Knowledge**

Semantic search is great for finding *relevant passages*, but it struggles with questions about *specific facts* and *relationships* scattered across multiple documents.
*   It can find a document mentioning "Dr. Hinton" and another mentioning "backpropagation."
*   It cannot easily answer: "Did Dr. Hinton *invent* backpropagation?"

This is where the `GraphStore` comes in. It reads the text chunks you've already indexed and uses an LLM to build a knowledge graph of the key **instances** (like the person "Geoffrey Hinton") and their **relationships** (like `PIONEERED` the concept "Backpropagation").

**Example: Upgrading Your RAG System with a Graph**

```python
from safe_store import GraphStore

# Assume 'store' object from Level 1 already exists and is populated.
# 1. Define an ontology to guide the LLM. It can be a simple string!
ontology = """
- Extract People, Concepts, and Companies.
- A Person can be a PIONEER_OF a Concept.
- A Person can WORK_FOR a Company.
"""

# 2. Create the GraphStore. It uses the same database and vectorizer.
graph_store = GraphStore(store=store, llm_executor_callback=my_llm, ontology=ontology)

# 3. Build the graph. This is a one-time process (per document).
with graph_store:
    graph_store.build_graph_for_all_documents()

# 4. Now, ask a precise, structured question.
graph_results = graph_store.query_graph(
    "Who pioneered backpropagation and where did they work?",
    output_mode="graph_only"
)

# The result is no longer just text, but a structured subgraph:
# {
#   "nodes": [
#     {"label": "Person", "properties": {"identifying_value": "Geoffrey Hinton", ...}},
#     {"label": "Concept", "properties": {"identifying_value": "Backpropagation", ...}},
#     {"label": "Company", "properties": {"identifying_value": "Google", ...}}
#   ],
#   "relationships": [
#     {"source": "Geoffrey Hinton", "type": "PIONEER_OF", "target": "Backpropagation"},
#     {"source": "Geoffrey Hinton", "type": "WORK_FOR", "target": "Google"}
#   ]
# }
```

### The Magic: Combining Semantic and Graph Search
The true power of `safe_store` lies in using both layers together. You can use a broad semantic search to find a starting point, then use a precise graph query to explore the entities within that context, creating a deeply informed RAG prompt.

---

## 🚀 Imaginative Use Cases in Action

#### 1. The Personal Knowledge Master
*   **Vector Search Alone:** You ask, "What are the core principles of Stoicism?" `safe_store` retrieves paragraphs from your notes on Seneca and Marcus Aurelius. **Good.**
*   **+ Knowledge Graph:** You then ask, "Show me the relationship between Seneca and Nero." The graph instantly returns a `TUTOR_OF` relationship—an explicit fact that vector search alone would never find. **Powerful.**

#### 2. The Codebase Archaeologist
*   **Vector Search Alone:** You search, "how to handle user authentication tokens." `safe_store` pulls up relevant code snippets that use JWT libraries and token validation logic. **Helpful.**
*   **+ Knowledge Graph:** You then ask, "Which API endpoints use the `validate_token` function?" The graph directly shows `'/api/v1/profile' --[USES]--> 'validate_token'` and `'/api/v1/settings' --[USES]--> 'validate_token'`. This is an instant dependency map that saves hours of manual code tracing. **Game-changing.**

#### 3. The AI-Powered Research Assistant
*   **Vector Search Alone:** You query, "papers discussing protein folding using deep learning." `safe_store` finds several PDFs, including AlphaFold's seminal paper. **Effective.**
*   **+ Knowledge Graph:** You follow up with, "Show me all the authors from DeepMind who co-authored papers with researchers from the University of Washington." The graph traverses affiliations and co-authorship links to reveal hidden collaborations across institutions. **Insightful.**

---

## 🏁 Quick Start Guide

This single script demonstrates the complete, two-level workflow.

```python
import safe_store
from safe_store import GraphStore, LogLevel
from lollms_client import LollmsClient
from pathlib import Path
import shutil

# --- 0. Configuration & Cleanup ---
DB_FILE = "quickstart.db"
DOC_DIR = Path("temp_docs_qs")
if DOC_DIR.exists(): shutil.rmtree(DOC_DIR)
DOC_DIR.mkdir()
Path(DB_FILE).unlink(missing_ok=True)

# --- 1. LLM Executor & Sample Document ---
def llm_executor(prompt: str) -> str:
    try:
        client = LollmsClient()
        return client.generate_code(prompt, language="json", temperature=0.1) or ""
    except Exception as e:
        raise ConnectionError(f"LLM call failed: {e}")

doc_path = DOC_DIR / "doc.txt"
doc_path.write_text("Dr. Aris Thorne is the CEO of QuantumLeap AI, a firm in Geneva.")

# --- 2. Level 1: Semantic Search with SafeStore ---
print("--- LEVEL 1: SEMANTIC SEARCH ---")
store = safe_store.SafeStore(db_path=DB_FILE, vectorizer_name="st", log_level=LogLevel.INFO)
with store:
    store.add_document(doc_path)
    results = store.query("who leads the AI firm in Geneva?", top_k=1)
    print(f"Semantic search result: '{results['chunk_text']}'")

# --- 3. Level 2: Knowledge Graph with GraphStore ---
print("\n--- LEVEL 2: KNOWLEDGE GRAPH ---")
ontology = "Extract People and Companies. A Person can be a CEO_OF a Company."
try:
    graph_store = GraphStore(store=store, llm_executor_callback=llm_executor, ontology=ontology)
    with graph_store:
        graph_store.build_graph_for_all_documents()
        graph_result = graph_store.query_graph("Who is the CEO of QuantumLeap AI?", output_mode="graph_only")
        
        print("Graph query result:")
        for rel in graph_result.get('relationships', []):
            source = rel['source_node']['properties'].get('identifying_value')
            target = rel['target_node']['properties'].get('identifying_value')
            print(f"- Relationship: '{source}' --[{rel['type']}]--> '{target}'")
except ConnectionError as e:
    print(f"[SKIP] GraphStore part failed: {e}")

store.close()
```

---

## ⚙️ Installation

```bash
pip install safe-store
```
Install optional dependencies for the features you need:
```bash
# For Sentence Transformers (recommended for local use)
pip install safe-store[sentence-transformers]

# For API-based vectorizers
pip install safe_store[openai,ollama,cohere]

# For parsing PDF, DOCX, etc.
pip install safe-store[parsing]

# For encryption
pip install safe-store[encryption]

# To install everything:
pip install safe-store[all] 
```
---

## 💡 API Highlights

#### `SafeStore` (The Foundation)
*   `__init__(db_path, vectorizer_name, ...)`: Creates or loads a database. The vectorizer is locked in at creation.
*   `add_document(path, ...)`: Parses, chunks, vectorizes, and stores a document or an entire folder.
*   `query(query_text, top_k, ...)`: Performs a semantic search and returns the most relevant text chunks for your RAG pipeline.

#### `GraphStore` (The Intelligence Layer)
*   `__init__(store, llm_executor_callback, ontology)`: Creates the graph manager on an existing `SafeStore` instance.
*   `build_graph_for_all_documents()`: Scans documents and uses an LLM to build the knowledge graph based on your ontology.
*   `query_graph(natural_language_query, ...)`: Translates a question into a graph traversal, returning nodes, relationships, and/or the original source text.
*   `add_node(...)`, `add_relationship(...)`: Manually edit the graph to add your own expert knowledge.

---

## 🤝 Contributing & License

Contributions are highly welcome! Please open an issue to discuss a new feature or submit a pull request on [GitHub](https://github.com/ParisNeo/safe_store).

Licensed under Apache 2.0. See [LICENSE](LICENSE).