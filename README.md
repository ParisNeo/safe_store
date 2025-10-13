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

`GraphStore` complements this by building a structured knowledge graph of the key **instances** (like the person "Geoffrey Hinton") and their **relationships** (like `PIONEERED` the concept "Backpropagation"). This allows you to ask precise, factual questions.

---

## Dynamic Vectorizer Discovery & Configuration

One of `safe_store`'s most powerful features is its ability to self-document. You don't need to guess which vectorizers are available or what parameters they need. You can discover everything at runtime.

This makes it easy to experiment with different embedding models and build interactive tools that guide users through the setup process.

### Step 1: Discovering Available Vectorizers

The `SafeStore.list_available_vectorizers()` class method scans the library for all built-in and custom vectorizers and returns their complete configuration metadata.

```python
import safe_store
import pprint

# Get a list of all available vectorizer configurations
available_vectorizers = safe_store.SafeStore.list_available_vectorizers()

# Pretty-print the result to see what's available
pprint.pprint(available_vectorizers)
```
This will produce a detailed output like this:
```
[{'author': 'ParisNeo',
  'class_name': 'CohereVectorizer',
  'creation_date': '2025-10-10',
  'description': "A vectorizer that uses Cohere's API...",
  'input_parameters': [{'default': 'embed-english-v3.0',
                        'description': 'The name of the Cohere embedding model...',
                        'mandatory': True,
                        'name': 'model'},
                       {'default': '',
                        'description': 'Your Cohere API key...',
                        'mandatory': False,
                        'name': 'api_key'},
                        ...],
  'last_update_date': '2025-10-10',
  'name': 'cohere',
  'title': 'Cohere Vectorizer'},
 {'author': 'ParisNeo',
  'class_name': 'OllamaVectorizer',
  'name': 'ollama',
  'title': 'Ollama Vectorizer',
  ...},
  ...
]
```

### Step 2: Listing Available Models for a Vectorizer

Once you know which vectorizer you want to use, you can ask `safe_store` what specific models it supports. This is especially useful for API-based or local server-based vectorizers like `ollama`, which can have many different models available.

```python
import safe_store

# Example: List all embedding models available from a running Ollama server
try:
    # This requires a running Ollama instance to succeed
    ollama_models = safe_store.SafeStore.list_models("ollama")
    print("Available Ollama embedding models:")
    for model in ollama_models:
        print(f"- {model}")
except Exception as e:
    print(f"Could not list Ollama models. Is the server running? Error: {e}")

```

### Step 3: Building an Interactive Configurator

You can use this metadata to create an interactive setup script, guiding the user to choose and configure their desired vectorizer on the fly.

**Full Interactive Example:**
Copy and run this script. It will guide you through selecting and configuring a vectorizer, then initialize `SafeStore` with your choices.

```python
# interactive_setup.py
import safe_store
import pprint

def interactive_vectorizer_setup():
    """
    An interactive CLI to guide the user through selecting and configuring a vectorizer.
    """
    print("--- Welcome to the safe_store Interactive Vectorizer Setup ---")
    
    # 1. List all available vectorizers
    vectorizers = safe_store.SafeStore.list_available_vectorizers()
    
    print("\nAvailable Vectorizers:")
    for i, vec in enumerate(vectorizers):
        print(f"  [{i+1}] {vec['name']} - {vec.get('title', 'No Title')}")

    # 2. Get user's choice
    choice = -1
    while choice < 0 or choice >= len(vectorizers):
        try:
            raw_choice = input(f"\nPlease select a vectorizer (1-{len(vectorizers)}): ")
            choice = int(raw_choice) - 1
            if not (0 <= choice < len(vectorizers)):
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

    selected_vectorizer = vectorizers[choice]
    selected_name = selected_vectorizer['name']
    
    print(f"\nYou have selected: {selected_name}")
    print(f"Description: {selected_vectorizer.get('description', 'N/A').strip()}")

    # 3. Dynamically build the configuration dictionary
    vectorizer_config = {}
    print("\nPlease provide the following configuration values (press Enter to use default):")
    
    params = selected_vectorizer.get('input_parameters', [])
    if not params:
        print("This vectorizer requires no special configuration.")
    else:
        for param in params:
            param_name = param['name']
            description = param.get('description', 'No description.')
            default_value = param.get('default', None)
            
            prompt = f"- {param_name} ({description})"
            if default_value is not None:
                prompt += f" [default: {default_value}]: "
            else:
                prompt += ": "
                
            user_input = input(prompt)
            
            # Use user input if provided, otherwise use default
            final_value = user_input if user_input else default_value
            
            # Simple type conversion for demonstration (can be expanded)
            if final_value is not None:
                if param.get('type') == 'int':
                    vectorizer_config[param_name] = int(final_value)
                elif param.get('type') == 'dict':
                    # For simplicity, we don't parse dicts here, but a real app might use json.loads
                    vectorizer_config[param_name] = final_value
                else:
                    vectorizer_config[param_name] = str(final_value)

    # 4. Initialize SafeStore with the dynamically created configuration
    print("\n--- Configuration Complete ---")
    print(f"Vectorizer Name: '{selected_name}'")
    print("Vectorizer Config:")
    pprint.pprint(vectorizer_config)
    
    try:
        print("\nInitializing SafeStore with your configuration...")
        store = safe_store.SafeStore(
            db_path=f"{selected_name}_store.db",
            vectorizer_name=selected_name,
            vectorizer_config=vectorizer_config
        )
        print("\n✅ SafeStore initialized successfully!")
        print(f"Database file is at: {selected_name}_store.db")
        store.close()
    except Exception as e:
        print(f"\n❌ Failed to initialize SafeStore: {e}")


if __name__ == "__main__":
    interactive_vectorizer_setup()
```
This script demonstrates how the self-documenting nature of `safe_store` enables you to build powerful, user-friendly applications on top of it.

---

## 🏁 Quick Start Guide

This example shows the end-to-end workflow: indexing a document, then building and querying a knowledge graph of its **instances** using a simple string-based ontology.

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
Install optional dependencies for the features you need:```bash
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