# examples/basic_usage.py
import safe_store
from pathlib import Path
import time
import shutil

# --- Configuration ---
# Activate or deactivate examples for each vectorizer type.
# Each example will create its own separate database file.
USE_ST = True       # Sentence-Transformers (local model)
USE_TFIDF = True    # TF-IDF (local, data-dependent)
USE_OLLAMA = True   # Ollama (requires running Ollama server)
USE_OPENAI = False  # OpenAI (requires API key)
USE_COHERE = False  # Cohere (requires API key)
USE_PARSING = True  # Set to False if parsing libs not installed

# --- Vectorizer Configurations ---
# The new way: define vectorizer type and its config separately.
st_config = {"model": "all-MiniLM-L6-v2"}
tfidf_config = {"name": "my_tfidf"} # 'name' is just an identifier for this fitted model
ollama_config = {"model": "qwen3-embedding:0.6b"} # Ensure you have pulled this model in Ollama
openai_config = {"model": "text-embedding-3-small"} # Key from OPENAI_API_KEY env var
cohere_config = {"model": "embed-english-v3.0"} # Key from COHERE_API_KEY env var

# --- Helper Functions ---
def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def cleanup_db_files(db_file):
    """Cleans up only the database and its associated files."""
    db_path = Path(db_file)
    paths_to_delete = [
        db_path,
        Path(f"{db_path}.lock"),
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm")
    ]
    for p in paths_to_delete:
        p.unlink(missing_ok=True)
    print(f"- Cleaned up database artifacts for {db_file}")

# --- Document Preparation ---
def prepare_documents(doc_dir="temp_docs_basic"):
    DOC_DIR = Path(doc_dir)
    # Clean up and recreate the directory from scratch at the beginning
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
    print_header("Preparing Sample Documents")
    DOC_DIR.mkdir(exist_ok=True)
    
    (DOC_DIR / "intro.txt").write_text(
        "safe_store is a Python library for local vector storage.", encoding='utf-8'
    )
    (DOC_DIR / "update_later.txt").write_text(
        "Initial content for update testing.", encoding='utf-8'
    )
    if USE_PARSING:
        (DOC_DIR / "web_snippet.html").write_text(
            "<html><body><p>Efficient retrieval is crucial for RAG pipelines.</p></body></html>",
            encoding='utf-8'
        )
    print(f"- Documents created in: {DOC_DIR.resolve()}")

# --- Main Script ---
if __name__ == "__main__":
    # --- Discover and Print Available Vectorizers ---
    print_header("Discovering Available Vectorizers")
    available_vectorizers = safe_store.SafeStore.list_available_vectorizers()
    for vec in available_vectorizers:
        print(f"\n- Vectorizer: {vec['name']} ({vec.get('title', 'No Title')})")
        print(f"  Description: {vec.get('description', 'N/A').strip()}")
        if vec.get('input_parameters'):
            print("  Parameters:")
            for param in vec['input_parameters']:
                default_val = f" (default: {param['default']})" if 'default' in param else ""
                mandatory_flag = "[MANDATORY]" if param.get('mandatory') else "[OPTIONAL]"
                print(f"    - {param['name']}: {param.get('description', 'N/A')} {mandatory_flag}{default_val}")

    DOC_DIR = Path("temp_docs_basic")
    prepare_documents(DOC_DIR)

    # --- Example 1: Sentence Transformer (ST) ---
    if USE_ST:
        db_file_st = "st_store.db"
        print_header(f"Sentence Transformer Example (DB: {db_file_st})")
        cleanup_db_files(db_file_st)
        try:
            store_st = safe_store.SafeStore(
                db_path=db_file_st,
                vectorizer_name="st",
                vectorizer_config=st_config,
                log_level=safe_store.LogLevel.INFO
            )
            with store_st:
                store_st.add_document(DOC_DIR / "intro.txt", metadata={"topic": "introduction"})
                if USE_PARSING:
                    store_st.add_document(DOC_DIR / "web_snippet.html", metadata={"source": "web"})

                results_st = store_st.query("local database library", top_k=1)
                if results_st:
                    res = results_st[0]
                    print(f"  Query Result: Score={res['similarity_percent']:.2f}%, Text='{res['chunk_text'][:60]}...'")
                
                # NEW: Demonstrate vectorizing with metadata
                print("\n  Demonstrating vectorization with metadata...")
                store_st.add_text(
                    unique_id="metadata_vectorization_test",
                    text="This text is about oranges and lemons.",
                    metadata={"topic": "citrus fruits", "author": "test"},
                    vectorize_with_metadata=True, # This is the new option
                    force_reindex=True
                )
                # This query should be more similar to the metadata ("citrus") than the other documents.
                results_meta = store_st.query("information about citrus", top_k=1)
                if results_meta:
                    res = results_meta[0]
                    print(f"  Querying with metadata context ('citrus'): Score={res['similarity_percent']:.2f}%, Path='{res['file_path']}'")
                    if res['file_path'] == 'metadata_vectorization_test':
                        print("  SUCCESS: The most relevant result came from the document with vectorized metadata.")
                    else:
                        print("  NOTE: The top result was not the one with vectorized metadata, which might happen with some models.")

                print("\n  Demonstrating file update...")
                (DOC_DIR / "update_later.txt").write_text("This content is new and improved for re-indexing.")
                store_st.add_document(DOC_DIR / "update_later.txt", force_reindex=True)
                print("  'update_later.txt' has been re-indexed.")

        except safe_store.ConfigurationError as e:
            print(f"  [SKIP] Could not run ST example: {e}")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred: {e}")

    # --- Example 2: TF-IDF ---
    if USE_TFIDF:
        db_file_tfidf = "tfidf_store.db"
        print_header(f"TF-IDF Example (DB: {db_file_tfidf})")
        cleanup_db_files(db_file_tfidf)
        try:
            store_tfidf = safe_store.SafeStore(
                db_path=db_file_tfidf,
                vectorizer_name="tfidf",
                vectorizer_config=tfidf_config,
                chunking_strategy='character'
            )
            with store_tfidf:
                print("  Adding documents (this will fit the TF-IDF model)...")
                store_tfidf.add_document(DOC_DIR / "intro.txt")
                if USE_PARSING:
                    store_tfidf.add_document(DOC_DIR / "web_snippet.html")

                results_tfidf = store_tfidf.query("SQLite backend storage", top_k=1)
                if results_tfidf:
                    res = results_tfidf[0]
                    print(f"  Query Result: Score={res['similarity_percent']:.2f}%, Text='{res['chunk_text'][:60]}...'")
        
        except safe_store.ConfigurationError as e:
            print(f"  [SKIP] Could not run TF-IDF example: {e}")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred: {e}")

    # --- Example 3: Ollama ---
    if USE_OLLAMA:
        db_file_ollama = "ollama_store.db"
        print_header(f"Ollama Example with Custom Tokenizer (DB: {db_file_ollama})")
        cleanup_db_files(db_file_ollama)
        try:
            available_models = safe_store.SafeStore.list_models("ollama")
            print(f"  Found Ollama models: {available_models}")
            if ollama_config["model"] not in available_models:
                 print(f"  [SKIP] Model '{ollama_config['model']}' not found in Ollama.")
            else:
                store_ollama = safe_store.SafeStore(
                    db_path=db_file_ollama,
                    vectorizer_name="ollama",
                    vectorizer_config=ollama_config,
                    # --- NOUVEAUTÉ : Utiliser le chunking par token en fournissant un tokenizer personnalisé ---
                    chunking_strategy='token',
                    custom_tokenizer={"name": "tiktoken", "model": "cl100k_base"}
                )
                with store_ollama:
                    store_ollama.add_document(DOC_DIR / "intro.txt")
                    results_ollama = store_ollama.query("file-based vector db", top_k=1)
                    if results_ollama:
                        res = results_ollama[0]
                        print(f"  Query Result: Score={res['similarity_percent']:.2f}%, Text='{res['chunk_text'][:60]}...'")

        except safe_store.VectorizationError as e:
            print(f"  [SKIP] Could not connect to Ollama server: {e}")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred: {e}")


    # --- API-based examples ---
    if USE_OPENAI:
        db_file_openai = "openai_store.db"
        print_header(f"OpenAI Example (DB: {db_file_openai})")
        cleanup_db_files(db_file_openai)
        try:
            store_openai = safe_store.SafeStore(
                db_path=db_file_openai,
                vectorizer_name="openai",
                vectorizer_config=openai_config,
                chunking_strategy='character' # Also required for OpenAI
            )
            with store_openai:
                store_openai.add_document(DOC_DIR / "intro.txt")
                results_openai = store_openai.query("python tool for embeddings", top_k=1)
                if results_openai:
                    print(f"  Query Result: Score={results_openai[0]['similarity_percent']:.2f}%")
        except Exception as e:
            print(f"  [ERROR] OpenAI example failed: {e}")

    if USE_COHERE:
        db_file_cohere = "cohere_store.db"
        print_header(f"Cohere Example (DB: {db_file_cohere})")
        cleanup_db_files(db_file_cohere)
        try:
            store_cohere = safe_store.SafeStore(
                db_path=db_file_cohere,
                vectorizer_name="cohere",
                vectorizer_config=cohere_config,
                chunking_strategy='character' # Also required for Cohere
            )
            with store_cohere:
                store_cohere.add_document(DOC_DIR / "intro.txt")
                results_cohere = store_cohere.query("library for vector search", top_k=1)
                if results_cohere:
                    print(f"  Query Result: Score={results_cohere[0]['similarity_percent']:.2f}%")
        except Exception as e:
            print(f"  [ERROR] Cohere example failed: {e}")

    print("\n--- Final Cleanup ---")
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
        print(f"- Removed directory: {DOC_DIR}")
        
    print("\n--- End of Script ---")