# examples/basic_usage.py
import safe_store
from pathlib import Path
import time
import shutil

# --- Configuration ---
DB_FILE = "basic_usage_store.db"
DOC_DIR = Path("temp_docs_basic")
USE_ST = True       # Set to False if sentence-transformers not installed
USE_TFIDF = True    # Set to False if scikit-learn not installed
USE_OLLAMA = True   # deactivate if you don't have an ollama server with emdebbing models
USE_OPENAI = False   # deactivate of not using open ai or you don't have any key
USE_COHERE = False   # deactivate of not using cohere ai or you don't have any key
USE_PARSING = True  # Set to False if parsing libs not installed

st_vectorizer_name = "st:all-MiniLM-L6-v2"
tfidf_vectorizer_name="tfidf:my_tfidf"
ollama_vectorizer_name = "ollama:bge-m3:latest"
# option 1 : use your Envrionment vrairanble OPENAI_API_KEY
openai_vectorizer_name = "openai:text-embedding-3-small"
# option 2 : not advised as it will be saved to the sql database 
# openai_vectorizer_name = "openai:text-embedding-3-small::sk-replacewithyourkey" # Warning!! Replace with your key
# option 1 : use your Envrionment vrairanble COHERE_API_KEY
cohere_vectorizer_name = "cohere:embed-english-v3.0"
# option 2 : not advised as it will be saved to the sql database 
# cohere_vectorizer_name = "cohere:embed-english-v3.0::replacewithyourkey" # Warning!! Replace with your key
# --- Helper Functions ---
def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def cleanup():
    print_header("Cleaning Up")
    db_path = Path(DB_FILE)
    lock_path = Path(f"{DB_FILE}.lock")
    wal_path = Path(f"{DB_FILE}-wal")
    shm_path = Path(f"{DB_FILE}-shm")

    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
        print(f"- Removed directory: {DOC_DIR}")
    if db_path.exists():
        db_path.unlink()
        print(f"- Removed database: {db_path}")
    if lock_path.exists():
        lock_path.unlink(missing_ok=True)
        print(f"- Removed lock file: {lock_path}")
    if wal_path.exists():
        wal_path.unlink(missing_ok=True)
        print(f"- Removed WAL file: {wal_path}")
    if shm_path.exists():
        shm_path.unlink(missing_ok=True)
        print(f"- Removed SHM file: {shm_path}")

# --- Main Script ---
if __name__ == "__main__":
    cleanup() # Start fresh

    # --- 1. Prepare Sample Documents ---
    print_header("Preparing Documents")
    DOC_DIR.mkdir(exist_ok=True)

    # Document 1 (Text)
    doc1_path = DOC_DIR / "intro.txt"
    doc1_content = """
    safe_store is a Python library for local vector storage.
    It uses SQLite as its backend, making it lightweight and file-based.
    Key features include concurrency control and support for multiple vectorizers.
    """
    doc1_path.write_text(doc1_content.strip(), encoding='utf-8')
    print(f"- Created: {doc1_path.name}")

    # Document 2 (HTML) - Requires [parsing]
    doc2_path = DOC_DIR / "web_snippet.html"
    doc2_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Example Page</title></head>
    <body>
        <h1>Information Retrieval</h1>
        <p>Efficient retrieval is crucial for RAG pipelines.</p>
        <p>safe_store helps manage embeddings for semantic search.</p>
    </body>
    </html>
    """
    if USE_PARSING:
        doc2_path.write_text(doc2_content.strip(), encoding='utf-8')
        print(f"- Created: {doc2_path.name}")
    else:
        print(f"- Skipping {doc2_path.name} (requires [parsing])")

    # Document 3 (Text) - Will be added later to show updates
    doc3_path = DOC_DIR / "update_later.txt"
    doc3_content_v1 = "Initial content for update testing."
    doc3_path.write_text(doc3_content_v1, encoding='utf-8')
    print(f"- Created: {doc3_path.name}")

    print(f"Documents prepared in: {DOC_DIR.resolve()}")

    # --- 2. Initialize safe_store ---
    print_header("Initializing safe_store")
    # Use INFO level for less verbose output in basic example
    store = safe_store.SafeStore(DB_FILE, log_level=safe_store.LogLevel.INFO)

    # --- 3. Indexing Documents ---
    try:
        with store:
            print_header("Indexing Documents")
            # --- Index doc1 with Sentence Transformer (default) ---
            if USE_ST:
                print(f"\nIndexing {doc1_path.name} with ST...")
                try:
                    store.add_document(
                        doc1_path,
                        vectorizer_name=st_vectorizer_name, # Default, but explicit here
                        chunk_size=80,
                        chunk_overlap=15,
                        metadata={"source": "manual", "topic": "introduction"}
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not index with Sentence Transformer: {e}")
                    USE_ST = False # Disable ST for later steps if failed here
            else:
                 print(f"\nSkipping ST indexing for {doc1_path.name}")

            # --- Index doc2 (HTML) ---
            if USE_PARSING:
                print(f"\nIndexing {doc2_path.name} with ST...")
                if USE_ST:
                    try:
                        store.add_document(doc2_path, metadata={"source": "web", "language": "en"})
                    except safe_store.ConfigurationError as e:
                         print(f"  [SKIP] Could not index {doc2_path.name} with ST: {e}")
                else:
                    print(f"  [SKIP] ST vectorizer not available.")
            else:
                print(f"\nSkipping HTML indexing for {doc2_path.name}")

            # --- Index doc3 (initial version) ---
            print(f"\nIndexing {doc3_path.name} (v1) with ST...")
            if USE_ST:
                try:
                    store.add_document(doc3_path, metadata={"version": 1})
                except safe_store.ConfigurationError as e:
                     print(f"  [SKIP] Could not index {doc3_path.name} with ST: {e}")
            else:
                print(f"  [SKIP] ST vectorizer not available.")


            # --- Add TF-IDF Vectorization to all docs ---
            if USE_TFIDF:
                print_header("Adding TF-IDF Vectorization")
                try:
                    store.add_vectorization(
                        vectorizer_name=tfidf_vectorizer_name,
                        # Let safe_store handle fitting on all docs
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not add TF-IDF vectorization: {e}")
                    USE_TFIDF = False # Disable TFIDF if failed
            else:
                print_header("Skipping TF-IDF Vectorization")

            # --- Add OLLAMA Vectorization to all docs ---
            if USE_OLLAMA:
                print_header("Adding Ollama Vectorization")
                try:
                    store.add_vectorization(
                        vectorizer_name=ollama_vectorizer_name,
                        # Let safe_store handle fitting on all docs
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not add Ollama vectorization: {e}")
                    USE_OLLAMA = False # Disable TFIDF if failed
            else:
                print_header("Skipping Ollama Vectorization")

            # --- Add OpenAi Vectorization to all docs ---
            if USE_OPENAI:
                print_header("Adding OpenAi Vectorization")
                try:
                    store.add_vectorization(
                        vectorizer_name=openai_vectorizer_name,
                        # Let safe_store handle fitting on all docs
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not add OpenAi vectorization: {e}")
                    USE_OPENAI = False # Disable TFIDF if failed
            else:
                print_header("Skipping OpenAi Vectorization")

            # --- Add Cohere Vectorization to all docs ---
            if USE_COHERE:
                print_header("Adding Cohere Vectorization")
                try:
                    store.add_vectorization(
                        vectorizer_name=cohere_vectorizer_name,
                        # Let safe_store handle fitting on all docs
                    )
                except safe_store.ConfigurationError as e:
                    print(f"  [SKIP] Could not add Cohere vectorization: {e}")
                    USE_COHERE = False # Disable TFIDF if failed
            else:
                print_header("Skipping Cohere Vectorization")

            # --- 4. Querying ---
            print_header("Querying")
            query_text = "vector database features"

            if USE_ST:
                print("\nQuerying with Sentence Transformer...")
                results_st = store.query(query_text, vectorizer_name=st_vectorizer_name, top_k=2)
                if results_st:
                    for i, res in enumerate(results_st):
                        print(f"  ST Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping ST Query.")

            if USE_TFIDF:
                print("\nQuerying with TF-IDF...")
                results_tfidf = store.query(query_text, vectorizer_name=tfidf_vectorizer_name, top_k=2)
                if results_tfidf:
                    for i, res in enumerate(results_tfidf):
                        print(f"  TFIDF Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping TF-IDF Query.")

            if USE_OLLAMA:
                print("\nQuerying with OLLAMA...")
                results_tfidf = store.query(query_text, vectorizer_name=ollama_vectorizer_name, top_k=2)
                if results_tfidf:
                    for i, res in enumerate(results_tfidf):
                        print(f"  TFIDF Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping OLLAMA Query.")


            if USE_OPENAI:
                print("\nQuerying with OpenAi...")
                results_tfidf = store.query(query_text, vectorizer_name=openai_vectorizer_name, top_k=2)
                if results_tfidf:
                    for i, res in enumerate(results_tfidf):
                        print(f"  OPenAI Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping OpenAI Query.")

            if USE_COHERE:
                print("\nQuerying with Cohere...")
                results_tfidf = store.query(query_text, vectorizer_name=cohere_vectorizer_name, top_k=2)
                if results_tfidf:
                    for i, res in enumerate(results_tfidf):
                        print(f"  Cohere Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")
                        print(f"    Metadata: {res.get('metadata')}")
                else:
                    print("  No results found.")
            else:
                print("\nSkipping Cohere Query.")

            # --- 5. File Updates & Re-indexing ---
            print_header("Updating and Re-indexing")
            print(f"Updating content of {doc3_path.name}...")
            doc3_content_v2 = "This content has been significantly updated for testing re-indexing."
            doc3_path.write_text(doc3_content_v2, encoding='utf-8')
            time.sleep(0.1) # Ensure file timestamp changes

            print(f"Running add_document again for {doc3_path.name}...")
            if USE_ST:
                store.add_document(doc3_path, metadata={"version": 2}) # Should detect change and re-index with ST
            else:
                 print("  Skipping re-indexing (ST not available).")
            # Note: TF-IDF vectors for the old chunks of doc3 were deleted by the re-index.
            # If we wanted TF-IDF for the *new* chunks, we'd need to run add_vectorization again.
            # Or, ideally, add_document could optionally re-vectorize for *all* methods. (Future enhancement)

            # --- 6. Listing ---
            print_header("Listing Contents")
            print("\n--- Documents ---")
            docs = store.list_documents()
            for doc in docs:
                print(f"- ID: {doc['doc_id']}, Path: {Path(doc['file_path']).name}, Hash: {doc['file_hash'][:8]}..., Meta: {doc.get('metadata')}")

            print("\n--- Vectorization Methods ---")
            methods = store.list_vectorization_methods()
            for method in methods:
                print(f"- ID: {method['method_id']}, Name: {method['method_name']}, Type: {method['method_type']}, Dim: {method['vector_dim']}, Fitted: {method.get('params',{}).get('fitted', 'N/A')}")

            # --- 7. Removing a Vectorization ---
            if USE_TFIDF:
                 print_header("Removing Vectorization")
                 print("Removing TF-IDF vectors...")
                 store.remove_vectorization("tfidf:my_tfidf")

                 print("\n--- Vectorization Methods After Removal ---")
                 methods_after = store.list_vectorization_methods()
                 for method in methods_after:
                      print(f"- ID: {method['method_id']}, Name: {method['method_name']}")
            else:
                 print_header("Skipping Vectorization Removal")


    except safe_store.ConfigurationError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Please install the required extras (e.g., pip install safe_store[all])")
    except safe_store.ConcurrencyError as e:
        print(f"\n[ERROR] Lock timeout or concurrency issue: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc() # Print traceback for unexpected errors
    finally:
        # Connection is closed automatically by 'with' statement
        print("\n--- End of Script ---")
        # Optional: uncomment cleanup() to remove files after run
        # cleanup()