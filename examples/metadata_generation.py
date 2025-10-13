    # examples/metadata_generation.py
import safe_store
from safe_store import LogLevel
import pipmaster as pm
from pathlib import Path
import shutil
import json

# --- Configuration ---
DB_FILE = "metadata_example.db"
DOC_DIR = Path("temp_docs_metadata_example")
ENCRYPTION_KEY = "my-super-secret-key-for-testing" # Use a strong key in production


BINDING_NAME = "ollama"
HOST_ADDRESS = "http://localhost:11434"
MODEL_NAME = "mistral:latest"

# --- Example Setup ---
def setup():
    """Cleans up old files and creates new ones for the example."""
    print_header("Setting Up Example Environment")
    # Clean up DB
    db_path = Path(DB_FILE)
    paths_to_delete = [db_path, Path(f"{db_path}.lock")]
    for p in paths_to_delete:
        p.unlink(missing_ok=True)
    
    # Clean up and create doc directory
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
    DOC_DIR.mkdir(exist_ok=True)

    # Create a sample document
    article_content = """
    The Art of Quantum Computing: A Gentle Introduction

    Quantum computing represents a fundamental shift in computation. Unlike classical
    computers that use bits (0s and 1s), quantum computers use qubits, which can
    exist in a superposition of both 0 and 1 simultaneously. This property, along
    with entanglement, allows quantum computers to explore a vast number of
    possibilities at once, promising to solve complex problems in fields like
    medicine, materials science, and cryptography that are intractable for even the
    most powerful classical supercomputers. However, building and controlling stable
    qubits remains a significant engineering challenge due to their sensitivity to
    environmental noise.
    """
    (DOC_DIR / "quantum_intro.txt").write_text(article_content.strip())
    print(f"- Created sample document in: {DOC_DIR.resolve()}")
    return DOC_DIR / "quantum_intro.txt"

def print_header(title: str):
    print("\n" + "="*20 + f" {title} " + "="*20)

# --- Metadata Generation with Lollms ---
def generate_metadata_with_lollms(file_content: str) -> dict:
    """
    Uses lollms-client to generate a title and summary for the given text.
    """
    print_header("Generating Metadata with LOLLMS")
    try:
        pm.ensure_packages(["lollms_client"])
        from lollms_client import LollmsClient
        # Make sure you have a lollms-webui instance running with a model loaded.
        # This example assumes a local instance at the default port.
        client = LollmsClient(llm_binding_name=BINDING_NAME, llm_binding_config={"host_address": HOST_ADDRESS, "model_name": MODEL_NAME})
    except Exception as e:
        print(f"  [SKIP] Could not initialize LollmsClient. Is it installed and running? Error: {e}")
        return {"error": "LollmsClient not available"}

    prompt = f"""
    Analyze the following document and extract a concise title and a one-sentence summary.
    Your response MUST be in a raw JSON format with "title" and "summary" as keys.

    Document:
    ---
    {file_content}
    ---

    JSON Response:
    """
    
    print("  - Sending prompt to LLM for metadata extraction...")
    try:
        response = client.generate_text(prompt, max_new_tokens=150)
        print("  - Received response from LLM.")
        # The response should be a JSON string, let's parse it
        metadata = json.loads(response)
        print(f"  - Successfully parsed metadata: {metadata}")
        return metadata
    except Exception as e:
        print(f"  [ERROR] Failed to generate or parse metadata from LLM. Error: {e}")
        return {"error": f"LLM metadata generation failed: {e}"}

# --- Main Script ---
if __name__ == "__main__":
    sample_doc_path = setup()
    
    # 1. Generate Metadata
    document_text = sample_doc_path.read_text()
    generated_metadata = generate_metadata_with_lollms(document_text)

    if "error" in generated_metadata:
        print("\n  Proceeding with fallback metadata.")
        generated_metadata = {
            "title": "Fallback Title: Quantum Computing",
            "summary": "A fallback summary about qubits and their challenges."
        }

    # 2. Initialize SafeStore with Encryption
    print_header("Initializing SafeStore with Encryption")
    try:
        # Note: We are now passing the encryption key
        store = safe_store.SafeStore(
            db_path=DB_FILE,
            vectorizer_name="st",
            vectorizer_config={"model": "all-MiniLM-L6-v2"},
            log_level=LogLevel.INFO,
            encryption_key=ENCRYPTION_KEY
        )
        print("  - SafeStore initialized.")
    except Exception as e:
        print(f"  [FATAL] Could not initialize SafeStore: {e}")
        exit(1)

    # 3. Add the document WITH the generated metadata
    print_header("Adding Document with Generated Metadata")
    with store:
        store.add_document(
            file_path=sample_doc_path,
            metadata=generated_metadata
        )
        print(f"  - Document '{sample_doc_path.name}' added to the store.")

        # 4. List documents to verify metadata storage (and encryption)
        print("\n  --- Verifying Stored Documents ---")
        docs = store.list_documents()
        for doc in docs:
            print(f"  - Found Doc ID: {doc['doc_id']}, Path: {doc['file_path']}")
            print(f"    Metadata: {doc['metadata']}")

    # 5. Query the store and inspect the results
    print_header("Querying the Store")
    query = "What are the difficulties of building qubits?"
    print(f"  - Query: '{query}'")

    with store:
        results = store.query(query, top_k=1)

        if not results:
            print("  - No results found.")
        else:
            top_result = results[0]
            print("\n  --- Top Query Result ---")
            print(f"  - Similarity: {top_result['similarity_percent']:.2f}%")
            
            print("\n  - Document Metadata (from result object):")
            print(f"    {top_result['document_metadata']}")
            
            print("\n  - Full Chunk Text (with prepended context):")
            print("-" * 50)
            print(top_result['chunk_text'])
            print("-" * 50)
            
            # Verification
            assert "Document Context" in top_result['chunk_text']
            assert generated_metadata['title'] in top_result['chunk_text']
            print("\n  [SUCCESS] Verified that document metadata was prepended to the chunk text.")

    # 6. Final cleanup
    print_header("Final Cleanup")
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
        print(f"- Removed temporary directory: {DOC_DIR}")
    
    print("\n--- Example Finished ---")