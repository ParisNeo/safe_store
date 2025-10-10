# examples/encryption_usage.py
"""
Demonstrates using safe_store's encryption feature.

Requires 'cryptography' library: pip install safe_store[encryption]
"""
import safe_store
from pathlib import Path
import shutil
import sqlite3 # For direct DB check example

# --- Configuration ---
DB_FILE = "encrypted_example_store.db"
ENCRYPTION_KEY = "this-is-my-secret-Pa$$wOrd!" # !! Use a strong, managed key in real apps !!
DOC_DIR = Path("temp_docs_encrypted")
# --- Vectorizer Configuration ---
# Define a default vectorizer to be used in the example
DEFAULT_VECTORIZER_NAME = "st"
DEFAULT_VECTORIZER_CONFIG = {"model": "all-MiniLM-L6-v2"}

# --- Helper Functions ---
def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def cleanup():
    print_header("Cleaning Up")
    db_path = Path(DB_FILE)
    lock_path = Path(f"{DB_FILE}.lock")
    wal_path = Path(f"{DB_FILE}-wal")
    shm_path = Path(f"{DB_FILE}-shm")

    if DOC_DIR.exists(): shutil.rmtree(DOC_DIR)
    if db_path.exists(): db_path.unlink()
    if lock_path.exists(): lock_path.unlink(missing_ok=True)
    if wal_path.exists(): wal_path.unlink(missing_ok=True)
    if shm_path.exists(): shm_path.unlink(missing_ok=True)
    print("- Cleanup complete.")

# --- Main Script ---
if __name__ == "__main__":
    cleanup() # Start fresh

    # --- Prepare Sample Document ---
    print_header("Preparing Document")
    DOC_DIR.mkdir(exist_ok=True)
    doc_path = DOC_DIR / "secret_notes.txt"
    doc_content = """
    Project Phoenix: Launch date is Q4.
    Budget allocation: $1.5M.
    Key personnel: Alice, Bob.
    Security protocol: Level Gamma.
    """
    doc_path.write_text(doc_content.strip(), encoding='utf-8')
    print(f"- Created: {doc_path.name}")

    # --- 1. Initialize safe_store WITH Encryption Key ---
    print_header("Initializing Encrypted Store")
    try:
        store_encrypted = safe_store.SafeStore(
            DB_FILE,
            log_level=safe_store.LogLevel.INFO,
            encryption_key=ENCRYPTION_KEY
        )
    except safe_store.ConfigurationError as e:
        print(f"[ERROR] Failed to initialize: {e}")
        exit()

    # --- 2. Add Document to Encrypted Store ---
    print_header("Adding Document (Encrypted)")
    try:
        with store_encrypted:
            store_encrypted.add_document(
                doc_path,
                chunk_size=80,
                chunk_overlap=10,
                metadata={"sensitivity": "high"},
                vectorizer_name=DEFAULT_VECTORIZER_NAME,
                vectorizer_config=DEFAULT_VECTORIZER_CONFIG
            )
            print(f"- Added '{doc_path.name}'. Chunk text should be encrypted in '{DB_FILE}'.")

            # Direct DB check
            try:
                 conn = sqlite3.connect(store_encrypted.db_path)
                 cursor = conn.cursor()
                 cursor.execute("SELECT chunk_text, is_encrypted FROM chunks LIMIT 1")
                 row = cursor.fetchone()
                 conn.close()
                 if row and row[1] == 1:
                      print("[VERIFIED] Direct DB check: is_encrypted flag is set and text is not plaintext.")
                 else:
                      print("[WARNING] Direct DB check failed or text appears unencrypted.")
            except Exception as db_err:
                 print(f"[INFO] Could not perform direct DB check: {db_err}")

            # --- 3. Query Encrypted Store (With Key) ---
            print_header("Querying Encrypted Store (With Key)")
            query = "budget personnel"
            results = store_encrypted.query(query, top_k=2, vectorizer_name=DEFAULT_VECTORIZER_NAME, vectorizer_config=DEFAULT_VECTORIZER_CONFIG)
            print(f"Query: '{query}'")
            if results:
                for i, res in enumerate(results):
                    print(f"  Result {i+1}: Score={res['similarity_score']:.4f}")
                    print(f"    Text: '{res['chunk_text'][:60]}...'")
                    assert "[Encrypted" not in res['chunk_text']
            else:
                print("  No results found.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred while using the encrypted store: {e}")
        import traceback
        traceback.print_exc()

    # --- 4. Access Encrypted DB WITHOUT the Key ---
    print_header("Accessing Encrypted Store WITHOUT Key")
    try:
        store_no_key = safe_store.SafeStore(DB_FILE, log_level=safe_store.LogLevel.WARNING)
        with store_no_key:
            query = "security protocol"
            results_no_key = store_no_key.query(query, top_k=1, vectorizer_name=DEFAULT_VECTORIZER_NAME, vectorizer_config=DEFAULT_VECTORIZER_CONFIG)

            if results_no_key:
                 print(f"    Text: '{results_no_key[0]['chunk_text']}'")
                 assert results_no_key[0]['chunk_text'] == "[Encrypted - Key Unavailable]"
            
            print("\nAttempting to add TF-IDF vectorization (should fail)...")
            try:
                 store_no_key.add_vectorization(vectorizer_name="tfidf", vectorizer_config={"name": "fail_test"})
            except safe_store.ConfigurationError as e:
                 print(f"[EXPECTED ERROR] ConfigurationError: {e}")
                 assert "Cannot fit TF-IDF on encrypted chunks without the correct encryption key." in str(e)

    except Exception as e:
        print(f"\n[ERROR] An error occurred while using the store without key: {e}")

    # --- 5. Access Encrypted DB with WRONG Key ---
    print_header("Accessing Encrypted Store With WRONG Key")
    try:
        store_wrong_key = safe_store.SafeStore(DB_FILE, log_level=safe_store.LogLevel.WARNING, encryption_key="this-is-the-WRONG-key")
        with store_wrong_key:
            query = "launch date"
            results_wrong_key = store_wrong_key.query(query, top_k=1, vectorizer_name=DEFAULT_VECTORIZER_NAME, vectorizer_config=DEFAULT_VECTORIZER_CONFIG)
            if results_wrong_key:
                 print(f"    Text: '{results_wrong_key[0]['chunk_text']}'")
                 assert results_wrong_key[0]['chunk_text'] == "[Encrypted - Decryption Failed]"

    except Exception as e:
        print(f"\n[ERROR] An error occurred while using the store with wrong key: {e}")