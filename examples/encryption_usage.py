# examples/encryption_usage.py
"""
Demonstrates using safe_store's encryption feature.
"""
import safe_store
from pathlib import Path
import shutil
import sqlite3

# --- Configuration ---
DB_FILE = "encrypted_example_store.db"
ENCRYPTION_KEY = "this-is-my-secret-Pa$$wOrd!"
DOC_DIR = Path("temp_docs_encrypted")
VECTORIZER_NAME = "st"
VECTORIZER_CONFIG = {"model": "all-MiniLM-L6-v2"}

def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def cleanup():
    print_header("Cleaning Up")
    for p in [DB_FILE, f"{DB_FILE}.lock", f"{DB_FILE}-wal", f"{DB_FILE}-shm"]:
        Path(p).unlink(missing_ok=True)
    if DOC_DIR.exists(): shutil.rmtree(DOC_DIR)
    print("- Cleanup complete.")

if __name__ == "__main__":
    cleanup()

    DOC_DIR.mkdir(exist_ok=True)
    doc_path = DOC_DIR / "secret_notes.txt"
    doc_path.write_text("Project Phoenix: Launch date is Q4. Key personnel: Alice, Bob.")

    # --- 1. Initialize SafeStore WITH Encryption Key and Vectorizer ---
    print_header("Initializing Encrypted Store")
    store_encrypted = safe_store.SafeStore(
        DB_FILE,
        vectorizer_name=VECTORIZER_NAME,
        vectorizer_config=VECTORIZER_CONFIG,
        log_level=safe_store.LogLevel.INFO,
        encryption_key=ENCRYPTION_KEY
    )

    # --- 2. Add Document to Encrypted Store ---
    print_header("Adding Document (Encrypted)")
    with store_encrypted:
        store_encrypted.add_document(doc_path, metadata={"sensitivity": "high"})
        print(f"- Added '{doc_path.name}'.")

        # Direct DB check
        conn = sqlite3.connect(store_encrypted.db_path)
        is_encrypted_flag = conn.execute("SELECT is_encrypted FROM chunks LIMIT 1").fetchone()[0]
        conn.close()
        if is_encrypted_flag == 1:
            print("[VERIFIED] Direct DB check: is_encrypted flag is set.")
        else:
            print("[WARNING] Direct DB check: is_encrypted flag is NOT set.")

        # --- 3. Query Encrypted Store (With Key) ---
        print_header("Querying Encrypted Store (With Key)")
        query = "project personnel"
        results = store_encrypted.query(query, top_k=1)
        if results:
            print(f"    Text: '{results[0]['chunk_text']}'")
            assert "[Encrypted" not in results[0]['chunk_text']

    # --- 4. Access Encrypted DB WITHOUT the Key ---
    print_header("Accessing Encrypted Store WITHOUT Key")
    store_no_key = safe_store.SafeStore(DB_FILE, vectorizer_name=VECTORIZER_NAME, vectorizer_config=VECTORIZER_CONFIG)
    with store_no_key:
        results_no_key = store_no_key.query("security protocol", top_k=1)
        if results_no_key:
            print(f"    Text: '{results_no_key[0]['chunk_text']}'")
            assert results_no_key[0]['chunk_text'] == "[Encrypted - Key Unavailable]"

    # --- 5. Access Encrypted DB with WRONG Key ---
    print_header("Accessing Encrypted Store With WRONG Key")
    store_wrong_key = safe_store.SafeStore(DB_FILE, vectorizer_name=VECTORIZER_NAME, vectorizer_config=VECTORIZER_CONFIG, encryption_key="wrong-key")
    with store_wrong_key:
        results_wrong_key = store_wrong_key.query("launch date", top_k=1)
        if results_wrong_key:
            print(f"    Text: '{results_wrong_key[0]['chunk_text']}'")
            assert results_wrong_key[0]['chunk_text'] == "[Encrypted - Decryption Failed]"