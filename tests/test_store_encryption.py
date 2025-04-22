# tests/test_store_encryption.py
import pytest
import sqlite3
from pathlib import Path
import base64

from safe_store import safe_store, LogLevel
from safe_store.core.exceptions import EncryptionError, ConfigurationError, SafeStoreError
from safe_store.security.encryption import CRYPTOGRAPHY_AVAILABLE, Encryptor

# Conditionally skip tests if cryptography is not installed
pytestmark = pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="Requires cryptography library")

@pytest.fixture
def encryption_password() -> str:
    return "my-secret-rag-key-42"

@pytest.fixture
def encrypted_store(temp_db_path: Path, encryption_password: str) -> safe_store:
    """Provides a safe_store instance configured with encryption."""
    if temp_db_path.exists(): temp_db_path.unlink()
    lock_path = temp_db_path.with_suffix(".db.lock")
    if lock_path.exists(): lock_path.unlink()

    store = safe_store(
        db_path=temp_db_path,
        log_level=LogLevel.DEBUG,
        encryption_key=encryption_password
    )
    yield store
    store.close()

@pytest.fixture
def unencrypted_store(temp_db_path: Path) -> safe_store:
    """Provides a safe_store instance WITHOUT encryption."""
    # Use a different DB file or ensure cleanup
    db_path = temp_db_path.parent / "unencrypted_test.db"
    if db_path.exists(): db_path.unlink()
    lock_path = db_path.with_suffix(".db.lock")
    if lock_path.exists(): lock_path.unlink()

    store = safe_store(db_path=db_path, log_level=LogLevel.DEBUG)
    yield store
    store.close()

@pytest.fixture
def sample_encrypt_text_file(tmp_path: Path) -> Path:
    """Creates a temporary text file specifically for encryption tests."""
    p = tmp_path / "encrypt_me.txt"
    p.write_text("This text should be encrypted.\nIt has multiple lines.", encoding='utf-8')
    return p


def test_init_with_encryption(encrypted_store: safe_store):
    """Test that the store initializes correctly with a key."""
    assert encrypted_store.encryptor.is_enabled
    # Check log? (Already tested in Encryptor tests)

def test_add_document_encrypts_chunks(encrypted_store: safe_store, sample_encrypt_text_file: Path, encryption_password: str):
    """Test that chunk text is encrypted when adding a document."""
    store = encrypted_store
    original_content = sample_encrypt_text_file.read_text(encoding='utf-8')
    chunk1_text = "This text should be encrypted." # Expected first chunk
    chunk2_text = "It has multiple lines."       # Expected second chunk (approx)

    with store:
        store.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

    # --- Verify DB state directly ---
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (str(sample_encrypt_text_file.resolve()),))
    doc_id_res = cursor.fetchone()
    assert doc_id_res is not None
    doc_id = doc_id_res[0]

    cursor.execute("SELECT chunk_text, is_encrypted FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
    chunks_data = cursor.fetchall()
    assert len(chunks_data) > 0
    assert chunks_data[0][1] != chunk1_text.encode('utf-8') # Should not be plaintext bytes
    assert chunks_data[0][1] != chunk1_text # Should not be plaintext string
    assert chunks_data[0][1] is not None
    assert chunks_data[0][0] is not None # Ensure chunk_text is not None
    assert chunks_data[0][1] # is_encrypted flag should be True (1)

    # --- Verify decryption works ---
    # Use a separate Encryptor instance to mimic reading later
    verifier_encryptor = Encryptor(encryption_password)
    try:
        decrypted_chunk1 = verifier_encryptor.decrypt(chunks_data[0][0]) # Assuming chunk_text is bytes token
        assert decrypted_chunk1.startswith("This text should be encrypted")
    except Exception as e:
        pytest.fail(f"Decryption failed for chunk 0: {e}. Stored data: {chunks_data[0][0]!r}")

    # Check second chunk if it exists
    if len(chunks_data) > 1:
        assert chunks_data[1][1] # is_encrypted flag
        assert chunks_data[1][0] is not None
        try:
             decrypted_chunk2 = verifier_encryptor.decrypt(chunks_data[1][0])
             assert decrypted_chunk2.startswith("It has multiple lines")
        except Exception as e:
             pytest.fail(f"Decryption failed for chunk 1: {e}. Stored data: {chunks_data[1][0]!r}")


    conn.close()


def test_query_decrypts_chunks(encrypted_store: safe_store, sample_encrypt_text_file: Path):
    """Test that query results contain decrypted chunk text."""
    store = encrypted_store
    query_text = "multiple lines"
    expected_part_of_chunk = "It has multiple lines."

    with store:
        store.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)
        results = store.query(query_text, top_k=1)

    assert len(results) == 1
    result_chunk = results[0]
    assert "chunk_text" in result_chunk
    assert result_chunk["chunk_text"] == expected_part_of_chunk
    assert "[Encrypted" not in result_chunk["chunk_text"]


def test_query_encrypted_data_without_key(encrypted_store: safe_store, sample_encrypt_text_file: Path, temp_db_path: Path):
    """Test querying encrypted data with a new store instance *without* the key."""
    store1 = encrypted_store
    db_file = store1.db_path # Get the path used by the encrypted store
    query_text = "multiple lines"

    # Add encrypted data
    with store1:
        store1.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

    # Create new store instance for the SAME DB file, but without the key
    store2 = safe_store(db_file, log_level=LogLevel.DEBUG, encryption_key=None)
    assert not store2.encryptor.is_enabled

    with store2:
        results = store2.query(query_text, top_k=1)

    assert len(results) == 1
    result_chunk = results[0]
    assert "chunk_text" in result_chunk
    assert result_chunk["chunk_text"] == "[Encrypted - Key Unavailable]"


def test_query_encrypted_data_with_wrong_key(encrypted_store: safe_store, sample_encrypt_text_file: Path, temp_db_path: Path):
    """Test querying encrypted data with a new store instance with the WRONG key."""
    store1 = encrypted_store
    db_file = store1.db_path
    query_text = "multiple lines"

    # Add encrypted data
    with store1:
        store1.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

    # Create new store instance for the SAME DB file, with the WRONG key
    store2 = safe_store(db_file, log_level=LogLevel.DEBUG, encryption_key="wrong-password")
    assert store2.encryptor.is_enabled

    with store2:
        results = store2.query(query_text, top_k=1)

    assert len(results) == 1
    result_chunk = results[0]
    assert "chunk_text" in result_chunk
    assert result_chunk["chunk_text"] == "[Encrypted - Decryption Failed]"


def test_add_vectorization_decrypts_for_tfidf(encrypted_store: safe_store, sample_encrypt_text_file: Path):
    """Test add_vectorization decrypts text before fitting TF-IDF."""
    store = encrypted_store
    tfidf_name = "tfidf:encrypted_fit"

    with store:
        # Add encrypted document first
        store.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

        # Now add TF-IDF - this should decrypt the chunks for fitting
        store.add_vectorization(tfidf_name)

        # Verify TF-IDF was fitted and vectors were added
        methods = store.list_vectorization_methods()
        tfidf_method = next((m for m in methods if m["method_name"] == tfidf_name), None)
        assert tfidf_method is not None
        assert tfidf_method["params"]["fitted"] is True
        assert tfidf_method["vector_dim"] > 0

        conn = store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (tfidf_method["method_id"],))
        vector_count = cursor.fetchone()[0]
        assert vector_count > 0 # Should be equal to the number of chunks


def test_add_vectorization_fails_without_key(encrypted_store: safe_store, sample_encrypt_text_file: Path):
    """Test add_vectorization (TF-IDF fit) fails if data is encrypted and key is missing."""
    store1 = encrypted_store
    db_file = store1.db_path
    tfidf_name = "tfidf:encrypted_fit_fail"

    # Add encrypted data
    with store1:
        store1.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

    # New store without key
    store2 = safe_store(db_file, log_level=LogLevel.DEBUG, encryption_key=None)

    with store2:
        with pytest.raises(ConfigurationError, match="Cannot fit TF-IDF on encrypted chunks without the correct encryption key."):
            store2.add_vectorization(tfidf_name)


def test_unencrypted_store_ignores_encrypted_chunks(unencrypted_store: safe_store, encrypted_store: safe_store, sample_encrypt_text_file: Path):
    """ Test interaction: Add encrypted data, then try to read/query with unencrypted store"""
    # Setup: Add encrypted data using encrypted_store
    with encrypted_store:
        encrypted_store.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)
    db_path = encrypted_store.db_path # Get the path

    # Test: Open SAME DB with unencrypted store
    store_no_key = safe_store(db_path, log_level=LogLevel.DEBUG, encryption_key=None)
    with store_no_key:
        # Querying should return placeholders
        results = store_no_key.query("multiple lines", top_k=1)
        assert len(results) == 1
        assert results[0]['chunk_text'] == "[Encrypted - Key Unavailable]"

        # Adding a NEW vectorization method that requires reading text should fail
        tfidf_name = "tfidf:fail_on_encrypted"
        with pytest.raises(ConfigurationError, match="Cannot fit TF-IDF on encrypted chunks without the correct encryption key."):
             store_no_key.add_vectorization(tfidf_name)

    store_no_key.close()

