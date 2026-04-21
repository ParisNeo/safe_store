# tests/test_store_encryption.py
import pytest
from pathlib import Path
import sqlite3
import shutil

# Assuming safe_store and its components are importable
from safe_store import SafeStore, LogLevel
from safe_store.core.exceptions import ConfigurationError, EncryptionError
from safe_store.security.encryption import Encryptor

# --- Fixtures ---

@pytest.fixture
def encryption_password() -> str:
    """Provides a standard password for encryption tests."""
    return "my-secret-rag-key-42"

@pytest.fixture
def encrypted_store(tmp_path, encryption_password) -> SafeStore:
    """Creates a SafeStore instance initialized with encryption."""
    db_path = tmp_path / "encrypted_test_store.db"
    # Use DEBUG for potentially more info during test failures
    store = SafeStore(db_path, log_level=LogLevel.DEBUG, encryption_key=encryption_password)
    yield store # Provide the store to the test
    # Teardown: Close the store connection explicitly
    store.close()
    # Optional: Remove DB and lock file if needed, though tmp_path usually handles cleanup
    # db_path.unlink(missing_ok=True)
    # Path(f"{db_path}.lock").unlink(missing_ok=True)

@pytest.fixture
def unencrypted_store(tmp_path) -> SafeStore:
    """Creates a SafeStore instance initialized without encryption."""
    db_path = tmp_path / "unencrypted_test_store.db"
    store = SafeStore(db_path, log_level=LogLevel.DEBUG, encryption_key=None)
    yield store
    store.close()

@pytest.fixture
def sample_encrypt_text_file(tmp_path) -> Path:
    """Creates a sample text file for encryption tests."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    file_path = doc_dir / "encrypt_me.txt"
    # Content designed to split across chunks with size=30, overlap=5
    file_path.write_text("This text should be encrypted.\nIt has multiple lines.", encoding='utf-8')
    return file_path

# --- Test Cases ---

def test_encryptor_init_no_deps(mocker):
    """Test Encryptor raises ConfigurationError if cryptography is missing."""
    mocker.patch('safe_store.security.encryption.CRYPTOGRAPHY_AVAILABLE', False)
    with pytest.raises(ConfigurationError, match="Encryption features require 'cryptography'"):
        Encryptor("any_password")

def test_encryptor_init_with_password(encryption_password):
    """Test Encryptor initializes with a valid password."""
    try:
        encryptor = Encryptor(encryption_password)
        assert encryptor.is_enabled
    except Exception as e:
        pytest.fail(f"Encryptor init failed with valid password: {e}")

def test_encryptor_init_no_password():
    """Test Encryptor initializes correctly without a password."""
    encryptor = Encryptor(None)
    assert not encryptor.is_enabled

@pytest.mark.parametrize("invalid_password", ["", 123, None])
def test_encryptor_init_invalid_password(invalid_password):
    """Test Encryptor raises error on invalid password types."""
    # Encryptor handles None specifically, so test others here
    if invalid_password is None:
         encryptor = Encryptor(None)
         assert not encryptor.is_enabled
    else:
        with pytest.raises((ValueError, TypeError)): # Catch relevant init errors
            Encryptor(invalid_password)

def test_encryptor_encrypt_decrypt(encryption_password):
    """Test basic encrypt/decrypt round trip."""
    encryptor = Encryptor(encryption_password)
    original_text = "Secret message here! £$%^&*()"
    encrypted = encryptor.encrypt(original_text)
    assert isinstance(encrypted, bytes)
    assert encrypted != original_text.encode('utf-8')

    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == original_text

def test_encryptor_decrypt_wrong_key(encryption_password):
    """Test decryption fails with the wrong key."""
    encryptor1 = Encryptor(encryption_password)
    encryptor2 = Encryptor("a-different-password")
    original_text = "Top secret data"

    encrypted = encryptor1.encrypt(original_text)

    with pytest.raises(EncryptionError, match="Invalid token"):
        encryptor2.decrypt(encrypted)

def test_encryptor_decrypt_tampered(encryption_password):
    """Test decryption fails if the token is tampered with."""
    encryptor = Encryptor(encryption_password)
    original_text = "Cannot touch this"
    encrypted = encryptor.encrypt(original_text)

    # Tamper with the encrypted data (e.g., flip a bit)
    tampered_list = list(encrypted)
    tampered_list[-1] = (tampered_list[-1] + 1) % 256
    tampered_bytes = bytes(tampered_list)
    assert tampered_bytes != encrypted # Ensure tampering occurred

    with pytest.raises(EncryptionError, match="Invalid token"):
        encryptor.decrypt(tampered_bytes)

def test_encryptor_no_password_raises(encryption_password):
    """Test that encrypt/decrypt raise error if encryptor was initialized without a password."""
    encryptor_no_key = Encryptor(None)
    encryptor_with_key = Encryptor(encryption_password)
    data = "some data"
    encrypted_data = encryptor_with_key.encrypt(data)

    with pytest.raises(EncryptionError, match="Encryption is not enabled"):
        encryptor_no_key.encrypt(data)

    with pytest.raises(EncryptionError, match="Decryption is not enabled"):
        encryptor_no_key.decrypt(encrypted_data)

# --- SafeStore Integration Tests ---

def test_init_with_encryption(tmp_path, encryption_password):
    """Test SafeStore initialization with a valid encryption key."""
    db_path = tmp_path / "init_enc_test.db"
    try:
        store = SafeStore(db_path, encryption_key=encryption_password)
        assert store.encryptor is not None
        assert store.encryptor.is_enabled
        store.close()
    except Exception as e:
        pytest.fail(f"SafeStore init with encryption key failed: {e}")

def test_init_without_encryption(tmp_path):
    """Test SafeStore initialization without an encryption key."""
    db_path = tmp_path / "init_no_enc_test.db"
    try:
        store = SafeStore(db_path, encryption_key=None)
        assert store.encryptor is not None
        assert not store.encryptor.is_enabled
        store.close()
    except Exception as e:
        pytest.fail(f"SafeStore init without encryption key failed: {e}")

def test_init_encryption_no_deps(tmp_path, encryption_password, mocker):
    """Test SafeStore init fails if encryption requested but deps missing."""
    mocker.patch('safe_store.security.encryption.CRYPTOGRAPHY_AVAILABLE', False)
    db_path = tmp_path / "init_enc_fail_deps.db"
    with pytest.raises(ConfigurationError, match="Encryption features require 'cryptography'"):
        SafeStore(db_path, encryption_key=encryption_password)

# --- Tests Adjusted for Chunk Overlap ---

def test_add_document_encrypts_chunks(encrypted_store: SafeStore, sample_encrypt_text_file: Path, encryption_password: str):
    """Test that chunk text is encrypted when adding a document."""
    store = encrypted_store
    chunk1_text = "This text should be encrypted." 
    chunk2_expected_text = "pted.\nIt has multiple lines."

    with store:
        store.add_document(
            sample_encrypt_text_file, 
            chunk_size=30, 
            chunk_overlap=5, 
            chunking_strategy='character',
            vectorize_with_metadata=False
        )

    # --- Verify DB state directly ---
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (str(sample_encrypt_text_file.resolve()),))
    doc_id_res = cursor.fetchone()
    assert doc_id_res is not None
    doc_id = doc_id_res[0]

    # Fetch chunk_text as bytes, and is_encrypted flag
    conn.text_factory = bytes # Ensure TEXT columns are read as bytes if they contain BLOBs
    cursor.execute("SELECT chunk_text, is_encrypted FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
    chunks_data = cursor.fetchall()
    conn.close() # Close connection after fetching

    assert len(chunks_data) >= 2 # Expect at least two chunks for this text/size
    # Verify first chunk encryption
    assert isinstance(chunks_data[0][0], bytes), "Chunk 0 text should be bytes"
    assert chunks_data[0][1] == 1, "Chunk 0 is_encrypted flag should be 1"

    # --- Verify decryption works ---
    verifier_encryptor = Encryptor(encryption_password)
    try:
        decrypted_chunk1 = verifier_encryptor.decrypt(chunks_data[0][0])
        assert decrypted_chunk1 == chunk1_text, f"Decrypted chunk 0 mismatch. Got: '{decrypted_chunk1}' Expected: '{chunk1_text}'"
    except Exception as e:
        pytest.fail(f"Decryption failed for chunk 0: {e}. Stored data: {chunks_data[0][0]!r}")

    # Check second chunk
    assert isinstance(chunks_data[1][0], bytes), "Chunk 1 text should be bytes"
    assert chunks_data[1][1] == 1, "Chunk 1 is_encrypted flag should be 1"
    try:
        decrypted_chunk2 = verifier_encryptor.decrypt(chunks_data[1][0])
        # Modify assertion: Check if the expected overlapped content matches the actual decrypted content
        assert decrypted_chunk2 == chunk2_expected_text, f"Decrypted chunk 1 mismatch. Got: '{decrypted_chunk2}' Expected: '{chunk2_expected_text}'"
    except Exception as e:
        # Keep the original assertion in the fail message for context if it fails
        pytest.fail(f"Decryption failed for chunk 1 OR assertion failed. Decrypted='{decrypted_chunk2}', Expected='{chunk2_expected_text}'. Original Error: {e}. Stored data: {chunks_data[1][0]!r}")

def test_query_decrypts_chunks(encrypted_store: SafeStore, sample_encrypt_text_file: Path):
    """Test that query results contain decrypted chunk text."""
    store = encrypted_store
    query_text = "multiple lines"
    # Adjust expected result based on character strategy
    expected_full_chunk_text = "pted.\nIt has multiple lines."

    with store:
        store.add_document(
            sample_encrypt_text_file, 
            chunk_size=30, 
            chunk_overlap=5, 
            chunking_strategy='character',
            vectorize_with_metadata=False
        )
        results = store.query(query_text, top_k=1)

    assert len(results) == 1, "Query should return one result"
    result_chunk = results[0]
    assert "chunk_text" in result_chunk
    # Modify assertion: Check the full expected content of the chunk
    assert result_chunk["chunk_text"] == expected_full_chunk_text, f"Query result chunk text mismatch. Got: '{result_chunk['chunk_text']}' Expected: '{expected_full_chunk_text}'"

# --- Remaining Tests (Should Pass with previous fixes) ---

def test_query_no_key_placeholder(encrypted_store: SafeStore, sample_encrypt_text_file: Path):
    """Test query returns placeholder if store lacks the key."""
    store1 = encrypted_store
    db_file = store1.db_path

    # Add encrypted data
    with store1:
        store1.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

    # New store without key
    store2 = SafeStore(db_file, log_level=LogLevel.DEBUG, encryption_key=None)

    with store2:
        results = store2.query("multiple lines", top_k=1)
        assert len(results) == 1
        assert results[0]['chunk_text'] == "[Encrypted Chunk - Key Unavailable]"

def test_query_wrong_key_placeholder(encrypted_store: SafeStore, sample_encrypt_text_file: Path):
    """Test query returns placeholder if store has the wrong key."""
    store1 = encrypted_store
    db_file = store1.db_path

    # Add encrypted data
    with store1:
        store1.add_document(sample_encrypt_text_file, chunk_size=30, chunk_overlap=5)

    # New store with wrong key
    store2 = SafeStore(db_file, log_level=LogLevel.DEBUG, encryption_key="this-is-definitely-wrong")

    with store2:
        results = store2.query("multiple lines", top_k=1)
        assert len(results) == 1
        assert results[0]['chunk_text'] == "[Encrypted Chunk - Decryption Failed]"

