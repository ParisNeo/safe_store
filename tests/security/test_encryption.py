# tests/security/test_encryption.py
import pytest
from safe_store.security.encryption import Encryptor, CRYPTOGRAPHY_AVAILABLE
from safe_store.core.exceptions import EncryptionError, ConfigurationError

# Conditionally skip tests if cryptography is not installed
pytestmark = pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="Requires cryptography library")

@pytest.fixture
def password() -> str:
    return "test-password-123!"

@pytest.fixture
def encryptor_instance(password: str) -> Encryptor:
    return Encryptor(password)

def test_encryptor_init_no_password():
    encryptor = Encryptor(None)
    assert not encryptor.is_enabled
    with pytest.raises(EncryptionError, match="Encryption is not enabled"):
        encryptor.encrypt("test")
    with pytest.raises(EncryptionError, match="Decryption is not enabled"):
        encryptor.decrypt(b"somebytes")

def test_encryptor_init_empty_password():
    with pytest.raises(ValueError, match="non-empty string"):
        Encryptor("")

def test_encryptor_init_with_password(encryptor_instance: Encryptor):
    assert encryptor_instance.is_enabled
    assert encryptor_instance._fernet is not None

def test_derive_key_consistency(password: str):
    """Ensure the same password yields the same key (due to fixed salt)."""
    key1 = Encryptor._derive_key(password)
    key2 = Encryptor._derive_key(password)
    assert key1 == key2
    assert isinstance(key1, bytes)

def test_derive_key_different_passwords(password: str):
    key1 = Encryptor._derive_key(password)
    key2 = Encryptor._derive_key(password + "extra")
    assert key1 != key2

def test_encrypt_decrypt_success(encryptor_instance: Encryptor):
    original_data = "This is sensitive data."
    encrypted_token = encryptor_instance.encrypt(original_data)
    assert isinstance(encrypted_token, bytes)
    assert encrypted_token != original_data.encode('utf-8')

    decrypted_data = encryptor_instance.decrypt(encrypted_token)
    assert isinstance(decrypted_data, str)
    assert decrypted_data == original_data

def test_encrypt_non_string(encryptor_instance: Encryptor):
    with pytest.raises(TypeError, match="must be a string"):
        encryptor_instance.encrypt(b"bytes data") # type: ignore
    with pytest.raises(TypeError, match="must be a string"):
        encryptor_instance.encrypt(123) # type: ignore

def test_decrypt_non_bytes(encryptor_instance: Encryptor):
    with pytest.raises(TypeError, match="must be bytes"):
        encryptor_instance.decrypt("string data") # type: ignore
    with pytest.raises(TypeError, match="must be bytes"):
        encryptor_instance.decrypt(123) # type: ignore

def test_decrypt_invalid_token(encryptor_instance: Encryptor):
    invalid_token = b"not_a_valid_fernet_token"
    with pytest.raises(EncryptionError, match="Invalid token"):
        encryptor_instance.decrypt(invalid_token)

def test_decrypt_tampered_token(encryptor_instance: Encryptor):
    original_data = "Original message."
    encrypted_token = encryptor_instance.encrypt(original_data)
    # Tamper slightly (e.g., flip a bit - simplistic tamper)
    tampered_token = bytearray(encrypted_token)
    tampered_token[-1] = tampered_token[-1] ^ 1 # Flip last bit
    tampered_token_bytes = bytes(tampered_token)

    with pytest.raises(EncryptionError, match="Invalid token"):
        encryptor_instance.decrypt(tampered_token_bytes)

def test_decrypt_wrong_key(password: str):
    encryptor1 = Encryptor(password)
    encryptor2 = Encryptor(password + "_different")

    original_data = "Secret info."
    encrypted_token = encryptor1.encrypt(original_data)

    # Attempt decryption with the wrong key
    with pytest.raises(EncryptionError, match="Invalid token"):
        encryptor2.decrypt(encrypted_token)
