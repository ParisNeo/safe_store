    # safe_store/security/encryption.py
import base64
from typing import Optional, Tuple

from ascii_colors import ASCIIColors
from ..core.exceptions import EncryptionError, ConfigurationError

# Attempt import
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet, InvalidToken = None, None
    PBKDF2HMAC, hashes = None, None


SALT_SIZE = 16 # Standard salt size for PBKDF2
# Recommended PBKDF2 iterations (adjust based on security needs vs performance)
# OWASP recommendation as of 2023 is >= 600,000 for SHA256
PBKDF2_ITERATIONS = 600_000

class Encryptor:
    """
    Handles symmetric encryption and decryption using Fernet (AES-128-CBC + HMAC).

    Derives a valid Fernet key from a user-provided password using PBKDF2.
    """

    def __init__(self, password: Optional[str]):
        """
        Initializes the Encryptor.

        Args:
            password: The password to use for encryption/decryption. If None,
                      encryption/decryption methods will raise errors.

        Raises:
            ConfigurationError: If 'cryptography' is not installed.
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            msg = "Encryption features require 'cryptography'. Install with: pip install safe_store[encryption]"
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        if password is None:
            self._fernet = None
            ASCIIColors.debug("Encryptor initialized without a password. Encryption/decryption disabled.")
        else:
            if not isinstance(password, str) or not password:
                raise ValueError("Encryption password must be a non-empty string.")
            # Note: Storing the key directly is generally discouraged in production.
            # This simple implementation derives the key on init.
            # A more robust system might derive it on demand or use a dedicated key management system.
            key = self._derive_key(password)
            self._fernet = Fernet(key)
            ASCIIColors.debug("Encryptor initialized with password-derived key.")

    @staticmethod
    def _derive_key(password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derives a 32-byte key suitable for Fernet using PBKDF2HMAC-SHA256.

        A fixed salt is used here for simplicity, allowing the same password
        to always produce the same key. **WARNING:** In a real-world scenario,
        you'd typically generate a *unique* salt per encryption and store it
        alongside the ciphertext. However, for this use case (encrypting chunks
        all potentially decrypted with the same store instance/password), using a
        fixed derivative might be acceptable, though less ideal than per-chunk salts.
        Using a hardcoded salt makes it slightly less secure than generating one.
        Let's stick to a hardcoded one for simplicity of this library's scope,
        but document this limitation heavily.

        Args:
            password: The user-provided password string.
            salt: Optional salt (not used in this fixed-salt implementation).

        Returns:
            A URL-safe base64-encoded 32-byte key.
        """
        # --- !!! SECURITY WARNING !!! ---
        # Using a hardcoded salt is NOT best practice for general encryption.
        # It means the same password will always yield the same key, reducing
        # protection against rainbow table attacks compared to unique salts.
        # This is a simplification for this specific library context.
        # Consider generating and storing salts if higher security is needed.
        hardcoded_salt = b'safe_store_salt_' # 16 bytes

        if PBKDF2HMAC is None or hashes is None:
             # Should be caught by __init__ check, but defensive coding
             raise ConfigurationError("Cryptography library components missing for key derivation.")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32, # Fernet keys are 32 bytes
            salt=hardcoded_salt,
            iterations=PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        return key

    @property
    def is_enabled(self) -> bool:
        """Returns True if encryption is configured (password provided)."""
        return self._fernet is not None

    def encrypt(self, data: str) -> bytes:
        """
        Encrypts string data.

        Args:
            data: The plaintext string to encrypt.

        Returns:
            The encrypted data as bytes (Fernet token).

        Raises:
            EncryptionError: If encryption is not enabled or fails.
            TypeError: If input data is not a string.
        """
        if not self.is_enabled or self._fernet is None:
            raise EncryptionError("Encryption is not enabled (no password provided).")
        if not isinstance(data, str):
            raise TypeError("Data to encrypt must be a string.")

        try:
            encrypted_data = self._fernet.encrypt(data.encode('utf-8'))
            return encrypted_data
        except Exception as e:
            msg = f"Encryption failed: {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise EncryptionError(msg) from e

    def decrypt(self, token: bytes) -> str:
        """
        Decrypts a Fernet token back into a string.

        Args:
            token: The encrypted data (Fernet token) as bytes.

        Returns:
            The decrypted plaintext string.

        Raises:
            EncryptionError: If decryption is not enabled, the token is invalid
                             (tampered or wrong key), or decryption fails.
            TypeError: If input token is not bytes.
        """
        if not self.is_enabled or self._fernet is None:
            raise EncryptionError("Decryption is not enabled (no password provided).")
        if not isinstance(token, bytes):
            raise TypeError("Token to decrypt must be bytes.")

        try:
            decrypted_data = self._fernet.decrypt(token)
            return decrypted_data.decode('utf-8')
        except InvalidToken:
            msg = "Decryption failed: Invalid token (likely tampered or wrong key)."
            ASCIIColors.error(msg)
            raise EncryptionError(msg) from InvalidToken # Chain specific error
        except Exception as e:
            msg = f"Decryption failed: {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise EncryptionError(msg) from e