# safe_store/store.py
import sqlite3
import json
from pathlib import Path
import hashlib
import threading
from typing import Optional, List, Dict, Any, Tuple, Union, Literal

from filelock import FileLock, Timeout
import numpy as np

from .core import db
from .security.encryption import Encryptor
from .core.exceptions import (
    DatabaseError,
    FileHandlingError,
    ParsingError,
    ConfigurationError,
    VectorizationError,
    QueryError,
    ConcurrencyError,
    SafeStoreError,
    EncryptionError,
)
from .indexing import parser, chunking
from .search import similarity
from .vectorization.manager import VectorizationManager
from .vectorization.methods.tfidf import TfidfVectorizerWrapper
from ascii_colors import ASCIIColors, LogLevel


# Default lock timeout in seconds
DEFAULT_LOCK_TIMEOUT: int = 60

class SafeStore:
    """
    Manages a local vector store backed by an SQLite database.

    Provides functionalities for indexing documents (parsing, chunking,
    vectorizing), managing multiple vectorization methods, querying based on
    semantic similarity, deleting documents, and handling concurrent access
    safely using file locks. Includes optional encryption for chunk text.

    Designed for simplicity and efficiency in RAG pipelines.

    Attributes:
        db_path (str): The resolved absolute path to the SQLite database file.
        lock_path (str): The path to the file lock used for concurrency control.
        lock_timeout (int): The maximum time in seconds to wait for the lock.
        vectorizer_manager (VectorizationManager): Manages vectorizer instances.
        conn (Optional[sqlite3.Connection]): The active SQLite database connection.
        encryptor (Encryptor): Handles encryption/decryption if a key is provided.
    """
    DEFAULT_VECTORIZER: str = "st:all-MiniLM-L6-v2"

    def __init__(
        self,
        db_path: Union[str, Path] = "safe_store.db",
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        encryption_key: Optional[str] = None
    ):
        """
        Initializes the safe_store instance.

        Connects to the database (creating it if necessary), initializes the
        schema, sets up logging, and prepares concurrency controls.

        Args:
            db_path: Path to the SQLite database file. Defaults to "safe_store.db"
                     in the current working directory.
            log_level: Minimum log level for console output via `ascii_colors`.
                       Defaults to `LogLevel.INFO`.
            lock_timeout: Timeout in seconds for acquiring the inter-process
                          write lock. Defaults to 60 seconds. Set to 0 or
                          negative for non-blocking.
            encryption_key: Optional password used to derive an encryption key
                            for encrypting chunk text at rest using AES-128-CBC.
                            If provided, `cryptography` must be installed.
                            **IMPORTANT:** You are responsible for securely managing
                            this key. If lost, encrypted data is unrecoverable.

        Raises:
            DatabaseError: If the database connection or schema initialization fails.
            ConcurrencyError: If acquiring the initial lock for setup times out.
            ConfigurationError: If `encryption_key` is provided but `cryptography`
                                is not installed.
            ValueError: If `encryption_key` is provided but is empty.
        """
        self.db_path: str = str(Path(db_path).resolve())
        self.lock_timeout: int = lock_timeout
        _db_file_path = Path(self.db_path)
        self.lock_path: str = str(_db_file_path.parent / f"{_db_file_path.name}.lock")

        ASCIIColors.set_log_level(log_level)
        ASCIIColors.info(f"Initializing SafeStore with database: {self.db_path}")
        ASCIIColors.debug(f"Using lock file: {self.lock_path} with timeout: {self.lock_timeout}s")

        self.conn: Optional[sqlite3.Connection] = None
        self._is_closed: bool = True

        self.vectorizer_manager = VectorizationManager()
        self._file_hasher = hashlib.sha256

        try:
            self.encryptor = Encryptor(encryption_key)
            if self.encryptor.is_enabled:
                 ASCIIColors.info("Encryption enabled for chunk text.")
        except (ConfigurationError, ValueError) as e:
             ASCIIColors.critical(f"Encryptor initialization failed: {e}")
             raise e

        self._instance_lock = threading.RLock()
        self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)

        try:
            self._connect_and_initialize()
        except (DatabaseError, Timeout, ConcurrencyError) as e:
            ASCIIColors.critical(f"SafeStore initialization failed: {e}")
            raise

    def _connect_and_initialize(self) -> None:
        """Establishes the database connection and initializes the schema."""
        init_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        try:
            with init_lock:
                ASCIIColors.debug("Acquired init lock for connection/schema setup.")
                if self.conn is None or self._is_closed:
                     self.conn = db.connect_db(self.db_path)
                     db.initialize_schema(self.conn)
                     self._is_closed = False
                else:
                     ASCIIColors.debug("Connection already established.")
            ASCIIColors.debug("Released init lock.")
        except Timeout as e:
            msg = f"Timeout acquiring initial lock for DB connection/setup at {self.lock_path}"
            ASCIIColors.error(msg)
            if self.conn:
                 try: self.conn.close()
                 except Exception: pass
                 finally: self.conn = None; self._is_closed = True
            raise ConcurrencyError(msg) from e
        except DatabaseError as e:
            ASCIIColors.error(f"Database error during initial setup: {e}")
            if self.conn:
                 try: self.conn.close()
                 except Exception: pass
                 finally: self.conn = None; self._is_closed = True
            raise
        except Exception as e:
            msg = f"Unexpected error during initial DB connection/setup: {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn:
                 try: self.conn.close()
                 except Exception: pass
                 finally: self.conn = None; self._is_closed = True
            raise SafeStoreError(msg) from e

    def close(self) -> None:
        """Closes the database connection and clears the vectorizer cache."""
        with self._instance_lock:
            if self._is_closed:
                 ASCIIColors.debug("Connection already closed.")
                 return
            if self.conn:
                ASCIIColors.debug("Closing database connection.")
                try: self.conn.close()
                except Exception as e: ASCIIColors.warning(f"Error closing DB connection: {e}")
                finally: self.conn = None; self._is_closed = True

            if hasattr(self, 'vectorizer_manager'):
                self.vectorizer_manager.clear_cache()
            ASCIIColors.info("SafeStore connection closed.")

    def __enter__(self):
        """Enter the runtime context related to this object."""
        with self._instance_lock:
            if self._is_closed or self.conn is None:
                ASCIIColors.debug("Re-establishing connection on context manager entry.")
                try: self._connect_and_initialize()
                except (DatabaseError, ConcurrencyError, SafeStoreError) as e:
                    ASCIIColors.error(f"Failed to re-establish connection in __enter__: {e}")
                    raise
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        self.close()
        if exc_type: ASCIIColors.error(f"SafeStore context closed with error: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
        else: ASCIIColors.debug("SafeStore context closed cleanly.")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generates a SHA256 hash for the file content."""
        try:
            hasher = self._file_hasher()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192): hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError as e:
            msg = f"File not found when trying to hash: {file_path}"
            ASCIIColors.error(msg)
            raise FileHandlingError(msg) from e
        except OSError as e:
            msg = f"OS error reading file for hashing {file_path}: {e}"
            ASCIIColors.error(msg)
            raise FileHandlingError(msg) from e
        except Exception as e:
            msg = f"Unexpected error generating hash for {file_path}: {e}"
            ASCIIColors.warning(msg)
            raise FileHandlingError(msg) from e

    def _get_text_hash(self, text: Path) -> str:
        """Generates a SHA256 hash for the file content."""
        try:
            hasher = self._file_hasher()
            while chunk := text.encode("utf8"): hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError as e:
            msg = f"File not found when trying to hash: {file_path}"
            ASCIIColors.error(msg)
            raise FileHandlingError(msg) from e
        except OSError as e:
            msg = f"OS error reading file for hashing {file_path}: {e}"
            ASCIIColors.error(msg)
            raise FileHandlingError(msg) from e
        except Exception as e:
            msg = f"Unexpected error generating hash for {file_path}: {e}"
            ASCIIColors.warning(msg)
            raise FileHandlingError(msg) from e

    def _ensure_connection(self) -> None:
        """Checks if the connection is active, raises ConnectionError if not."""
        if self._is_closed or self.conn is None:
            raise ConnectionError("Database connection is closed or not available.")

    # === Write methods requiring locking ===

    def add_document(
        self,
        file_path: Union[str, Path],
        vectorizer_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        vectorizer_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Adds or updates a document in the SafeStore.

        Handles parsing, chunking, optional encryption, vectorization, and storage.
        Detects file changes via hash and re-indexes automatically. Skips if
        unchanged and vectors exist. Acquires an exclusive write lock.

        Args:
            file_path: Path to the document file.
            vectorizer_name: Vectorizer to use (e.g., 'st:model', 'tfidf:name'). Defaults to `DEFAULT_VECTORIZER`.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks. Must be less than `chunk_size`.
            metadata: Optional JSON-serializable metadata dictionary.
            force_reindex: If True, re-process even if hash matches.
            vectorizer_params: Optional parameters for vectorizer initialization (e.g., TF-IDF).

        Raises:
            ValueError: If chunk parameters are invalid.
            FileHandlingError: For file read/hash errors or file not found.
            ParsingError: If document parsing fails.
            ConfigurationError: For missing dependencies or unsupported types.
            VectorizationError: If vector generation fails.
            DatabaseError: For database interaction errors.
            ConcurrencyError: If write lock times out.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
            EncryptionError: If encryption operations fail.
        """
        _file_path = Path(file_path)
        if chunk_overlap >= chunk_size:
             raise ValueError("chunk_overlap must be smaller than chunk_size")

        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for add_document: {_file_path.name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for add_document: {_file_path.name}")
                    self._ensure_connection()
                    self._add_document_impl(
                        _file_path, vectorizer_name, chunk_size, chunk_overlap,
                        metadata, force_reindex, vectorizer_params
                    )
                ASCIIColors.debug(f"Write lock released for add_document: {_file_path.name}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for add_document: {_file_path.name}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, FileHandlingError, ParsingError, ConfigurationError,
                    VectorizationError, EncryptionError, QueryError,
                    ValueError, ConnectionError, SafeStoreError) as e:
                ASCIIColors.error(f"Error during add_document: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during add_document (lock scope) for '{_file_path.name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _add_document_impl(
        self,
        file_path: Path,
        vectorizer_name: Optional[str],
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]],
        force_reindex: bool,
        vectorizer_params: Optional[Dict[str, Any]]
    ) -> None:
        """Internal implementation of add_document logic."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        abs_file_path = str(file_path.resolve())

        ASCIIColors.info(f"Starting indexing process for: {file_path.name}")
        ASCIIColors.debug(f"Params: vectorizer='{_vectorizer_name}', chunk_size={chunk_size}, overlap={chunk_overlap}, force={force_reindex}, encryption={'enabled' if self.encryptor.is_enabled else 'disabled'}")

        try: current_hash = self._get_file_hash(file_path)
        except FileHandlingError as e: raise e

        existing_doc_id: Optional[int] = None
        existing_hash: Optional[str] = None
        needs_parsing_chunking = True
        needs_vectorization = True

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (abs_file_path,))
            result = cursor.fetchone()
            if result:
                existing_doc_id, existing_hash = result
                ASCIIColors.debug(f"Document '{file_path.name}' found in DB (doc_id={existing_doc_id}). Hash: {existing_hash}/{current_hash}")

                if force_reindex:
                    ASCIIColors.warning(f"Force re-indexing requested for '{file_path.name}'.")
                    cursor.execute("BEGIN")
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    self.conn.commit()
                    ASCIIColors.debug(f"Deleted old chunks/vectors for forced re-index of doc_id={existing_doc_id}.")
                elif existing_hash == current_hash:
                    ASCIIColors.info(f"Document '{file_path.name}' is unchanged.")
                    needs_parsing_chunking = False
                    try: _, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn, vectorizer_params)
                    except (ConfigurationError, VectorizationError, DatabaseError) as e: raise SafeStoreError(f"Failed to get vectorizer info for existence check: {e}") from e
                    cursor.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? AND v.method_id = ? LIMIT 1", (existing_doc_id, method_id))
                    if cursor.fetchone() is not None:
                        ASCIIColors.success(f"Vectorization '{_vectorizer_name}' already exists for unchanged '{file_path.name}'. Skipping.")
                        needs_vectorization = False
                    else:
                         ASCIIColors.info(f"Document '{file_path.name}' exists and is unchanged, but needs vectorization '{_vectorizer_name}'.")
                else:
                    ASCIIColors.warning(f"Document '{file_path.name}' has changed (hash mismatch). Re-indexing...")
                    cursor.execute("BEGIN")
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    self.conn.commit()
                    ASCIIColors.debug(f"Deleted old chunks/vectors for changed doc_id={existing_doc_id}.")
            else:
                 ASCIIColors.info(f"Document '{file_path.name}' is new.")

        except (sqlite3.Error, DatabaseError) as e:
            msg = f"Database error checking/updating document state for '{file_path.name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            raise DatabaseError(msg) from e
        except SafeStoreError as e: raise e
        except Exception as e:
             msg = f"Unexpected error preparing indexing for '{file_path.name}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             raise SafeStoreError(msg) from e

        if not needs_parsing_chunking and not needs_vectorization: return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            doc_id: Optional[int] = existing_doc_id
            full_text: Optional[str] = None
            chunks_data: List[Tuple[str, int, int]] = []
            chunk_ids: List[int] = []
            chunk_texts_for_vectorization: List[str] = []

            if needs_parsing_chunking:
                ASCIIColors.debug(f"Parsing document: {file_path.name}")
                try: full_text = parser.parse_document(file_path)
                except (ParsingError, FileHandlingError, ConfigurationError, ValueError) as e: raise e
                except Exception as e: raise ParsingError(f"Unexpected error parsing {file_path.name}: {e}") from e
                ASCIIColors.debug(f"Parsed document '{file_path.name}'. Length: {len(full_text)} chars.")
                metadata_str = json.dumps(metadata) if metadata else None

                if doc_id is None:
                    doc_id = db.add_document_record(self.conn, abs_file_path, full_text, current_hash, metadata_str)
                else:
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?", (current_hash, full_text, metadata_str, doc_id))

                try: chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                except ValueError as e: raise e

                if not chunks_data:
                    ASCIIColors.warning(f"No chunks generated for {file_path.name}. Document record saved, but skipping vectorization.")
                    self.conn.commit(); return

                ASCIIColors.info(f"Generated {len(chunks_data)} chunks for '{file_path.name}'. Storing chunks...")
                should_encrypt = self.encryptor.is_enabled
                logged_encryption_status = False

                for i, (text, start, end) in enumerate(chunks_data):
                    text_to_store: Union[str, bytes] = text
                    is_encrypted_flag = False
                    encrypted_metadata = None
                    if should_encrypt:
                        try:
                            encrypted_token = self.encryptor.encrypt(text)
                            text_to_store = encrypted_token
                            is_encrypted_flag = True
                            if not logged_encryption_status: ASCIIColors.debug("Encrypting chunk text."); logged_encryption_status = True
                        except EncryptionError as e: raise e

                    chunk_id = db.add_chunk_record(self.conn, doc_id, text_to_store, start, end, i, is_encrypted=is_encrypted_flag, encryption_metadata=encrypted_metadata)
                    chunk_ids.append(chunk_id)
                    chunk_texts_for_vectorization.append(text)
            else:
                if doc_id is None: raise SafeStoreError(f"Inconsistent state: doc_id is None but parsing/chunking was skipped for {file_path.name}")
                ASCIIColors.debug(f"Retrieving existing chunks for doc_id={doc_id} to add new vectors...")
                cursor.execute("SELECT c.chunk_id, c.chunk_text, c.is_encrypted FROM chunks c WHERE c.doc_id = ? ORDER BY c.chunk_seq", (doc_id,))
                results = cursor.fetchall()

                if not results:
                      ASCIIColors.warning(f"Document {doc_id} ('{file_path.name}') exists but has no stored chunks. Cannot add vectorization '{_vectorizer_name}'.")
                      needs_vectorization = False
                else:
                    ASCIIColors.debug(f"Processing {len(results)} existing chunks for vectorization (decrypting if needed)...")
                    logged_decryption_status = False
                    for chunk_id_db, text_data_db, is_encrypted_flag_db in results:
                        chunk_ids.append(chunk_id_db)
                        text_for_vec: str
                        if is_encrypted_flag_db:
                            if self.encryptor.is_enabled:
                                try:
                                    if not isinstance(text_data_db, bytes): raise TypeError(f"Chunk {chunk_id_db} marked encrypted but data is not bytes.")
                                    text_for_vec = self.encryptor.decrypt(text_data_db)
                                    if not logged_decryption_status: ASCIIColors.debug("Decrypting existing chunk text for vectorization."); logged_decryption_status = True
                                except (EncryptionError, TypeError) as e: raise e
                            else: raise ConfigurationError(f"Cannot get text for vectorization: Chunk {chunk_id_db} is encrypted, but no encryption key provided.")
                        else:
                             if not isinstance(text_data_db, str):
                                  ASCIIColors.warning(f"Chunk {chunk_id_db} not marked encrypted, but data is not string. Attempting decode.")
                                  try: text_for_vec = text_data_db.decode('utf-8') if isinstance(text_data_db, bytes) else str(text_data_db)
                                  except Exception: text_for_vec = str(text_data_db)
                             else: text_for_vec = text_data_db
                        chunk_texts_for_vectorization.append(text_for_vec)
                    ASCIIColors.debug(f"Retrieved {len(chunk_ids)} existing chunk IDs and obtained text for vectorization.")

            if needs_vectorization:
                if not chunk_ids or not chunk_texts_for_vectorization:
                     ASCIIColors.warning(f"No valid chunk text available to vectorize for '{file_path.name}'. Skipping vectorization.")
                     self.conn.commit(); return

                try: vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn, vectorizer_params)
                except (ConfigurationError, VectorizationError, DatabaseError) as e: raise e

                if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                     ASCIIColors.warning(f"TF-IDF vectorizer '{_vectorizer_name}' is not fitted. Fitting ONLY on chunks from '{file_path.name}'.")
                     try:
                         vectorizer.fit(chunk_texts_for_vectorization)
                         new_params = vectorizer.get_params_to_store()
                         self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                         ASCIIColors.debug(f"TF-IDF '{_vectorizer_name}' fitted on document chunks and params updated in DB.")
                     except (VectorizationError, DatabaseError) as e: raise e
                     except Exception as e: raise VectorizationError(f"Failed to fit TF-IDF model '{_vectorizer_name}' on '{file_path.name}': {e}") from e

                ASCIIColors.info(f"Vectorizing {len(chunk_texts_for_vectorization)} chunks using '{_vectorizer_name}' (method_id={method_id})...")
                try: vectors = vectorizer.vectorize(chunk_texts_for_vectorization)
                except VectorizationError as e: raise e
                except Exception as e: raise VectorizationError(f"Unexpected error during vectorization with '{_vectorizer_name}': {e}") from e

                if vectors.shape[0] != len(chunk_ids): raise VectorizationError(f"Mismatch between chunks ({len(chunk_ids)}) and vectors ({vectors.shape[0]}) for '{file_path.name}'!")

                ASCIIColors.debug(f"Adding {len(vectors)} vector records to DB (method_id={method_id})...")
                for chunk_id_vec, vector_data in zip(chunk_ids, vectors):
                    vector_contiguous = np.ascontiguousarray(vector_data, dtype=vectorizer.dtype)
                    db.add_vector_record(self.conn, chunk_id_vec, method_id, vector_contiguous)

            self.conn.commit()
            ASCIIColors.success(f"Successfully processed '{file_path.name}' with vectorizer '{_vectorizer_name}'.")

        except (sqlite3.Error, DatabaseError, FileHandlingError, ParsingError, ConfigurationError, VectorizationError, EncryptionError, ValueError, SafeStoreError) as e:
            ASCIIColors.error(f"Error during indexing transaction for '{file_path.name}': {e.__class__.__name__}: {e}", exc_info=False)
            if self.conn: self.conn.rollback()
            ASCIIColors.debug("Transaction rolled back due to error.")
            raise
        except Exception as e:
            msg = f"Unexpected error during indexing transaction for '{file_path.name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            ASCIIColors.debug("Transaction rolled back due to unexpected error.")
            raise SafeStoreError(msg) from e


    def add_text(
        self,
        unique_id:str,
        text: str,
        vectorizer_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        vectorizer_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Adds or updates a text content in the SafeStore using a unique ID.

        Handles chunking, optional encryption, vectorization, and storage.
        Detects text changes via hash and re-indexes automatically. Skips if
        unchanged and vectors exist. Acquires an exclusive write lock.

        Args:
            unique_id: A unique identifier for the text content. This will be stored
                       where file_path is stored for documents.
            text: The text content to add.
            vectorizer_name: Vectorizer to use (e.g., 'st:model', 'tfidf:name'). Defaults to `DEFAULT_VECTORIZER`.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks. Must be less than `chunk_size`.
            metadata: Optional JSON-serializable metadata dictionary.
            force_reindex: If True, re-process even if hash matches.
            vectorizer_params: Optional parameters for vectorizer initialization (e.g., TF-IDF).

        Raises:
            ValueError: If chunk parameters are invalid or unique_id/text is empty/None.
            FileHandlingError: For file-like operation errors (less direct, but possible from underlying components).
            ParsingError: If any component unexpectedly requires parsing (less direct now that text is pre-parsed).
            ConfigurationError: For missing dependencies or unsupported types.
            VectorizationError: If vector generation fails.
            DatabaseError: For database interaction errors.
            ConcurrencyError: If write lock times out.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors (e.g., text hashing failure).
            EncryptionError: If encryption operations fail.
        """
        if not unique_id:
            raise ValueError("unique_id cannot be empty.")
        if text is None: # Check for None; empty string might be permissible.
            raise ValueError("text content cannot be None.")
        if chunk_overlap >= chunk_size:
             raise ValueError("chunk_overlap must be smaller than chunk_size")

        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for add_text: {unique_id}")
            try:
                self._ensure_connection()
                self._add_text_impl(
                    unique_id, text, vectorizer_name, chunk_size, chunk_overlap,
                    metadata, force_reindex, vectorizer_params
                )
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for add_text: {unique_id}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, FileHandlingError, ParsingError, ConfigurationError,
                    VectorizationError, EncryptionError, QueryError,
                    ValueError, ConnectionError, SafeStoreError) as e:
                # Kept FileHandlingError & ParsingError for consistency with error hierarchy,
                # though their direct causes from file I/O or parsing are removed for add_text.
                ASCIIColors.error(f"Error during add_text for ID '{unique_id}': {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during add_text (lock scope) for ID '{unique_id}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _add_text_impl(
        self,
        unique_id: str,
        text_content: str,
        vectorizer_name: Optional[str],
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]],
        force_reindex: bool,
        vectorizer_params: Optional[Dict[str, Any]]
    ) -> None:
        """Internal implementation of add_text logic."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        # `unique_id` serves as the replacement for `abs_file_path`

        ASCIIColors.info(f"Starting indexing process for text ID: {unique_id}")
        ASCIIColors.debug(f"Params: vectorizer='{_vectorizer_name}', chunk_size={chunk_size}, overlap={chunk_overlap}, force={force_reindex}, encryption={'enabled' if self.encryptor.is_enabled else 'disabled'}")

        try:
            current_hash = self._get_text_hash(text_content)
        except SafeStoreError as e: # Raised by _get_text_hash for its internal errors
            ASCIIColors.error(f"Failed to compute hash for text ID '{unique_id}': {e}", exc_info=True)
            raise # Re-raise to be caught by the main try-except in add_text

        existing_doc_id: Optional[int] = None
        existing_hash: Optional[str] = None
        needs_content_processing_chunking = True # Renamed from needs_parsing_chunking
        needs_vectorization = True

        try:
            cursor = self.conn.cursor()
            # Use unique_id in the 'file_path' column for queries
            cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (unique_id,))
            result = cursor.fetchone()
            if result:
                existing_doc_id, existing_hash = result
                ASCIIColors.debug(f"Text ID '{unique_id}' found in DB (doc_id={existing_doc_id}). Hash: {existing_hash}/{current_hash}")

                if force_reindex:
                    ASCIIColors.warning(f"Force re-indexing requested for text ID '{unique_id}'.")
                    cursor.execute("BEGIN")
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    self.conn.commit() # Commit this delete operation
                    ASCIIColors.debug(f"Deleted old chunks/vectors for forced re-index of doc_id={existing_doc_id}.")
                elif existing_hash == current_hash:
                    ASCIIColors.info(f"Text ID '{unique_id}' is unchanged.")
                    needs_content_processing_chunking = False
                    try: _, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn, vectorizer_params)
                    except (ConfigurationError, VectorizationError, DatabaseError) as e: raise SafeStoreError(f"Failed to get vectorizer info for existence check: {e}") from e
                    cursor.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? AND v.method_id = ? LIMIT 1", (existing_doc_id, method_id))
                    if cursor.fetchone() is not None:
                        ASCIIColors.success(f"Vectorization '{_vectorizer_name}' already exists for unchanged text ID '{unique_id}'. Skipping.")
                        needs_vectorization = False
                    else:
                         ASCIIColors.info(f"Text ID '{unique_id}' exists and is unchanged, but needs vectorization '{_vectorizer_name}'.")
                else: # Hash mismatch
                    ASCIIColors.warning(f"Text ID '{unique_id}' has changed (hash mismatch). Re-indexing...")
                    cursor.execute("BEGIN")
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    self.conn.commit() # Commit this delete operation
                    ASCIIColors.debug(f"Deleted old chunks/vectors for changed doc_id={existing_doc_id}.")
            else: # New unique_id
                 ASCIIColors.info(f"Text ID '{unique_id}' is new.")

        except (sqlite3.Error, DatabaseError) as e:
            msg = f"Database error checking/updating document state for text ID '{unique_id}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback() # Rollback if transaction was started and failed
            raise DatabaseError(msg) from e
        except SafeStoreError as e: raise e # Propagate SafeStoreError (e.g., from get_vectorizer)
        except Exception as e: # Catch any other unexpected errors during this stage
             msg = f"Unexpected error preparing indexing for text ID '{unique_id}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             # Consider rollback if a transaction might have been started by DB implicitly
             # if self.conn and self.conn.in_transaction: self.conn.rollback()
             raise SafeStoreError(msg) from e

        if not needs_content_processing_chunking and not needs_vectorization:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN") # Start main transaction for additions/updates
            doc_id: Optional[int] = existing_doc_id
            full_text: str # Will be assigned from text_content
            chunks_data: List[Tuple[str, int, int]] = []
            chunk_ids: List[int] = []
            chunk_texts_for_vectorization: List[str] = []

            if needs_content_processing_chunking:
                ASCIIColors.debug(f"Processing provided text for ID: {unique_id}")
                full_text = text_content # Text is provided directly
                # No parsing step like parser.parse_document(file_path)
                ASCIIColors.debug(f"Using text for ID '{unique_id}'. Length: {len(full_text)} chars.")
                metadata_str = json.dumps(metadata) if metadata else None

                if doc_id is None: # New text ID
                    # Store unique_id in the file_path column
                    doc_id = db.add_document_record(self.conn, unique_id, full_text, current_hash, metadata_str)
                else: # Existing text ID, content changed or forced reindex
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?", (current_hash, full_text, metadata_str, doc_id))

                try:
                    chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                except ValueError as e: # Raised by chunk_text for invalid params
                    raise e

                if not chunks_data:
                    ASCIIColors.warning(f"No chunks generated for text ID '{unique_id}'. Document record saved, but skipping vectorization.")
                    self.conn.commit() # Commit document record changes
                    return

                ASCIIColors.info(f"Generated {len(chunks_data)} chunks for text ID '{unique_id}'. Storing chunks...")
                should_encrypt = self.encryptor.is_enabled
                logged_encryption_status = False

                for i, (text_chunk_content, start, end) in enumerate(chunks_data): # Renamed 'text' to 'text_chunk_content'
                    text_to_store: Union[str, bytes] = text_chunk_content
                    is_encrypted_flag = False
                    encrypted_metadata = None # This was None in original too
                    if should_encrypt:
                        try:
                            encrypted_token = self.encryptor.encrypt(text_chunk_content)
                            text_to_store = encrypted_token
                            is_encrypted_flag = True
                            if not logged_encryption_status: ASCIIColors.debug("Encrypting chunk text."); logged_encryption_status = True
                        except EncryptionError as e: raise e

                    chunk_id = db.add_chunk_record(self.conn, doc_id, text_to_store, start, end, i, is_encrypted=is_encrypted_flag, encryption_metadata=encrypted_metadata)
                    chunk_ids.append(chunk_id)
                    chunk_texts_for_vectorization.append(text_chunk_content) # Original text for vectorization
            else: # Content unchanged, but needs new vectorization
                if doc_id is None:
                    raise SafeStoreError(f"Inconsistent state: doc_id is None but content processing/chunking was skipped for text ID '{unique_id}'")
                ASCIIColors.debug(f"Retrieving existing chunks for doc_id={doc_id} (text ID: '{unique_id}') to add new vectors...")
                cursor.execute("SELECT c.chunk_id, c.chunk_text, c.is_encrypted FROM chunks c WHERE c.doc_id = ? ORDER BY c.chunk_seq", (doc_id,))
                results = cursor.fetchall()

                if not results:
                      ASCIIColors.warning(f"Text ID {doc_id} ('{unique_id}') exists but has no stored chunks. Cannot add vectorization '{_vectorizer_name}'.")
                      needs_vectorization = False # Cannot proceed if no chunks
                else:
                    ASCIIColors.debug(f"Processing {len(results)} existing chunks for vectorization (decrypting if needed)...")
                    logged_decryption_status = False
                    for chunk_id_db, text_data_db, is_encrypted_flag_db in results:
                        chunk_ids.append(chunk_id_db)
                        text_for_vec: str
                        if is_encrypted_flag_db:
                            if self.encryptor.is_enabled:
                                try:
                                    if not isinstance(text_data_db, bytes): raise TypeError(f"Chunk {chunk_id_db} (text ID '{unique_id}') marked encrypted but data is not bytes.")
                                    text_for_vec = self.encryptor.decrypt(text_data_db)
                                    if not logged_decryption_status: ASCIIColors.debug("Decrypting existing chunk text for vectorization."); logged_decryption_status = True
                                except (EncryptionError, TypeError) as e: raise e
                            else: raise ConfigurationError(f"Cannot get text for vectorization: Chunk {chunk_id_db} (text ID '{unique_id}') is encrypted, but no encryption key provided.")
                        else: # Not encrypted
                             if not isinstance(text_data_db, str):
                                  ASCIIColors.warning(f"Chunk {chunk_id_db} (text ID '{unique_id}') not marked encrypted, but data is not string. Attempting decode.")
                                  try: text_for_vec = text_data_db.decode('utf-8') if isinstance(text_data_db, bytes) else str(text_data_db)
                                  except Exception: text_for_vec = str(text_data_db) # Best effort
                             else: text_for_vec = text_data_db
                        chunk_texts_for_vectorization.append(text_for_vec)
                    ASCIIColors.debug(f"Retrieved {len(chunk_ids)} existing chunk IDs and obtained text for vectorization for text ID '{unique_id}'.")

            if needs_vectorization:
                if not chunk_ids or not chunk_texts_for_vectorization:
                     ASCIIColors.warning(f"No valid chunk text available to vectorize for text ID '{unique_id}'. Skipping vectorization.")
                     self.conn.commit() # Commit any document/chunk records made so far
                     return

                try: vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn, vectorizer_params)
                except (ConfigurationError, VectorizationError, DatabaseError) as e: raise e

                if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                     ASCIIColors.warning(f"TF-IDF vectorizer '{_vectorizer_name}' is not fitted. Fitting ONLY on chunks from text ID '{unique_id}'.")
                     try:
                         vectorizer.fit(chunk_texts_for_vectorization)
                         new_params = vectorizer.get_params_to_store()
                         self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                         ASCIIColors.debug(f"TF-IDF '{_vectorizer_name}' fitted on text ID '{unique_id}' chunks and params updated in DB.")
                     except (VectorizationError, DatabaseError) as e: raise e
                     except Exception as e: raise VectorizationError(f"Failed to fit TF-IDF model '{_vectorizer_name}' on text ID '{unique_id}': {e}") from e

                ASCIIColors.info(f"Vectorizing {len(chunk_texts_for_vectorization)} chunks using '{_vectorizer_name}' (method_id={method_id}) for text ID '{unique_id}'...")
                try: vectors = vectorizer.vectorize(chunk_texts_for_vectorization)
                except VectorizationError as e: raise e
                except Exception as e: raise VectorizationError(f"Unexpected error during vectorization with '{_vectorizer_name}' for text ID '{unique_id}': {e}") from e

                if vectors.shape[0] != len(chunk_ids):
                    raise VectorizationError(f"Mismatch between chunks ({len(chunk_ids)}) and vectors ({vectors.shape[0]}) for text ID '{unique_id}'!")

                ASCIIColors.debug(f"Adding {len(vectors)} vector records to DB (method_id={method_id}) for text ID '{unique_id}'...")
                for chunk_id_vec, vector_data in zip(chunk_ids, vectors):
                    vector_contiguous = np.ascontiguousarray(vector_data, dtype=vectorizer.dtype)
                    db.add_vector_record(self.conn, chunk_id_vec, method_id, vector_contiguous)

            self.conn.commit() # Commit the main transaction
            ASCIIColors.success(f"Successfully processed text ID '{unique_id}' with vectorizer '{_vectorizer_name}'.")

        except (sqlite3.Error, DatabaseError, FileHandlingError, ParsingError, ConfigurationError, VectorizationError, EncryptionError, ValueError, SafeStoreError) as e:
            ASCIIColors.error(f"Error during indexing transaction for text ID '{unique_id}': {e.__class__.__name__}: {e}", exc_info=False)
            if self.conn: self.conn.rollback()
            ASCIIColors.debug(f"Transaction for text ID '{unique_id}' rolled back due to error.")
            raise
        except Exception as e:
            msg = f"Unexpected error during indexing transaction for text ID '{unique_id}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            ASCIIColors.debug(f"Transaction for text ID '{unique_id}' rolled back due to unexpected error.")
            raise SafeStoreError(msg) from e

    
    def add_vectorization(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[Union[str, Path]] = None,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        batch_size: int = 64
    ) -> None:
        """
        Adds vector embeddings using a specified method to existing documents.

        Fits TF-IDF if needed (potentially decrypting text). Processes in batches.
        Acquires an exclusive write lock.

        Args:
            vectorizer_name: Vectorizer to add (e.g., 'st:new-model', 'tfidf:variant').
            target_doc_path: If specified, only adds vectors for this document. Otherwise all.
            vectorizer_params: Optional parameters for vectorizer init (mainly TF-IDF).
            batch_size: Number of chunks to process per batch.

        Raises:
            FileHandlingError: If target_doc_path not found in DB.
            ConfigurationError: For missing vectorizer dependencies or issues.
            VectorizationError: If vector generation or fitting fails.
            DatabaseError: For database interaction errors.
            ConcurrencyError: If write lock times out.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
            EncryptionError: If required decryption fails.
        """
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for add_vectorization: {vectorizer_name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for add_vectorization: {vectorizer_name}")
                    self._ensure_connection()
                    self._add_vectorization_impl(vectorizer_name, target_doc_path, vectorizer_params, batch_size)
                ASCIIColors.debug(f"Write lock released for add_vectorization: {vectorizer_name}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for add_vectorization: {vectorizer_name}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, FileHandlingError, ConfigurationError, VectorizationError, EncryptionError, QueryError, ValueError, ConnectionError, SafeStoreError) as e:
                ASCIIColors.error(f"Error during add_vectorization: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during add_vectorization (lock scope) for '{vectorizer_name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _add_vectorization_impl(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[Union[str, Path]],
        vectorizer_params: Optional[Dict[str, Any]],
        batch_size: int
    ) -> None:
        """Internal implementation of add_vectorization."""
        assert self.conn is not None
        ASCIIColors.info(f"Starting process to add vectorization '{vectorizer_name}'.")
        resolved_target_doc_path: Optional[str] = None
        target_doc_id: Optional[int] = None

        if target_doc_path:
             resolved_target_doc_path = str(Path(target_doc_path).resolve())
             ASCIIColors.info(f"Targeting specific document: {resolved_target_doc_path}")
             cursor_check = self.conn.cursor()
             try:
                 cursor_check.execute("SELECT doc_id FROM documents WHERE file_path = ?", (resolved_target_doc_path,))
                 target_doc_id_result = cursor_check.fetchone()
                 if not target_doc_id_result: raise FileHandlingError(f"Target document '{resolved_target_doc_path}' not found in the database.")
                 target_doc_id = target_doc_id_result[0]
                 ASCIIColors.debug(f"Target document ID resolved: {target_doc_id}")
             except sqlite3.Error as e: raise DatabaseError(f"Database error resolving target document ID for '{resolved_target_doc_path}': {e}") from e
        else: ASCIIColors.info("Targeting all documents in the store.")

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, self.conn, vectorizer_params)

            if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' requires fitting.")
                fit_sql_base = "SELECT c.chunk_text, c.is_encrypted FROM chunks c"
                fit_params_list: List[Any] = []
                if target_doc_id is not None: fit_sql = fit_sql_base + " WHERE c.doc_id = ?"; fit_params_list.append(target_doc_id)
                else: fit_sql = fit_sql_base
                ASCIIColors.info(f"Fetching chunks for fitting {'from doc ' + str(target_doc_id) if target_doc_id else 'all docs'}...")
                cursor.execute(fit_sql, tuple(fit_params_list))
                texts_to_fit_raw = cursor.fetchall()
                if not texts_to_fit_raw: ASCIIColors.warning("No text chunks found to fit the TF-IDF model."); self.conn.commit(); return

                texts_to_fit: List[str] = []
                ASCIIColors.debug(f"Processing {len(texts_to_fit_raw)} chunks for TF-IDF fitting (decrypting if needed)...")
                logged_decryption_status_fit = False
                for text_data, is_encrypted_flag in texts_to_fit_raw:
                    text_for_fit: str
                    if is_encrypted_flag:
                        if self.encryptor.is_enabled:
                            try:
                                if not isinstance(text_data, bytes): raise TypeError("Chunk marked encrypted but data is not bytes.")
                                text_for_fit = self.encryptor.decrypt(text_data)
                                if not logged_decryption_status_fit: ASCIIColors.debug("Decrypting chunk text for TF-IDF fitting."); logged_decryption_status_fit = True
                            except (EncryptionError, TypeError) as e: raise EncryptionError(f"Failed to decrypt chunk for fitting: {e}") from e
                        else: raise ConfigurationError("Cannot fit TF-IDF on encrypted chunks without the correct encryption key.")
                    else:
                        if not isinstance(text_data, str):
                             ASCIIColors.warning(f"Chunk not marked encrypted, but data is not string. Attempting decode.")
                             try: text_for_fit = text_data.decode('utf-8') if isinstance(text_data, bytes) else str(text_data)
                             except Exception: text_for_fit = str(text_data)
                        else: text_for_fit = text_data
                    texts_to_fit.append(text_for_fit)

                try:
                    vectorizer.fit(texts_to_fit)
                    new_params = vectorizer.get_params_to_store()
                    self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                    ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' fitted successfully using {len(texts_to_fit)} chunks.")
                except (VectorizationError, DatabaseError) as e: raise e
                except Exception as e: raise VectorizationError(f"Failed to fit TF-IDF model '{vectorizer_name}': {e}") from e

            chunks_to_vectorize_sql_base = "SELECT c.chunk_id, c.chunk_text, c.is_encrypted FROM chunks c LEFT JOIN vectors v ON c.chunk_id = v.chunk_id AND v.method_id = ? WHERE v.vector_id IS NULL"
            sql_params: List[Any] = [method_id]
            if target_doc_id is not None: chunks_to_vectorize_sql = chunks_to_vectorize_sql_base + " AND c.doc_id = ?"; sql_params.append(target_doc_id)
            else: chunks_to_vectorize_sql = chunks_to_vectorize_sql_base
            ASCIIColors.info(f"Fetching chunks missing '{vectorizer_name}' vectors{' for doc ' + str(target_doc_id) if target_doc_id else ''}...")
            cursor.execute(chunks_to_vectorize_sql, tuple(sql_params))
            chunks_data_raw = cursor.fetchall()

            if not chunks_data_raw: ASCIIColors.success(f"No chunks found needing vectorization for '{vectorizer_name}'."); self.conn.commit(); return
            total_chunks = len(chunks_data_raw)
            ASCIIColors.info(f"Found {total_chunks} chunks to vectorize.")

            num_added = 0
            try:
                logged_decryption_status_vec = False
                for i in range(0, total_chunks, batch_size):
                    batch_raw = chunks_data_raw[i : i + batch_size]
                    batch_ids = [item[0] for item in batch_raw]
                    batch_texts: List[str] = []
                    ASCIIColors.debug(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_raw)} chunks)...")
                    for _, text_data, is_encrypted_flag in batch_raw:
                        text_for_vec: str
                        if is_encrypted_flag:
                            if self.encryptor.is_enabled:
                                try:
                                    if not isinstance(text_data, bytes): raise TypeError("Chunk marked encrypted but data is not bytes.")
                                    text_for_vec = self.encryptor.decrypt(text_data)
                                    if not logged_decryption_status_vec: ASCIIColors.debug("Decrypting chunk text for vectorization batch."); logged_decryption_status_vec = True
                                except (EncryptionError, TypeError) as e: raise EncryptionError(f"Failed to decrypt chunk for vectorization: {e}") from e
                            else: raise ConfigurationError("Cannot vectorize encrypted chunks without the correct encryption key.")
                        else:
                            if not isinstance(text_data, str):
                                 ASCIIColors.warning(f"Chunk not marked encrypted, but data is not string. Attempting decode.")
                                 try: text_for_vec = text_data.decode('utf-8') if isinstance(text_data, bytes) else str(text_data)
                                 except Exception: text_for_vec = str(text_data)
                            else: text_for_vec = text_data
                        batch_texts.append(text_for_vec)

                    try:
                         vectors = vectorizer.vectorize(batch_texts)
                         if vectors.shape[0] != len(batch_ids): raise VectorizationError(f"Vectorization output count ({vectors.shape[0]}) mismatch batch size ({len(batch_ids)}).")
                    except VectorizationError as e: raise e
                    except Exception as e: raise VectorizationError(f"Unexpected error during vectorization batch for '{vectorizer_name}': {e}") from e

                    for chunk_id_vec, vector_data in zip(batch_ids, vectors):
                         vector_contiguous = np.ascontiguousarray(vector_data, dtype=vectorizer.dtype)
                         db.add_vector_record(self.conn, chunk_id_vec, method_id, vector_contiguous)
                    num_added += len(batch_ids)
                    ASCIIColors.debug(f"Added {len(batch_ids)} vectors for batch.")
            except (sqlite3.Error, DatabaseError, VectorizationError, EncryptionError) as e: raise e
            except Exception as e: raise SafeStoreError(f"Unexpected error during vectorization batch processing for '{vectorizer_name}': {e}") from e

            self.conn.commit()
            ASCIIColors.success(f"Successfully added {num_added} vector embeddings using '{vectorizer_name}'.")

        except (sqlite3.Error, DatabaseError, FileHandlingError, ConfigurationError, VectorizationError, EncryptionError, SafeStoreError) as e:
             ASCIIColors.error(f"Error during add_vectorization transaction: {e.__class__.__name__}: {e}", exc_info=False)
             if self.conn: self.conn.rollback()
             ASCIIColors.debug("Transaction rolled back due to error.")
             raise
        except Exception as e:
             msg = f"Unexpected error during add_vectorization transaction for '{vectorizer_name}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             if self.conn: self.conn.rollback()
             ASCIIColors.debug("Transaction rolled back due to unexpected error.")
             raise SafeStoreError(msg) from e

    def remove_vectorization(self, vectorizer_name: str) -> None:
        """
        Removes a vectorization method and its associated vectors.

        Acquires an exclusive write lock.

        Args:
            vectorizer_name: The name of the vectorization method to remove.

        Raises:
            DatabaseError: For database interaction errors.
            ConcurrencyError: If write lock times out.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
        """
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for remove_vectorization: {vectorizer_name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for remove_vectorization: {vectorizer_name}")
                    self._ensure_connection()
                    self._remove_vectorization_impl(vectorizer_name)
                ASCIIColors.debug(f"Write lock released for remove_vectorization: {vectorizer_name}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for remove_vectorization: {vectorizer_name}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, ConnectionError, SafeStoreError) as e:
                ASCIIColors.error(f"Error during remove_vectorization: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during remove_vectorization (lock scope) for '{vectorizer_name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _remove_vectorization_impl(self, vectorizer_name: str) -> None:
        """Internal implementation of remove_vectorization."""
        assert self.conn is not None
        ASCIIColors.warning(f"Attempting to remove vectorization method '{vectorizer_name}' and all associated vectors.")
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_name,))
            result = cursor.fetchone()
            if not result: ASCIIColors.warning(f"Vectorization method '{vectorizer_name}' not found."); return
            method_id = result[0]
            ASCIIColors.debug(f"Found method_id {method_id} for '{vectorizer_name}'.")

            cursor.execute("BEGIN")
            cursor.execute("DELETE FROM vectors WHERE method_id = ?", (method_id,))
            deleted_vectors = cursor.rowcount; ASCIIColors.debug(f"Deleted {deleted_vectors} vector records.")
            cursor.execute("DELETE FROM vectorization_methods WHERE method_id = ?", (method_id,))
            deleted_methods = cursor.rowcount; ASCIIColors.debug(f"Deleted {deleted_methods} vectorization method record.")
            self.conn.commit()

            self.vectorizer_manager.remove_from_cache_by_id(method_id)
            ASCIIColors.success(f"Successfully removed vectorization method '{vectorizer_name}' (ID: {method_id}) and {deleted_vectors} associated vectors.")
        except sqlite3.Error as e:
             msg = f"Database error during removal of '{vectorizer_name}': {e}"
             ASCIIColors.error(msg, exc_info=True); self.conn.rollback(); raise DatabaseError(msg) from e
        except Exception as e:
             msg = f"Unexpected error during removal of '{vectorizer_name}': {e}"
             ASCIIColors.error(msg, exc_info=True); self.conn.rollback(); raise SafeStoreError(msg) from e

    def delete_document_by_id(self, doc_id: int) -> None:
        """
        Deletes a document and all its associated chunks and vectors by its ID.

        Uses `ON DELETE CASCADE` defined in the schema. Acquires an exclusive write lock.

        Args:
            doc_id: The integer ID of the document to delete.

        Raises:
            DatabaseError: For database interaction errors.
            ConcurrencyError: If write lock times out.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
        """
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for delete_document_by_id: {doc_id}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for delete_document_by_id: {doc_id}")
                    self._ensure_connection()
                    self._delete_document_by_id_impl(doc_id)
                ASCIIColors.debug(f"Write lock released for delete_document_by_id: {doc_id}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for delete_document_by_id: {doc_id}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, ConnectionError, SafeStoreError) as e:
                ASCIIColors.error(f"Error during delete_document_by_id: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during delete_document_by_id (lock scope) for ID {doc_id}: {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _delete_document_by_id_impl(self, doc_id: int) -> None:
        """Internal implementation of deleting a document by ID."""
        assert self.conn is not None
        ASCIIColors.warning(f"Attempting to delete document with ID: {doc_id}")
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            rows_affected = cursor.rowcount
            self.conn.commit()

            if rows_affected > 0:
                ASCIIColors.success(f"Successfully deleted document ID {doc_id} (and associated chunks/vectors via CASCADE).")
            else:
                ASCIIColors.warning(f"Document with ID {doc_id} not found. Nothing deleted.")
        except sqlite3.Error as e:
            msg = f"Database error during deletion of document ID {doc_id}: {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            raise DatabaseError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during deletion of document ID {doc_id}: {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            raise SafeStoreError(msg) from e

    def delete_document_by_path(self, file_path: Union[str, Path]) -> None:
        """
        Deletes a document and all its associated data by its file path.

        Finds the document ID based on the path, then deletes it.
        Acquires an exclusive write lock.

        Args:
            file_path: The file path of the document to delete.

        Raises:
            DatabaseError: For database interaction errors.
            ConcurrencyError: If write lock times out.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
        """
        _file_path = Path(file_path)
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for delete_document_by_path: {_file_path.name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for delete_document_by_path: {_file_path.name}")
                    self._ensure_connection()
                    self._delete_document_by_path_impl(_file_path)
                ASCIIColors.debug(f"Write lock released for delete_document_by_path: {_file_path.name}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for delete_document_by_path: {_file_path.name}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, ConnectionError, SafeStoreError, FileHandlingError) as e: # Added FileHandlingError
                ASCIIColors.error(f"Error during delete_document_by_path: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during delete_document_by_path (lock scope) for path '{_file_path.name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _delete_document_by_path_impl(self, file_path: Path) -> None:
        """Internal implementation of deleting a document by path."""
        assert self.conn is not None
        abs_file_path = str(file_path.resolve())
        ASCIIColors.warning(f"Attempting to delete document by path: {abs_file_path}")
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (abs_file_path,))
            result = cursor.fetchone()
            if result:
                doc_id = result[0]
                ASCIIColors.debug(f"Found document ID {doc_id} for path '{abs_file_path}'. Proceeding with deletion.")
                # Reuse the by-ID implementation within the same transaction/lock context
                self._delete_document_by_id_impl(doc_id)
            else:
                ASCIIColors.warning(f"Document with path '{abs_file_path}' not found. Nothing deleted.")
                # Raise FileHandlingError if not found? Or just log? Let's just log warning for now.
                # raise FileHandlingError(f"Document with path '{abs_file_path}' not found.")

        except sqlite3.Error as e:
            msg = f"Database error finding/deleting document by path '{abs_file_path}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            # Rollback might be handled by _delete_document_by_id_impl if called, otherwise handle here
            if self.conn: self.conn.rollback()
            raise DatabaseError(msg) from e
        except Exception as e:
            msg = f"Unexpected error finding/deleting document by path '{abs_file_path}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            raise SafeStoreError(msg) from e


    # === Read methods ===

    def query(
        self,
        query_text: str,
        vectorizer_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries the store for chunks semantically similar to the query text.

        Uses the specified vectorizer and cosine similarity. This is primarily a
        read operation and uses an instance-level lock for thread safety.

        Args:
            query_text: The text to search for.
            vectorizer_name: The vectorization method name. Defaults to `DEFAULT_VECTORIZER`.
            top_k: Maximum number of results to return.

        Returns:
            A list of dictionaries, each representing a relevant chunk:
            - 'chunk_id': (int) ID of the chunk.
            - 'chunk_text': (str) Text content (decrypted if applicable).
            - 'similarity': (float) Cosine similarity score (-1.0 to 1.0).
            - 'similarity_percent': (float) Similarity score scaled to 0-100.
            - 'doc_id': (int) ID of the source document.
            - 'file_path': (str) Path to the source document file.
            - 'start_pos': (int) Start character offset in the original document.
            - 'end_pos': (int) End character offset in the original document.
            - 'chunk_seq': (int) Sequence number of the chunk within the document.
            - 'metadata': (dict | None) Metadata associated with the document.

        Raises:
            ConfigurationError: If vectorizer dependencies are missing.
            VectorizationError: If query vectorization fails.
            DatabaseError: If fetching data fails.
            QueryError: For similarity calculation or logic errors.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
            EncryptionError: If result decryption fails.
        """
        with self._instance_lock:
            self._ensure_connection()
            try: return self._query_impl(query_text, vectorizer_name, top_k)
            except (DatabaseError, ConfigurationError, VectorizationError, QueryError, EncryptionError, ValueError, ConnectionError, SafeStoreError) as e:
                ASCIIColors.error(f"Error during query: {e.__class__.__name__}: {e}", exc_info=False); raise
            except Exception as e:
                msg = f"Unexpected error during query for '{query_text[:50]}...': {e}"; ASCIIColors.error(msg, exc_info=True); raise SafeStoreError(msg) from e

    def _query_impl(
        self,
        query_text: str,
        vectorizer_name: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Internal implementation of query logic."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        ASCIIColors.info(f"Received query. Searching with '{_vectorizer_name}', top_k={top_k}.")
        cursor = self.conn.cursor()
        try:
            try: vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn, None)
            except (ConfigurationError, VectorizationError, DatabaseError) as e: raise e
            ASCIIColors.debug(f"Using vectorizer '{_vectorizer_name}' (method_id={method_id})")
            ASCIIColors.debug("Vectorizing query text...")
            try:
                 query_vector_list = vectorizer.vectorize([query_text])
                 if not isinstance(query_vector_list, np.ndarray) or query_vector_list.ndim != 2 or query_vector_list.shape[0] != 1: raise VectorizationError("Vectorizer did not return a single 2D vector for the query.")
                 query_vector = np.ascontiguousarray(query_vector_list[0], dtype=vectorizer.dtype)
            except VectorizationError as e: raise e
            except Exception as e: raise VectorizationError(f"Unexpected error vectorizing query text with '{_vectorizer_name}': {e}") from e
            ASCIIColors.debug(f"Query vector generated. Shape: {query_vector.shape}, Dtype: {query_vector.dtype}")

            ASCIIColors.debug(f"Loading all vectors for method_id {method_id}...")
            try:
                cursor.execute("SELECT v.chunk_id, v.vector_data FROM vectors v WHERE v.method_id = ?", (method_id,))
                all_vectors_data = cursor.fetchall()
            except sqlite3.Error as e: raise DatabaseError(f"Database error loading vectors for method '{_vectorizer_name}' (ID: {method_id}): {e}") from e
            if not all_vectors_data: ASCIIColors.warning(f"No vectors found for method '{_vectorizer_name}'."); return []
            chunk_ids_ordered: List[int] = [row[0] for row in all_vectors_data]
            vector_blobs: List[bytes] = [row[1] for row in all_vectors_data]

            method_details = self.vectorizer_manager._get_method_details_from_db(self.conn, _vectorizer_name)
            if not method_details: raise DatabaseError(f"Could not retrieve method details for '{_vectorizer_name}'.")
            vector_dtype_str = method_details['vector_dtype']
            vector_dim_expected = method_details['vector_dim']
            ASCIIColors.debug(f"Reconstructing {len(vector_blobs)} vectors from BLOBs with dtype '{vector_dtype_str}'...")
            try:
                 candidate_vectors_list = [db.reconstruct_vector(blob, vector_dtype_str) for blob in vector_blobs]
                 if not candidate_vectors_list: candidate_vectors = np.empty((0, vector_dim_expected or 0), dtype=np.dtype(vector_dtype_str))
                 else: candidate_vectors = np.stack(candidate_vectors_list, axis=0)
            except (DatabaseError, ValueError, TypeError) as e: raise QueryError(f"Failed to reconstruct one or more vectors for method '{_vectorizer_name}': {e}") from e
            ASCIIColors.debug(f"Candidate vectors loaded. Matrix shape: {candidate_vectors.shape}")

            ASCIIColors.debug("Calculating similarity scores...")
            try:
                if candidate_vectors.shape[0] == 0: scores = np.array([], dtype=query_vector.dtype)
                else: scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            except (ValueError, TypeError) as e: raise QueryError(f"Error calculating cosine similarity: {e}") from e
            ASCIIColors.debug(f"Similarity scores calculated. Shape: {scores.shape}")

            num_candidates = len(scores); k = min(top_k, num_candidates) if top_k > 0 else 0
            if k <= 0: ASCIIColors.info("Top-k is 0 or no candidates found."); return []
            if k < num_candidates // 2 : top_k_indices_unsorted = np.argpartition(scores, -k)[-k:]; top_k_indices = top_k_indices_unsorted[np.argsort(scores[top_k_indices_unsorted])[::-1]]
            else: top_k_indices = np.argsort(scores)[::-1][:k]
            ASCIIColors.debug(f"Identified top {k} indices.")

            top_chunk_ids = [chunk_ids_ordered[i] for i in top_k_indices]; top_scores = [scores[i] for i in top_k_indices]
            if not top_chunk_ids: return []

            placeholders = ','.join('?' * len(top_chunk_ids))
            sql_chunk_details = f"SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.chunk_seq, c.is_encrypted, d.doc_id, d.file_path, d.metadata FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE c.chunk_id IN ({placeholders})"
            try:
                original_text_factory = self.conn.text_factory
                self.conn.text_factory = bytes
                cursor.execute(sql_chunk_details, top_chunk_ids)
                chunk_details_list_raw = cursor.fetchall()
                self.conn.text_factory = original_text_factory
            except sqlite3.Error as e: self.conn.text_factory = original_text_factory; raise DatabaseError(f"Database error fetching chunk details for top-k results: {e}") from e

            chunk_details_map: Dict[int, Dict[str, Any]] = {}
            ASCIIColors.debug(f"Processing {len(chunk_details_list_raw)} chunk details (decrypting if needed)...")
            logged_decryption_status_query = False

            for row in chunk_details_list_raw:
                 chunk_id, chunk_text_data, start_pos, end_pos, chunk_seq, is_encrypted_flag, doc_id, file_path_bytes, metadata_json_bytes = row
                 chunk_text_final: str; file_path = file_path_bytes.decode('utf-8') if isinstance(file_path_bytes, bytes) else file_path_bytes; metadata_json = metadata_json_bytes.decode('utf-8') if isinstance(metadata_json_bytes, bytes) else metadata_json_bytes

                 if is_encrypted_flag:
                      if self.encryptor.is_enabled:
                           try:
                                if not isinstance(chunk_text_data, bytes): chunk_text_final = "[Encrypted - Decryption Failed: Invalid Type]"; ASCIIColors.error(f"Cannot decrypt chunk {chunk_id}: data type {type(chunk_text_data)}.")
                                else: chunk_text_final = self.encryptor.decrypt(chunk_text_data);
                                if not logged_decryption_status_query: ASCIIColors.debug("Decrypting result chunk text."); logged_decryption_status_query = True
                           except EncryptionError as e: chunk_text_final = "[Encrypted - Decryption Failed]"; ASCIIColors.error(f"Failed to decrypt result chunk {chunk_id}: {e}")
                      else: chunk_text_final = "[Encrypted - Key Unavailable]"; ASCIIColors.warning(f"Chunk {chunk_id} is encrypted, but no key provided.")
                 else:
                      if isinstance(chunk_text_data, bytes):
                           ASCIIColors.debug(f"Chunk {chunk_id} not marked encrypted, but read as bytes. Decoding.")
                           try: chunk_text_final = chunk_text_data.decode('utf-8')
                           except UnicodeDecodeError: chunk_text_final = "[Data Decode Error]"; ASCIIColors.error(f"Failed to decode non-encrypted bytes for chunk {chunk_id}.")
                      elif isinstance(chunk_text_data, str): chunk_text_final = chunk_text_data
                      else: chunk_text_final = str(chunk_text_data); ASCIIColors.warning(f"Chunk {chunk_id} data type unexpected ({type(chunk_text_data)}).")

                 metadata_dict = None
                 if metadata_json:
                      try: metadata_dict = json.loads(metadata_json)
                      except json.JSONDecodeError: metadata_dict = {"error": "invalid JSON"}; ASCIIColors.warning(f"Failed to decode metadata JSON for chunk {chunk_id}")
                 chunk_details_map[chunk_id] = {"chunk_id": chunk_id, "chunk_text": chunk_text_final, "start_pos": start_pos, "end_pos": end_pos, "chunk_seq": chunk_seq, "doc_id": doc_id, "file_path": file_path, "metadata": metadata_dict}

            results: List[Dict[str, Any]] = []
            for chunk_id_res, score_res in zip(top_chunk_ids, top_scores):
                if chunk_id_res in chunk_details_map:
                    result_item = chunk_details_map[chunk_id_res].copy()
                    similarity_value = float(np.float64(score_res))
                    result_item["similarity"] = similarity_value
                    result_item["similarity_percent"] = round(((similarity_value + 1) / 2) * 100, 2) # Scale -1..1 to 0..100
                    results.append(result_item)
                else: ASCIIColors.warning(f"Could not find details for chunk_id {chunk_id_res}. Skipping.")

            ASCIIColors.success(f"Query successful. Found {len(results)} relevant chunks.")
            return results

        except (DatabaseError, ConfigurationError, VectorizationError, QueryError, EncryptionError, ValueError, SafeStoreError) as e: raise e
        except Exception as e: raise SafeStoreError(f"Unexpected error during query implementation for '{query_text[:50]}...': {e}") from e

    def query_all(
        self,
        query_text: str,
        top_k: int = 5,
        mode: Literal['union', 'intersection'] = 'union'
    ) -> List[Dict[str, Any]]:
        """
        Queries the store using *all* available vectorization methods.

        Combines results based on the specified mode ('union' or 'intersection').

        Args:
            query_text: The text to search for.
            top_k: The maximum number of results *per vectorizer* to consider before combining.
            mode: How to combine results:
                  - 'union': Returns unique chunks found by any method, ranked by the highest
                             score achieved by any method finding that chunk. (Default)
                  - 'intersection': Returns only chunks found by *all* queried methods, ranked
                                    by the average score across those methods.

        Returns:
            A list of dictionaries, similar to `query`, but with an additional field:
            - 'found_by_methods': (List[str]) List of vectorizer names that retrieved this chunk.
            - 'similarity': (float) Max score in 'union' mode, Average score in 'intersection' mode.
            - 'similarity_percent': (float) Corresponding percentage score.

        Raises:
            ValueError: If the mode is invalid.
            ConfigurationError: If vectorizer dependencies are missing.
            VectorizationError: If query vectorization fails for any method.
            DatabaseError: If fetching data fails.
            QueryError: For similarity calculation or logic errors.
            ConnectionError: If database connection is closed.
            SafeStoreError: For other unexpected errors.
            EncryptionError: If result decryption fails.
        """
        if mode not in ['union', 'intersection']:
            raise ValueError("Invalid mode specified. Must be 'union' or 'intersection'.")

        with self._instance_lock:
            self._ensure_connection()
            try:
                return self._query_all_impl(query_text, top_k, mode)
            except (DatabaseError, ConfigurationError, VectorizationError, QueryError,
                    EncryptionError, ValueError, ConnectionError, SafeStoreError) as e:
                ASCIIColors.error(f"Error during query_all: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during query_all for '{query_text[:50]}...': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e

    def _query_all_impl(
        self,
        query_text: str,
        top_k: int,
        mode: Literal['union', 'intersection']
    ) -> List[Dict[str, Any]]:
        """Internal implementation of query_all logic."""
        assert self.conn is not None
        ASCIIColors.info(f"Received query_all (mode={mode}). Searching across all methods, top_k={top_k} per method.")

        all_methods = self.list_vectorization_methods()
        if not all_methods:
            ASCIIColors.warning("No vectorization methods found in the database. Cannot perform query_all.")
            return []

        method_names = [m['method_name'] for m in all_methods]
        ASCIIColors.debug(f"Querying across methods: {method_names}")

        # Store results per chunk_id
        # For union: {chunk_id: {'max_score': float, 'details': dict, 'methods': set}}
        # For intersection: {chunk_id: {'scores': list, 'details': dict, 'methods': set}}
        combined_results: Dict[int, Dict[str, Any]] = {}
        successful_method_count = 0

        for method_name in method_names:
            try:
                ASCIIColors.debug(f"Querying with method: {method_name}")
                # Use _query_impl which handles vectorization, similarity, details, decryption
                method_results = self._query_impl(query_text, method_name, top_k)
                successful_method_count += 1

                for res in method_results:
                    chunk_id = res['chunk_id']
                    score = res['similarity']

                    if mode == 'union':
                        if chunk_id not in combined_results:
                            combined_results[chunk_id] = {'max_score': score, 'details': res, 'methods': {method_name}}
                        else:
                            # Update if this score is higher, always add method name
                            if score > combined_results[chunk_id]['max_score']:
                                combined_results[chunk_id]['max_score'] = score
                                combined_results[chunk_id]['details'] = res # Update details if score is higher
                            combined_results[chunk_id]['methods'].add(method_name)
                    elif mode == 'intersection':
                        if chunk_id not in combined_results:
                            combined_results[chunk_id] = {'scores': [score], 'details': res, 'methods': {method_name}}
                        else:
                            combined_results[chunk_id]['scores'].append(score)
                            combined_results[chunk_id]['methods'].add(method_name)

            except (DatabaseError, ConfigurationError, VectorizationError, QueryError, EncryptionError) as e:
                ASCIIColors.warning(f"Skipping method '{method_name}' in query_all due to error: {e}")
            except Exception as e:
                ASCIIColors.warning(f"Skipping method '{method_name}' in query_all due to unexpected error: {e}", exc_info=True)

        if successful_method_count == 0:
             ASCIIColors.error("query_all failed: No vectorization methods could be successfully queried.")
             return []

        # Final processing based on mode
        final_results: List[Dict[str, Any]] = []
        if mode == 'union':
            for chunk_id, data in combined_results.items():
                details = data['details']
                final_score = data['max_score']
                details['similarity'] = final_score # Overwrite with max score
                details['similarity_percent'] = round(((final_score + 1) / 2) * 100, 2)
                details['found_by_methods'] = sorted(list(data['methods']))
                final_results.append(details)
            # Sort by the max score (descending)
            final_results.sort(key=lambda x: x['similarity'], reverse=True)

        elif mode == 'intersection':
            for chunk_id, data in combined_results.items():
                # Only include chunks found by all successfully queried methods
                if len(data['methods']) == successful_method_count:
                    details = data['details']
                    avg_score = sum(data['scores']) / len(data['scores'])
                    details['similarity'] = avg_score # Use average score
                    details['similarity_percent'] = round(((avg_score + 1) / 2) * 100, 2)
                    details['found_by_methods'] = sorted(list(data['methods']))
                    final_results.append(details)
            # Sort by the average score (descending)
            final_results.sort(key=lambda x: x['similarity'], reverse=True)

        ASCIIColors.success(f"query_all ({mode}) successful. Found {len(final_results)} combined results across {successful_method_count} methods.")
        # Note: Final list length might be more or less than top_k depending on overlap and mode
        return final_results


    # --- Helper Methods ---
    def list_documents(self) -> List[Dict[str, Any]]:
         """Lists all documents currently stored in the database."""
         with self._instance_lock:
              self._ensure_connection(); assert self.conn is not None; cursor = self.conn.cursor()
              try:
                   cursor.execute("SELECT doc_id, file_path, file_hash, added_timestamp, metadata FROM documents ORDER BY added_timestamp")
                   docs = []
                   for row in cursor.fetchall():
                        metadata_dict = None
                        if row[4]:
                            try: 
                                metadata_dict = json.loads(row[4])
                            except json.JSONDecodeError: 
                                pass
                        docs.append({"doc_id": row[0], "file_path": row[1], "file_hash": row[2], "added_timestamp": row[3], "metadata": metadata_dict})
                   return docs
              except sqlite3.Error as e: raise DatabaseError(f"Database error listing documents: {e}") from e

    def list_vectorization_methods(self) -> List[Dict[str, Any]]:
         """Lists all registered vectorization methods."""
         with self._instance_lock:
              self._ensure_connection(); assert self.conn is not None; cursor = self.conn.cursor()
              try:
                   cursor.execute("SELECT method_id, method_name, method_type, vector_dim, vector_dtype, params FROM vectorization_methods ORDER BY method_name")
                   methods = []
                   for row in cursor.fetchall():
                        params_dict = None
                        if row[5]: 
                            try: 
                                params_dict = json.loads(row[5]) 
                            except json.JSONDecodeError:
                                pass
                        methods.append({"method_id": row[0], "method_name": row[1], "method_type": row[2], "vector_dim": row[3], "vector_dtype": row[4], "params": params_dict})
                   return methods
              except sqlite3.Error as e: raise DatabaseError(f"Database error listing vectorization methods: {e}") from e

    @staticmethod
    def list_possible_vectorizer_names() -> List[str]:
        """
        Provides example and common vectorizer names. See documentation for details.
        - 'st:...' : Use any model from huggingface.co/models?library=sentence-transformers
        - 'tfidf:<your_custom_name>' : Fitted on your data during add/vectorize.
        """
        return [
            "st:all-MiniLM-L6-v2", "st:all-mpnet-base-v2", "st:multi-qa-MiniLM-L6-cos-v1",
            "st:paraphrase-multilingual-MiniLM-L12-v2", "st:sentence-t5-base",
            "tfidf:<your_custom_name> (e.g., tfidf:my_project_tfidf)"
        ]
