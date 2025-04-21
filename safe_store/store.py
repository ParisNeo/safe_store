# safe_store/store.py
import sqlite3
import json
from pathlib import Path
import hashlib
import threading
from typing import Optional, List, Dict, Any, Tuple, Union

from filelock import FileLock, Timeout

from .core import db
from .core.exceptions import ( # Import specific exceptions
    DatabaseError,
    FileHandlingError,
    ParsingError,
    ConfigurationError,
    VectorizationError,
    QueryError,
    ConcurrencyError,
    safe_storeError,
)
from .indexing import parser, chunking
from .search import similarity
from .vectorization.manager import VectorizationManager
from .vectorization.methods.tfidf import TfidfVectorizerWrapper
from ascii_colors import ASCIIColors, LogLevel
import numpy as np

# Default lock timeout in seconds
DEFAULT_LOCK_TIMEOUT: int = 60

class safe_store:
    """
    Manages a local vector store backed by an SQLite database.

    Provides functionalities for indexing documents (parsing, chunking,
    vectorizing), managing multiple vectorization methods, querying based on
    semantic similarity, and handling concurrent access safely using file locks.

    Designed for simplicity and efficiency in RAG pipelines.

    Attributes:
        db_path (str): The resolved absolute path to the SQLite database file.
        lock_path (str): The path to the file lock used for concurrency control.
        lock_timeout (int): The maximum time in seconds to wait for the lock.
        vectorizer_manager (VectorizationManager): Manages vectorizer instances.
        conn (Optional[sqlite3.Connection]): The active SQLite database connection.
    """
    DEFAULT_VECTORIZER: str = "st:all-MiniLM-L6-v2"

    def __init__(
        self,
        db_path: Union[str, Path] = "safe_store.db",
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT
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

        Raises:
            DatabaseError: If the database connection or schema initialization fails.
            ConcurrencyError: If acquiring the initial lock for setup times out.
        """
        self.db_path: str = str(Path(db_path).resolve())
        self.lock_timeout: int = lock_timeout
        # Create lock file path next to the db file
        _db_file_path = Path(self.db_path)
        self.lock_path: str = str(_db_file_path.parent / f"{_db_file_path.name}.lock")

        ASCIIColors.set_log_level(log_level)
        ASCIIColors.info(f"Initializing safe_store with database: {self.db_path}")
        ASCIIColors.debug(f"Using lock file: {self.lock_path} with timeout: {self.lock_timeout}s")

        self.conn: Optional[sqlite3.Connection] = None
        self._is_closed: bool = True # Start as closed, connect will set to False

        self.vectorizer_manager = VectorizationManager()
        self._file_hasher = hashlib.sha256

        # Instance-level lock for thread safety within the same process
        self._instance_lock = threading.RLock()
        # File lock for inter-process safety
        self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)

        try:
            self._connect_and_initialize()
        except (DatabaseError, Timeout, ConcurrencyError) as e:
            ASCIIColors.critical(f"safe_store initialization failed: {e}")
            raise # Re-raise critical initialization errors

    def _connect_and_initialize(self) -> None:
        """
        Establishes the database connection and initializes the schema.

        Internal method called during `__init__`. Uses a temporary lock
        for safe initialization.

        Raises:
            DatabaseError: If connection or schema setup fails.
            ConcurrencyError: If the initialization lock times out.
        """
        init_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        try:
            with init_lock:
                ASCIIColors.debug("Acquired init lock for connection/schema setup.")
                if self.conn is None or self._is_closed: # Check if we need to connect
                     self.conn = db.connect_db(self.db_path)
                     db.initialize_schema(self.conn)
                     self._is_closed = False
                else:
                     ASCIIColors.debug("Connection already established.")

            ASCIIColors.debug("Released init lock.")
        except Timeout as e: # Capture the original Timeout exception
            msg = f"Timeout acquiring initial lock for DB connection/setup at {self.lock_path}"
            ASCIIColors.error(msg)
            # Explicitly close connection if partially opened before timeout
            if self.conn:
                 try:
                     self.conn.close()
                 except Exception:
                     pass # Ignore errors during cleanup
                 finally:
                     self.conn = None
                     self._is_closed = True
            raise ConcurrencyError(msg) from e # Use custom error, chain original
        except DatabaseError as e: # Catch specific DB errors
            ASCIIColors.error(f"Database error during initial setup: {e}")
            if self.conn:
                 try:
                     self.conn.close()
                 except Exception:
                     pass
                 finally:
                     self.conn = None
                     self._is_closed = True
            raise # Re-raise DatabaseError
        except Exception as e:
            # Catch any other unexpected errors during initialization
            msg = f"Unexpected error during initial DB connection/setup: {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn:
                 try:
                     self.conn.close()
                 except Exception:
                     pass
                 finally:
                     self.conn = None
                     self._is_closed = True
            raise safe_storeError(msg) from e # Wrap in generic safe_storeError

    def close(self) -> None:
        """
        Closes the database connection and clears the vectorizer cache.

        It's recommended to use `safe_store` as a context manager (`with safe_store(...)`)
        to ensure the connection is closed automatically.
        """
        with self._instance_lock:
            if self._is_closed:
                 ASCIIColors.debug("Connection already closed.")
                 return
            if self.conn:
                ASCIIColors.debug("Closing database connection.")
                try:
                    self.conn.close()
                except Exception as e:
                    # Log warning but don't raise, as we are closing anyway
                    ASCIIColors.warning(f"Error closing DB connection: {e}")
                finally:
                    self.conn = None
                    self._is_closed = True

            # Clear vectorizer cache upon closing
            if hasattr(self, 'vectorizer_manager'):
                self.vectorizer_manager.clear_cache()
            ASCIIColors.info("safe_store connection closed.")

    def __enter__(self):
        """Enter the runtime context related to this object."""
        with self._instance_lock:
            if self._is_closed or self.conn is None:
                ASCIIColors.debug("Re-establishing connection on context manager entry.")
                try:
                    self._connect_and_initialize()
                except (DatabaseError, ConcurrencyError, safe_storeError) as e:
                    ASCIIColors.error(f"Failed to re-establish connection in __enter__: {e}")
                    raise # Re-raise critical errors
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        self.close()
        if exc_type:
            ASCIIColors.error(f"safe_store context closed with error: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
        else:
            ASCIIColors.debug("safe_store context closed cleanly.")

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Generates a SHA256 hash for the file content.

        Args:
            file_path: Path object pointing to the file.

        Returns:
            The hex digest of the file hash.

        Raises:
            FileHandlingError: If the file cannot be read or found.
        """
        try:
            hasher = self._file_hasher()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
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
            # Catch unexpected errors during hashing
            msg = f"Unexpected error generating hash for {file_path}: {e}"
            ASCIIColors.warning(msg) # Log as warning, but raise for clarity
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
        Adds or updates a document in the safe_store.

        This method handles parsing the file based on its extension, chunking the
        text content, generating vector embeddings using the specified vectorizer,
        and storing the document, chunks, and vectors in the database.

        It automatically detects changes in the file content (using SHA256 hash)
        and will efficiently skip processing if the file hasn't changed and the
        specified vectorization already exists. If the file has changed, it
        removes the old data and re-indexes the new content.

        This operation acquires an exclusive write lock on the database file
        to prevent race conditions with other processes.

        Args:
            file_path: Path to the document file to add.
            vectorizer_name: Name of the vectorizer method to use (e.g.,
                             'st:all-MiniLM-L6-v2', 'tfidf:my_tfidf'). Defaults
                             to `safe_store.DEFAULT_VECTORIZER`.
            chunk_size: Target size of text chunks in characters. Defaults to 1000.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
                           Defaults to 150. Must be less than `chunk_size`.
            metadata: Optional dictionary of metadata to associate with the document.
                      Must be JSON serializable. Defaults to None.
            force_reindex: If True, forces re-parsing, re-chunking, and
                           re-vectorizing the document even if the file hash
                           matches the stored hash. Defaults to False.
            vectorizer_params: Optional dictionary of parameters specific to the
                               vectorizer initialization (currently primarily for
                               TF-IDF setup). Defaults to None.

        Raises:
            ValueError: If `chunk_overlap` is not less than `chunk_size`.
            FileHandlingError: If there's an error reading or hashing the file, or file not found.
            ParsingError: If the document parsing fails (e.g., invalid file format).
            ConfigurationError: If a required optional dependency for parsing or
                                vectorization is missing (e.g., `pypdf` for PDFs) or file type unsupported.
            VectorizationError: If vector generation fails.
            DatabaseError: If there's an error interacting with the database.
            ConcurrencyError: If the write lock cannot be acquired within the timeout.
            ConnectionError: If the database connection is not available.
            safe_storeError: For other unexpected errors during the process.
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

            except Timeout as e: # Catch original Timeout
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for add_document: {_file_path.name}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e # Wrap in ConcurrencyError
            except (DatabaseError, FileHandlingError, ParsingError, ConfigurationError,
                    VectorizationError, QueryError, ValueError, ConnectionError, safe_storeError) as e:
                # Log the caught known error type and message
                ASCIIColors.error(f"Error during add_document: {e.__class__.__name__}: {e}", exc_info=False)
                raise # Re-raise the caught exception
            except Exception as e:
                msg = f"Unexpected error during add_document (lock scope) for '{_file_path.name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise safe_storeError(msg) from e


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
        """Internal implementation of add_document logic (assumes lock is held and connection is valid)."""
        assert self.conn is not None

        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        abs_file_path = str(file_path.resolve())

        ASCIIColors.info(f"Starting indexing process for: {file_path.name}")
        ASCIIColors.debug(f"Params: vectorizer='{_vectorizer_name}', chunk_size={chunk_size}, overlap={chunk_overlap}, force={force_reindex}")

        # --- Get Hash ---
        try:
            current_hash = self._get_file_hash(file_path)
        except FileHandlingError as e:
            # Log the error, but re-raise it to be caught by the caller (add_document)
            ASCIIColors.error(f"Failed to generate hash for {file_path.name}. Aborting. Error: {e}")
            raise e

        existing_doc_id: Optional[int] = None
        existing_hash: Optional[str] = None
        needs_parsing_chunking: bool = True
        needs_vectorization: bool = True

        # --- Check existing document state ---
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (abs_file_path,))
            result = cursor.fetchone()
            if result:
                existing_doc_id, existing_hash = result
                ASCIIColors.debug(f"Document '{file_path.name}' found in DB (doc_id={existing_doc_id}). Stored Hash: {existing_hash}, Current Hash: {current_hash}")

                if force_reindex:
                    ASCIIColors.warning(f"Force re-indexing requested for '{file_path.name}'.")
                elif existing_hash == current_hash:
                    ASCIIColors.info(f"Document '{file_path.name}' is unchanged.")
                    needs_parsing_chunking = False
                    # Pass vectorizer_params correctly as keyword 'initial_params'
                    _vec_instance, method_id = self.vectorizer_manager.get_vectorizer(
                        name=_vectorizer_name, conn=self.conn, initial_params=vectorizer_params
                    )
                    cursor.execute("""
                        SELECT 1 FROM vectors v
                        JOIN chunks c ON v.chunk_id = c.chunk_id
                        WHERE c.doc_id = ? AND v.method_id = ?
                        LIMIT 1
                    """, (existing_doc_id, method_id))
                    vector_exists = cursor.fetchone() is not None
                    if vector_exists:
                        ASCIIColors.success(f"Vectorization '{_vectorizer_name}' already exists for unchanged '{file_path.name}'. Skipping.")
                        needs_vectorization = False
                    else:
                         ASCIIColors.info(f"Document '{file_path.name}' exists and is unchanged, but needs vectorization '{_vectorizer_name}'.")
                else:
                    ASCIIColors.warning(f"Document '{file_path.name}' has changed (hash mismatch). Re-indexing...")
                    cursor.execute("BEGIN")
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    metadata_str = json.dumps(metadata) if metadata else None
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?",
                                   (current_hash, None, metadata_str, existing_doc_id))
                    self.conn.commit()
                    ASCIIColors.debug(f"Deleted old chunks/vectors and updated document record for changed doc_id={existing_doc_id}.")
            else:
                 ASCIIColors.info(f"Document '{file_path.name}' is new.")

        except (sqlite3.Error, DatabaseError) as e:
            msg = f"Database error checking/updating document state for '{file_path.name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            raise DatabaseError(msg) from e
        except safe_storeError as e: raise e
        except Exception as e:
             msg = f"Unexpected error preparing indexing for '{file_path.name}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             raise safe_storeError(msg) from e

        if not needs_parsing_chunking and not needs_vectorization:
             return

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            doc_id: Optional[int] = existing_doc_id
            full_text: Optional[str] = None
            chunks_data: List[Tuple[str, int, int]] = []
            chunk_ids: List[int] = []
            chunk_texts: List[str] = []

            # --- Parsing and Chunking ---
            if needs_parsing_chunking:
                ASCIIColors.debug(f"Parsing document: {file_path.name}")
                try:
                    full_text = parser.parse_document(file_path)
                    ASCIIColors.debug(f"Parsed document '{file_path.name}'. Length: {len(full_text)} chars.")
                except (ParsingError, FileHandlingError, ConfigurationError, ValueError) as e:
                    raise e # Re-raise specific errors
                except Exception as e:
                    msg = f"Unexpected error parsing {file_path.name}: {e}"
                    ASCIIColors.error(msg, exc_info=True)
                    raise ParsingError(msg) from e

                metadata_str = json.dumps(metadata) if metadata else None
                if doc_id is None:
                    doc_id = db.add_document_record(
                        self.conn, abs_file_path, full_text, current_hash, metadata_str
                    )
                else:
                    cursor.execute("UPDATE documents SET full_text = ?, metadata = ? WHERE doc_id = ?",
                                   (full_text, metadata_str, doc_id))

                try:
                    chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                except ValueError as e: raise e

                if not chunks_data:
                    ASCIIColors.warning(f"No chunks generated for {file_path.name}. Document record saved, but skipping vectorization.")
                    self.conn.commit()
                    return

                ASCIIColors.info(f"Generated {len(chunks_data)} chunks for '{file_path.name}'. Storing chunks...")
                for i, (text, start, end) in enumerate(chunks_data):
                    chunk_id = db.add_chunk_record(self.conn, doc_id, text, start, end, i)
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(text)

            # --- Retrieve Existing Chunks if Only Vectorizing ---
            else:
                if doc_id is None:
                     raise safe_storeError(f"Inconsistent state: doc_id is None but parsing/chunking was skipped for {file_path.name}")
                ASCIIColors.debug(f"Retrieving existing chunks for doc_id={doc_id} to add new vectors...")
                cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
                results = cursor.fetchall()
                if not results:
                      ASCIIColors.warning(f"Document {doc_id} ('{file_path.name}') exists but has no stored chunks. Cannot add vectorization '{_vectorizer_name}'.")
                      needs_vectorization = False
                      self.conn.commit()
                      return
                chunk_ids = [row[0] for row in results]
                chunk_texts = [row[1] for row in results]
                ASCIIColors.debug(f"Retrieved {len(chunk_ids)} existing chunk IDs and texts.")

            # --- Vectorization ---
            if needs_vectorization:
                if not chunk_ids or not chunk_texts:
                     ASCIIColors.warning(f"No chunks available to vectorize for '{file_path.name}'.")
                     self.conn.commit()
                     return

                try:
                     # Pass vectorizer_params correctly as keyword 'initial_params'
                     vectorizer, method_id = self.vectorizer_manager.get_vectorizer(
                         name=_vectorizer_name, conn=self.conn, initial_params=vectorizer_params
                     )
                except (ConfigurationError, VectorizationError, DatabaseError) as e:
                     raise e

                if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                     ASCIIColors.warning(f"TF-IDF vectorizer '{_vectorizer_name}' is not fitted. Fitting ONLY on chunks from '{file_path.name}'. Consider using add_vectorization for global fitting.")
                     try:
                         vectorizer.fit(chunk_texts)
                         new_params = vectorizer.get_params_to_store()
                         self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                     except (VectorizationError, DatabaseError) as e:
                         raise e
                     except Exception as e:
                         msg = f"Failed to fit TF-IDF model '{_vectorizer_name}' on '{file_path.name}': {e}"
                         ASCIIColors.error(msg, exc_info=True)
                         raise VectorizationError(msg) from e

                ASCIIColors.info(f"Vectorizing {len(chunk_texts)} chunks using '{_vectorizer_name}' (method_id={method_id})...")

                try:
                     vectors = vectorizer.vectorize(chunk_texts)
                except VectorizationError as e: raise e
                except Exception as e:
                     msg = f"Unexpected error during vectorization with '{_vectorizer_name}': {e}"
                     ASCIIColors.error(msg, exc_info=True)
                     raise VectorizationError(msg) from e

                if vectors.shape[0] != len(chunk_ids):
                    msg = f"Mismatch between number of chunks ({len(chunk_ids)}) and generated vectors ({vectors.shape[0]}) for '{file_path.name}'!"
                    ASCIIColors.error(msg)
                    raise VectorizationError(msg)

                ASCIIColors.debug(f"Adding {len(vectors)} vector records to DB (method_id={method_id})...")
                for chunk_id, vector in zip(chunk_ids, vectors):
                    vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                    db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)

            # --- Commit Transaction ---
            self.conn.commit()
            ASCIIColors.success(f"Successfully processed '{file_path.name}' with vectorizer '{_vectorizer_name}'.")

        except (sqlite3.Error, DatabaseError, FileHandlingError, ParsingError, ConfigurationError,
                VectorizationError, ValueError, safe_storeError) as e:
            ASCIIColors.error(f"Error during indexing transaction for '{file_path.name}': {e.__class__.__name__}: {e}", exc_info=False)
            if self.conn: self.conn.rollback()
            raise
        except Exception as e:
            msg = f"Unexpected error during indexing transaction for '{file_path.name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn: self.conn.rollback()
            raise safe_storeError(msg) from e

    # --- Add Vectorization ---
    def add_vectorization(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[Union[str, Path]] = None,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        batch_size: int = 64
    ) -> None:
        """
        Adds vector embeddings using a specified vectorization method to documents
        already present in the store.

        This is useful for adding embeddings from a new model or method without
        re-parsing and re-chunking the original documents. If the vectorizer
        (e.g., TF-IDF) requires fitting and hasn't been fitted yet, this method
        will fit it on the specified documents (or all documents if `target_doc_path`
        is None).

        It processes documents in batches for potentially better performance and
        memory management during vectorization.

        This operation acquires an exclusive write lock.

        Args:
            vectorizer_name: The name of the vectorizer method to add
                             (e.g., 'st:new-model', 'tfidf:variant').
            target_doc_path: If specified, only adds vectors for this specific
                             document file path. If None, adds vectors for all
                             documents in the store that don't already have them
                             for this `vectorizer_name`. Defaults to None.
            vectorizer_params: Optional parameters for vectorizer initialization,
                               primarily for TF-IDF. Defaults to None.
            batch_size: Number of chunks to process in each vectorization batch.
                        Defaults to 64.

        Raises:
            FileHandlingError: If `target_doc_path` is specified but the document
                              is not found in the database.
            ConfigurationError: If a required optional dependency for the
                                vectorizer is missing.
            VectorizationError: If vector generation or fitting fails.
            DatabaseError: If there's an error interacting with the database.
            ConcurrencyError: If the write lock cannot be acquired within the timeout.
            ConnectionError: If the database connection is not available.
            safe_storeError: For other unexpected errors during the process.
        """
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for add_vectorization: {vectorizer_name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for add_vectorization: {vectorizer_name}")
                    self._ensure_connection()
                    self._add_vectorization_impl(
                        vectorizer_name, target_doc_path, vectorizer_params, batch_size
                    )
                ASCIIColors.debug(f"Write lock released for add_vectorization: {vectorizer_name}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for add_vectorization: {vectorizer_name}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
            except (DatabaseError, FileHandlingError, ConfigurationError, VectorizationError,
                     QueryError, ValueError, ConnectionError, safe_storeError) as e:
                ASCIIColors.error(f"Error during add_vectorization: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during add_vectorization (lock scope) for '{vectorizer_name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise safe_storeError(msg) from e

    def _add_vectorization_impl(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[Union[str, Path]],
        vectorizer_params: Optional[Dict[str, Any]],
        batch_size: int
    ) -> None:
        """Internal implementation of add_vectorization (assumes lock is held and connection is valid)."""
        assert self.conn is not None

        ASCIIColors.info(f"Starting process to add vectorization '{vectorizer_name}'.")
        resolved_target_doc_path: Optional[str] = None
        if target_doc_path:
             resolved_target_doc_path = str(Path(target_doc_path).resolve())
             ASCIIColors.info(f"Targeting specific document: {resolved_target_doc_path}")
        else:
             ASCIIColors.info("Targeting all documents in the store.")

        cursor = self.conn.cursor()
        target_doc_id: Optional[int] = None
        # Define SQL string base here
        chunks_to_vectorize_sql: str = f"""
            SELECT c.chunk_id, c.chunk_text -- TODO: Add decryption later
            FROM chunks c
            LEFT JOIN vectors v ON c.chunk_id = v.chunk_id AND v.method_id = ?
            WHERE v.vector_id IS NULL
        """

        try:
            cursor.execute("BEGIN") # Start transaction here

            # --- Get Vectorizer & Handle Fitting ---
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, self.conn, vectorizer_params)

            texts_to_fit: List[str] = [] # Define outside the if block

            if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' requires fitting.")
                fit_sql = "SELECT c.chunk_text FROM chunks c" # Base SQL
                fit_params_list: List[Any] = []

                if resolved_target_doc_path:
                    cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (resolved_target_doc_path,))
                    target_doc_id_result = cursor.fetchone()
                    if not target_doc_id_result:
                        msg = f"Target document '{resolved_target_doc_path}' not found in the database."
                        ASCIIColors.error(msg)
                        raise FileHandlingError(msg)
                    target_doc_id = target_doc_id_result[0]
                    fit_sql += " JOIN documents d ON c.doc_id = d.doc_id WHERE d.doc_id = ?"
                    fit_params_list.append(target_doc_id)
                    ASCIIColors.info(f"Fetching chunks for fitting ONLY from document ID {target_doc_id} ('{resolved_target_doc_path}').")
                else:
                    ASCIIColors.info("Fetching all chunks from database for fitting TF-IDF...")

                cursor.execute(fit_sql, tuple(fit_params_list))
                texts_to_fit = [row[0] for row in cursor.fetchall()] # Assign here

                if not texts_to_fit: # Check if list is empty
                    ASCIIColors.warning("No text chunks found to fit the TF-IDF model. Aborting vectorization add.")
                    self.conn.commit() # Commit potential method add/update
                    return

                try:
                    vectorizer.fit(texts_to_fit)
                    new_params = vectorizer.get_params_to_store()
                    self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                    ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' fitted successfully.")
                except (VectorizationError, DatabaseError) as e:
                    raise e
                except Exception as e:
                    msg = f"Failed to fit TF-IDF model '{vectorizer_name}': {e}" # Removed texts_to_fit reference
                    ASCIIColors.error(msg, exc_info=True)
                    raise VectorizationError(msg) from e
                # End of fitting logic

            # --- Find Chunks Missing Vectors ---
            sql_params: List[Any] = [method_id] # Base params

            if resolved_target_doc_path:
                if target_doc_id is None: # Fetch if not already fetched during fitting
                     cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (resolved_target_doc_path,))
                     target_doc_id_result = cursor.fetchone()
                     if not target_doc_id_result:
                         msg = f"Target document '{resolved_target_doc_path}' not found (second check)."
                         ASCIIColors.error(msg)
                         raise FileHandlingError(msg)
                     target_doc_id = target_doc_id_result[0]

                # Modify SQL string correctly
                current_chunks_sql = chunks_to_vectorize_sql + " AND c.doc_id = ?" # Use temporary variable
                sql_params.append(target_doc_id)
                ASCIIColors.info(f"Fetching chunks missing '{vectorizer_name}' vectors for document ID {target_doc_id}...")
                cursor.execute(current_chunks_sql, tuple(sql_params)) # Execute modified SQL
            else:
                 ASCIIColors.info(f"Fetching all chunks missing '{vectorizer_name}' vectors...")
                 cursor.execute(chunks_to_vectorize_sql, tuple(sql_params)) # Execute base SQL

            chunks_data = cursor.fetchall()

            if not chunks_data:
                ASCIIColors.success(f"No chunks found needing vectorization for '{vectorizer_name}'. Process complete.")
                self.conn.commit() # Commit potential method add/update
                return

            total_chunks = len(chunks_data)
            ASCIIColors.info(f"Found {total_chunks} chunks to vectorize.")

            # --- Vectorize and Store in Batches ---
            num_added = 0
            try: # Inner try for batch processing errors
                for i in range(0, total_chunks, batch_size):
                    batch = chunks_data[i : i + batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_texts = [item[1] for item in batch]
                    ASCIIColors.debug(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_texts)} chunks)")

                    try:
                         vectors = vectorizer.vectorize(batch_texts)
                         if vectors.shape[0] != len(batch_ids):
                              msg = f"Vectorization output count ({vectors.shape[0]}) doesn't match batch size ({len(batch_ids)})."
                              raise VectorizationError(msg)
                    except VectorizationError as e: raise e
                    except Exception as e:
                         msg = f"Unexpected error during vectorization batch for '{vectorizer_name}': {e}"
                         ASCIIColors.error(msg, exc_info=True)
                         raise VectorizationError(msg) from e

                    for chunk_id, vector in zip(batch_ids, vectors):
                         vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                         db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)
                    num_added += len(batch_ids)
                    ASCIIColors.debug(f"Added {len(batch_ids)} vectors for batch.")

            except (sqlite3.Error, DatabaseError, VectorizationError) as e:
                 ASCIIColors.error(f"Error during vectorization/storage batch processing: {e.__class__.__name__}: {e}", exc_info=False)
                 raise # Re-raise to be caught by outer handler
            except Exception as e:
                 msg = f"Unexpected error during vectorization batch processing for '{vectorizer_name}': {e}"
                 ASCIIColors.error(msg, exc_info=True)
                 raise safe_storeError(msg) from e

            # --- Commit the entire transaction ---
            self.conn.commit()
            ASCIIColors.success(f"Successfully added {num_added} vector embeddings using '{vectorizer_name}'.")

        except (sqlite3.Error, DatabaseError, FileHandlingError, ConfigurationError, VectorizationError, safe_storeError) as e:
             ASCIIColors.error(f"Error during add_vectorization transaction: {e.__class__.__name__}: {e}", exc_info=False)
             if self.conn: self.conn.rollback()
             raise
        except Exception as e:
             msg = f"Unexpected error during add_vectorization transaction for '{vectorizer_name}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             if self.conn: self.conn.rollback()
             raise safe_storeError(msg) from e

    # --- Remove Vectorization ---
    def remove_vectorization(self, vectorizer_name: str) -> None:
        """
        Removes a vectorization method and all associated vector embeddings from
        the database and the internal cache.

        This operation acquires an exclusive write lock.

        Args:
            vectorizer_name: The name of the vectorization method to remove.

        Raises:
            DatabaseError: If there's an error interacting with the database.
            ConcurrencyError: If the write lock cannot be acquired within the timeout.
            ConnectionError: If the database connection is not available.
            safe_storeError: For other unexpected errors during the process.
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
            except (DatabaseError, ConnectionError, safe_storeError) as e:
                ASCIIColors.error(f"Error during remove_vectorization: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during remove_vectorization (lock scope) for '{vectorizer_name}': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise safe_storeError(msg) from e

    def _remove_vectorization_impl(self, vectorizer_name: str) -> None:
        """Internal implementation of remove_vectorization (assumes lock is held and connection is valid)."""
        assert self.conn is not None
        ASCIIColors.warning(f"Attempting to remove vectorization method '{vectorizer_name}' and all associated vectors.")

        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_name,))
            result = cursor.fetchone()
            if not result:
                ASCIIColors.warning(f"Vectorization method '{vectorizer_name}' not found in the database. Nothing to remove.")
                return
            method_id = result[0]
            ASCIIColors.debug(f"Found method_id {method_id} for '{vectorizer_name}'.")

            cursor.execute("BEGIN")
            cursor.execute("DELETE FROM vectors WHERE method_id = ?", (method_id,))
            deleted_vectors = cursor.rowcount
            ASCIIColors.debug(f"Deleted {deleted_vectors} vector records.")
            cursor.execute("DELETE FROM vectorization_methods WHERE method_id = ?", (method_id,))
            deleted_methods = cursor.rowcount
            ASCIIColors.debug(f"Deleted {deleted_methods} vectorization method record.")
            self.conn.commit()

            self.vectorizer_manager.remove_from_cache_by_id(method_id)
            ASCIIColors.success(f"Successfully removed vectorization method '{vectorizer_name}' (ID: {method_id}) and {deleted_vectors} associated vectors.")

        except sqlite3.Error as e:
             msg = f"Database error during removal of '{vectorizer_name}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             if self.conn: self.conn.rollback()
             raise DatabaseError(msg) from e
        except Exception as e:
             msg = f"Unexpected error during removal of '{vectorizer_name}': {e}"
             ASCIIColors.error(msg, exc_info=True)
             if self.conn: self.conn.rollback()
             raise safe_storeError(msg) from e


    # === Read methods ===
    def query(
        self,
        query_text: str,
        vectorizer_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries the store for document chunks semantically similar to the query text.

        Uses the specified vectorizer to embed the query text and compares it
        against stored vectors using cosine similarity. Returns the top_k most
        similar chunks.

        This is primarily a read operation. With SQLite's WAL mode enabled, it
        should generally not block or be blocked by write operations, but uses
        an instance-level lock for thread safety within the current process.

        Args:
            query_text: The text to search for.
            vectorizer_name: The name of the vectorization method to use for the
                             query. Must match a method used during indexing.
                             Defaults to `safe_store.DEFAULT_VECTORIZER`.
            top_k: The maximum number of similar chunks to return. Defaults to 5.

        Returns:
            A list of dictionaries, where each dictionary represents a relevant
            chunk and contains:
            - 'chunk_id': (int) ID of the chunk.
            - 'chunk_text': (str) The text content of the chunk.
            - 'similarity': (float) The cosine similarity score (between -1 and 1).
            - 'doc_id': (int) ID of the source document.
            - 'file_path': (str) Path to the source document file.
            - 'start_pos': (int) Start character offset in the original document.
            - 'end_pos': (int) End character offset in the original document.
            - 'chunk_seq': (int) Sequence number of the chunk within the document.
            - 'metadata': (dict | None) Metadata associated with the document.

        Raises:
            ConfigurationError: If the specified vectorizer requires a missing dependency.
            VectorizationError: If vectorizing the query text fails.
            DatabaseError: If fetching vectorizer details or stored vectors fails.
            QueryError: If calculating similarity fails or other query logic errors occur.
            ConnectionError: If the database connection is not available.
            safe_storeError: For other unexpected errors.
        """
        with self._instance_lock:
            self._ensure_connection()
            try:
                return self._query_impl(query_text, vectorizer_name, top_k)
            except (DatabaseError, ConfigurationError, VectorizationError, QueryError,
                     ValueError, ConnectionError, safe_storeError) as e:
                ASCIIColors.error(f"Error during query: {e.__class__.__name__}: {e}", exc_info=False)
                raise
            except Exception as e:
                msg = f"Unexpected error during query for '{query_text[:50]}...': {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise safe_storeError(msg) from e


    def _query_impl(
        self,
        query_text: str,
        vectorizer_name: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Internal implementation of query logic (assumes instance lock held and connection is valid)."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        ASCIIColors.info(f"Received query. Searching with '{_vectorizer_name}', top_k={top_k}.")

        results: List[Dict[str, Any]] = []
        cursor = self.conn.cursor()

        try:
            # --- 1. Get Vectorizer & Query Vector ---
            try:
                vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn, None)
            except (ConfigurationError, VectorizationError, DatabaseError) as e: raise e

            ASCIIColors.debug(f"Using vectorizer '{_vectorizer_name}' (method_id={method_id})")
            ASCIIColors.debug(f"Vectorizing query text...")
            try:
                 query_vector_list = vectorizer.vectorize([query_text])
                 if not isinstance(query_vector_list, np.ndarray) or query_vector_list.ndim != 2 or query_vector_list.shape[0] != 1:
                      raise VectorizationError(f"Vectorizer did not return a single 2D vector for the query. Shape: {getattr(query_vector_list, 'shape', 'N/A')}")
                 query_vector = query_vector_list[0]
                 query_vector = np.ascontiguousarray(query_vector, dtype=vectorizer.dtype)
            except VectorizationError as e: raise e
            except Exception as e:
                 msg = f"Unexpected error vectorizing query text with '{_vectorizer_name}': {e}"
                 ASCIIColors.error(msg, exc_info=True)
                 raise VectorizationError(msg) from e
            ASCIIColors.debug(f"Query vector generated. Shape: {query_vector.shape}, Dtype: {query_vector.dtype}")

            # --- 2. Load Candidate Vectors ---
            ASCIIColors.debug(f"Loading all vectors for method_id {method_id} from database...")
            try:
                cursor.execute("SELECT v.chunk_id, v.vector_data FROM vectors v WHERE v.method_id = ?", (method_id,))
                all_vectors_data = cursor.fetchall()
            except sqlite3.Error as e:
                msg = f"Database error loading vectors for method '{_vectorizer_name}' (ID: {method_id}): {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise DatabaseError(msg) from e

            if not all_vectors_data:
                ASCIIColors.warning(f"No vectors found in the database for method '{_vectorizer_name}' (ID: {method_id}). Cannot perform query.")
                return []

            chunk_ids_ordered: List[int] = [row[0] for row in all_vectors_data]
            vector_blobs: List[bytes] = [row[1] for row in all_vectors_data]

            # --- 3. Reconstruct Vectors ---
            method_details = self.vectorizer_manager._get_method_details_from_db(self.conn, _vectorizer_name)
            if not method_details:
                 raise DatabaseError(f"Could not retrieve method details for '{_vectorizer_name}' after getting instance.")
            vector_dtype_str = method_details['vector_dtype']
            vector_dim_expected = method_details['vector_dim'] # Get expected dimension
            ASCIIColors.debug(f"Reconstructing {len(vector_blobs)} vectors from BLOBs with dtype '{vector_dtype_str}'...")
            try:
                 candidate_vectors_list = [db.reconstruct_vector(blob, vector_dtype_str) for blob in vector_blobs]
                 # Handle empty list after reconstruction if all blobs were invalid/empty
                 if not candidate_vectors_list:
                      candidate_vectors = np.empty((0, vector_dim_expected or 0), dtype=np.dtype(vector_dtype_str))
                 else:
                      candidate_vectors = np.stack(candidate_vectors_list, axis=0)
            except (DatabaseError, ValueError, TypeError) as e:
                 msg = f"Failed to reconstruct one or more vectors for method '{_vectorizer_name}': {e}"
                 ASCIIColors.error(msg, exc_info=False)
                 raise QueryError(msg) from e
            ASCIIColors.debug(f"Candidate vectors loaded. Matrix shape: {candidate_vectors.shape}")

            # --- 4. Calculate Similarity ---
            ASCIIColors.debug("Calculating similarity scores...")
            try:
                if candidate_vectors.shape[0] == 0: # Handle empty candidate matrix
                    scores = np.array([], dtype=query_vector.dtype)
                else:
                    scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            except (ValueError, TypeError) as e:
                 msg = f"Error calculating cosine similarity: {e}"
                 ASCIIColors.error(msg, exc_info=True)
                 raise QueryError(msg) from e
            ASCIIColors.debug(f"Similarity scores calculated. Shape: {scores.shape}")

            # --- 5. Get Top-K Results ---
            num_candidates = len(scores)
            k = min(top_k, num_candidates)
            if k <= 0:
                 ASCIIColors.info("Top-k is 0 or no candidates found, returning empty results.")
                 return []

            if k < num_candidates // 2 :
                 top_k_indices_unsorted = np.argpartition(scores, -k)[-k:]
                 top_k_indices = top_k_indices_unsorted[np.argsort(scores[top_k_indices_unsorted])[::-1]]
            else:
                 top_k_indices = np.argsort(scores)[::-1][:k]
            ASCIIColors.debug(f"Identified top {k} indices.")

            # --- 6. Retrieve Chunk Details ---
            top_chunk_ids = [chunk_ids_ordered[i] for i in top_k_indices]
            top_scores = [scores[i] for i in top_k_indices]
            placeholders = ','.join('?' * len(top_chunk_ids))
            sql_chunk_details = f"""
                SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.chunk_seq,
                       d.doc_id, d.file_path, d.metadata
                FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.chunk_id IN ({placeholders})
            """
            try:
                cursor.execute(sql_chunk_details, top_chunk_ids)
                chunk_details_list = cursor.fetchall()
            except sqlite3.Error as e:
                msg = f"Database error fetching chunk details for top-k results: {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise DatabaseError(msg) from e

            # --- 7. Combine Results ---
            chunk_details_map: Dict[int, Dict[str, Any]] = {}
            for row in chunk_details_list:
                 metadata_dict = None
                 if row[7]:
                      try: metadata_dict = json.loads(row[7])
                      except json.JSONDecodeError: pass
                 chunk_details_map[row[0]] = {
                    "chunk_id": row[0], "chunk_text": row[1], "start_pos": row[2],
                    "end_pos": row[3], "chunk_seq": row[4], "doc_id": row[5],
                    "file_path": row[6], "metadata": metadata_dict
                 }

            for chunk_id, score in zip(top_chunk_ids, top_scores):
                if chunk_id in chunk_details_map:
                    result_item = chunk_details_map[chunk_id].copy()
                    result_item["similarity"] = float(np.float64(score))
                    results.append(result_item)
                else:
                     ASCIIColors.warning(f"Could not find details for chunk_id {chunk_id} which was in top-k. Skipping.")

            ASCIIColors.success(f"Query successful. Found {len(results)} relevant chunks.")
            return results

        except (DatabaseError, VectorizationError, QueryError, ValueError) as e: raise e
        except Exception as e:
             msg = f"Unexpected error during query implementation for '{query_text[:50]}...': {e}"
             ASCIIColors.error(msg, exc_info=True)
             raise safe_storeError(msg) from e

    # --- Helper Methods ---
    def list_documents(self) -> List[Dict[str, Any]]:
         """Lists all documents currently stored in the database."""
         with self._instance_lock:
              self._ensure_connection()
              assert self.conn is not None
              cursor = self.conn.cursor()
              try:
                   cursor.execute("SELECT doc_id, file_path, file_hash, added_timestamp, metadata FROM documents ORDER BY added_timestamp")
                   docs = []
                   for row in cursor.fetchall():
                        metadata_dict = None
                        if row[4]:
                             try: metadata_dict = json.loads(row[4])
                             except json.JSONDecodeError: pass
                        docs.append({
                             "doc_id": row[0], "file_path": row[1], "file_hash": row[2],
                             "added_timestamp": row[3], "metadata": metadata_dict,
                        })
                   return docs
              except sqlite3.Error as e:
                   msg = f"Database error listing documents: {e}"
                   ASCIIColors.error(msg, exc_info=True)
                   raise DatabaseError(msg) from e

    def list_vectorization_methods(self) -> List[Dict[str, Any]]:
         """Lists all registered vectorization methods."""
         with self._instance_lock:
              self._ensure_connection()
              assert self.conn is not None
              cursor = self.conn.cursor()
              try:
                   cursor.execute("SELECT method_id, method_name, method_type, vector_dim, vector_dtype, params FROM vectorization_methods ORDER BY method_name")
                   methods = []
                   for row in cursor.fetchall():
                        params_dict = None
                        if row[5]:
                             try: params_dict = json.loads(row[5])
                             except json.JSONDecodeError: pass
                        methods.append({
                             "method_id": row[0], "method_name": row[1], "method_type": row[2],
                             "vector_dim": row[3], "vector_dtype": row[4], "params": params_dict,
                        })
                   return methods
              except sqlite3.Error as e:
                   msg = f"Database error listing vectorization methods: {e}"
                   ASCIIColors.error(msg, exc_info=True)
                   raise DatabaseError(msg) from e