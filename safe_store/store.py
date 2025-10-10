import sqlite3
import json
from pathlib import Path
import hashlib
import threading
from typing import Optional, List, Dict, Any, Tuple, Union, Literal, ContextManager
import tempfile
import os
from contextlib import contextmanager

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
from ascii_colors import ASCIIColors, LogLevel, trace_exception


DEFAULT_LOCK_TIMEOUT: int = 60
TEMP_FILE_DB_INDICATOR = ":tempfile:"
IN_MEMORY_DB_INDICATOR = ":memory:"

class SafeStore:
    """
    Manages a local vector store backed by an SQLite database.

    Provides functionalities for indexing documents (parsing, chunking,
    vectorizing), managing multiple vectorization methods, querying based on
    semantic similarity, deleting documents, and handling concurrent access
    safely using file locks. Includes optional encryption for chunk text.
    """
    DEFAULT_VECTORIZER_NAME: str = "st"
    DEFAULT_VECTORIZER_CONFIG: Dict[str, Any] = {"model": "all-MiniLM-L6-v2"}

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = "safe_store.db",
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        encryption_key: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initializes the safe_store instance.
        """
        self.lock_timeout: int = lock_timeout
        self._is_in_memory: bool = False
        self._is_temp_file_db: bool = False
        self._temp_db_actual_path: Optional[str] = None
        self._file_lock: Optional[FileLock] = None

        self.name: Optional[str] = name
        self.description: Optional[str] = description
        self.metadata: Optional[Dict[str, Any]] = metadata

        actual_db_path_str: str
        lock_path_str: Optional[str] = None

        db_path_input_str = str(db_path).lower() if db_path is not None else IN_MEMORY_DB_INDICATOR

        ASCIIColors.set_log_level(log_level)

        if db_path_input_str == IN_MEMORY_DB_INDICATOR:
            actual_db_path_str = IN_MEMORY_DB_INDICATOR
            self._is_in_memory = True
            ASCIIColors.info("Initializing SafeStore with an in-memory SQLite database.")
        elif db_path_input_str == TEMP_FILE_DB_INDICATOR:
            try:
                with tempfile.NamedTemporaryFile(suffix=".db", prefix="safestore_temp_", delete=False) as tmp_f:
                    self._temp_db_actual_path = tmp_f.name
                actual_db_path_str = self._temp_db_actual_path
                self._is_temp_file_db = True
                ASCIIColors.info(f"Initializing SafeStore with a temporary database file: {actual_db_path_str}")
                _db_file_path_obj = Path(actual_db_path_str)
                lock_path_str = str(_db_file_path_obj.parent / f"{_db_file_path_obj.name}.lock")
                self._file_lock = FileLock(lock_path_str, timeout=self.lock_timeout)
            except Exception as e:
                msg = f"Failed to create temporary database file: {e}"
                self._manual_cleanup_temp_files_on_error()
                raise ConfigurationError(msg) from e
        else:
            actual_db_path_str = str(Path(db_path).resolve()) # type: ignore
            ASCIIColors.info(f"Initializing SafeStore with persistent database: {actual_db_path_str}")
            _db_file_path_obj = Path(actual_db_path_str)
            try:
                _db_file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise FileHandlingError(f"Failed to create parent directory for database '{actual_db_path_str}': {e}") from e
            lock_path_str = str(_db_file_path_obj.parent / f"{_db_file_path_obj.name}.lock")
            self._file_lock = FileLock(lock_path_str, timeout=self.lock_timeout)

        self.db_path: str = actual_db_path_str
        self.lock_path: Optional[str] = lock_path_str

        if self.name is None:
            self.name = "in_memory_store" if self._is_in_memory else Path(self.db_path).stem

        if self.lock_path:
            ASCIIColors.debug(f"Using lock file: {self.lock_path} with timeout: {self.lock_timeout}s")
        
        self.conn: Optional[sqlite3.Connection] = None
        self._is_closed: bool = True
        self.vectorizer_manager = VectorizationManager(cache_folder=cache_folder)
        self._file_hasher = hashlib.sha256

        try:
            self.encryptor = Encryptor(encryption_key)
            if self.encryptor.is_enabled:
                 ASCIIColors.info("Encryption enabled for chunk text.")
        except (ConfigurationError, ValueError) as e:
             self._manual_cleanup_temp_files_on_error()
             raise e

        self._instance_lock = threading.RLock()
        try:
            self._connect_and_initialize()
        except (DatabaseError, Timeout, ConcurrencyError, SafeStoreError) as e:
            self._manual_cleanup_temp_files_on_error()
            raise

    def _connect_and_initialize(self) -> None:
        """Establishes the database connection, initializes the schema, and loads store properties."""
        try:
            with self._optional_file_lock_context("DB connection/schema setup"):
                if self.conn is None or self._is_closed:
                     self.conn = db.connect_db(self.db_path)
                     db.initialize_schema(self.conn)
                     self._is_closed = False
                     ASCIIColors.debug(f"Database connection established and schema initialized for: {self.db_path}")

                self._load_or_initialize_store_properties()

        except (DatabaseError, Timeout, ConcurrencyError) as e:
            if self.conn:
                 try: self.conn.close()
                 except Exception: pass
                 finally: self.conn = None; self._is_closed = True
            raise
        except Exception as e:
            msg = f"Unexpected error during initial DB connection/setup: {e}"
            if self.conn:
                 try: self.conn.close()
                 except Exception: pass
                 finally: self.conn = None; self._is_closed = True
            raise SafeStoreError(msg) from e

    def _load_or_initialize_store_properties(self) -> None:
        """
        Loads store properties (name, description, metadata) from the database.
        If they don't exist (new store or old DB version), it writes the current
        instance properties to the database. This acts as an "upgrade" step.
        This method must be called within a write lock and after `self.conn` is set.
        """
        assert self.conn is not None, "Connection must be established before loading properties."
        
        try:
            self.conn.execute("BEGIN")
            
            db_name = db.get_store_metadata(self.conn, "store_name")
            db_description = db.get_store_metadata(self.conn, "store_description")
            db_metadata_json = db.get_store_metadata(self.conn, "store_metadata")

            if db_name is None:
                ASCIIColors.info("Store properties not found in DB. Initializing them now (first run or upgrade).")
                if self.name is not None:
                    db.set_store_metadata(self.conn, "store_name", self.name)
                if self.description is not None:
                    db.set_store_metadata(self.conn, "store_description", self.description)
                if self.metadata is not None:
                    db.set_store_metadata(self.conn, "store_metadata", json.dumps(self.metadata))
            else:
                ASCIIColors.debug("Loading existing store properties from DB.")
                self.name = db_name
                self.description = db_description
                self.metadata = json.loads(db_metadata_json) if db_metadata_json else None

            self.conn.commit()
            ASCIIColors.debug(f"Store properties loaded/initialized. Name: '{self.name}'")

        except (DatabaseError, ConfigurationError) as e:
            if self.conn and self.conn.in_transaction: self.conn.rollback()
            raise
        except Exception as e:
            if self.conn and self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError(f"Unexpected error loading/initializing store properties: {e}") from e


    def get_properties(self) -> Dict[str, Any]:
        """
        Retrieves the name, description, and metadata of the store.
        """
        with self._instance_lock:
            self._ensure_connection()
            return {
                "name": self.name,
                "description": self.description,
                "metadata": self.metadata
            }

    def update_properties(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite_metadata: bool = False
    ) -> None:
        """
        Updates the store's name, description, and/or metadata.
        """
        if name is None and description is None and metadata is None:
            return

        with self._instance_lock:
            with self._optional_file_lock_context("update_properties"):
                self._ensure_connection()
                assert self.conn is not None

                try:
                    self.conn.execute("BEGIN")

                    if name is not None:
                        db.set_store_metadata(self.conn, "store_name", name)
                        self.name = name

                    if description is not None:
                        db.set_store_metadata(self.conn, "store_description", description)
                        self.description = description
                    
                    if metadata is not None:
                        new_metadata = metadata
                        if not overwrite_metadata and isinstance(self.metadata, dict):
                            merged = self.metadata.copy()
                            merged.update(metadata)
                            new_metadata = merged
                        
                        db.set_store_metadata(self.conn, "store_metadata", json.dumps(new_metadata))
                        self.metadata = new_metadata

                    self.conn.commit()
                    ASCIIColors.success("Store properties updated successfully.")

                except (DatabaseError, ConfigurationError) as e:
                    if self.conn.in_transaction: self.conn.rollback()
                    raise
                except Exception as e:
                    if self.conn.in_transaction: self.conn.rollback()
                    raise SafeStoreError(f"Unexpected error updating store properties: {e}") from e

    def _manual_cleanup_temp_files_on_error(self):
        """Helper to clean up temp files if __init__ fails after their creation."""
        if self._is_temp_file_db and self._temp_db_actual_path:
            path_to_del = self._temp_db_actual_path
            lock_path_to_del = self.lock_path
            self._temp_db_actual_path = None
            self._is_temp_file_db = False
            try: Path(path_to_del).unlink(missing_ok=True)
            except OSError: pass
            if lock_path_to_del:
                try: Path(lock_path_to_del).unlink(missing_ok=True)
                except OSError: pass

    @contextmanager
    def _optional_file_lock_context(self, description: Optional[str] = None) -> ContextManager[None]:
        """Acquires the file lock if configured, otherwise yields immediately."""
        if self._file_lock:
            op_desc = f" for {description}" if description else ""
            try:
                with self._file_lock:
                    ASCIIColors.info(f"File lock acquired{op_desc}.")
                    yield
                ASCIIColors.debug(f"File lock released{op_desc}.")
            except Timeout as e:
                raise ConcurrencyError(f"Timeout acquiring file lock{op_desc} (path: {self.lock_path})") from e
        else:
            yield

    def close(self) -> None:
        """Closes the database connection, clears vectorizer cache, and cleans up temporary files if any."""
        with self._instance_lock:
            if self._is_closed and not (self._is_temp_file_db and self._temp_db_actual_path):
                 return

            if self.conn:
                try: self.conn.close()
                except Exception: pass
                finally: self.conn = None

            self._is_closed = True

            if hasattr(self, 'vectorizer_manager'):
                self.vectorizer_manager.clear_cache()
            ASCIIColors.info("SafeStore resources (connection, cache) released.")

            if self._is_temp_file_db and self._temp_db_actual_path:
                path_to_del, lock_path_to_del = self._temp_db_actual_path, self.lock_path
                self._temp_db_actual_path, self._is_temp_file_db = None, False
                try: Path(path_to_del).unlink(missing_ok=True)
                except OSError: pass
                if lock_path_to_del:
                    try: Path(lock_path_to_del).unlink(missing_ok=True)
                    except OSError: pass

    def __enter__(self):
        with self._instance_lock:
            if self._is_closed or self.conn is None:
                self._connect_and_initialize()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_file_hash(self, file_path: Path) -> str:
        """Generates a SHA256 hash for the file content."""
        try:
            hasher = self._file_hasher()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192): hasher.update(chunk)
            return hasher.hexdigest()
        except (FileNotFoundError, OSError) as e:
            raise FileHandlingError(f"Error reading file for hashing {file_path}: {e}") from e

    def _get_text_hash(self, text: str) -> str:
        """Generates a SHA256 hash for the given text."""
        hasher = self._file_hasher()
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def _ensure_connection(self) -> None:
        """Checks if the connection is active, raises ConnectionError if not."""
        if self._is_closed or self.conn is None:
            raise ConnectionError("Database connection is closed. Use SafeStore as a context manager or call connect().")

    def preload_vectorizer(self,
                           vectorizer_name: str,
                           vectorizer_config: Optional[Dict[str, Any]] = None) -> None:
        """Preloads a vectorizer for future use."""
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER_NAME
        _vectorizer_config = vectorizer_config or self.DEFAULT_VECTORIZER_CONFIG
        
        with self._instance_lock:
            with self._optional_file_lock_context(f"preloading vectorizer '{_vectorizer_name}'"):
                self._ensure_connection()
                assert self.conn is not None
                self.vectorizer_manager.get_vectorizer(
                    _vectorizer_name,
                    _vectorizer_config,
                    self.conn
                )
                ASCIIColors.success(f"Vectorizer '{_vectorizer_name}' preloaded successfully.")

    def add_document(
        self,
        file_path: Union[str, Path],
        vectorizer_name: Optional[str] = None,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False
    ) -> None:
        """Adds or updates a document in the SafeStore."""
        if chunk_overlap >= chunk_size:
             raise ValueError("chunk_overlap must be smaller than chunk_size")

        with self._instance_lock:
            with self._optional_file_lock_context(f"add_document: {Path(file_path).name}"):
                self._ensure_connection()
                self._add_document_impl(
                    Path(file_path), vectorizer_name, vectorizer_config,
                    chunk_size, chunk_overlap, metadata, force_reindex
                )

    def _add_document_impl(
        self,
        file_path: Path,
        vectorizer_name: Optional[str],
        vectorizer_config: Optional[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]],
        force_reindex: bool
    ) -> None:
        """Internal implementation of add_document logic."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER_NAME
        _vectorizer_config = vectorizer_config if vectorizer_config is not None else self.DEFAULT_VECTORIZER_CONFIG
        abs_file_path = str(file_path.resolve())

        ASCIIColors.info(f"Starting indexing for: {file_path.name}")
        current_hash = self._get_file_hash(file_path)

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            existing_doc_id, existing_hash = None, None
            result = cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (abs_file_path,)).fetchone()
            if result:
                existing_doc_id, existing_hash = result

            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, _vectorizer_config, self.conn)

            # Determine if re-indexing or vectorization is needed
            is_unchanged = not force_reindex and existing_hash == current_hash
            needs_parsing = not is_unchanged
            
            if is_unchanged:
                vector_exists = cursor.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? AND v.method_id = ? LIMIT 1", (existing_doc_id, method_id)).fetchone()
                if vector_exists:
                    ASCIIColors.success(f"Document '{file_path.name}' is unchanged and vectorized. Skipping.")
                    self.conn.commit()
                    return
            
            if needs_parsing:
                ASCIIColors.info(f"Parsing and chunking '{file_path.name}'...")
                if existing_doc_id:
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                
                full_text = parser.parse_document(file_path)
                metadata_str = json.dumps(metadata) if metadata else None
                
                if existing_doc_id is None:
                    doc_id = db.add_document_record(self.conn, abs_file_path, full_text, current_hash, metadata_str)
                else:
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?", (current_hash, full_text, metadata_str, existing_doc_id))
                    doc_id = existing_doc_id
                
                chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                if not chunks_data:
                    ASCIIColors.warning(f"No chunks generated for {file_path.name}.")
                    self.conn.commit()
                    return

                should_encrypt = self.encryptor.is_enabled
                for i, (text, start, end) in enumerate(chunks_data):
                    text_to_store: Union[str, bytes] = self.encryptor.encrypt(text) if should_encrypt else text
                    db.add_chunk_record(self.conn, doc_id, text_to_store, start, end, i, is_encrypted=should_encrypt)
            
            doc_id = existing_doc_id or doc_id
            cursor.execute("SELECT chunk_id, chunk_text, is_encrypted FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
            chunks_to_vectorize = cursor.fetchall()
            
            chunk_ids = [c[0] for c in chunks_to_vectorize]
            chunk_texts = [self.encryptor.decrypt(c[1]) if c[2] else c[1] for c in chunks_to_vectorize]

            if not chunk_texts:
                self.conn.commit()
                return

            ASCIIColors.info(f"Vectorizing {len(chunk_texts)} chunks...")
            vectors = vectorizer.vectorize(chunk_texts)
            
            for chunk_id, vector_data in zip(chunk_ids, vectors):
                db.add_vector_record(self.conn, chunk_id, method_id, np.ascontiguousarray(vector_data, dtype=vectorizer.dtype))

            self.conn.commit()
            ASCIIColors.success(f"Successfully processed '{file_path.name}'.")

        except Exception as e:
            if self.conn and self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError(f"Transaction failed for '{file_path.name}': {e}") from e

    def add_text(
        self,
        unique_id: str,
        text: str,
        vectorizer_name: Optional[str] = None,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False
    ) -> None:
        """Adds or updates a text content in the SafeStore."""
        if not unique_id or text is None:
            raise ValueError("unique_id and text must be provided.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        with self._instance_lock:
            with self._optional_file_lock_context(f"add_text: {unique_id}"):
                self._ensure_connection()
                # Reusing the document implementation logic by treating text as a document
                self._add_document_impl(
                    file_path=Path(unique_id), # Using unique_id as a virtual path
                    vectorizer_name=vectorizer_name,
                    vectorizer_config=vectorizer_config,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata=metadata,
                    force_reindex=force_reindex,
                    # Override file-based functions with text-based ones
                    _get_hash_func=self._get_text_hash,
                    _get_content_func=lambda p: text
                )

    def _add_document_impl(
        self,
        file_path: Path,
        vectorizer_name: Optional[str],
        vectorizer_config: Optional[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[Dict[str, Any]],
        force_reindex: bool,
        _get_hash_func=None,
        _get_content_func=None
    ) -> None:
        """Generalized internal implementation for adding content."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER_NAME
        _vectorizer_config = vectorizer_config if vectorizer_config is not None else self.DEFAULT_VECTORIZER_CONFIG
        
        content_id = str(file_path.resolve()) if _get_content_func is None else str(file_path)
        get_hash = _get_hash_func or self._get_file_hash
        get_content = _get_content_func or parser.parse_document

        ASCIIColors.info(f"Starting indexing for content ID: {content_id}")
        content_to_hash = get_content(file_path) if _get_hash_func else file_path
        current_hash = get_hash(content_to_hash)

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            result = cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (content_id,)).fetchone()
            existing_doc_id, existing_hash = result if result else (None, None)

            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, _vectorizer_config, self.conn)

            is_unchanged = not force_reindex and existing_hash == current_hash
            needs_processing = not is_unchanged

            if is_unchanged:
                if cursor.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? AND v.method_id = ? LIMIT 1", (existing_doc_id, method_id)).fetchone():
                    ASCIIColors.success(f"Content '{content_id}' is unchanged and vectorized. Skipping.")
                    self.conn.commit()
                    return

            if needs_processing:
                if existing_doc_id:
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                
                full_text = content_to_hash # It's already the content for add_text
                metadata_str = json.dumps(metadata) if metadata else None
                
                if existing_doc_id is None:
                    doc_id = db.add_document_record(self.conn, content_id, full_text, current_hash, metadata_str)
                else:
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?", (current_hash, full_text, metadata_str, existing_doc_id))
                    doc_id = existing_doc_id
                
                chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                if not chunks_data:
                    self.conn.commit(); return

                should_encrypt = self.encryptor.is_enabled
                for i, (text, start, end) in enumerate(chunks_data):
                    text_to_store = self.encryptor.encrypt(text) if should_encrypt else text
                    db.add_chunk_record(self.conn, doc_id, text_to_store, start, end, i, is_encrypted=should_encrypt)
            
            doc_id = existing_doc_id or doc_id # type: ignore
            
            cursor.execute("SELECT chunk_id, chunk_text, is_encrypted FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
            chunks_to_vectorize = cursor.fetchall()
            chunk_ids = [c[0] for c in chunks_to_vectorize]
            chunk_texts = [(self.encryptor.decrypt(c[1]) if c[2] else c[1].decode('utf-8') if isinstance(c[1], bytes) else c[1]) for c in chunks_to_vectorize]

            if not chunk_texts:
                self.conn.commit(); return

            vectors = vectorizer.vectorize(chunk_texts)
            for chunk_id, vector_data in zip(chunk_ids, vectors):
                db.add_vector_record(self.conn, chunk_id, method_id, np.ascontiguousarray(vector_data, dtype=vectorizer.dtype))

            self.conn.commit()
            ASCIIColors.success(f"Successfully processed content ID '{content_id}'.")
        except Exception as e:
            if self.conn and self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError(f"Transaction failed for '{content_id}': {e}") from e


    def add_vectorization(
        self,
        vectorizer_name: str,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        target_doc_path: Optional[Union[str, Path]] = None,
        batch_size: int = 64
    ) -> None:
        """Adds vector embeddings using a specified method to existing documents."""
        with self._instance_lock:
            with self._optional_file_lock_context(f"add_vectorization: {vectorizer_name}"):
                self._ensure_connection()
                self._add_vectorization_impl(vectorizer_name, vectorizer_config, target_doc_path, batch_size)

    def _add_vectorization_impl(
        self,
        vectorizer_name: str,
        vectorizer_config: Optional[Dict[str, Any]],
        target_doc_path: Optional[Union[str, Path]],
        batch_size: int
    ) -> None:
        """Internal implementation of add_vectorization."""
        assert self.conn is not None
        _vectorizer_config = vectorizer_config or {}

        target_doc_id = None
        if target_doc_path:
             resolved_path = str(Path(target_doc_path).resolve())
             cursor_check = self.conn.cursor()
             res = cursor_check.execute("SELECT doc_id FROM documents WHERE file_path = ?", (resolved_path,)).fetchone()
             if not res:
                 raise FileHandlingError(f"Target document '{resolved_path}' not found.")
             target_doc_id = res[0]

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, _vectorizer_config, self.conn)

            if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                fit_sql = "SELECT c.chunk_text, c.is_encrypted FROM chunks c" + (" WHERE c.doc_id = ?" if target_doc_id else "")
                texts_to_fit_raw = cursor.execute(fit_sql, (target_doc_id,) if target_doc_id else ()).fetchall()
                
                if not texts_to_fit_raw:
                    self.conn.commit(); return
                
                texts_to_fit = [(self.encryptor.decrypt(t[0]) if t[1] else t[0].decode('utf-8') if isinstance(t[0], bytes) else t[0]) for t in texts_to_fit_raw]
                
                vectorizer.fit(texts_to_fit)
                new_params = vectorizer.get_params_to_store()
                self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
            
            # Fetch chunks that need vectorization
            sql_base = "SELECT c.chunk_id, c.chunk_text, c.is_encrypted FROM chunks c LEFT JOIN vectors v ON c.chunk_id = v.chunk_id AND v.method_id = ? WHERE v.vector_id IS NULL"
            sql_params: List[Any] = [method_id]
            if target_doc_id:
                sql_base += " AND c.doc_id = ?"
                sql_params.append(target_doc_id)
            
            chunks_data_raw = cursor.execute(sql_base, sql_params).fetchall()
            if not chunks_data_raw:
                self.conn.commit(); return
            
            for i in range(0, len(chunks_data_raw), batch_size):
                batch_raw = chunks_data_raw[i : i + batch_size]
                batch_ids = [item[0] for item in batch_raw]
                batch_texts = [(self.encryptor.decrypt(item[1]) if item[2] else item[1].decode('utf-8') if isinstance(item[1], bytes) else item[1]) for item in batch_raw]

                vectors = vectorizer.vectorize(batch_texts)
                for chunk_id_vec, vector_data in zip(batch_ids, vectors):
                    db.add_vector_record(self.conn, chunk_id_vec, method_id, np.ascontiguousarray(vector_data, dtype=vectorizer.dtype))
            
            self.conn.commit()
            ASCIIColors.success(f"Successfully added {len(chunks_data_raw)} vector embeddings.")
        except Exception as e:
            if self.conn and self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError(f"Add vectorization failed: {e}") from e

    def remove_vectorization(self, vectorizer_name: str, vectorizer_config: Optional[Dict[str, Any]] = None) -> None:
        """Removes a vectorization method and its associated vectors."""
        with self._instance_lock:
            with self._optional_file_lock_context(f"remove_vectorization"):
                self._ensure_connection()
                assert self.conn is not None
                
                unique_name = self.vectorizer_manager._create_unique_name(vectorizer_name, vectorizer_config)
                ASCIIColors.warning(f"Attempting to remove vectorization '{unique_name}'.")
                
                cursor = self.conn.cursor()
                try:
                    res = cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (unique_name,)).fetchone()
                    if not res:
                        ASCIIColors.warning(f"Vectorization '{unique_name}' not found.")
                        return
                    
                    method_id = res[0]
                    cursor.execute("BEGIN")
                    deleted_vectors = cursor.execute("DELETE FROM vectors WHERE method_id = ?", (method_id,)).rowcount
                    cursor.execute("DELETE FROM vectorization_methods WHERE method_id = ?", (method_id,))
                    self.conn.commit()
                    
                    self.vectorizer_manager.remove_from_cache_by_id(method_id)
                    ASCIIColors.success(f"Successfully removed '{unique_name}' and {deleted_vectors} vectors.")
                except sqlite3.Error as e:
                    if self.conn.in_transaction: self.conn.rollback()
                    raise DatabaseError(f"DB error during removal of '{unique_name}': {e}") from e

    def delete_document_by_id(self, doc_id: int) -> None:
        """Deletes a document and all its associated data by its ID."""
        with self._instance_lock:
            with self._optional_file_lock_context(f"delete_document_by_id: {doc_id}"):
                self._ensure_connection()
                assert self.conn is not None
                try:
                    self.conn.execute("BEGIN")
                    rows_affected = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,)).rowcount
                    self.conn.commit()
                    if rows_affected > 0:
                        ASCIIColors.success(f"Successfully deleted document ID {doc_id}.")
                    else:
                        ASCIIColors.warning(f"Document with ID {doc_id} not found.")
                except sqlite3.Error as e:
                    if self.conn.in_transaction: self.conn.rollback()
                    raise DatabaseError(f"DB error during deletion of document ID {doc_id}: {e}") from e

    def delete_document_by_path(self, file_path: Union[str, Path]) -> None:
        """Deletes a document by its file path or unique_id."""
        _path_or_id = str(Path(file_path).resolve() if isinstance(file_path, Path) else file_path)
        with self._instance_lock:
            with self._optional_file_lock_context(f"delete_document_by_path/id: {_path_or_id}"):
                self._ensure_connection()
                assert self.conn is not None
                res = self.conn.execute("SELECT doc_id FROM documents WHERE file_path = ?", (_path_or_id,)).fetchone()
                if res:
                    self.delete_document_by_id(res[0])
                else:
                    ASCIIColors.warning(f"Document with path/id '{_path_or_id}' not found.")


    def query(
        self,
        query_text: str,
        vectorizer_name: Optional[str] = None,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_similarity_percent: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Queries the store for chunks semantically similar to the query text."""
        if not (0.0 <= min_similarity_percent <= 100.0):
            raise ValueError("min_similarity_percent must be between 0.0 and 100.0.")

        with self._instance_lock:
            with self._optional_file_lock_context("query"):
                self._ensure_connection()
                return self._query_impl(query_text, vectorizer_name, vectorizer_config, top_k, min_similarity_percent)

    def _query_impl(
        self,
        query_text: str,
        vectorizer_name: Optional[str],
        vectorizer_config: Optional[Dict[str, Any]],
        top_k: int,
        min_similarity_percent: float,
    ) -> List[Dict[str, Any]]:
        """Internal implementation of query logic."""
        assert self.conn is not None
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER_NAME
        _vectorizer_config = vectorizer_config if vectorizer_config is not None else self.DEFAULT_VECTORIZER_CONFIG
        
        try:
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, _vectorizer_config, self.conn)

            query_vector = vectorizer.vectorize([query_text])[0]
            
            cursor = self.conn.cursor()
            all_vectors_data = cursor.execute("SELECT v.chunk_id, v.vector_data FROM vectors v WHERE v.method_id = ?", (method_id,)).fetchall()
            if not all_vectors_data:
                return []

            chunk_ids, vector_blobs = zip(*all_vectors_data)
            candidate_vectors = np.array([db.reconstruct_vector(blob, vectorizer.dtype.name) for blob in vector_blobs])

            scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            
            score_threshold = (min_similarity_percent / 50.0) - 1.0
            pass_mask = scores >= score_threshold
            
            scores_passing = scores[pass_mask]
            chunk_ids_passing = np.array(chunk_ids)[pass_mask]

            if len(scores_passing) == 0:
                return []

            k_to_select = min(top_k, len(scores_passing)) if top_k > 0 else len(scores_passing)
            top_indices = np.argsort(scores_passing)[::-1][:k_to_select]
            
            top_chunk_ids = chunk_ids_passing[top_indices]
            top_scores = scores_passing[top_indices]

            placeholders = ','.join('?' * len(top_chunk_ids))
            sql = f"SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.is_encrypted, d.file_path, d.metadata FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE c.chunk_id IN ({placeholders})"
            
            original_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            details_raw = cursor.execute(sql, tuple(top_chunk_ids.tolist())).fetchall()
            self.conn.text_factory = original_factory

            details_map = {}
            for row in details_raw:
                chunk_id, text_data, start, end, is_encrypted, path, meta = row
                text = self.encryptor.decrypt(text_data) if is_encrypted else text_data.decode('utf-8')
                details_map[chunk_id] = {"chunk_text": text, "start_pos": start, "end_pos": end, "file_path": path.decode('utf-8'), "metadata": json.loads(meta.decode('utf-8')) if meta else None}

            results = []
            for chunk_id, score in zip(top_chunk_ids, top_scores):
                res = details_map.get(chunk_id, {})
                res.update({
                    "chunk_id": chunk_id,
                    "similarity_score": float(score),
                    "similarity_percent": round(((score + 1) / 2) * 100, 2)
                })
                results.append(res)
            
            return results
        except Exception as e:
            raise QueryError(f"Query failed: {e}") from e


    def query_all(
        self,
        query_text: str,
        top_k: int = 5,
        mode: Literal['union', 'intersection'] = 'union',
        min_similarity_percent: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Queries the store using *all* available vectorization methods."""
        if mode not in ['union', 'intersection']:
            raise ValueError("Invalid mode. Must be 'union' or 'intersection'.")

        with self._instance_lock:
            all_methods = self.list_vectorization_methods()
            if not all_methods: return []

            combined_results = {}
            for method in all_methods:
                unique_name = method['method_name']
                vectorizer_name, config_str = unique_name.split(":", 1)
                vectorizer_config = json.loads(config_str)
                
                method_results = self.query(query_text, vectorizer_name, vectorizer_config, top_k, min_similarity_percent)
                for res in method_results:
                    chunk_id = res['chunk_id']
                    if chunk_id not in combined_results:
                        combined_results[chunk_id] = {'max_score': -2.0, 'details': res, 'methods': set()}
                    
                    if res['similarity_score'] > combined_results[chunk_id]['max_score']:
                        combined_results[chunk_id]['max_score'] = res['similarity_score']
                        combined_results[chunk_id]['details'] = res
                    combined_results[chunk_id]['methods'].add(unique_name)

            final_list = list(combined_results.values())
            
            if mode == 'intersection':
                final_list = [res for res in final_list if len(res['methods']) == len(all_methods)]
            
            final_list.sort(key=lambda x: x['max_score'], reverse=True)
            return [res['details'] for res in final_list]


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
                    metadata_dict = json.loads(row[4]) if row[4] else None
                    docs.append({
                        "doc_id": row[0], "file_path": row[1], "file_hash": row[2], 
                        "added_timestamp": row[3], "metadata": metadata_dict
                    })
                return docs
            except sqlite3.Error as e:
                raise DatabaseError(f"Database error listing documents: {e}") from e

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
                    params_dict = json.loads(row[5]) if row[5] else None
                    methods.append({
                        "method_id": row[0], "method_name": row[1], "method_type": row[2], 
                        "vector_dim": row[3], "vector_dtype": row[4], "params": params_dict
                    })
                return methods
            except sqlite3.Error as e:
                raise DatabaseError(f"Database error listing vectorization methods: {e}") from e

    def vectorize_text(self, text_to_vectorize: str, vectorizer_name: Optional[str] = None, vectorizer_config: Optional[Dict[str, Any]] = None):
        """Vectorizes a single string of text."""
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER_NAME
        _vectorizer_config = vectorizer_config or self.DEFAULT_VECTORIZER_CONFIG
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None
            vectorizer, _ = self.vectorizer_manager.get_vectorizer(_vectorizer_name, _vectorizer_config, self.conn)
            return vectorizer.vectorize([text_to_vectorize])

    @staticmethod
    def list_possible_vectorizer_names() -> List[str]:
        """Lists the names of available vectorizer types."""
        return ["st", "tfidf", "openai", "ollama", "lollms", "cohere"]