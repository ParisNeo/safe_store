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
from .vectorization.base import BaseVectorizer
from ascii_colors import ASCIIColors, LogLevel, trace_exception


DEFAULT_LOCK_TIMEOUT: int = 60
TEMP_FILE_DB_INDICATOR = ":tempfile:"
IN_MEMORY_DB_INDICATOR = ":memory:"

class SafeStore:
    """
    Manages a local vector store backed by an SQLite database, tied to a single
    vectorization method defined at initialization.

    Provides functionalities for indexing documents, querying based on semantic
    similarity, and handling concurrent access safely.
    """
    DEFAULT_VECTORIZER_NAME: str = "st"
    DEFAULT_VECTORIZER_CONFIG: Dict[str, Any] = {"model": "all-MiniLM-L6-v2"}

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = "safe_store.db",
        vectorizer_name: str = "st",
        vectorizer_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        encryption_key: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initializes the SafeStore instance with a specific vectorizer.

        Args:
            db_path: Path to the SQLite database file.
            vectorizer_name: The name of the vectorizer type (e.g., 'st', 'openai').
            vectorizer_config: Configuration for the vectorizer (e.g., `{"model": "name"}`).
            name: A human-readable name for the store.
            description: A description of the store's purpose.
            metadata: Custom metadata for the store.
            log_level: Minimum log level for console output.
            lock_timeout: Timeout in seconds for the inter-process write lock.
            encryption_key: Optional password to encrypt chunk text at rest.
            cache_folder: Optional folder to cache downloaded models.
        """
        self.lock_timeout: int = lock_timeout
        self._is_in_memory: bool = False
        self._is_temp_file_db: bool = False
        self._temp_db_actual_path: Optional[str] = None
        self._file_lock: Optional[FileLock] = None

        self.name: Optional[str] = name
        self.description: Optional[str] = description
        self.metadata: Optional[Dict[str, Any]] = metadata
        
        if vectorizer_name == self.DEFAULT_VECTORIZER_NAME and vectorizer_config is None:
            self.vectorizer_config = self.DEFAULT_VECTORIZER_CONFIG
        else:
            self.vectorizer_config = vectorizer_config or {}
        self.vectorizer_name = vectorizer_name

        ASCIIColors.set_log_level(log_level)
        db_path_input_str = str(db_path).lower() if db_path is not None else IN_MEMORY_DB_INDICATOR

        if db_path_input_str == IN_MEMORY_DB_INDICATOR:
            self.db_path = IN_MEMORY_DB_INDICATOR
            self._is_in_memory = True
            self.lock_path = None
        elif db_path_input_str == TEMP_FILE_DB_INDICATOR:
            try:
                tmp_f = tempfile.NamedTemporaryFile(suffix=".db", prefix="safestore_temp_", delete=False)
                self.db_path = self._temp_db_actual_path = tmp_f.name
                self._is_temp_file_db = True
                db_file_obj = Path(self.db_path)
                self.lock_path = str(db_file_obj.parent / f"{db_file_obj.name}.lock")
                self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
            except Exception as e:
                raise ConfigurationError(f"Failed to create temporary database file: {e}") from e
        else:
            self.db_path = str(Path(db_path).resolve())
            db_file_obj = Path(self.db_path)
            db_file_obj.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path = str(db_file_obj.parent / f"{db_file_obj.name}.lock")
            self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)

        if self.name is None:
            self.name = "in_memory_store" if self._is_in_memory else Path(self.db_path).stem
        
        self.conn: Optional[sqlite3.Connection] = None
        self._is_closed: bool = True
        self.vectorizer_manager = VectorizationManager(cache_folder=cache_folder)
        self._file_hasher = hashlib.sha256
        self.encryptor = Encryptor(encryption_key)
        self._instance_lock = threading.RLock()
        self.vectorizer: BaseVectorizer

        try:
            self._connect_and_initialize()
            self._initialize_and_verify_vectorizer()
        except Exception as e:
            self._manual_cleanup_temp_files_on_error()
            raise e

    @classmethod
    def list_available_models(cls, vectorizer_name: str, **kwargs) -> List[str]:
        """
        Lists available models for a given vectorizer type.

        - For 'ollama', this method dynamically queries the Ollama server.
        - For 'st', 'openai', etc., it returns a curated list of common models.
        - For 'tfidf', it returns an empty list as models are user-defined.

        Args:
            vectorizer_name: The name of the vectorizer (e.g., 'ollama', 'st').
            **kwargs: Additional arguments, such as 'host' for the Ollama client.

        Returns:
            A list of available model name strings.

        Raises:
            ConfigurationError: If a required library for a vectorizer is not installed.
            VectorizationError: If it fails to connect to a service like Ollama.
            ValueError: If the vectorizer_name is unknown.
        """
        if vectorizer_name == "ollama":
            try:
                import ollama
            except ImportError:
                raise ConfigurationError("Ollama support is not installed. Please run: pip install safe_store[ollama]")
            
            host = kwargs.get("host", "http://localhost:11434")
            try:
                client = ollama.Client(host=host)
                response = client.list()
                return [model['name'] for model in response.get('models', [])]
            except Exception as e:
                raise VectorizationError(f"Could not connect to Ollama server at '{host}'. Please ensure it is running. Error: {e}") from e

        elif vectorizer_name == "st":
            # Returning a curated list of popular models
            return [
                "all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1",
                "all-distilroberta-v1", "paraphrase-albert-small-v2", "LaBSE"
            ]
        
        elif vectorizer_name == "openai":
            return ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

        elif vectorizer_name == "cohere":
            return ["embed-english-v3.0", "embed-english-light-v3.0", "embed-multilingual-v3.0"]

        elif vectorizer_name == "tfidf":
            return [] # TF-IDF models are not pre-existing, they are fitted on data.
        
        elif vectorizer_name == "lollms":
            ASCIIColors.warning("Model listing for 'lollms' is dynamic. Please check your Lollms server for available embedding models.")
            return []

        else:
            raise ValueError(f"Unknown vectorizer name: '{vectorizer_name}'")

    def _initialize_and_verify_vectorizer(self):
        with self._optional_file_lock_context("initialize and verify vectorizer"):
            self._ensure_connection()
            assert self.conn is not None
            
            current_vectorizer = self.vectorizer_manager.get_vectorizer(self.vectorizer_name, self.vectorizer_config)
            stored_info_json = db.get_store_metadata(self.conn, "vectorizer_info")
            
            if stored_info_json:
                stored_info = json.loads(stored_info_json)
                unique_name_from_instance = self.vectorizer_manager._create_unique_name(self.vectorizer_name, self.vectorizer_config)
                
                if stored_info.get("unique_name") != unique_name_from_instance:
                    raise ConfigurationError(
                        f"Database at '{self.db_path}' is already configured with a different vectorizer: '{stored_info.get('unique_name')}'. "
                        f"This instance is configured with '{unique_name_from_instance}'. Use a new DB file for a new vectorizer."
                    )
                ASCIIColors.info("Instance vectorizer matches the one stored in the database.")
            else:
                ASCIIColors.info("No vectorizer info found in DB. Storing current vectorizer configuration.")
                vectorizer_info = {
                    "unique_name": self.vectorizer_manager._create_unique_name(self.vectorizer_name, self.vectorizer_config),
                    "vectorizer_name": self.vectorizer_name,
                    "vectorizer_config": self.vectorizer_config,
                    "dim": current_vectorizer.dim,
                    "dtype": current_vectorizer.dtype.name,
                }
                db.set_store_metadata(self.conn, "vectorizer_info", json.dumps(vectorizer_info))

            self.vectorizer = current_vectorizer
            ASCIIColors.success(f"SafeStore is ready with vectorizer '{self.vectorizer.vectorizer_name}'.")

    def _connect_and_initialize(self) -> None:
        """Establishes the database connection, initializes the schema, and loads store properties."""
        with self._optional_file_lock_context("DB connection/schema setup"):
            if self.conn is None or self._is_closed:
                self.conn = db.connect_db(self.db_path)
                db.initialize_schema(self.conn)
                self._is_closed = False
            self._load_or_initialize_store_properties()

    def _load_or_initialize_store_properties(self) -> None:
        """Loads or sets up store properties in the database."""
        assert self.conn is not None
        try:
            self.conn.execute("BEGIN")
            db_name = db.get_store_metadata(self.conn, "store_name")
            if db_name is None:
                if self.name: db.set_store_metadata(self.conn, "store_name", self.name)
                if self.description: db.set_store_metadata(self.conn, "store_description", self.description)
                if self.metadata: db.set_store_metadata(self.conn, "store_metadata", json.dumps(self.metadata))
            else:
                self.name = db_name
                self.description = db.get_store_metadata(self.conn, "store_description")
                meta_json = db.get_store_metadata(self.conn, "store_metadata")
                self.metadata = json.loads(meta_json) if meta_json else None
            self.conn.commit()
        except Exception as e:
            if self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError("Failed to load/initialize store properties") from e
    
    @contextmanager
    def _optional_file_lock_context(self, description: Optional[str] = None) -> ContextManager[None]:
        if self._file_lock:
            try:
                with self._file_lock:
                    yield
            except Timeout as e:
                raise ConcurrencyError(f"Timeout acquiring file lock for {description}") from e
        else:
            yield
    
    def close(self) -> None:
        with self._instance_lock:
            if self.conn:
                self.conn.close()
                self.conn = None
            self._is_closed = True
            self.vectorizer_manager.clear_cache()
            if self._is_temp_file_db and self._temp_db_actual_path:
                self._manual_cleanup_temp_files_on_error()

    def _manual_cleanup_temp_files_on_error(self):
        if self._temp_db_actual_path:
            Path(self._temp_db_actual_path).unlink(missing_ok=True)
            if self.lock_path:
                Path(self.lock_path).unlink(missing_ok=True)
            self._temp_db_actual_path = None

    def __enter__(self):
        with self._instance_lock:
            if self._is_closed or self.conn is None:
                self._connect_and_initialize()
                self._initialize_and_verify_vectorizer()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_connection(self) -> None:
        if self._is_closed or self.conn is None or not hasattr(self, 'vectorizer'):
            raise ConnectionError("Database connection is closed or vectorizer not initialized.")

    def _get_file_hash(self, file_path: Path) -> str:
        hasher = self._file_hasher()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192): hasher.update(chunk)
        return hasher.hexdigest()

    def _get_text_hash(self, text: str) -> str:
        hasher = self._file_hasher()
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def add_document(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False
    ) -> None:
        """Adds or updates a document using the instance's configured vectorizer."""
        with self._instance_lock, self._optional_file_lock_context(f"add_document: {Path(file_path).name}"):
            self._ensure_connection()
            self._add_content_impl(
                content_id=str(Path(file_path).resolve()),
                content_loader=lambda: parser.parse_document(file_path),
                hash_loader=lambda: self._get_file_hash(Path(file_path)),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                force_reindex=force_reindex
            )

    def add_text(
        self,
        unique_id: str,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False
    ) -> None:
        """Adds or updates a text content using the instance's configured vectorizer."""
        with self._instance_lock, self._optional_file_lock_context(f"add_text: {unique_id}"):
            self._ensure_connection()
            self._add_content_impl(
                content_id=unique_id,
                content_loader=lambda: text,
                hash_loader=lambda: self._get_text_hash(text),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                force_reindex=force_reindex
            )

    def _add_content_impl(self, content_id, content_loader, hash_loader, chunk_size, chunk_overlap, metadata, force_reindex):
        assert self.conn and self.vectorizer is not None
        
        current_hash = hash_loader()
        cursor = self.conn.cursor()

        try:
            cursor.execute("BEGIN")
            res = cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (content_id,)).fetchone()
            existing_doc_id, existing_hash = res if res else (None, None)

            is_unchanged = not force_reindex and existing_hash == current_hash
            if is_unchanged and existing_doc_id:
                if cursor.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? LIMIT 1", (existing_doc_id,)).fetchone():
                    self.conn.commit()
                    ASCIIColors.info(f"Content '{content_id}' is unchanged and vectorized. Skipping.")
                    return

            if existing_doc_id:
                cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))

            full_text = content_loader()
            doc_id = existing_doc_id
            if doc_id is None:
                doc_id = db.add_document_record(self.conn, content_id, full_text, current_hash, json.dumps(metadata) if metadata else None)
            else:
                cursor.execute("UPDATE documents SET file_hash=?, full_text=?, metadata=? WHERE doc_id=?", (current_hash, full_text, json.dumps(metadata) if metadata else None, doc_id))
            
            chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
            if not chunks_data:
                self.conn.commit(); return
            
            if isinstance(self.vectorizer, TfidfVectorizerWrapper) and not self.vectorizer._fitted:
                texts_to_fit = [text for text, _, _ in chunks_data]
                self.vectorizer.fit(texts_to_fit)
                stored_info = json.loads(db.get_store_metadata(self.conn, "vectorizer_info") or '{}')
                stored_info["dim"] = self.vectorizer.dim
                stored_info["params"] = self.vectorizer.get_params_to_store()
                db.set_store_metadata(self.conn, "vectorizer_info", json.dumps(stored_info))

            chunk_ids, chunk_texts = [], []
            for i, (text, start, end) in enumerate(chunks_data):
                text_to_store = self.encryptor.encrypt(text) if self.encryptor.is_enabled else text
                chunk_id = db.add_chunk_record(self.conn, doc_id, text_to_store, start, end, i, is_encrypted=self.encryptor.is_enabled)
                chunk_ids.append(chunk_id)
                chunk_texts.append(text)
            
            vectors = self.vectorizer.vectorize(chunk_texts)
            for chunk_id, vector_data in zip(chunk_ids, vectors):
                db.add_vector_record(self.conn, chunk_id, np.ascontiguousarray(vector_data, dtype=self.vectorizer.dtype))

            self.conn.commit()
        except Exception as e:
            if self.conn and self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError(f"Transaction failed for '{content_id}': {e}") from e

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity_percent: float = 0.0
    ) -> List[Dict[str, Any]]:
        with self._instance_lock, self._optional_file_lock_context("query"):
            self._ensure_connection()
            assert self.conn and self.vectorizer is not None

            query_vector = self.vectorizer.vectorize([query_text])[0]
            cursor = self.conn.cursor()
            
            all_vectors_data = cursor.execute("SELECT v.chunk_id, v.vector_data FROM vectors v").fetchall()
            if not all_vectors_data: return []

            chunk_ids, vector_blobs = zip(*all_vectors_data)
            candidate_vectors = np.array([db.reconstruct_vector(blob, self.vectorizer.dtype.name) for blob in vector_blobs])

            scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            score_threshold = (min_similarity_percent / 50.0) - 1.0
            pass_mask = scores >= score_threshold
            
            scores_passing = scores[pass_mask]
            if len(scores_passing) == 0: return []
                
            chunk_ids_passing = np.array(chunk_ids)[pass_mask]
            
            k = min(top_k, len(scores_passing)) if top_k > 0 else len(scores_passing)
            top_indices = np.argsort(scores_passing)[::-1][:k]
            
            top_chunk_ids, top_scores = chunk_ids_passing[top_indices], scores_passing[top_indices]

            placeholders = ','.join('?' * len(top_chunk_ids))
            sql = f"SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.is_encrypted, d.file_path, d.metadata FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE c.chunk_id IN ({placeholders})"
            
            details_map = {}
            original_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            details_raw = cursor.execute(sql, tuple(top_chunk_ids.tolist())).fetchall()
            self.conn.text_factory = original_factory
            
            for row in details_raw:
                chunk_id, text_data, start, end, is_enc, path, meta = row
                text = self.encryptor.decrypt(text_data) if is_enc and self.encryptor.is_enabled else (
                    "[Encrypted - Key Unavailable]" if is_enc else text_data.decode('utf-8')
                )
                path_str = path.decode('utf-8')
                meta_dict = json.loads(meta.decode('utf-8')) if meta else None
                details_map[chunk_id] = {"chunk_text": text, "start_pos": start, "end_pos": end, "file_path": path_str, "metadata": meta_dict}
            
            ordered_results = []
            for cid, s in zip(top_chunk_ids, top_scores):
                res = details_map.get(cid, {})
                res.update({"chunk_id": cid, "similarity_score": float(s), "similarity_percent": round(((s + 1) / 2) * 100, 2)})
                ordered_results.append(res)
            return ordered_results

    def get_vectorization_details(self) -> Optional[Dict[str, Any]]:
         """Returns the details of the vectorizer configured for this store."""
         with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None
            info_json = db.get_store_metadata(self.conn, "vectorizer_info")
            return json.loads(info_json) if info_json else None

    def delete_document_by_id(self, doc_id: int) -> None:
        """Deletes a document and all its associated data by its ID."""
        with self._instance_lock, self._optional_file_lock_context(f"delete_document: {doc_id}"):
            self._ensure_connection()
            assert self.conn is not None
            try:
                self.conn.execute("BEGIN")
                rows = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,)).rowcount
                self.conn.commit()
                if rows > 0: ASCIIColors.success(f"Deleted document ID {doc_id}.")
            except sqlite3.Error as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise DatabaseError from e

    def delete_document_by_path(self, file_path: Union[str, Path]) -> None:
        """Deletes a document by its file path or unique_id."""
        _path_or_id = str(Path(file_path).resolve() if isinstance(file_path, Path) else file_path)
        with self._instance_lock, self._optional_file_lock_context(f"delete_document: {_path_or_id}"):
            self._ensure_connection()
            assert self.conn is not None
            res = self.conn.execute("SELECT doc_id FROM documents WHERE file_path = ?", (_path_or_id,)).fetchone()
            if res:
                self.delete_document_by_id(res[0])
            else:
                ASCIIColors.warning(f"Document '{_path_or_id}' not found.")

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None
            docs = []
            for r in self.conn.execute("SELECT doc_id, file_path, file_hash, added_timestamp, metadata FROM documents"):
                 docs.append({"doc_id": r[0], "file_path": r[1], "file_hash": r[2], "added_timestamp": r[3], "metadata": json.loads(r[4]) if r[4] else None})
            return docs

    def vectorize_text(self, text_to_vectorize: str):
        self._ensure_connection()
        assert self.vectorizer is not None
        return self.vectorizer.vectorize([text_to_vectorize])