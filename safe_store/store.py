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

        # Database path and lock setup... (same as before)
        if db_path_input_str == IN_MEMORY_DB_INDICATOR:
            self.db_path = IN_MEMORY_DB_INDICATOR; self._is_in_memory = True; self.lock_path = None
        elif db_path_input_str == TEMP_FILE_DB_INDICATOR:
            tmp_f = tempfile.NamedTemporaryFile(suffix=".db", prefix="safestore_temp_", delete=False)
            self.db_path = self._temp_db_actual_path = tmp_f.name
            self._is_temp_file_db = True
            db_file_obj = Path(self.db_path)
            self.lock_path = str(db_file_obj.parent / f"{db_file_obj.name}.lock")
            self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        else:
            self.db_path = str(Path(db_path).resolve())
            db_file_obj = Path(self.db_path)
            db_file_obj.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path = str(db_file_obj.parent / f"{db_file_obj.name}.lock")
            self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)

        if self.name is None: self.name = "in_memory_store" if self._is_in_memory else Path(self.db_path).stem
        
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

    # ... (core methods like _connect, close, __enter__, etc. are unchanged) ...
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
                text = self.encryptor.decrypt(text_data) if is_enc else text_data.decode('utf-8')
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
    
    # Other methods like delete_document_by_id, list_documents etc. remain largely unchanged.