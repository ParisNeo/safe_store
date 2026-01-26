import sqlite3
import json
from pathlib import Path
import hashlib
import threading
from typing import Optional, List, Dict, Any, Union, Literal, ContextManager, Callable
import tempfile
from contextlib import contextmanager

from filelock import FileLock, Timeout
import numpy as np

from safe_store.core import db
from safe_store.security.encryption import Encryptor
from safe_store.core.exceptions import (
    DatabaseError, FileHandlingError, ParsingError, ConfigurationError,
    VectorizationError, QueryError, ConcurrencyError, SafeStoreError, EncryptionError
)
from safe_store.indexing import parser, chunking
from safe_store.search import similarity
from safe_store.vectorization.manager import VectorizationManager
from safe_store.vectorization.base import BaseVectorizer
from safe_store.vectorization.utils import load_vectorizer_module
from safe_store.processing.text_cleaning import get_cleaner
from safe_store.processing.tokenizers import get_tokenizer
from ascii_colors import ASCIIColors, LogLevel

DEFAULT_LOCK_TIMEOUT: int = 60
TEMP_FILE_DB_INDICATOR = ":tempfile:"
IN_MEMORY_DB_INDICATOR = ":memory:"

class SafeStore:
    """
    Manages a local vector store with a single, fixed vectorizer and chunking strategy.
    """
    DEFAULT_VECTORIZER_NAME: str = "st"
    DEFAULT_VECTORIZER_CONFIG: Dict[str, Any] = {"model": "all-MiniLM-L6-v2"}

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = "safe_store.db",
        vectorizer_name: str = "st",
        vectorizer_config: Optional[Dict[str, Any]] = None,
        custom_vectorizers_path: Optional[str] = None,
        chunk_size: int = 384,
        chunk_overlap: int = 50,
        chunking_strategy: Literal['character', 'token', 'paragraph', 'semantic', 'recursive'] = 'token',
        custom_tokenizer: Optional[Dict[str, Any]] = None,
        expand_before: int = 0,
        expand_after: int = 0,
        text_cleaner: Union[str, Callable[[str], str], None] = 'basic',
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        encryption_key: Optional[str] = None,
        cache_folder: Optional[str] = None,
        chunking_kwargs: Optional[Dict[str, Any]] = None
    ):
        ASCIIColors.set_log_level(log_level)

        self.vectorizer_name = vectorizer_name
        self.vectorizer_config = vectorizer_config if vectorizer_config is not None else (self.DEFAULT_VECTORIZER_CONFIG if vectorizer_name == self.DEFAULT_VECTORIZER_NAME else {})
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.custom_tokenizer = custom_tokenizer
        self.expand_before = expand_before
        self.expand_after = expand_after
        self.text_cleaner = get_cleaner(text_cleaner)
        self.chunking_kwargs = chunking_kwargs or {}
        
        self.name = name
        self.description = description
        self.metadata = metadata
        self.lock_timeout = lock_timeout

        self._is_in_memory: bool = False
        self._is_temp_file_db: bool = False
        self._temp_db_actual_path: Optional[str] = None
        self._file_lock: Optional[FileLock] = None
        
        self._setup_paths_and_locks(db_path)
        
        self.conn: Optional[sqlite3.Connection] = None
        self._is_closed: bool = True
        self.vectorizer_manager = VectorizationManager(
            cache_folder=cache_folder,
            custom_vectorizers_path=custom_vectorizers_path
        )
        self._file_hasher = hashlib.sha256
        self.encryptor = Encryptor(encryption_key)
        self._instance_lock = threading.RLock()
        self.vectorizer: BaseVectorizer
        self.tokenizer_for_chunking: Optional[Any] = None

        try:
            self._connect_and_initialize()
            self._initialize_and_verify_vectorizer()
        except Exception as e:
            self._manual_cleanup_temp_files_on_error()
            raise e

    def _setup_paths_and_locks(self, db_path):
        db_path_input_str = str(db_path).lower() if db_path is not None else IN_MEMORY_DB_INDICATOR
        if db_path_input_str == IN_MEMORY_DB_INDICATOR:
            self.db_path = IN_MEMORY_DB_INDICATOR
            self._is_in_memory = True
            self.lock_path = None
            self._file_lock = None
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
            
        self._cleanup_stale_locks()
        if self.name is None:
            self.name = "in_memory_store" if self._is_in_memory else Path(self.db_path).stem

    @classmethod
    def list_available_vectorizers(cls, custom_vectorizers_path: Optional[str] = None) -> List[Dict[str, Any]]:
        manager = VectorizationManager(custom_vectorizers_path=custom_vectorizers_path)
        return manager.list_vectorizers()

    @classmethod
    def list_models(cls, vectorizer_name: str, custom_vectorizers_path: Optional[str] = None, **kwargs) -> List[str]:
        try:
            module = load_vectorizer_module(vectorizer_name, custom_vectorizers_path)
            VectorizerClass = None
            if hasattr(module, 'class_name'):
                VectorizerClass = getattr(module, module.class_name, None)

            if VectorizerClass and issubclass(VectorizerClass, BaseVectorizer) and hasattr(VectorizerClass, 'list_models'):
                return VectorizerClass.list_models(**kwargs)
            else:
                ASCIIColors.warning(f"Vectorizer module '{vectorizer_name}' does not have a valid 'list_models' static method or class setup.")
                if hasattr(module, 'list_models'):
                    return module.list_models(**kwargs)
                return []
        except (FileNotFoundError, ConfigurationError, VectorizationError) as e:
            raise e
        except Exception as e:
            raise SafeStoreError(f"An unexpected error occurred while listing models for '{vectorizer_name}': {e}") from e

    def _cleanup_stale_locks(self):
        """
        Attempts to clean up lock files that exist but are not held by any process.
        """
        if self._file_lock and self.lock_path and Path(self.lock_path).exists():
            try:
                self._file_lock.acquire(timeout=0.01)
                self._file_lock.release()
                try:
                    Path(self.lock_path).unlink()
                except OSError:
                    pass
            except (Timeout, OSError, Exception):
                pass

    def _initialize_and_verify_vectorizer(self):
        self.vectorizer = self.vectorizer_manager.get_vectorizer(self.vectorizer_name, self.vectorizer_config)

        if self.chunking_strategy in ['token', 'paragraph', 'semantic', 'recursive']:
            tokenizer = self.vectorizer.get_tokenizer()
            if tokenizer is not None:
                self.tokenizer_for_chunking = tokenizer
                ASCIIColors.info("Using tokenizer provided by the vectorizer model for chunking.")
            elif self.custom_tokenizer is not None:
                self.tokenizer_for_chunking = get_tokenizer(self.custom_tokenizer)
                ASCIIColors.warning(
                    f"Using custom tokenizer '{self.custom_tokenizer.get('name')}' for chunking. "
                    "Note: This may not perfectly match the remote vectorizer's internal tokenizer, "
                    "but is a close approximation."
                )
            else:
                ASCIIColors.warning(
                    f"Vectorizer '{self.vectorizer_name}' does not provide a client-side tokenizer. "
                    "Defaulting to 'tiktoken' for accurate sizing. This is a common choice for OpenAI-compatible models."
                )
                self.tokenizer_for_chunking = get_tokenizer({"name": "tiktoken", "model": "cl100k_base"})

        with self._optional_file_lock_context("verify vectorizer compatibility"):
            assert self.conn is not None, "Database must be connected before verifying vectorizer."
            stored_info_json = db.get_store_metadata(self.conn, "vectorizer_info")
            
            if stored_info_json:
                stored_info = json.loads(stored_info_json)
                unique_name_from_instance = self.vectorizer_manager._create_unique_name(self.vectorizer_name, self.vectorizer_config)
                
                if stored_info.get("unique_name") != unique_name_from_instance:
                    raise ConfigurationError(
                        f"Database at '{self.db_path}' is already configured with a different vectorizer: '{stored_info.get('unique_name')}'. "
                        f"This instance is configured with '{unique_name_from_instance}'. Use a new DB file for a new vectorizer."
                    )
            else:
                ASCIIColors.info("No vectorizer info found in DB. Storing current vectorizer configuration.")
                vectorizer_info = {
                    "unique_name": self.vectorizer_manager._create_unique_name(self.vectorizer_name, self.vectorizer_config),
                    "vectorizer_name": self.vectorizer_name,
                    "vectorizer_config": self.vectorizer_config,
                    "dim": self.vectorizer.dim,
                    "dtype": self.vectorizer.dtype.name,
                }
                try:
                    self.conn.execute("BEGIN")
                    db.set_store_metadata(self.conn, "vectorizer_info", json.dumps(vectorizer_info))
                    self.conn.commit()
                except Exception as e:
                    if self.conn.in_transaction: self.conn.rollback()
                    raise SafeStoreError("Failed to store vectorizer info in database") from e
            
            ASCIIColors.success(f"SafeStore is ready with vectorizer '{self.vectorizer_name}'.")

    def _connect_and_initialize(self) -> None:
        with self._optional_file_lock_context("DB connection/schema setup"):
            if self.conn is None or self._is_closed:
                self.conn = db.connect_db(self.db_path)
                db.initialize_schema(self.conn)
                self._is_closed = False
            self._load_or_initialize_store_properties()

    def _load_or_initialize_store_properties(self) -> None:
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
        if self._is_closed or self.conn is None:
            self._connect_and_initialize()
            self._initialize_and_verify_vectorizer()

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
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        vectorize_with_metadata: bool = True,
        chunk_processor: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        skip_chunking: bool = False
    ) -> Dict[str, int]:
        """
        Adds a document to the store.
        Returns:
            Dict[str, int]: {"num_chunks_added": int, "num_chunks_ignored": int}
        """
        with self._instance_lock:
            self._ensure_connection()
            return self._add_content_impl(
                content_id=str(Path(file_path).resolve()),
                content_loader=lambda: parser.parse_document(file_path),
                hash_loader=lambda: self._get_file_hash(Path(file_path)),
                metadata=metadata,
                force_reindex=force_reindex,
                vectorize_with_metadata=vectorize_with_metadata,
                chunk_processor=chunk_processor,
                skip_chunking=skip_chunking
            )

    def add_text(
        self,
        unique_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        vectorize_with_metadata: bool = True,
        chunk_processor: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        skip_chunking: bool = False
    ) -> Dict[str, int]:
        """
        Adds raw text to the store.
        Returns:
            Dict[str, int]: {"num_chunks_added": int, "num_chunks_ignored": int}
        """
        with self._instance_lock:
            self._ensure_connection()
            return self._add_content_impl(
                content_id=unique_id,
                content_loader=lambda: text,
                hash_loader=lambda: self._get_text_hash(text),
                metadata=metadata,
                force_reindex=force_reindex,
                vectorize_with_metadata=vectorize_with_metadata,
                chunk_processor=chunk_processor,
                skip_chunking=skip_chunking
            )

    def _add_content_impl(self, content_id, content_loader, hash_loader, metadata, force_reindex, vectorize_with_metadata, chunk_processor, skip_chunking) -> Dict[str, int]:
        assert self.conn and self.vectorizer is not None
        
        try:
            current_hash = hash_loader()
        except (OSError, FileNotFoundError) as e:
            ASCIIColors.warning(f"File '{content_id}' is empty or not readable. Error: {e}")
            return {"num_chunks_added": 0, "num_chunks_ignored": 0}

        with self._optional_file_lock_context(f"checking document status for {content_id}"):
            self._ensure_connection()
            assert self.conn is not None
            res = self.conn.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (content_id,)).fetchone()
            existing_doc_id, existing_hash = res if res else (None, None)

            if not force_reindex and existing_hash == current_hash and existing_doc_id:
                if self.conn.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? LIMIT 1", (existing_doc_id,)).fetchone():
                    ASCIIColors.info(f"'{content_id}' is already up-to-date. Skipping.")
                    return {"num_chunks_added": 0, "num_chunks_ignored": 0}

        ASCIIColors.info(f"Processing '{content_id}' for indexing...")
        
        try:
            full_text = content_loader()
        except ConfigurationError:
            ASCIIColors.warning(f"Unknown file format for '{content_id}'. Skipping.")
            return {"num_chunks_added": 0, "num_chunks_ignored": 0}
        except ParsingError as e:
            if "encrypted" in str(e).lower():
                ASCIIColors.warning(f"File '{content_id}' content is encrypted. Skipping.")
                return {"num_chunks_added": 0, "num_chunks_ignored": 0}
            ASCIIColors.warning(f"File '{content_id}' is empty or not readable. Error: {e}")
            return {"num_chunks_added": 0, "num_chunks_ignored": 0}
        except (FileHandlingError, OSError) as e:
            ASCIIColors.warning(f"File '{content_id}' is empty or not readable. Error: {e}")
            return {"num_chunks_added": 0, "num_chunks_ignored": 0}
        except Exception as e:
             ASCIIColors.error(f"Unexpected error processing '{content_id}': {e}")
             return {"num_chunks_added": 0, "num_chunks_ignored": 0}

        if not full_text or not full_text.strip():
             ASCIIColors.warning(f"File '{content_id}' is empty or not readable.")
             return {"num_chunks_added": 0, "num_chunks_ignored": 0}

        cleaned_text = self.text_cleaner(full_text)
        
        raw_chunks_data = []
        if skip_chunking:
            storage_text = cleaned_text
            if self.tokenizer_for_chunking:
                tokens = self.tokenizer_for_chunking.encode(cleaned_text)
                if len(tokens) > self.chunk_size:
                    ASCIIColors.warning(f"Document '{content_id}' exceeds chunk_size ({self.chunk_size}). Cropping vector text, but storing full content.")
                    vector_text = self.tokenizer_for_chunking.decode(tokens[:self.chunk_size])
                else:
                    vector_text = cleaned_text
            else:
                if len(cleaned_text) > self.chunk_size:
                     ASCIIColors.warning(f"Document '{content_id}' exceeds chunk_size ({self.chunk_size}). Cropping vector text, but storing full content.")
                     vector_text = cleaned_text[:self.chunk_size]
                else:
                     vector_text = cleaned_text
            raw_chunks_data = [(vector_text, storage_text)]
        else:
            raw_chunks_data = chunking.generate_chunks(
                text=cleaned_text, 
                strategy=self.chunking_strategy, 
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap, 
                expand_before=self.expand_before,
                expand_after=self.expand_after, 
                tokenizer=self.tokenizer_for_chunking,
                vectorizer_fn=self.vectorizer.vectorize if self.chunking_strategy == 'semantic' else None,
                **self.chunking_kwargs
            )

        processed_chunks_data = []
        if chunk_processor:
            doc_metadata = metadata or {}
            for vector_text, storage_text in raw_chunks_data:
                processed_text = chunk_processor(vector_text, doc_metadata)
                # Keep tuple structure even if processed
                processed_chunks_data.append((processed_text, processed_text))
        else:
            processed_chunks_data = raw_chunks_data

        # Filter valid chunks and count ignored
        valid_chunks_data = [chunk for chunk in processed_chunks_data if chunk[0] and chunk[0].strip()]
        num_ignored = len(processed_chunks_data) - len(valid_chunks_data)

        if not valid_chunks_data:
            ASCIIColors.warning(f"No valid chunks generated for '{content_id}'. Deleting if it exists.")
            if existing_doc_id:
                self.delete_document_by_id(existing_doc_id)
            return {"num_chunks_added": 0, "num_chunks_ignored": num_ignored}

        vector_texts = [item[0] for item in valid_chunks_data]
        storage_texts = [item[1] for item in valid_chunks_data]
        
        if vectorize_with_metadata and metadata:
            metadata_string = "--- Document Context ---\n"
            for k, v in metadata.items(): metadata_string += f"{str(k).title()}: {str(v)}\n"
            metadata_string += "------------------------\n\n"
            vector_texts = [metadata_string + text for text in vector_texts]

        if hasattr(self.vectorizer, 'fit') and hasattr(self.vectorizer, '_fitted') and not self.vectorizer._fitted:
            self.vectorizer.fit(vector_texts)
            if hasattr(self.vectorizer, 'get_params_to_store'):
                with self._optional_file_lock_context("updating vectorizer params"):
                    self._ensure_connection()
                    assert self.conn is not None
                    try:
                        self.conn.execute("BEGIN")
                        stored_info_json = db.get_store_metadata(self.conn, "vectorizer_info") or '{}'
                        stored_info = json.loads(stored_info_json)
                        stored_info["dim"] = self.vectorizer.dim
                        stored_info["params"] = self.vectorizer.get_params_to_store()
                        db.set_store_metadata(self.conn, "vectorizer_info", json.dumps(stored_info))
                        self.conn.commit()
                    except Exception as e:
                        if self.conn and self.conn.in_transaction: self.conn.rollback()
                        raise DatabaseError("Failed to store updated vectorizer params") from e

        vectors = self.vectorizer.vectorize(vector_texts)

        with self._optional_file_lock_context(f"writing document {content_id} to DB"):
            self._ensure_connection()
            assert self.conn is not None
            try:
                self.conn.execute("BEGIN")
                
                res = self.conn.execute("SELECT doc_id FROM documents WHERE file_path = ?", (content_id,)).fetchone()
                doc_id = res[0] if res else None
                
                metadata_blob: Optional[bytes] = None
                if metadata:
                    metadata_json = json.dumps(metadata)
                    metadata_blob = self.encryptor.encrypt(metadata_json) if self.encryptor.is_enabled else metadata_json.encode('utf-8')
                
                if doc_id:
                    self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                    self.conn.execute("UPDATE documents SET file_hash=?, metadata=?, is_encrypted=? WHERE doc_id=?",
                                     (current_hash, metadata_blob, 1 if self.encryptor.is_enabled else 0, doc_id))
                else:
                    doc_id = db.add_document_record(self.conn, file_path=content_id, file_hash=current_hash,
                                                     metadata=metadata_blob, is_encrypted=self.encryptor.is_enabled)
                
                for i, storage_text in enumerate(storage_texts):
                    text_to_store = self.encryptor.encrypt(storage_text) if self.encryptor.is_enabled else storage_text
                    chunk_id = db.add_chunk_record(self.conn, doc_id, text_to_store, 0, 0, i, is_encrypted=self.encryptor.is_enabled)
                    db.add_vector_record(self.conn, chunk_id, np.ascontiguousarray(vectors[i], dtype=self.vectorizer.dtype))

                self.conn.commit()
                ASCIIColors.success(f"Successfully indexed '{content_id}'. Added {len(valid_chunks_data)} chunks, ignored {num_ignored}.")
                return {"num_chunks_added": len(valid_chunks_data), "num_chunks_ignored": num_ignored}

            except Exception as e:
                if self.conn and self.conn.in_transaction: self.conn.rollback()
                raise SafeStoreError(f"Database transaction failed for '{content_id}': {e}") from e
        
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity_percent: float = 0.0
    ) -> List[Dict[str, Any]]:
        with self._instance_lock:
            # --- Phase 1: Vectorize query (No lock) ---
            self._ensure_connection()
            assert self.conn and self.vectorizer is not None
            query_vector = self.vectorizer.vectorize([query_text])[0]
            
            # --- Phase 2: Fetch all vectors (Short lock) ---
            with self._optional_file_lock_context("query - fetch vectors"):
                self._ensure_connection()
                assert self.conn is not None
                all_vectors_data = self.conn.execute("SELECT v.chunk_id, v.vector_data FROM vectors v").fetchall()
            
            if not all_vectors_data: return []

            # --- Phase 3: Compute similarity (No lock) ---
            chunk_ids, vector_blobs = zip(*all_vectors_data)
            candidate_vectors = np.array([db.reconstruct_vector(blob, self.vectorizer.dtype.name) for blob in vector_blobs])

            scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            score_threshold = (min_similarity_percent / 50.0) - 1.0
            pass_mask = scores >= score_threshold
            
            if not np.any(pass_mask): return []
                
            scores_passing = scores[pass_mask]
            chunk_ids_passing = np.array(chunk_ids)[pass_mask]
            
            k = min(top_k, len(scores_passing)) if top_k > 0 else len(scores_passing)
            top_indices = np.argsort(scores_passing)[::-1][:k]
            top_chunk_ids, top_scores = chunk_ids_passing[top_indices], scores_passing[top_indices]

            # --- Phase 4: Fetch details for top chunks (Short lock) ---
            with self._optional_file_lock_context("query - fetch details"):
                self._ensure_connection()
                assert self.conn is not None
                placeholders = ','.join('?' * len(top_chunk_ids))
                sql = f"""
                    SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos,
                           c.is_encrypted AS chunk_is_encrypted, d.file_path,
                           d.metadata AS doc_metadata, d.is_encrypted AS doc_is_encrypted
                    FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.chunk_id IN ({placeholders})
                """
                
                details_map = {}
                original_factory = self.conn.text_factory
                self.conn.text_factory = bytes
                details_raw = self.conn.execute(sql, tuple(top_chunk_ids.tolist())).fetchall()
                self.conn.text_factory = original_factory

            # --- Phase 5: Process details (No lock) ---
            for row in details_raw:
                chunk_id, chunk_text_data, start, end, chunk_is_enc, path, doc_meta_data, doc_is_enc = row
                
                chunk_text: str
                if chunk_is_enc:
                    chunk_text = "[Encrypted Chunk - Decryption Failed]"
                    if self.encryptor.is_enabled:
                        try: chunk_text = self.encryptor.decrypt(chunk_text_data)
                        except EncryptionError: pass
                    else: chunk_text = "[Encrypted Chunk - Key Unavailable]"
                else:
                    chunk_text = chunk_text_data.decode('utf-8')
                
                doc_metadata_text, meta_dict = "", None
                if doc_meta_data:
                    meta_json_str: Optional[str] = None
                    if doc_is_enc:
                        meta_dict = {"error": "Encrypted metadata but key is unavailable"}
                        if self.encryptor.is_enabled:
                            try: meta_json_str = self.encryptor.decrypt(doc_meta_data)
                            except EncryptionError: meta_dict = {"error": "Failed to decrypt document metadata"}
                    else:
                        meta_json_str = doc_meta_data.decode('utf-8')
                    
                    if meta_json_str:
                        try: meta_dict = json.loads(meta_json_str)
                        except json.JSONDecodeError: meta_dict = {"error": "Could not parse metadata JSON"}
                
                if isinstance(meta_dict, dict) and "error" not in meta_dict:
                    doc_metadata_text += "--- Document Context ---\n"
                    for k, v in meta_dict.items(): doc_metadata_text += f"{str(k).title()}: {str(v)}\n"
                    doc_metadata_text += "------------------------\n\n"

                details_map[chunk_id] = {
                    "chunk_text": doc_metadata_text + chunk_text, "start_pos": start, "end_pos": end,
                    "file_path": path.decode('utf-8'), "document_metadata": meta_dict
                }
            
            ordered_results = []
            for cid, s in zip(top_chunk_ids, top_scores):
                res = details_map.get(cid, {})
                res.update({"chunk_id": cid, "similarity_score": float(s), "similarity_percent": float(round(((s + 1) / 2) * 100, 2))})
                ordered_results.append(res)
            return ordered_results

    def get_vectorization_details(self) -> Optional[Dict[str, Any]]:
         with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None
            info_json = db.get_store_metadata(self.conn, "vectorizer_info")
            return json.loads(info_json) if info_json else None

    def delete_document_by_id(self, doc_id: int) -> None:
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
            original_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            rows = self.conn.execute("SELECT doc_id, file_path, file_hash, added_timestamp, metadata, is_encrypted FROM documents").fetchall()
            self.conn.text_factory = original_factory

            for r in rows:
                doc_id, file_path_bytes, file_hash_bytes, ts, meta_blob, is_enc = r
                
                meta_dict = None
                if meta_blob:
                    if is_enc:
                        if self.encryptor.is_enabled:
                            try:
                                meta_json = self.encryptor.decrypt(meta_blob)
                                meta_dict = json.loads(meta_json)
                            except (EncryptionError, json.JSONDecodeError):
                                meta_dict = {"error": "Failed to decrypt or parse metadata"}
                        else:
                            meta_dict = {"error": "Encrypted metadata but key unavailable"}
                    else:
                        try:
                            meta_dict = json.loads(meta_blob.decode('utf-8'))
                        except json.JSONDecodeError:
                            meta_dict = {"error": "Failed to parse metadata"}

                docs.append({
                    "doc_id": doc_id,
                    "file_path": file_path_bytes.decode('utf-8'),
                    "file_hash": file_hash_bytes.decode('utf-8') if file_hash_bytes else None,
                    "added_timestamp": ts,
                    "metadata": meta_dict
                })
            return docs

    def vectorize_text(self, text_to_vectorize: str):
        self._ensure_connection()
        assert self.vectorizer is not None
        return self.vectorizer.vectorize([text_to_vectorize])

    def reconstruct_document_text(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Reconstructs the content of a document by fetching and joining its chunks.
        """
        _path_or_id = str(Path(file_path).resolve() if isinstance(file_path, Path) else file_path)
        
        with self._instance_lock, self._optional_file_lock_context(f"reconstruct_document: {_path_or_id}"):
            self._ensure_connection()
            assert self.conn is not None

            doc_id = db.get_document_id_by_path(self.conn, _path_or_id)
            if doc_id is None:
                ASCIIColors.warning(f"Document not found for reconstruction: '{_path_or_id}'")
                return None
            
            sql = "SELECT chunk_text, is_encrypted FROM chunks WHERE doc_id = ? ORDER BY chunk_seq ASC"
            original_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            cursor = self.conn.cursor()
            rows = cursor.execute(sql, (doc_id,)).fetchall()
            self.conn.text_factory = original_factory

            if not rows:
                return ""

            decrypted_chunks = []
            for chunk_text_data, is_enc in rows:
                if is_enc:
                    if self.encryptor.is_enabled:
                        try:
                            decrypted_chunks.append(self.encryptor.decrypt(chunk_text_data))
                        except EncryptionError:
                            decrypted_chunks.append("[Encrypted Chunk - Decryption Failed]")
                    else:
                        decrypted_chunks.append("[Encrypted Chunk - Key Unavailable]")
                else:
                    decrypted_chunks.append(chunk_text_data.decode('utf-8'))
            
            return "\n".join(decrypted_chunks)

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        with self._instance_lock, self._optional_file_lock_context(f"get_chunk_by_id: {chunk_id}"):
            self._ensure_connection()
            assert self.conn is not None
            
            row = db.get_chunk_raw_details_by_id(self.conn, chunk_id)
            if not row:
                return None

            # Unpack the raw row from the database
            _, chunk_text_data, chunk_is_enc, path_bytes, doc_meta_data, doc_is_enc = row

            # Decrypt chunk text if necessary
            chunk_text: str
            if chunk_is_enc:
                if self.encryptor.is_enabled:
                    try: chunk_text = self.encryptor.decrypt(chunk_text_data)
                    except EncryptionError: chunk_text = "[Encrypted Chunk - Decryption Failed]"
                else: chunk_text = "[Encrypted Chunk - Key Unavailable]"
            else:
                chunk_text = chunk_text_data.decode('utf-8')

            # Decrypt and parse document metadata if necessary
            meta_dict = None
            if doc_meta_data:
                if doc_is_enc:
                    if self.encryptor.is_enabled:
                        try:
                            meta_json = self.encryptor.decrypt(doc_meta_data)
                            meta_dict = json.loads(meta_json)
                        except (EncryptionError, json.JSONDecodeError):
                            meta_dict = {"error": "Failed to decrypt or parse metadata"}
                    else:
                        meta_dict = {"error": "Encrypted metadata but key unavailable"}
                else:
                    try: meta_dict = json.loads(doc_meta_data.decode('utf-8'))
                    except json.JSONDecodeError: meta_dict = {"error": "Failed to parse metadata"}

            return {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "file_path": path_bytes.decode('utf-8'),
                "document_metadata": meta_dict
            }

    def export_point_cloud(self, output_format: Literal['json_str', 'dict', 'csv'] = 'json_str') -> Union[str, List[Dict[str, Any]]]:
        """
        Exports all vectorized chunks as a 2D point cloud using PCA.
        """
        try:
            from sklearn.decomposition import PCA
            import pandas as pd
        except ImportError:
            raise ConfigurationError("'export_point_cloud' requires 'scikit-learn' and 'pandas'. Install with: pip install scikit-learn pandas")

        with self._instance_lock:
            # --- Phase 1: Fetch all data from DB (Short Lock) ---
            with self._optional_file_lock_context("export_point_cloud - fetch data"):
                self._ensure_connection()
                assert self.conn and self.vectorizer is not None
                all_data = db.get_all_vectors_with_doc_info(self.conn)
                vectorizer_details = self.get_vectorization_details()

            if not all_data:
                raise SafeStoreError("No vectors found in the store to export.")
            if not vectorizer_details:
                raise SafeStoreError("Could not retrieve vectorizer details from the database.")

            # --- Phase 2: Process data (No Lock) ---
            chunk_ids, vectors, doc_paths, doc_metadatas = [], [], [], []
            dtype_str = vectorizer_details["dtype"]

            for row in all_data:
                chunk_id, vector_blob, path_bytes, meta_blob, is_enc = row
                chunk_ids.append(chunk_id)
                vectors.append(db.reconstruct_vector(vector_blob, dtype_str))
                doc_paths.append(path_bytes.decode('utf-8'))
                
                meta_dict = None
                if meta_blob:
                    if is_enc:
                        if self.encryptor.is_enabled:
                            try: meta_dict = json.loads(self.encryptor.decrypt(meta_blob))
                            except (EncryptionError, json.JSONDecodeError): meta_dict = {"error": "decryption_failed"}
                        else: meta_dict = {"error": "key_unavailable"}
                    else:
                        try: meta_dict = json.loads(meta_blob.decode('utf-8'))
                        except json.JSONDecodeError: meta_dict = {"error": "parsing_failed"}
                doc_metadatas.append(meta_dict or {})
            
            X = np.array(vectors)
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)

            point_cloud_data = [{
                "x": float(X_2d[i, 0]), "y": float(X_2d[i, 1]), "chunk_id": int(chunk_id),
                "document_title": Path(doc_paths[i]).name, "document_path": doc_paths[i], "metadata": doc_metadatas[i]
            } for i, chunk_id in enumerate(chunk_ids)]

            if output_format == 'dict':
                return point_cloud_data
            elif output_format == 'csv':
                df = pd.DataFrame(point_cloud_data)
                if any(d['metadata'] for d in point_cloud_data):
                    meta_df = pd.json_normalize([d['metadata'] for d in point_cloud_data]).add_prefix('meta_')
                    df = pd.concat([df.drop('metadata', axis=1), meta_df], axis=1)
                return df.to_csv(index=False)
            else:
                return json.dumps(point_cloud_data)
