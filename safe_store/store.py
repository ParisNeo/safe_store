import sqlite3
import json
import base64
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
from safe_store.indexing.page_index import PageIndex
from safe_store.search import similarity
from safe_store.search.bm25 import BM25Retriever
from safe_store.search.fusion import reciprocal_rank_fusion
from .datalake.viewer import DatalakeViewer
from safe_store.vectorization.manager import VectorizationManager
from safe_store.vectorization.base import BaseVectorizer
from safe_store.vectorization.utils import load_vectorizer_module
from safe_store.processing.text_cleaning import get_cleaner
from safe_store.processing.tokenizers import get_tokenizer
from enum import IntEnum
from ascii_colors import ASCIIColors

try:
    from ascii_colors import LogLevel as _ASCIILogLevel
    LogLevel = _ASCIILogLevel
except (ImportError, AttributeError):
    class LogLevel(IntEnum):
        DEBUG = 10
        INFO = 20
        WARNING = 30
        ERROR = 40
        CRITICAL = 50

DEFAULT_LOCK_TIMEOUT: int = 60
TEMP_FILE_DB_INDICATOR = ":tempfile:"
IN_MEMORY_DB_INDICATOR = ":memory:"

class SafeStore:
    """
    Manages a local vector store with a single, fixed vectorizer and chunking strategy.
    Supports dense vector similarity, lexical BM25 search, and tri-modal hybrid retrieval.
    """
    DEFAULT_VECTORIZER_NAME: str = "st"
    DEFAULT_VECTORIZER_CONFIG: Dict[str, Any] = {"model": "all-MiniLM-L6-v2"}

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = "safe_store.db",
        vectorizer_name: Optional[str] = None,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        custom_vectorizers_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: Optional[Literal['character', 'token', 'paragraph', 'semantic', 'recursive', 'structure', 'markdown', 'contextual', 'late']] = None,
        custom_tokenizer: Optional[Dict[str, Any]] = None,
        expand_before: Optional[int] = None,
        expand_after: Optional[int] = None,
        text_cleaner: Optional[Union[str, Callable[[str], str]]] = None,
        remove_line_returns: bool = False,
        context_enricher: Optional[Callable[[str, str], str]] = None,
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

        explicit_kwargs = {}
        if vectorizer_name is not None: explicit_kwargs['vectorizer_name'] = vectorizer_name
        if vectorizer_config is not None: explicit_kwargs['vectorizer_config'] = vectorizer_config
        if chunk_size is not None: explicit_kwargs['chunk_size'] = chunk_size
        if chunk_overlap is not None: explicit_kwargs['chunk_overlap'] = chunk_overlap
        if chunking_strategy is not None: explicit_kwargs['chunking_strategy'] = chunking_strategy
        if custom_tokenizer is not None: explicit_kwargs['custom_tokenizer'] = custom_tokenizer
        if expand_before is not None: explicit_kwargs['expand_before'] = expand_before
        if expand_after is not None: explicit_kwargs['expand_after'] = expand_after
        if text_cleaner is not None: explicit_kwargs['text_cleaner'] = text_cleaner
        if remove_line_returns: explicit_kwargs['remove_line_returns'] = remove_line_returns
        if name is not None: explicit_kwargs['name'] = name
        if description is not None: explicit_kwargs['description'] = description
        if metadata is not None: explicit_kwargs['metadata'] = metadata
        if chunking_kwargs is not None: explicit_kwargs['chunking_kwargs'] = chunking_kwargs

        db_path_input_str = str(db_path).lower() if db_path is not None else ":memory:"
        stored_config: Dict[str, Any] = {}

        if db_path_input_str not in (":memory:", ":tempfile:"):
            db_path_resolved = str(Path(db_path).resolve())
            if Path(db_path_resolved).exists():
                try:
                    conn = db.connect_db(db_path_resolved)
                    try:
                        raw_config = db.get_store_metadata(conn, "store_config")
                        if raw_config: stored_config = json.loads(raw_config)
                    except Exception: pass
                    finally: conn.close()
                except Exception: pass

        merged_config = {**stored_config, **explicit_kwargs}

        self.vectorizer_name = merged_config.get('vectorizer_name', 'st')
        self.vectorizer_config = merged_config.get('vectorizer_config', self.DEFAULT_VECTORIZER_CONFIG if self.vectorizer_name == self.DEFAULT_VECTORIZER_NAME else {})
        self.chunk_size = merged_config.get('chunk_size', 384)
        self.chunk_overlap = merged_config.get('chunk_overlap', 50)
        self.chunking_strategy = merged_config.get('chunking_strategy', 'token')
        self.custom_tokenizer = merged_config.get('custom_tokenizer', None)
        self.expand_before = merged_config.get('expand_before', 0)
        self.expand_after = merged_config.get('expand_after', 0)
        self.remove_line_returns = merged_config.get('remove_line_returns', False)
        self.context_enricher = context_enricher
        self.text_cleaner_name = merged_config.get('text_cleaner', 'basic')
        self.text_cleaner = get_cleaner(self.text_cleaner_name, remove_line_returns=self.remove_line_returns)
        self.chunking_kwargs = merged_config.get('chunking_kwargs', {})

        self.name = merged_config.get('name', name)
        self.description = merged_config.get('description', description)
        self.metadata = merged_config.get('metadata', metadata)
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
        self._page_index: Optional[PageIndex] = None
        self._datalake_viewer: Optional[DatalakeViewer] = None

        try:
            self._connect_and_initialize()
            self._initialize_and_verify_vectorizer()
        except Exception as e:
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
            self._is_closed = True
            self._manual_cleanup_temp_files_on_error()
            raise e

    @classmethod
    def open(cls, db_path: Union[str, Path], **kwargs) -> "SafeStore":
        return cls.from_db(db_path, **kwargs)

    @classmethod
    def from_db(cls, db_path: Union[str, Path], **kwargs) -> "SafeStore":
        return cls(db_path=db_path, **kwargs)

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
            VectorizerClass = getattr(module, module.class_name, None) if hasattr(module, 'class_name') else None
            if VectorizerClass and issubclass(VectorizerClass, BaseVectorizer) and hasattr(VectorizerClass, 'list_models'):
                return VectorizerClass.list_models(**kwargs)
            return []
        except Exception as e:
            raise SafeStoreError(f"Error listing models for '{vectorizer_name}': {e}") from e

    def _cleanup_stale_locks(self):
        if self._file_lock and self.lock_path and Path(self.lock_path).exists():
            try:
                self._file_lock.acquire(timeout=0.01)
                self._file_lock.release()
                Path(self.lock_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _initialize_and_verify_vectorizer(self):
        self.vectorizer = self.vectorizer_manager.get_vectorizer(self.vectorizer_name, self.vectorizer_config)

        if self.chunking_strategy in ['token', 'paragraph', 'semantic', 'recursive']:
            tokenizer = self.vectorizer.get_tokenizer()
            if tokenizer is not None:
                self.tokenizer_for_chunking = tokenizer
            elif self.custom_tokenizer is not None:
                self.tokenizer_for_chunking = get_tokenizer(self.custom_tokenizer)
            else:
                self.tokenizer_for_chunking = get_tokenizer({"name": "tiktoken", "model": "cl100k_base"})

        with self._optional_file_lock_context("verify vectorizer compatibility"):
            assert self.conn is not None
            stored_info_json = db.get_store_metadata(self.conn, "vectorizer_info")
            
            if stored_info_json:
                stored_info = json.loads(stored_info_json)
                unique_name_from_instance = self.vectorizer_manager._create_unique_name(self.vectorizer_name, self.vectorizer_config)
                
                if stored_info.get("unique_name") != unique_name_from_instance:
                    raise ConfigurationError(
                        f"Database at '{self.db_path}' has an incompatible vectorizer: '{stored_info.get('unique_name')}'. "
                        f"This instance is configured with '{unique_name_from_instance}'."
                    )
            else:
                vectorizer_info = {
                    "unique_name": self.vectorizer_manager._create_unique_name(self.vectorizer_name, self.vectorizer_config),
                    "name": self.vectorizer_name,
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

            existing_config = db.get_store_metadata(self.conn, "store_config")
            if existing_config is None:
                store_config = {
                    'vectorizer_name': self.vectorizer_name,
                    'vectorizer_config': self.vectorizer_config,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'chunking_strategy': self.chunking_strategy,
                    'expand_before': self.expand_before,
                    'expand_after': self.expand_after,
                    'text_cleaner': self.text_cleaner_name,
                    'chunking_kwargs': self.chunking_kwargs,
                }
                db.set_store_metadata(self.conn, "store_config", json.dumps(store_config))

            self.conn.commit()
        except Exception as e:
            if self.conn.in_transaction: self.conn.rollback()
            raise SafeStoreError("Failed to load/initialize store properties") from e
    
    @contextmanager
    def _optional_file_lock_context(self, description: Optional[str] = None) -> ContextManager[None]:
        if self._file_lock:
            try:
                op_name = description.split(':')[0]
                ASCIIColors.debug(f"Attempting to acquire write lock for {op_name}")
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for {op_name}")
                    yield
                ASCIIColors.debug(f"Write lock released for {description}")
            except Timeout as e:
                msg = f"Timeout ({self.lock_timeout}s) acquiring write lock for {description}"
                ASCIIColors.error(msg)
                raise ConcurrencyError(msg) from e
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
            ASCIIColors.info("safe_store connection closed.")

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
        ASCIIColors.debug("safe_store context closed cleanly.")
        self.close()

    def _ensure_connection(self) -> None:
        if self._is_closed or self.conn is None:
            self._connect_and_initialize()
            self._initialize_and_verify_vectorizer()

    @property
    def page_index(self) -> PageIndex:
        if self._page_index is None:
            self._page_index = PageIndex(self)
        return self._page_index

    @property
    def datalake(self) -> DatalakeViewer:
        if self._datalake_viewer is None:
            self._datalake_viewer = DatalakeViewer(self)
        return self._datalake_viewer

    def get_database_info(self, print_summary: bool = False) -> Dict[str, Any]:
        """
        Gathers comprehensive diagnostics and metadata about the database:
        - Vectorizer info (name, config, dimension, dtype)
        - Chunking and cleaning configuration
        - Documents summary with individual chunk counts and metadata
        - Ontology schema presence & summary
        - Knowledge Graph metrics (total nodes, relationships, label counts)
        """
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None

            cursor = self.conn.cursor()

            # 1. Document metrics & chunk counts per document
            cursor.execute("""
                SELECT d.doc_id, d.file_path, d.file_hash, d.added_timestamp, d.is_encrypted,
                       COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.doc_id = c.doc_id
                GROUP BY d.doc_id
                ORDER BY d.doc_id ASC
            """)
            raw_docs = cursor.fetchall()

            docs_info = []
            total_chunks = 0
            for r in raw_docs:
                doc_id, f_path, f_hash, ts, is_enc, c_count = r
                total_chunks += (c_count or 0)

                # Fetch metadata for this doc
                doc_row = db.get_document_record_by_id(self.conn, doc_id)
                meta_dict = None
                if doc_row and doc_row[4]:
                    meta_b = doc_row[4]
                    if is_enc and self.encryptor.is_enabled:
                        try: meta_dict = json.loads(self.encryptor.decrypt(meta_b))
                        except Exception: meta_dict = {"encrypted": True}
                    elif not is_enc:
                        try: meta_dict = json.loads(meta_b.decode('utf-8'))
                        except Exception: meta_dict = {}

                docs_info.append({
                    "doc_id": doc_id,
                    "file_path": f_path,
                    "document_title": Path(f_path).name,
                    "chunk_count": c_count or 0,
                    "file_hash": f_hash,
                    "added_timestamp": ts,
                    "is_encrypted": bool(is_enc),
                    "metadata": meta_dict
                })

            # 2. Vectorizer Details
            v_details = self.get_vectorization_details() or {
                "name": self.vectorizer_name,
                "config": self.vectorizer_config,
                "dim": getattr(self.vectorizer, 'dim', None),
                "dtype": str(getattr(self.vectorizer, 'dtype', 'float32'))
            }

            # 3. Knowledge Graph Metrics
            graph_info = None
            try:
                cursor.execute("SELECT COUNT(*) FROM graph_nodes")
                total_nodes = cursor.fetchone()[0] or 0

                cursor.execute("SELECT COUNT(*) FROM graph_relationships")
                total_rels = cursor.fetchone()[0] or 0

                cursor.execute("SELECT node_label, COUNT(*) FROM graph_nodes GROUP BY node_label")
                node_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

                cursor.execute("SELECT relationship_type, COUNT(*) FROM graph_relationships GROUP BY relationship_type")
                rel_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

                cursor.execute("SELECT COUNT(*) FROM node_chunk_links")
                total_links = cursor.fetchone()[0] or 0

                graph_info = {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "total_provenance_links": total_links,
                    "nodes_by_label": node_breakdown,
                    "relationships_by_type": rel_breakdown
                }
            except Exception:
                graph_info = {"total_nodes": 0, "total_relationships": 0, "nodes_by_label": {}, "relationships_by_type": {}}

            # 4. Ontology Inspection
            ontology_info = None
            if self.metadata and isinstance(self.metadata, dict) and "ontology" in self.metadata:
                ont = self.metadata["ontology"]
                if isinstance(ont, dict):
                    ontology_info = {
                        "defined_classes": list(ont.get("nodes", {}).keys()),
                        "defined_relationships": list(ont.get("relationships", {}).keys())
                    }
                else:
                    ontology_info = {"raw": str(ont)[:300]}

            info_payload = {
                "database_path": self.db_path,
                "store_name": self.name,
                "store_description": self.description,
                "is_in_memory": self._is_in_memory,
                "encryption_enabled": self.encryptor.is_enabled,
                "vectorizer": v_details,
                "chunking_configuration": {
                    "strategy": self.chunking_strategy,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "expand_before": self.expand_before,
                    "expand_after": self.expand_after,
                    "text_cleaner": self.text_cleaner_name
                },
                "documents": {
                    "total_documents": len(docs_info),
                    "total_chunks": total_chunks,
                    "list": docs_info
                },
                "knowledge_graph": graph_info,
                "ontology": ontology_info
            }

            if print_summary:
                self._print_database_info_summary(info_payload)

            return info_payload

    def info(self, print_summary: bool = True) -> Dict[str, Any]:
        """Convenience alias for get_database_info(). Default prints formatted summary."""
        return self.get_database_info(print_summary=print_summary)

    def get_infos(self, print_summary: bool = False) -> Dict[str, Any]:
        """Convenience alias for get_database_info()."""
        return self.get_database_info(print_summary=print_summary)

    def _print_database_info_summary(self, info: Dict[str, Any]) -> None:
        """Prints a human-readable, diagnostic ASCII summary of the database state."""
        ASCIIColors.panel(f"""[bold cyan]SafeStore Database Information[/bold cyan]
Database Path : {info['database_path']}
Store Name    : {info['store_name']} ({info['store_description'] or 'No description'})
Encrypted     : {'[green]YES (Fernet)[/green]' if info['encryption_enabled'] else '[yellow]NO[/yellow]'}

[bold yellow]── Vectorizer ──[/bold yellow]
Name          : {info['vectorizer'].get('name', 'st')}
Dimension     : {info['vectorizer'].get('dim', 'N/A')}
Dtype         : {info['vectorizer'].get('dtype', 'N/A')}

[bold yellow]── Chunking & Processing ──[/bold yellow]
Strategy      : {info['chunking_configuration']['strategy']}
Chunk Size    : {info['chunking_configuration']['chunk_size']} tokens/chars (Overlap: {info['chunking_configuration']['chunk_overlap']})
Text Cleaner  : {info['chunking_configuration']['text_cleaner']}

[bold yellow]── Documents ({info['documents']['total_documents']} docs, {info['documents']['total_chunks']} chunks) ──[/bold yellow]""" + "".join(
    f"\n  • [ID {d['doc_id']}] {d['document_title']}: {d['chunk_count']} chunks ({'Encrypted' if d['is_encrypted'] else 'Plaintext'})"
    for d in info['documents']['list']
) + f"""

[bold yellow]── Knowledge Graph ──[/bold yellow]
Total Nodes   : {info['knowledge_graph']['total_nodes']} ({info['knowledge_graph']['nodes_by_label']})
Total Edges   : {info['knowledge_graph']['total_relationships']} ({info['knowledge_graph']['relationships_by_type']})
Chunk Links   : {info['knowledge_graph']['total_provenance_links']}

[bold yellow]── Ontology ──[/bold yellow]
{info['ontology'] if info['ontology'] else 'None configured'}
""", "[bold][magenta]DATABASE DIAGNOSTICS[/bold][/magenta]")

    def get_properties(self) -> Dict[str, Any]:
        """Returns the high-level configuration and metadata of the store."""
        with self._instance_lock:
            self._ensure_connection()
            return {
                "name": self.name,
                "description": self.description,
                "metadata": self.metadata,
                "vectorizer_name": self.vectorizer_name,
                "vectorizer_config": self.vectorizer_config,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunking_strategy": self.chunking_strategy
            }

    def update_properties(
        self,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite_metadata: bool = False
    ) -> None:
        """Updates store description or metadata properties with persistence."""
        with self._instance_lock, self._optional_file_lock_context("update_properties"):
            self._ensure_connection()
            assert self.conn is not None
            try:
                self.conn.execute("BEGIN")
                if properties:
                    if "name" in properties:
                        self.name = properties["name"]
                        db.set_store_metadata(self.conn, "store_name", self.name or "")
                    if "description" in properties:
                        self.description = properties["description"]
                        db.set_store_metadata(self.conn, "store_description", self.description or "")
                    if "metadata" in properties:
                        metadata = properties["metadata"]
                if metadata is not None:
                    if overwrite_metadata or self.metadata is None:
                        self.metadata = metadata
                    else:
                        self.metadata.update(metadata)
                    db.set_store_metadata(self.conn, "store_metadata", json.dumps(self.metadata))
                self.conn.commit()
            except Exception as e:
                if self.conn.in_transaction:
                    self.conn.rollback()
                raise SafeStoreError(f"Failed to update store properties: {e}") from e

    def clear_datalake_cache(self) -> int:
        """Clears cached 2D/3D projections from the database."""
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None
            return db.clear_projection_cache(self.conn)

    def _get_file_hash(self, file_path: Path) -> str:
        hasher = self._file_hasher()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192): hasher.update(chunk)
        return hasher.hexdigest()

    def _get_text_hash(self, text: str) -> str:
        hasher = self._file_hasher()
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    @property
    def DEFAULT_VECTORIZER(self):
        return self.vectorizer_name

    def add_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        force_reindex: bool = False,
        vectorize_with_metadata: bool = True,
        chunk_processor: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        skip_chunking: bool = False,
        remove_line_returns: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ) -> Dict[str, int]:
        with self._instance_lock:
            self._ensure_connection()
            return self._add_content_impl(
                content_id=str(Path(file_path).resolve()),
                content_loader=lambda: parser.parse_document(file_path),
                hash_loader=lambda: self._get_file_hash(Path(file_path)),
                metadata=metadata,
                tags=tags,
                force_reindex=force_reindex,
                vectorize_with_metadata=vectorize_with_metadata,
                chunk_processor=chunk_processor,
                skip_chunking=skip_chunking,
                op_chunk_size=chunk_size,
                op_chunk_overlap=chunk_overlap,
                op_chunking_strategy=chunking_strategy
            )

    def add_text(
        self,
        unique_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        force_reindex: bool = False,
        vectorize_with_metadata: bool = True,
        chunk_processor: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        skip_chunking: bool = False,
        remove_line_returns: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: Optional[str] = None
    ) -> Dict[str, int]:
        with self._instance_lock:
            self._ensure_connection()
            return self._add_content_impl(
                content_id=unique_id,
                content_loader=lambda: text,
                hash_loader=lambda: self._get_text_hash(text),
                metadata=metadata,
                tags=tags,
                force_reindex=force_reindex,
                vectorize_with_metadata=vectorize_with_metadata,
                chunk_processor=chunk_processor,
                skip_chunking=skip_chunking,
                op_chunk_size=chunk_size,
                op_chunk_overlap=chunk_overlap,
                op_chunking_strategy=chunking_strategy
            )

    def _add_content_impl(self, content_id, content_loader, hash_loader, metadata, tags, force_reindex, vectorize_with_metadata, chunk_processor, skip_chunking, remove_line_returns=None, op_chunk_size=None, op_chunk_overlap=None, op_chunking_strategy=None) -> Dict[str, int]:
        self._ensure_connection()
        assert self.conn and self.vectorizer is not None
        filename_for_log = Path(content_id).name

        with self._optional_file_lock_context(f"add_document: {filename_for_log}"):
            try:
                current_hash = hash_loader()
            except (OSError, FileNotFoundError) as e:
                msg = f"File not found when trying to hash: {content_id}"
                ASCIIColors.error(f"Error during add_document: FileHandlingError: {msg}")
                raise FileHandlingError(msg) from e
            except FileHandlingError as e:
                ASCIIColors.error(f"Error during add_document: FileHandlingError: {e}")
                raise

            res = self.conn.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (content_id,)).fetchone()
            existing_doc_id, existing_hash = res if res else (None, None)

            if not force_reindex and existing_hash == current_hash and existing_doc_id:
                if self.conn.execute("SELECT 1 FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? LIMIT 1", (existing_doc_id,)).fetchone():
                    ASCIIColors.info(f"Document '{filename_for_log}' is unchanged.")
                    ASCIIColors.success(f"Vectorization '{self.vectorizer_name}' already exists for unchanged '{filename_for_log}'. Skipping.")
                    return {"num_chunks_added": 0, "num_chunks_ignored": 0}
            
            if existing_doc_id and existing_hash != current_hash:
                ASCIIColors.warning(f"Document '{filename_for_log}' has changed (hash mismatch). Re-indexing...")
            elif existing_doc_id and force_reindex:
                ASCIIColors.info(f"Force re-indexing requested for '{filename_for_log}'.")
            else:
                ASCIIColors.info(f"Document '{filename_for_log}' is new.")

            ASCIIColors.info(f"Starting indexing process for: {filename_for_log}")
            
            try:
                full_text = content_loader()
            except ConfigurationError as e:
                ASCIIColors.warning(str(e))
                ASCIIColors.error(f"Error during add_document: ConfigurationError: {e}")
                raise e
            except ParsingError as e:
                if "encrypted" in str(e).lower():
                    ASCIIColors.warning(f"File '{content_id}' content is encrypted. Skipping.")
                    return {"num_chunks_added": 0, "num_chunks_ignored": 0}
                ASCIIColors.warning(f"File '{content_id}' is empty or not readable. Error: {e}")
                return {"num_chunks_added": 0, "num_chunks_ignored": 0}
            except (FileHandlingError, OSError) as e:
                ASCIIColors.error(f"Error during add_document: FileHandlingError: {e}")
                raise e
            except Exception as e:
                 ASCIIColors.error(f"Error during add_document: {type(e).__name__}: {e}")
                 raise e

            if not full_text or not full_text.strip():
                 ASCIIColors.warning(f"No chunks generated for {filename_for_log}. Document record saved, but skipping vectorization.")
                 try:
                    self.conn.execute("BEGIN")
                    db.add_document_record(self.conn, file_path=content_id, file_hash=current_hash, full_text=full_text, is_encrypted=self.encryptor.is_enabled)
                    self.conn.commit()
                 except Exception:
                    if self.conn.in_transaction: self.conn.rollback()
                 return {"num_chunks_added": 0, "num_chunks_ignored": 0}

            should_remove_lr = self.remove_line_returns if remove_line_returns is None else remove_line_returns
            cleaner = get_cleaner(self.text_cleaner_name, remove_line_returns=should_remove_lr) if should_remove_lr != self.remove_line_returns else self.text_cleaner
            cleaned_text = cleaner(full_text)
            
            c_size = op_chunk_size if op_chunk_size is not None else self.chunk_size
            c_overlap = op_chunk_overlap if op_chunk_overlap is not None else self.chunk_overlap
            c_strategy = op_chunking_strategy if op_chunking_strategy is not None else self.chunking_strategy

            raw_chunks_data = []
            if skip_chunking:
                storage_text = cleaned_text
                v_text = self.tokenizer_for_chunking.decode(self.tokenizer_for_chunking.encode(cleaned_text)[:c_size]) if self.tokenizer_for_chunking else cleaned_text[:c_size]
                raw_chunks_data = [(v_text, storage_text)]
            else:
                raw_chunks_data = chunking.generate_chunks(
                    text=cleaned_text, 
                    strategy=c_strategy, 
                    chunk_size=c_size,
                    chunk_overlap=c_overlap, 
                    expand_before=self.expand_before,
                    expand_after=self.expand_after, 
                    tokenizer=self.tokenizer_for_chunking,
                    vectorizer_fn=self.vectorizer.vectorize if c_strategy == 'semantic' else None,
                    **self.chunking_kwargs
                )

            processed_chunks_data = [(chunk_processor(v, metadata or {}) if chunk_processor else v, chunk_processor(s, metadata or {}) if chunk_processor else s) for v, s in raw_chunks_data]
            valid_chunks_data = [chunk for chunk in processed_chunks_data if chunk[0] and chunk[0].strip()]
            num_ignored = len(processed_chunks_data) - len(valid_chunks_data)

            if not valid_chunks_data:
                ASCIIColors.warning(f"No chunks generated for {filename_for_log}. Document record saved, but skipping vectorization.")
                return {"num_chunks_added": 0, "num_chunks_ignored": num_ignored}

            ASCIIColors.info(f"Generated {len(valid_chunks_data)} chunks for {filename_for_log}")
            ASCIIColors.info(f"Vectorizing {len(valid_chunks_data)} chunks using '{self.vectorizer_name}'")

            vector_texts = [item[0] for item in valid_chunks_data]
            storage_texts = [item[1] for item in valid_chunks_data]
            
            if vectorize_with_metadata and metadata:
                metadata_string = "--- Document Context ---\n" + "\n".join(f"{str(k).title()}: {str(v)}" for k, v in metadata.items()) + "\n------------------------\n\n"
                vector_texts = [metadata_string + text for text in vector_texts]

            if hasattr(self.vectorizer, 'fit') and hasattr(self.vectorizer, '_fitted') and not self.vectorizer._fitted:
                if self.vectorizer_name in ['tf_idf', 'tfidf']:
                    ASCIIColors.warning(f"TF-IDF vectorizer '{self.vectorizer_name}' is not fitted. Fitting ONLY on chunks from '{filename_for_log}'")
                self.vectorizer.fit(vector_texts)

            vectors = self.vectorizer.vectorize(vector_texts)

            try:
                if not self.conn.in_transaction:
                    self.conn.execute("BEGIN")
                doc_id = db.get_document_id_by_path(self.conn, content_id)
                if doc_id:
                    self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                    ASCIIColors.debug("Deleted old chunks/vectors")

                meta_blob = self.encryptor.encrypt(json.dumps(metadata)) if metadata and self.encryptor.is_enabled else (json.dumps(metadata).encode('utf-8') if metadata else None)

                if doc_id:
                    self.conn.execute("UPDATE documents SET file_hash=?, full_text=?, metadata=?, is_encrypted=? WHERE doc_id=?", (current_hash, full_text, meta_blob, 1 if self.encryptor.is_enabled else 0, doc_id))
                else:
                    doc_id = db.add_document_record(self.conn, content_id, current_hash, full_text, meta_blob, self.encryptor.is_enabled)

                tags_str = ",".join(sorted(list(set(tags)))) if tags else None
                for i, storage_text in enumerate(storage_texts):
                    t_store = self.encryptor.encrypt(storage_text) if self.encryptor.is_enabled else storage_text
                    cid = db.add_chunk_record(self.conn, doc_id, t_store, 0, 0, i, tags=tags_str, is_encrypted=self.encryptor.is_enabled)
                    db.add_vector_record(self.conn, cid, np.ascontiguousarray(vectors[i], dtype=self.vectorizer.dtype))

                self.clear_datalake_cache()
                self.conn.commit()

                if doc_id and hasattr(self.vectorizer, 'on_document_indexed') and callable(getattr(self.vectorizer, 'on_document_indexed')):
                    try:
                        self.vectorizer.on_document_indexed(self.conn, doc_id, storage_texts)
                    except Exception as hook_err:
                        ASCIIColors.warning(f"Vectorizer hook 'on_document_indexed' failed: {hook_err}")

                ASCIIColors.success(f"Successfully processed '{filename_for_log}' with vectorizer '{self.vectorizer_name}'")
                return {"num_chunks_added": len(valid_chunks_data), "num_chunks_ignored": num_ignored}

            except Exception as e:
                if self.conn and self.conn.in_transaction: self.conn.rollback()
                raise SafeStoreError(f"Database transaction failed for '{content_id}': {e}") from e

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_relevance_percent: float = 0.0,
        min_similarity_percent: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector store using dense vector similarity.
        Every result contains a standardized 0-100 grade (relevance_score & similarity_percent).
        Results below min_relevance_percent are excluded.
        """
        threshold = min_similarity_percent if min_similarity_percent is not None else min_relevance_percent
        with self._instance_lock:
            ASCIIColors.info(f"Received query. Searching with '{self.vectorizer_name}', top_k={top_k}, threshold={threshold}%")

            self._ensure_connection()
            assert self.conn and self.vectorizer is not None
            custom_search = getattr(self.vectorizer, 'custom_search', None)
            if callable(custom_search):
                return custom_search(self.conn, query_text, top_k, threshold)

            query_vector = self.vectorizer.vectorize([query_text])[0]

            with self._optional_file_lock_context("query - fetch vectors"):
                self._ensure_connection()
                assert self.conn is not None
                all_vectors_data = self.conn.execute("SELECT v.chunk_id, v.vector_data FROM vectors v").fetchall()

            if not all_vectors_data: 
                return []

            chunk_ids, vector_blobs = zip(*all_vectors_data)
            candidate_vectors = np.array([db.reconstruct_vector(blob, self.vectorizer.dtype.name) for blob in vector_blobs])

            scores = similarity.cosine_similarity(query_vector, candidate_vectors)

            # Map cosine similarity [-1.0, 1.0] to [0.0, 100.0]% grade
            percent_grades = np.clip(((scores + 1.0) / 2.0) * 100.0, 0.0, 100.0)
            pass_mask = percent_grades >= threshold

            if not np.any(pass_mask): 
                return []

            scores_passing = scores[pass_mask]
            grades_passing = percent_grades[pass_mask]
            chunk_ids_passing = np.array(chunk_ids)[pass_mask]

            k = min(top_k, len(scores_passing)) if top_k > 0 else len(scores_passing)
            top_indices = np.argsort(grades_passing)[::-1][:k]
            top_chunk_ids = chunk_ids_passing[top_indices]
            top_scores = scores_passing[top_indices]
            top_grades = grades_passing[top_indices]

            with self._optional_file_lock_context("query - fetch details"):
                self._ensure_connection()
                assert self.conn is not None
                placeholders = ','.join('?' * len(top_chunk_ids))
                sql = f"""
                    SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos,
                           c.is_encrypted AS chunk_is_encrypted, d.file_path,
                           d.metadata AS doc_metadata, d.is_encrypted AS doc_is_encrypted,
                           c.doc_id
                    FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.chunk_id IN ({placeholders})
                """
                
                details_map = {}
                original_factory = self.conn.text_factory
                self.conn.text_factory = bytes
                details_raw = self.conn.execute(sql, tuple(top_chunk_ids.tolist())).fetchall()
                self.conn.text_factory = original_factory

            for row in details_raw:
                chunk_id, chunk_text_data, start, end, chunk_is_enc, path, doc_meta_data, doc_is_enc, doc_id = row
                
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
                    for kd, vd in meta_dict.items(): doc_metadata_text += f"{str(kd).title()}: {str(vd)}\n"
                    doc_metadata_text += "------------------------\n\n"

                details_map[chunk_id] = {
                    "chunk_id": int(chunk_id),
                    "doc_id": int(doc_id),
                    "chunk_text": doc_metadata_text + chunk_text, "start_pos": start, "end_pos": end,
                    "file_path": path.decode('utf-8'), "document_metadata": meta_dict
                }
            
            ordered_results = []
            for cid, s, g in zip(top_chunk_ids, top_scores, top_grades):
                cid_int = int(cid)
                res = dict(details_map.get(cid_int, {}))
                grade_val = float(round(g, 2))
                res.update({
                    "chunk_id": cid_int,
                    "similarity_score": float(s),
                    "similarity_percent": grade_val,
                    "relevance_score": grade_val
                })
                ordered_results.append(res)
            return ordered_results

    def hybrid_query(
        self,
        query_text: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        min_relevance_percent: float = 0.0,
        min_similarity_percent: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a Tri-Modal Hybrid query combining Dense Vector search and Sparse BM25 lexical search via RRF.
        Standardizes relevance onto a 0-100 grade and filters out results below the threshold.
        """
        threshold = min_similarity_percent if min_similarity_percent is not None else min_relevance_percent
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None

            # 1. Fetch Dense Vector Results (without early false-cutoff)
            dense_results = self.query(query_text, top_k=top_k * 3, min_relevance_percent=0.0)

            # 2. Fetch BM25 Lexical Results
            bm25_retriever = BM25Retriever(self.conn)
            bm25_results = bm25_retriever.search(query_text, top_k=top_k * 3, min_relevance_percent=0.0)

            # 3. Fuse Results with Score-Calibrated Reciprocal Rank Fusion
            fused = reciprocal_rank_fusion(
                ranked_lists=[dense_results, bm25_results],
                weights=[dense_weight, bm25_weight],
                k=rrf_k,
                top_k=top_k,
                min_relevance_percent=threshold
            )

            return fused

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
                rows = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                self.clear_datalake_cache()
                self.conn.commit()
                if rows > 0: ASCIIColors.success(f"Deleted document ID {doc_id}.")
            except sqlite3.Error as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise DatabaseError from e

    def revectorize_database(
        self,
        new_vectorizer_name: str,
        new_vectorizer_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 50
    ) -> None:
        """
        Re-embeds all chunks in the database using a new vectorizer.
        Atomically updates the vectorizer info and all vector records.
        """
        with self._instance_lock, self._optional_file_lock_context("revectorize_database"):
            self._ensure_connection()
            assert self.conn is not None

            ASCIIColors.info(f"Initializing new vectorizer '{new_vectorizer_name}' for re-vectorization...")
            new_vectorizer = self.vectorizer_manager.get_vectorizer(new_vectorizer_name, new_vectorizer_config or {})

            if new_vectorizer.dim is None:
                raise ConfigurationError(f"Vectorizer '{new_vectorizer_name}' has an unknown dimension.")

            # Fetch all chunks
            cursor = self.conn.cursor()
            cursor.execute("SELECT chunk_id, chunk_text, is_encrypted FROM chunks ORDER BY chunk_id ASC")
            all_chunks = cursor.fetchall()

            if not all_chunks:
                ASCIIColors.warning("No chunks found in the database to re-vectorize.")
                return

            ASCIIColors.info(f"Extracting and decrypting {len(all_chunks)} chunks...")
            chunk_ids = []
            texts_to_vectorize = []

            for cid, text_data, is_enc in all_chunks:
                if is_enc:
                    if not self.encryptor.is_enabled:
                        raise EncryptionError("Cannot re-vectorize encrypted chunks without the decryption key.")
                    try:
                        text = self.encryptor.decrypt(text_data)
                    except EncryptionError as e:
                        raise EncryptionError(f"Failed to decrypt chunk {cid} during re-vectorization: {e}")
                else:
                    text = text_data.decode('utf-8') if isinstance(text_data, bytes) else str(text_data)

                chunk_ids.append(cid)
                texts_to_vectorize.append(text)

            # Vectorize in batches
            new_vectors = []
            total = len(texts_to_vectorize)
            for i in range(0, total, batch_size):
                batch = texts_to_vectorize[i:i + batch_size]
                ASCIIColors.info(f"Vectorizing batch {i//batch_size + 1}/{(total//batch_size)+1} ({len(batch)} chunks)...")

                if hasattr(new_vectorizer, 'fit') and not getattr(new_vectorizer, '_fitted', True):
                    new_vectorizer.fit(batch)

                batch_vecs = new_vectorizer.vectorize(batch)
                new_vectors.extend(batch_vecs)

            if len(new_vectors) != total:
                raise VectorizationError(f"Vectorizer returned {len(new_vectors)} vectors, expected {total}.")

            # Atomically update database
            try:
                self.conn.execute("BEGIN")

                # Update vectorizer metadata
                unique_name = self.vectorizer_manager._create_unique_name(new_vectorizer_name, new_vectorizer_config)
                vectorizer_info = {
                    "unique_name": unique_name,
                    "name": new_vectorizer_name,
                    "vectorizer_name": new_vectorizer_name,
                    "vectorizer_config": new_vectorizer_config or {},
                    "dim": new_vectorizer.dim,
                    "dtype": new_vectorizer.dtype.name,
                }
                db.set_store_metadata(self.conn, "vectorizer_info", json.dumps(vectorizer_info))

                # Store new config in store_config
                raw_store_config = db.get_store_metadata(self.conn, "store_config")
                store_config = json.loads(raw_store_config) if raw_store_config else {}
                store_config['vectorizer_name'] = new_vectorizer_name
                store_config['vectorizer_config'] = new_vectorizer_config or {}
                db.set_store_metadata(self.conn, "store_config", json.dumps(store_config))

                # Update vectors
                ASCIIColors.info("Updating vector records in database...")
                for cid, vec in zip(chunk_ids, new_vectors):
                    contiguous_vec = np.ascontiguousarray(vec, dtype=new_vectorizer.dtype)
                    self.conn.execute("UPDATE vectors SET vector_data = ? WHERE chunk_id = ?", (contiguous_vec, cid))

                self.clear_datalake_cache()
                self.conn.commit()

                # Update instance state
                self.vectorizer_name = new_vectorizer_name
                self.vectorizer_config = new_vectorizer_config or {}
                self.vectorizer = new_vectorizer

                ASCIIColors.success(f"Database successfully re-vectorized using '{new_vectorizer_name}'.")

            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise SafeStoreError(f"Failed to commit re-vectorization to database: {e}") from e

    def export_database(self, output_path: Union[str, Path], decrypt: bool = False) -> None:
        """
        Exports the entire database state to a portable JSON file.
        If decrypt is True, encrypted chunks and metadata are exported as plaintext.
        """
        with self._instance_lock, self._optional_file_lock_context("export_database"):
            self._ensure_connection()
            assert self.conn is not None

            if decrypt and not self.encryptor.is_enabled:
                # Check if there is actually encrypted data
                cursor = self.conn.execute("SELECT 1 FROM chunks WHERE is_encrypted = 1 LIMIT 1")
                if cursor.fetchone():
                    raise EncryptionError("Cannot export decrypted data: encryption key not provided.")

            export_data = {
                "version": "3.6.0",
                "metadata": {},
                "documents": [],
                "chunks": [],
                "vectors": [],
                "graph_nodes": [],
                "graph_relationships": [],
                "node_chunk_links": []
            }

            original_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            cursor = self.conn.cursor()

            try:
                # Metadata
                cursor.execute("SELECT key, value FROM store_metadata")
                for k, v in cursor.fetchall():
                    export_data["metadata"][k.decode('utf-8')] = v.decode('utf-8')

                # Documents
                cursor.execute("SELECT doc_id, file_path, file_hash, full_text, metadata, is_encrypted, added_timestamp FROM documents")
                for row in cursor.fetchall():
                    doc_id, path_b, hash_b, text_b, meta_b, is_enc, ts = row
                    full_text = self._decrypt_payload(text_b, bool(is_enc), "Encrypted Document") if decrypt else (text_b.decode('utf-8') if text_b else None)
                    metadata = self._decrypt_payload(meta_b, bool(is_enc), "Encrypted Metadata") if decrypt else (meta_b.decode('utf-8') if meta_b else None)

                    export_data["documents"].append({
                        "doc_id": doc_id,
                        "file_path": path_b.decode('utf-8'),
                        "file_hash": hash_b.decode('utf-8') if hash_b else None,
                        "full_text": full_text,
                        "metadata": metadata,
                        "is_encrypted": bool(is_enc) if not decrypt else False,
                        "added_timestamp": ts.decode('utf-8') if isinstance(ts, bytes) else ts
                    })

                # Chunks
                cursor.execute("SELECT chunk_id, doc_id, chunk_text, start_pos, end_pos, chunk_seq, tags, is_encrypted, encryption_metadata, graph_processed_at FROM chunks")
                for row in cursor.fetchall():
                    cid, did, text_b, start, end, seq, tags_b, is_enc, enc_meta_b, gpa = row
                    text = self._decrypt_payload(text_b, bool(is_enc), "Encrypted Chunk") if decrypt else (text_b.decode('utf-8') if text_b else "")

                    export_data["chunks"].append({
                        "chunk_id": cid,
                        "doc_id": did,
                        "chunk_text": text,
                        "start_pos": start,
                        "end_pos": end,
                        "chunk_seq": seq,
                        "tags": tags_b.decode('utf-8') if tags_b else None,
                        "is_encrypted": bool(is_enc) if not decrypt else False,
                        "encryption_metadata": enc_meta_b.decode('utf-8') if enc_meta_b else None,
                        "graph_processed_at": gpa.decode('utf-8') if isinstance(gpa, bytes) else gpa
                    })

                # Vectors
                cursor.execute("SELECT chunk_id, vector_data FROM vectors")
                for cid, v_blob in cursor.fetchall():
                    export_data["vectors"].append({
                        "chunk_id": cid,
                        "vector_data": base64.b64encode(v_blob).decode('utf-8')
                    })

                # Graph Nodes
                cursor.execute("SELECT node_id, node_label, node_properties, unique_signature, node_vector FROM graph_nodes")
                for nid, label, props, sig, vec_blob in cursor.fetchall():
                    export_data["graph_nodes"].append({
                        "node_id": nid,
                        "node_label": label,
                        "node_properties": props,
                        "unique_signature": sig,
                        "node_vector": base64.b64encode(vec_blob).decode('utf-8') if vec_blob else None
                    })

                # Graph Relationships
                cursor.execute("SELECT relationship_id, source_node_id, target_node_id, relationship_type, relationship_properties FROM graph_relationships")
                for rid, src, tgt, rtype, props in cursor.fetchall():
                    export_data["graph_relationships"].append({
                        "relationship_id": rid,
                        "source_node_id": src,
                        "target_node_id": tgt,
                        "relationship_type": rtype,
                        "relationship_properties": props
                    })

                # Node Chunk Links
                cursor.execute("SELECT node_id, chunk_id FROM node_chunk_links")
                for nid, cid in cursor.fetchall():
                    export_data["node_chunk_links"].append({
                        "node_id": nid,
                        "chunk_id": cid
                    })

                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(export_data, indent=2), encoding='utf-8')
                ASCIIColors.success(f"Database successfully exported to {out_path}")

            finally:
                self.conn.text_factory = original_factory

    @classmethod
    def import_database(
        cls,
        input_path: Union[str, Path],
        db_path: Union[str, Path],
        decryption_key: Optional[str] = None,
        encryption_key: Optional[str] = None,
        **kwargs
    ) -> "SafeStore":
        """
        Imports a database state from a JSON export file.
        If the data is encrypted, decryption_key must be provided.
        If encryption_key is provided, the new database will be encrypted.
        """
        in_path = Path(input_path)
        if not in_path.exists():
            raise FileHandlingError(f"Import file not found: {in_path}")

        try:
            export_data = json.loads(in_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            raise SafeStoreError(f"Invalid import file format: {e}")

        # Initialize new store
        store = cls(db_path=db_path, encryption_key=encryption_key, **kwargs)

        with store._instance_lock, store._optional_file_lock_context("import_database"):
            store._ensure_connection()
            assert store.conn is not None

            try:
                store.conn.execute("BEGIN")
                cursor = store.conn.cursor()

                # Import Metadata
                for k, v in export_data.get("metadata", {}).items():
                    db.set_store_metadata(store.conn, k, v)

                # Import Documents
                doc_id_map = {}
                for doc in export_data.get("documents", []):
                    full_text = doc["full_text"]
                    metadata = doc["metadata"]
                    is_enc = doc["is_encrypted"]

                    if is_enc and decryption_key:
                        temp_enc = Encryptor(decryption_key)
                        full_text = temp_enc.decrypt(full_text.encode('utf-8')) if full_text else None
                        metadata = temp_enc.decrypt(metadata.encode('utf-8')) if metadata else None
                        is_enc = False

                    # Re-encrypt if target store has encryption
                    final_is_enc = store.encryptor.is_enabled
                    t_store = store.encryptor.encrypt(full_text) if final_is_enc and full_text else full_text
                    m_store = store.encryptor.encrypt(metadata) if final_is_enc and metadata else metadata

                    cursor.execute(
                        "INSERT INTO documents (file_path, file_hash, full_text, metadata, is_encrypted, added_timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                        (doc["file_path"], doc["file_hash"], t_store, m_store, 1 if final_is_enc else 0, doc["added_timestamp"])
                    )
                    doc_id_map[doc["doc_id"]] = cursor.lastrowid

                # Import Chunks
                chunk_id_map = {}
                for chunk in export_data.get("chunks", []):
                    text = chunk["chunk_text"]
                    is_enc = chunk["is_encrypted"]

                    if is_enc and decryption_key:
                        temp_enc = Encryptor(decryption_key)
                        text = temp_enc.decrypt(text.encode('utf-8'))
                        is_enc = False

                    final_is_enc = store.encryptor.is_enabled
                    t_store = store.encryptor.encrypt(text) if final_is_enc else text

                    cursor.execute(
                        "INSERT INTO chunks (doc_id, chunk_text, start_pos, end_pos, chunk_seq, tags, is_encrypted, encryption_metadata, graph_processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (doc_id_map[chunk["doc_id"]], t_store, chunk["start_pos"], chunk["end_pos"], chunk["chunk_seq"], chunk["tags"], 1 if final_is_enc else 0, chunk["encryption_metadata"], chunk["graph_processed_at"])
                    )
                    new_cid = cursor.lastrowid
                    chunk_id_map[chunk["chunk_id"]] = new_cid

                    # Sync FTS5
                    try:
                        cursor.execute("INSERT INTO chunks_fts(rowid, chunk_text) VALUES (?, ?)", (new_cid, text))
                    except Exception:
                        pass

                # Import Vectors
                for vec in export_data.get("vectors", []):
                    if vec["chunk_id"] in chunk_id_map:
                        v_blob = base64.b64decode(vec["vector_data"])
                        cursor.execute("INSERT INTO vectors (chunk_id, vector_data) VALUES (?, ?)", (chunk_id_map[vec["chunk_id"]], v_blob))

                # Import Graph Nodes
                node_id_map = {}
                for node in export_data.get("graph_nodes", []):
                    vec_blob = base64.b64decode(node["node_vector"]) if node["node_vector"] else None
                    cursor.execute(
                        "INSERT INTO graph_nodes (node_label, node_properties, unique_signature, node_vector) VALUES (?, ?, ?, ?)",
                        (node["node_label"], node["node_properties"], node["unique_signature"], vec_blob)
                    )
                    node_id_map[node["node_id"]] = cursor.lastrowid

                # Import Graph Relationships
                for rel in export_data.get("graph_relationships", []):
                    if rel["source_node_id"] in node_id_map and rel["target_node_id"] in node_id_map:
                        cursor.execute(
                            "INSERT INTO graph_relationships (source_node_id, target_node_id, relationship_type, relationship_properties) VALUES (?, ?, ?, ?)",
                            (node_id_map[rel["source_node_id"]], node_id_map[rel["target_node_id"]], rel["relationship_type"], rel["relationship_properties"])
                        )

                # Import Node Chunk Links
                for link in export_data.get("node_chunk_links", []):
                    if link["node_id"] in node_id_map and link["chunk_id"] in chunk_id_map:
                        cursor.execute(
                            "INSERT INTO node_chunk_links (node_id, chunk_id) VALUES (?, ?)",
                            (node_id_map[link["node_id"]], chunk_id_map[link["chunk_id"]])
                        )

                store.conn.commit()
                ASCIIColors.success(f"Database successfully imported from {in_path}")
                return store

            except Exception as e:
                if store.conn.in_transaction: store.conn.rollback()
                store.close()
                Path(db_path).unlink(missing_ok=True)
                raise SafeStoreError(f"Database import failed: {e}") from e

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

    def list_vectorization_methods(self) -> List[Dict[str, Any]]:
        self._ensure_connection()
        assert self.conn is not None
        cursor = self.conn.execute("SELECT 1 FROM vectors LIMIT 1")
        if cursor.fetchone() is None:
            return []
        return [{
            "method_id": 0,
            "method_name": self.vectorizer_name,
            "method_type": "sentence_transformer" if self.vectorizer_name == 'st' else self.vectorizer_name,
            "vector_dim": self.vectorizer.dim,
            "vector_dtype": self.vectorizer.dtype.name,
            "params": {}
        }]

    def _decrypt_payload(self, data: Optional[bytes], is_encrypted: bool, fallback_label: str = "Encrypted") -> Optional[str]:
        """Helper to decrypt byte blobs safely or decode UTF-8."""
        if data is None:
            return None
        if not is_encrypted:
            return data.decode('utf-8', errors='ignore')
        if not self.encryptor.is_enabled:
            return f"[{fallback_label} - Key Unavailable]"
        try:
            return self.encryptor.decrypt(data)
        except EncryptionError:
            return f"[{fallback_label} - Decryption Failed]"

    def reconstruct_document_text(self, file_path_or_id: Union[str, Path, int]) -> Optional[str]:
        """Reconstructs the full document text by doc_id or path, decrypting safely."""
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None

            doc_id: Optional[int] = None
            if isinstance(file_path_or_id, int):
                doc_id = file_path_or_id
            else:
                _path_or_id = str(Path(file_path_or_id).resolve() if isinstance(file_path_or_id, Path) else file_path_or_id)
                doc_id = db.get_document_id_by_path(self.conn, _path_or_id)

            if doc_id is None:
                return None

            doc_row = db.get_document_record_by_id(self.conn, doc_id)
            if not doc_row:
                return None

            _, _, _, full_text_bytes, _, doc_enc, _ = doc_row
            if full_text_bytes:
                decrypted_full = self._decrypt_payload(full_text_bytes, bool(doc_enc), "Encrypted Document")
                if decrypted_full and not decrypted_full.startswith("[Encrypted"):
                    return decrypted_full

            # Fallback: Assemble from sequentially ordered chunks
            sql = "SELECT chunk_text, is_encrypted FROM chunks WHERE doc_id = ? ORDER BY chunk_seq ASC"
            original_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            rows = self.conn.execute(sql, (doc_id,)).fetchall()
            self.conn.text_factory = original_factory

            if not rows:
                return ""

            decrypted_chunks = [self._decrypt_payload(chunk_b, bool(c_enc), "Encrypted Chunk") or "" for chunk_b, c_enc in rows]
            return "\n".join(decrypted_chunks)

    def query_full_documents(
        self,
        query_text: str,
        top_k_docs: int = 3,
        search_mode: Literal['dense', 'bm25', 'hybrid'] = 'hybrid',
        top_k_chunks: int = 20,
        min_relevance_percent: float = 0.0,
        min_similarity_percent: Optional[float] = None,
        include_hit_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Queries chunks across the database, aggregates hits by document, and returns the
        highest-scoring documents along with their complete reconstructed full text.
        Excludes documents whose aggregated relevance score is below min_relevance_percent.
        """
        threshold = min_similarity_percent if min_similarity_percent is not None else min_relevance_percent
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None

            # Retrieve candidate chunks with a relaxed pre-filter so document-level aggregation can evaluate properly
            chunk_retrieval_floor = max(0.0, threshold - 20.0) if threshold > 0 else 0.0

            if search_mode == 'hybrid':
                chunk_hits = self.hybrid_query(query_text, top_k=top_k_chunks * 2, min_relevance_percent=chunk_retrieval_floor)
            elif search_mode == 'bm25':
                bm25_retriever = BM25Retriever(self.conn)
                chunk_hits = bm25_retriever.search(query_text, top_k=top_k_chunks * 2, min_relevance_percent=chunk_retrieval_floor)
            elif search_mode == 'dense':
                chunk_hits = self.query(query_text, top_k=top_k_chunks * 2, min_relevance_percent=chunk_retrieval_floor)
            else:
                raise ValueError(f"Unknown search_mode: '{search_mode}'. Supported: 'dense', 'bm25', 'hybrid'.")

            if not chunk_hits:
                return []

            doc_hits_map: Dict[int, List[Dict[str, Any]]] = {}
            doc_path_to_id: Dict[str, int] = {}

            for hit in chunk_hits:
                doc_id = hit.get("doc_id")
                file_path = hit.get("file_path", "")
                if doc_id is None and file_path:
                    if file_path in doc_path_to_id:
                        doc_id = doc_path_to_id[file_path]
                    else:
                        doc_id = db.get_document_id_by_path(self.conn, file_path)
                        if doc_id:
                            doc_path_to_id[file_path] = doc_id

                if doc_id is None:
                    continue

                if doc_id not in doc_hits_map:
                    doc_hits_map[doc_id] = []
                doc_hits_map[doc_id].append(hit)

            scored_documents = []
            for doc_id, hits in doc_hits_map.items():
                grades = [float(h.get("relevance_score", h.get("similarity_percent", 0.0))) for h in hits]
                max_grade = max(grades) if grades else 0.0

                # Composite aggregate relevance grade (0-100 scale)
                agg_grade = min(100.0, max_grade + min(15.0, (len(hits) - 1) * 3.0))
                agg_grade_rounded = round(agg_grade, 2)

                if agg_grade_rounded < threshold:
                    continue

                # Filter matching chunks to those meeting the contextual relevance floor
                filtered_hits = [h for h in hits if float(h.get("relevance_score", h.get("similarity_percent", 0.0))) >= chunk_retrieval_floor]

                scored_documents.append({
                    "doc_id": doc_id,
                    "relevance_score": agg_grade_rounded,
                    "similarity_percent": agg_grade_rounded,
                    "top_chunk_score": round(max_grade, 2),
                    "hit_count": len(hits),
                    "chunk_hits": filtered_hits if filtered_hits else hits
                })

            scored_documents.sort(key=lambda d: d["relevance_score"], reverse=True)
            top_docs = scored_documents[:top_k_docs]

            results = []
            for doc_entry in top_docs:
                doc_id = doc_entry["doc_id"]
                doc_row = db.get_document_record_by_id(self.conn, doc_id)
                if not doc_row:
                    continue

                _, path_bytes, hash_bytes, full_text_bytes, meta_bytes, doc_enc, ts = doc_row
                file_path = path_bytes.decode('utf-8', errors='ignore')
                full_text = self.reconstruct_document_text(doc_id)

                meta_dict = None
                if meta_bytes:
                    decrypted_meta = self._decrypt_payload(meta_bytes, bool(doc_enc), "Encrypted Metadata")
                    if decrypted_meta:
                        try:
                            meta_dict = json.loads(decrypted_meta)
                        except Exception:
                            meta_dict = {"raw": decrypted_meta}

                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
                total_chunks = cursor.fetchone()[0] or 0

                doc_payload = {
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "document_title": Path(file_path).name,
                    "aggregate_score": doc_entry["relevance_score"],
                    "relevance_score": doc_entry["relevance_score"],
                    "similarity_percent": doc_entry["similarity_percent"],
                    "top_chunk_score": doc_entry["top_chunk_score"],
                    "hit_chunk_count": doc_entry["hit_count"],
                    "total_chunk_count": total_chunks,
                    "full_text": full_text,
                    "metadata": meta_dict,
                    "added_timestamp": ts
                }
                if include_hit_chunks:
                    doc_payload["matching_chunks"] = doc_entry["chunk_hits"]

                results.append(doc_payload)

            return results

    def query_document_content_window(
        self,
        query_text: str,
        top_k_hits: int = 3,
        window_before: int = 1,
        window_after: int = 1,
        search_mode: Literal['dense', 'bm25', 'hybrid'] = 'hybrid',
        min_relevance_percent: float = 0.0,
        min_similarity_percent: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Locates top matching chunks and expands their context window by retrieving
        adjacent preceding and succeeding chunks in sequential order.
        Excludes matches below min_relevance_percent.
        """
        threshold = min_similarity_percent if min_similarity_percent is not None else min_relevance_percent
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None

            if search_mode == 'hybrid':
                chunk_hits = self.hybrid_query(query_text, top_k=top_k_hits, min_relevance_percent=threshold)
            elif search_mode == 'bm25':
                bm25_retriever = BM25Retriever(self.conn)
                chunk_hits = bm25_retriever.search(query_text, top_k=top_k_hits, min_relevance_percent=threshold)
            elif search_mode == 'dense':
                chunk_hits = self.query(query_text, top_k=top_k_hits, min_relevance_percent=threshold)
            else:
                raise ValueError(f"Unknown search_mode: '{search_mode}'")

            results = []
            for hit in chunk_hits:
                chunk_id = hit.get("chunk_id")
                if chunk_id is None:
                    continue

                cursor = self.conn.cursor()
                cursor.execute("SELECT doc_id, chunk_seq FROM chunks WHERE chunk_id = ?", (chunk_id,))
                c_row = cursor.fetchone()
                if not c_row:
                    continue

                doc_id, target_seq = c_row
                start_seq = max(0, target_seq - window_before)
                end_seq = target_seq + window_after

                window_rows = db.get_document_chunks_by_seq_range(self.conn, doc_id, start_seq, end_seq)
                surrounding_chunks = []
                stitched_parts = []

                for cid, did, chunk_b, start_p, end_p, seq, tags, is_enc in window_rows:
                    decrypted_txt = self._decrypt_payload(chunk_b, bool(is_enc), "Encrypted Chunk") or ""
                    is_target = (seq == target_seq)
                    surrounding_chunks.append({
                        "chunk_id": cid,
                        "chunk_seq": seq,
                        "is_target_hit": is_target,
                        "chunk_text": decrypted_txt
                    })
                    stitched_parts.append(decrypted_txt)

                doc_row = db.get_document_record_by_id(self.conn, doc_id)
                file_path = doc_row[1].decode('utf-8', errors='ignore') if doc_row else hit.get("file_path", "")

                rel_grade = float(hit.get("relevance_score", hit.get("similarity_percent", 0.0)))

                results.append({
                    "target_chunk_id": chunk_id,
                    "target_chunk_seq": target_seq,
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "document_title": Path(file_path).name,
                    "hit_score": hit.get("fused_score", hit.get("similarity_score", hit.get("score", 0.0))),
                    "similarity_percent": rel_grade,
                    "relevance_score": rel_grade,
                    "stitched_window_text": "\n\n".join(stitched_parts),
                    "surrounding_chunks": surrounding_chunks
                })

            return results

    def get_document_content_paginated(
        self,
        doc_id_or_path: Union[int, str, Path],
        page: int = 1,
        page_size: int = 5,
        highlight_chunk_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Returns a paginated slice of chunks for a document, along with pagination metadata
        and continuous stitched page text.
        """
        with self._instance_lock:
            self._ensure_connection()
            assert self.conn is not None

            doc_id: Optional[int] = None
            if isinstance(doc_id_or_path, int):
                doc_id = doc_id_or_path
            else:
                _path_or_id = str(Path(doc_id_or_path).resolve() if isinstance(doc_id_or_path, Path) else doc_id_or_path)
                doc_id = db.get_document_id_by_path(self.conn, _path_or_id)

            if doc_id is None:
                raise SafeStoreError(f"Document '{doc_id_or_path}' not found.")

            if page < 1:
                page = 1
            if page_size < 1:
                page_size = 5

            offset = (page - 1) * page_size
            rows, total_chunks = db.get_document_chunks_paginated(self.conn, doc_id, offset=offset, limit=page_size)

            total_pages = max(1, (total_chunks + page_size - 1) // page_size)
            highlight_set = set(highlight_chunk_ids or [])

            chunks = []
            stitched_parts = []
            for cid, did, chunk_b, start_p, end_p, seq, tags, is_enc in rows:
                text = self._decrypt_payload(chunk_b, bool(is_enc), "Encrypted Chunk") or ""
                chunks.append({
                    "chunk_id": cid,
                    "chunk_seq": seq,
                    "chunk_text": text,
                    "is_highlighted": cid in highlight_set
                })
                stitched_parts.append(text)

            doc_row = db.get_document_record_by_id(self.conn, doc_id)
            file_path = doc_row[1].decode('utf-8', errors='ignore') if doc_row else ""

            return {
                "doc_id": doc_id,
                "file_path": file_path,
                "document_title": Path(file_path).name,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "has_previous_page": page > 1,
                "has_next_page": page < total_pages,
                "chunks": chunks,
                "stitched_text": "\n\n".join(stitched_parts)
            }

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        with self._instance_lock, self._optional_file_lock_context(f"get_chunk_by_id: {chunk_id}"):
            self._ensure_connection()
            assert self.conn is not None
            
            row = db.get_chunk_raw_details_by_id(self.conn, chunk_id)
            if not row: return None

            _, chunk_text_data, chunk_is_enc, path_bytes, doc_meta_data, doc_is_enc = row

            chunk_text: str
            if chunk_is_enc:
                if self.encryptor.is_enabled:
                    try: chunk_text = self.encryptor.decrypt(chunk_text_data)
                    except EncryptionError: chunk_text = "[Encrypted Chunk - Decryption Failed]"
                else: chunk_text = "[Encrypted Chunk - Key Unavailable]"
            else:
                chunk_text = chunk_text_data.decode('utf-8')

            meta_dict = None
            if doc_meta_data:
                if doc_is_enc:
                    if self.encryptor.is_enabled:
                        try: meta_dict = json.loads(self.encryptor.decrypt(doc_meta_data))
                        except (EncryptionError, json.JSONDecodeError): meta_dict = {"error": "Failed to decrypt or parse metadata"}
                    else: meta_dict = {"error": "Encrypted metadata but key unavailable"}
                else:
                    try: meta_dict = json.loads(doc_meta_data.decode('utf-8'))
                    except json.JSONDecodeError: meta_dict = {"error": "Failed to parse metadata"}

            return {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "file_path": path_bytes.decode('utf-8'),
                "document_metadata": meta_dict
            }

    def get_datalake_view(
        self,
        method: Literal['pca', 'tsne', 'umap', 'incremental_pca'] = 'pca',
        n_components: int = 2,
        use_cache: bool = True,
        sample_size: Optional[int] = None,
        filter_doc_ids: Optional[List[int]] = None,
        output_format: Literal['dict', 'json_str', 'csv', 'dataframe'] = 'dict',
        include_chunk_text: bool = True
    ) -> Union[List[Dict[str, Any]], str, Any]:
        """
        Retrieves a 2D or 3D datalake semantic projection using PCA, t-SNE, or UMAP.
        Supports instant cached retrieval, sampling, and filtering.
        """
        with self._instance_lock:
            self._ensure_connection()
            return self.datalake.get_datalake_view(
                method=method,
                n_components=n_components,
                use_cache=use_cache,
                sample_size=sample_size,
                filter_doc_ids=filter_doc_ids,
                output_format=output_format,
                include_chunk_text=include_chunk_text
            )

    def stream_datalake_chunks(
        self,
        batch_size: int = 500,
        method: Literal['pca', 'incremental_pca'] = 'incremental_pca',
        n_components: int = 2
    ):
        """Streams datalake points in incremental batches without loading full vector matrices into RAM."""
        with self._instance_lock:
            self._ensure_connection()
            return self.datalake.stream_datalake_chunks(
                batch_size=batch_size,
                method=method,
                n_components=n_components
            )

    def export_datalake_html(
        self,
        output_file: Union[str, Path] = "datalake_view.html",
        title: str = "SafeStore Semantic Datalake Explorer",
        method: Literal['pca', 'tsne', 'umap'] = 'pca',
        n_components: int = 2,
        sample_size: Optional[int] = None
    ) -> Path:
        """Exports a standalone, interactive HTML 2D/3D visualizer for the entire datalake."""
        with self._instance_lock:
            self._ensure_connection()
            return self.datalake.export_datalake_html(
                output_file=output_file,
                title=title,
                method=method,
                n_components=n_components,
                sample_size=sample_size
            )

    def export_point_cloud(
        self,
        output_format: Literal['json_str', 'dict', 'csv'] = 'json_str',
        method: Literal['pca', 'tsne', 'umap'] = 'pca',
        n_components: int = 2,
        use_cache: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Exports a point-cloud projection of document chunks (backward-compatible and enhanced).
        """
        with self._instance_lock:
            self._ensure_connection()
            return self.datalake.get_datalake_view(
                method=method,
                n_components=n_components,
                use_cache=use_cache,
                output_format=output_format
            )