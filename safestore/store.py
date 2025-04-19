# safestore/store.py
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import hashlib
import threading # For instance-level lock (optional but good practice)
from filelock import FileLock, Timeout # For inter-process lock

from .core import db
from .indexing import parser, chunking
from .search import similarity
from .vectorization.manager import VectorizationManager
from .vectorization.methods.tfidf import TfidfVectorizerWrapper
from ascii_colors import ASCIIColors, LogLevel

# Default lock timeout in seconds
DEFAULT_LOCK_TIMEOUT = 60

class SafeStore:
    """
    Main class for interacting with the SafeStore database.
    Manages document indexing, vectorization, and querying.
    Includes basic file-based locking for concurrency control.
    """
    DEFAULT_VECTORIZER = "st:all-MiniLM-L6-v2"

    def __init__(
        self,
        db_path: str | Path = "safestore.db",
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT
    ):
        """
        Initializes the SafeStore.

        Args:
            db_path: Path to the SQLite database file.
            log_level: Minimum log level for ascii_colors output (default INFO).
            lock_timeout: Timeout in seconds for acquiring the file lock (default 60).
                          Set to 0 or negative for non-blocking, or adjust as needed.
        """
        self.db_path = str(Path(db_path).resolve()) # Use resolved path
        self.lock_timeout = lock_timeout
        # Create lock file path next to the db file
        self.lock_path = str(Path(self.db_path).parent / f"{Path(self.db_path).name}.lock")

        ASCIIColors.set_log_level(log_level)
        ASCIIColors.info(f"Initializing SafeStore with database: {self.db_path}")
        ASCIIColors.debug(f"Using lock file: {self.lock_path} with timeout: {self.lock_timeout}s")

        self.conn: Optional[sqlite3.Connection] = None # Initialize conn as None
        self._connect_and_initialize() # Connect and setup schema

        self.vectorizer_manager = VectorizationManager()
        self._file_hasher = hashlib.sha256

        # Instance-level lock for thread safety within the same process (basic)
        self._instance_lock = threading.RLock()
        # File lock for inter-process safety
        self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)

    def _connect_and_initialize(self):
        """Connects to the DB and initializes schema if needed."""
        # Basic locking during connection/initialization itself (optional, but safer)
        # Using a temporary FileLock instance here as self._file_lock might not be fully ready
        init_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        try:
            with init_lock:
                 ASCIIColors.debug("Acquired init lock for connection/schema setup.")
                 self.conn = db.connect_db(self.db_path)
                 db.initialize_schema(self.conn)
            ASCIIColors.debug("Released init lock.")
        except Timeout:
             ASCIIColors.error(f"Timeout acquiring initial lock for DB connection at {self.lock_path}")
             raise Timeout(f"Could not acquire lock for DB initialization: {self.lock_path}")
        except Exception as e:
             ASCIIColors.error(f"Error during initial DB connection/setup: {e}")
             if self.conn:
                 self.conn.close()
                 self.conn = None
             raise # Re-raise the exception


    def close(self):
        """Closes the database connection."""
        # Use instance lock for thread safety during close
        with self._instance_lock:
            if self.conn:
                ASCIIColors.debug("Closing database connection.")
                try:
                    self.conn.close()
                except Exception as e:
                     ASCIIColors.warning(f"Error closing DB connection: {e}")
                finally:
                    self.conn = None
            # Clear vectorizer cache
            if hasattr(self, 'vectorizer_manager'):
                 self.vectorizer_manager.clear_cache()

    def __enter__(self):
        # Ensure connected on entry, although init should handle it
        if self.conn is None:
            self._connect_and_initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type:
             ASCIIColors.error(f"SafeStore context closed with error: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
        else:
             ASCIIColors.debug("SafeStore context closed cleanly.")


    def _get_file_hash(self, file_path: Path) -> str:
        """Generates a hash for the file content."""
        try:
            hasher = self._file_hasher()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            ASCIIColors.error(f"File not found when trying to hash: {file_path}")
            raise
        except Exception as e:
            ASCIIColors.warning(f"Could not generate hash for {file_path}: {e}")
            return ""


    # === Write methods requiring locking ===

    def add_document(
        self,
        file_path: str | Path,
        vectorizer_name: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[dict] = None,
        force_reindex: bool = False,
        vectorizer_params: Optional[dict] = None
    ):
        """
        Adds or updates a document in the SafeStore (acquires write lock).
        [Rest of docstring remains the same]
        """
        # Acquire instance lock first (thread safety), then file lock (process safety)
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for add_document: {Path(file_path).name}")
            try:
                with self._file_lock: # Acquire exclusive file lock
                    ASCIIColors.info(f"Write lock acquired for add_document: {Path(file_path).name}")
                    # Check connection health before proceeding
                    if not self.conn:
                        ASCIIColors.warning("Connection lost, attempting to reconnect...")
                        self._connect_and_initialize()
                        if not self.conn:
                             raise ConnectionError("Database connection is not available.")

                    # --- Call the actual implementation ---
                    self._add_document_impl(
                        file_path, vectorizer_name, chunk_size, chunk_overlap,
                        metadata, force_reindex, vectorizer_params
                    )
                    ASCIIColors.debug(f"Write lock released for add_document: {Path(file_path).name}")

            except Timeout:
                ASCIIColors.error(f"Timeout ({self.lock_timeout}s) acquiring write lock for add_document: {Path(file_path).name}")
                raise Timeout(f"Could not acquire write lock for {Path(file_path).name} within {self.lock_timeout}s")
            except Exception as e:
                 # Ensure lock is released if error happens after acquisition but before exit
                 if self._file_lock.is_locked:
                     try:
                         self._file_lock.release()
                         ASCIIColors.debug("Force-released write lock due to error in add_document.")
                     except Exception as release_err:
                         ASCIIColors.warning(f"Error trying to force-release lock: {release_err}")
                 ASCIIColors.error(f"Error during add_document (lock scope): {e}", exc_info=True)
                 raise # Re-raise the original error

    def _add_document_impl(
        self,
        file_path: str | Path,
        vectorizer_name: str | None,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[dict],
        force_reindex: bool,
        vectorizer_params: Optional[dict]
    ):
        """Internal implementation of add_document logic (assumes lock is held)."""
        # [This contains the exact same logic as the previous add_document method]
        # ... (Copy the entire logic from the previous add_document here) ...
        # START COPY from previous version
        file_path = Path(file_path)
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        abs_file_path = str(file_path.resolve()) # Store absolute path

        ASCIIColors.info(f"Starting indexing process for: {file_path.name}")
        ASCIIColors.debug(f"Params: vectorizer='{_vectorizer_name}', chunk_size={chunk_size}, overlap={chunk_overlap}, force={force_reindex}")

        if not file_path.exists():
            ASCIIColors.error(f"File not found: {abs_file_path}")
            raise FileNotFoundError(f"Source file not found: {abs_file_path}")

        current_hash = self._get_file_hash(file_path)
        if not current_hash: # Handle hashing failure
            ASCIIColors.error(f"Failed to generate hash for {file_path.name}. Aborting.")
            return

        existing_doc_id = None
        existing_hash = None
        needs_parsing_chunking = True
        needs_vectorization = True

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
                    _vec_instance, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn)
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
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?",
                                   (current_hash, None, json.dumps(metadata) if metadata else None, existing_doc_id))
                    self.conn.commit()
                    ASCIIColors.debug(f"Deleted old chunks/vectors and updated document record for changed doc_id={existing_doc_id}.")
            else:
                 ASCIIColors.info(f"Document '{file_path.name}' is new.")

        except sqlite3.Error as e:
            ASCIIColors.error(f"Database error checking/updating document '{file_path.name}': {e}", exc_info=True)
            self.conn.rollback()
            raise
        except Exception as e:
             ASCIIColors.error(f"Error preparing indexing for '{file_path.name}': {e}", exc_info=True)
             raise

        if not needs_parsing_chunking and not needs_vectorization:
             return # Nothing more to do

        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            doc_id = existing_doc_id
            full_text = None
            chunks_data = []
            chunk_ids = []
            chunk_texts = []

            if needs_parsing_chunking:
                ASCIIColors.debug(f"Parsing document: {file_path.name}")
                try:
                    full_text = parser.parse_document(file_path) # PARSER UPDATE NEEDED LATER
                    ASCIIColors.debug(f"Parsed document '{file_path.name}'. Length: {len(full_text)} chars.")
                except Exception as e:
                    ASCIIColors.error(f"Failed to parse {file_path.name}: {e}")
                    raise

                if doc_id is None:
                     _doc_id = db.add_document_record(
                         self.conn, abs_file_path, full_text, current_hash, json.dumps(metadata) if metadata else None
                     )
                     if _doc_id is None: raise RuntimeError("Failed to get doc_id for new document.")
                     doc_id = _doc_id
                else:
                    cursor.execute("UPDATE documents SET full_text = ? WHERE doc_id = ?", (full_text, doc_id)) # ENCRYPTION NEEDED LATER

                chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                if not chunks_data:
                    ASCIIColors.warning(f"No chunks generated for {file_path.name}. Skipping vectorization.")
                    self.conn.commit()
                    return

                ASCIIColors.info(f"Generated {len(chunks_data)} chunks for '{file_path.name}'. Storing chunks...")

                for i, (text, start, end) in enumerate(chunks_data):
                    chunk_id = db.add_chunk_record(self.conn, doc_id, text, start, end, i) # ENCRYPTION NEEDED LATER
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(text)

            else:
                ASCIIColors.debug(f"Retrieving existing chunks for doc_id={doc_id} to add new vectors...")
                cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,)) # DECRYPTION NEEDED LATER
                results = cursor.fetchall()
                if not results:
                      ASCIIColors.error(f"Document {doc_id} exists but no chunks found! Inconsistent state.")
                      raise RuntimeError(f"Inconsistent state: No chunks found for existing document ID {doc_id}")
                chunk_ids = [row[0] for row in results]
                chunk_texts = [row[1] for row in results]
                ASCIIColors.debug(f"Retrieved {len(chunk_ids)} existing chunk IDs and texts.")


            if needs_vectorization:
                if not chunk_ids or not chunk_texts:
                     ASCIIColors.warning(f"No chunks available to vectorize for '{file_path.name}'.")
                     self.conn.commit()
                     return

                vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn)

                if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                     ASCIIColors.warning(f"TF-IDF vectorizer '{_vectorizer_name}' is not fitted. Fitting on chunks from '{file_path.name}' only.")
                     try:
                         vectorizer.fit(chunk_texts)
                         new_params = vectorizer.get_params_to_store()
                         self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                     except Exception as e:
                         ASCIIColors.error(f"Failed to fit TF-IDF model '{_vectorizer_name}' on '{file_path.name}': {e}")
                         raise


                ASCIIColors.info(f"Vectorizing {len(chunk_texts)} chunks using '{_vectorizer_name}' (method_id={method_id})...")

                try:
                     vectors = vectorizer.vectorize(chunk_texts)
                except Exception as e:
                     ASCIIColors.error(f"Vectorization failed for '{_vectorizer_name}': {e}", exc_info=True)
                     raise


                if vectors.shape[0] != len(chunk_ids):
                    ASCIIColors.error(f"Mismatch between number of chunks ({len(chunk_ids)}) and generated vectors ({vectors.shape[0]})!")
                    raise ValueError("Chunk and vector count mismatch during indexing.")

                ASCIIColors.debug(f"Adding {len(vectors)} vector records to DB (method_id={method_id})...")
                for chunk_id, vector in zip(chunk_ids, vectors):
                    vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                    db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)

            self.conn.commit()
            ASCIIColors.success(f"Successfully processed '{file_path.name}' with vectorizer '{_vectorizer_name}'.")

        except Exception as e:
            ASCIIColors.error(f"Error during indexing of '{file_path.name}': {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            raise # Re-raise the exception
        # END COPY


    def add_vectorization(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[str | Path] = None,
        vectorizer_params: Optional[dict] = None,
        batch_size: int = 64
    ):
        """
        Adds vector embeddings using a new or existing method (acquires write lock).
        [Rest of docstring remains the same]
        """
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for add_vectorization: {vectorizer_name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for add_vectorization: {vectorizer_name}")
                    if not self.conn:
                        ASCIIColors.warning("Connection lost, attempting to reconnect...")
                        self._connect_and_initialize()
                        if not self.conn:
                             raise ConnectionError("Database connection is not available.")
                    # --- Call the actual implementation ---
                    self._add_vectorization_impl(
                        vectorizer_name, target_doc_path, vectorizer_params, batch_size
                    )
                    ASCIIColors.debug(f"Write lock released for add_vectorization: {vectorizer_name}")
            except Timeout:
                ASCIIColors.error(f"Timeout ({self.lock_timeout}s) acquiring write lock for add_vectorization: {vectorizer_name}")
                raise Timeout(f"Could not acquire write lock for {vectorizer_name} within {self.lock_timeout}s")
            except Exception as e:
                 if self._file_lock.is_locked:
                     try:
                         self._file_lock.release()
                         ASCIIColors.debug("Force-released write lock due to error in add_vectorization.")
                     except Exception as release_err:
                         ASCIIColors.warning(f"Error trying to force-release lock: {release_err}")
                 ASCIIColors.error(f"Error during add_vectorization (lock scope): {e}", exc_info=True)
                 raise

    def _add_vectorization_impl(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[str | Path],
        vectorizer_params: Optional[dict],
        batch_size: int
    ):
        """Internal implementation of add_vectorization (assumes lock is held)."""
        # [This contains the exact same logic as the previous add_vectorization method]
        # ... (Copy the entire logic from the previous add_vectorization here) ...
        # START COPY
        ASCIIColors.info(f"Starting process to add vectorization '{vectorizer_name}'.")
        if target_doc_path:
             target_doc_path = Path(target_doc_path).resolve()
             ASCIIColors.info(f"Targeting specific document: {target_doc_path}")
        else:
             ASCIIColors.info("Targeting all documents in the store.")

        cursor = self.conn.cursor()
        try:
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, self.conn)

            if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' requires fitting.")
                fit_sql = "SELECT chunk_text FROM chunks" # DECRYPTION NEEDED LATER
                fit_params: List[Any] = []
                target_doc_id = None # Initialize target_doc_id
                if target_doc_path:
                     cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (str(target_doc_path),))
                     target_doc_id_result = cursor.fetchone()
                     if not target_doc_id_result:
                         ASCIIColors.error(f"Target document '{target_doc_path}' not found in the database.")
                         return
                     target_doc_id = target_doc_id_result[0]
                     fit_sql += " WHERE doc_id = ?"
                     fit_params.append(target_doc_id)
                     ASCIIColors.info(f"Fetching chunks for fitting from document ID {target_doc_id}...")
                else:
                     ASCIIColors.info("Fetching all chunks from database for fitting...")

                cursor.execute(fit_sql, tuple(fit_params))
                texts_to_fit = [row[0] for row in cursor.fetchall()] # Texts might be encrypted later

                if not texts_to_fit:
                     ASCIIColors.warning("No text chunks found to fit the TF-IDF model. Aborting vectorization add.")
                     return

                try:
                    vectorizer.fit(texts_to_fit)
                    new_params = vectorizer.get_params_to_store()
                    self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                    ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' fitted successfully.")
                except Exception as e:
                    ASCIIColors.error(f"Failed to fit TF-IDF model '{vectorizer_name}': {e}", exc_info=True)
                    return


            chunks_to_vectorize_sql = f"""
                SELECT c.chunk_id, c.chunk_text -- DECRYPTION NEEDED LATER
                FROM chunks c
                LEFT JOIN vectors v ON c.chunk_id = v.chunk_id AND v.method_id = ?
                WHERE v.vector_id IS NULL
            """
            sql_params: List[Any] = [method_id]

            if target_doc_path:
                # Fetch doc_id again if not fetched during fit
                if target_doc_id is None:
                     cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (str(target_doc_path),))
                     target_doc_id_result = cursor.fetchone()
                     if not target_doc_id_result:
                         ASCIIColors.error(f"Target document '{target_doc_path}' not found in the database (second check).")
                         return
                     target_doc_id = target_doc_id_result[0]

                chunks_to_vectorize_sql += " AND c.doc_id = ?"
                sql_params.append(target_doc_id)
                ASCIIColors.info(f"Fetching chunks missing '{vectorizer_name}' vectors for document ID {target_doc_id}...")
            else:
                 ASCIIColors.info(f"Fetching all chunks missing '{vectorizer_name}' vectors...")


            cursor.execute(chunks_to_vectorize_sql, tuple(sql_params))
            chunks_data = cursor.fetchall()

            if not chunks_data:
                ASCIIColors.success(f"No chunks found needing vectorization for '{vectorizer_name}'. Process complete.")
                return

            total_chunks = len(chunks_data)
            ASCIIColors.info(f"Found {total_chunks} chunks to vectorize.")

            num_added = 0
            cursor.execute("BEGIN")
            try:
                for i in range(0, total_chunks, batch_size):
                    batch = chunks_data[i : i + batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_texts = [item[1] for item in batch] # Texts might be encrypted later

                    ASCIIColors.debug(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_texts)} chunks)")

                    try:
                         vectors = vectorizer.vectorize(batch_texts)
                         if vectors.shape[0] != len(batch_ids):
                              raise ValueError(f"Vectorization output count ({vectors.shape[0]}) doesn't match batch size ({len(batch_ids)}).")
                    except Exception as e:
                         ASCIIColors.error(f"Vectorization failed for batch: {e}", exc_info=True)
                         raise RuntimeError("Vectorization failed, aborting add_vectorization.") from e


                    for chunk_id, vector in zip(batch_ids, vectors):
                         vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                         db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)
                    num_added += len(batch_ids)
                    ASCIIColors.debug(f"Added {len(batch_ids)} vectors for batch.")

                self.conn.commit()
                ASCIIColors.success(f"Successfully added {num_added} vector embeddings using '{vectorizer_name}'.")

            except Exception as e:
                 ASCIIColors.error(f"Error during vectorization/storage: {e}", exc_info=True)
                 self.conn.rollback()
                 raise

        except sqlite3.Error as e:
             ASCIIColors.error(f"Database error during add_vectorization: {e}", exc_info=True)
             raise
        except Exception as e:
             ASCIIColors.error(f"An unexpected error occurred during add_vectorization: {e}", exc_info=True)
             raise
        # END COPY


    def remove_vectorization(self, vectorizer_name: str):
        """
        Removes a vectorization method and associated vectors (acquires write lock).
        [Rest of docstring remains the same]
        """
        with self._instance_lock:
            ASCIIColors.debug(f"Attempting to acquire write lock for remove_vectorization: {vectorizer_name}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"Write lock acquired for remove_vectorization: {vectorizer_name}")
                    if not self.conn:
                        ASCIIColors.warning("Connection lost, attempting to reconnect...")
                        self._connect_and_initialize()
                        if not self.conn:
                             raise ConnectionError("Database connection is not available.")
                    # --- Call the actual implementation ---
                    self._remove_vectorization_impl(vectorizer_name)
                    ASCIIColors.debug(f"Write lock released for remove_vectorization: {vectorizer_name}")
            except Timeout:
                ASCIIColors.error(f"Timeout ({self.lock_timeout}s) acquiring write lock for remove_vectorization: {vectorizer_name}")
                raise Timeout(f"Could not acquire write lock for {vectorizer_name} within {self.lock_timeout}s")
            except Exception as e:
                if self._file_lock.is_locked:
                    try:
                        self._file_lock.release()
                        ASCIIColors.debug("Force-released write lock due to error in remove_vectorization.")
                    except Exception as release_err:
                        ASCIIColors.warning(f"Error trying to force-release lock: {release_err}")
                ASCIIColors.error(f"Error during remove_vectorization (lock scope): {e}", exc_info=True)
                raise

    def _remove_vectorization_impl(self, vectorizer_name: str):
        """Internal implementation of remove_vectorization (assumes lock is held)."""
        # [This contains the exact same logic as the previous remove_vectorization method]
        # ... (Copy the entire logic from the previous remove_vectorization here) ...
        # START COPY
        ASCIIColors.warning(f"Attempting to remove vectorization method '{vectorizer_name}' and all associated vectors.")

        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_name,))
            result = cursor.fetchone()
            if not result:
                ASCIIColors.error(f"Vectorization method '{vectorizer_name}' not found in the database. Cannot remove.")
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

            # Use list() to avoid modifying dict during iteration
            cached_items_to_remove = [name for name, (_, mid, _) in self.vectorizer_manager._cache.items() if mid == method_id]
            for name in cached_items_to_remove:
                 if name in self.vectorizer_manager._cache:
                     del self.vectorizer_manager._cache[name]
                     ASCIIColors.debug(f"Removed '{name}' from vectorizer cache.")


            ASCIIColors.success(f"Successfully removed vectorization method '{vectorizer_name}' (ID: {method_id}) and {deleted_vectors} associated vectors.")

        except sqlite3.Error as e:
             ASCIIColors.error(f"Database error during removal of '{vectorizer_name}': {e}", exc_info=True)
             self.conn.rollback()
             raise
        except Exception as e:
             ASCIIColors.error(f"An unexpected error occurred during removal of '{vectorizer_name}': {e}", exc_info=True)
             raise
        # END COPY


    # === Read methods (no explicit lock needed with WAL, but instance lock for thread safety) ===

    def query(
        self,
        query_text: str,
        vectorizer_name: str | None = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries the store for similar chunks (read operation, uses instance lock).
        [Docstring remains largely the same, mention read lock briefly if added later]
        """
        # Use instance lock for thread safety, especially around manager/connection access
        with self._instance_lock:
            # Check connection health
            if not self.conn:
                 ASCIIColors.error("Database connection is not available for query.")
                 # Option 1: Try to reconnect (might fail if init fails)
                 # try:
                 #     self._connect_and_initialize()
                 # except Exception as e:
                 #     raise ConnectionError("Database connection is not available and reconnect failed.") from e
                 # Option 2: Just raise error
                 raise ConnectionError("Database connection is not available for query.")

            # WAL mode should allow concurrent reads without explicit file lock,
            # but acquiring a *shared* lock could prevent issues if other processes
            # are doing schema changes or intensive writes. For now, rely on WAL.
            # ASCIIColors.debug(f"Attempting to acquire shared lock for query...")
            # with self._file_lock.read_lock(): # Example if using shared lock
            #     ASCIIColors.debug(f"Shared lock acquired for query.")
            #     return self._query_impl(query_text, vectorizer_name, top_k)

            # --- Call the actual implementation (without explicit file lock for now) ---
            return self._query_impl(query_text, vectorizer_name, top_k)


    def _query_impl(
        self,
        query_text: str,
        vectorizer_name: str | None,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Internal implementation of query logic (assumes instance lock held)."""
        # [This contains the exact same logic as the previous query method]
        # ... (Copy the entire logic from the previous query method here) ...
        # START COPY
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        ASCIIColors.info(f"Received query. Searching with '{_vectorizer_name}', top_k={top_k}.")

        results = []
        cursor = self.conn.cursor()

        try:
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn)
            ASCIIColors.debug(f"Using vectorizer '{_vectorizer_name}' (method_id={method_id})")

            ASCIIColors.debug(f"Vectorizing query text...")
            query_vector = vectorizer.vectorize([query_text])[0]
            query_vector = np.ascontiguousarray(query_vector, dtype=vectorizer.dtype)
            ASCIIColors.debug(f"Query vector generated. Shape: {query_vector.shape}, Dtype: {query_vector.dtype}")

            ASCIIColors.debug(f"Loading all vectors for method_id {method_id} from database...")
            cursor.execute("""
                SELECT v.chunk_id, v.vector_data
                FROM vectors v
                WHERE v.method_id = ?
            """, (method_id,))
            all_vectors_data = cursor.fetchall()

            if not all_vectors_data:
                ASCIIColors.warning(f"No vectors found in the database for method '{_vectorizer_name}' (ID: {method_id}). Cannot perform query.")
                return []

            chunk_ids_ordered = [row[0] for row in all_vectors_data]
            vector_blobs = [row[1] for row in all_vectors_data]

            method_details = self.vectorizer_manager._get_method_details_from_db(self.conn, _vectorizer_name)
            if not method_details:
                 raise RuntimeError(f"Could not retrieve method details for '{_vectorizer_name}' after getting instance.")
            vector_dtype = method_details['vector_dtype']

            ASCIIColors.debug(f"Reconstructing {len(vector_blobs)} vectors from BLOBs with dtype '{vector_dtype}'...")
            try:
                 candidate_vectors = np.array([db.reconstruct_vector(blob, vector_dtype) for blob in vector_blobs])
            except ValueError as e:
                 ASCIIColors.error(f"Failed to reconstruct one or more vectors: {e}")
                 raise

            ASCIIColors.debug(f"Candidate vectors loaded. Matrix shape: {candidate_vectors.shape}")

            ASCIIColors.debug("Calculating similarity scores...")
            scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            ASCIIColors.debug(f"Similarity scores calculated. Shape: {scores.shape}")

            num_candidates = len(scores)
            k = min(top_k, num_candidates)
            top_k_indices = np.argsort(scores)[::-1][:k]

            ASCIIColors.debug(f"Identified top {k} indices.")

            if k > 0:
                top_chunk_ids = [chunk_ids_ordered[i] for i in top_k_indices]
                top_scores = [scores[i] for i in top_k_indices]

                placeholders = ','.join('?' * len(top_chunk_ids))
                sql_chunk_details = f"""
                    SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.chunk_seq, -- DECRYPTION NEEDED LATER
                           d.doc_id, d.file_path
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.chunk_id IN ({placeholders})
                """
                cursor.execute(sql_chunk_details, top_chunk_ids)
                chunk_details_list = cursor.fetchall()

                chunk_details_map = {
                     row[0]: {
                        "chunk_id": row[0],
                        "chunk_text": row[1], # Needs decryption later if enabled
                        "start_pos": row[2],
                        "end_pos": row[3],
                        "chunk_seq": row[4],
                        "doc_id": row[5],
                        "file_path": row[6]
                     } for row in chunk_details_list
                }

                for chunk_id, score in zip(top_chunk_ids, top_scores):
                    if chunk_id in chunk_details_map:
                        result_item = chunk_details_map[chunk_id].copy()
                        result_item["similarity"] = float(score)
                        results.append(result_item)
                    else:
                         ASCIIColors.warning(f"Could not find details for chunk_id {chunk_id} which was in top-k. Skipping.")

            ASCIIColors.success(f"Query successful. Found {len(results)} relevant chunks.")
            return results

        except sqlite3.Error as e:
             ASCIIColors.error(f"Database error during query: {e}", exc_info=True)
             raise
        except Exception as e:
             ASCIIColors.error(f"An unexpected error occurred during query: {e}", exc_info=True)
             raise
        # END COPY

    # --- TODO (Phase 3/4): Implement reindex, encryption helpers ---
    # def reindex(self, ...):
    #     with self._instance_lock:
    #         ASCIIColors.debug("Attempting to acquire write lock for reindex...")
    #         try:
    #             with self._file_lock:
    #                 ASCIIColors.info("Write lock acquired for reindex.")
    #                 # ... implementation ...
    #                 ASCIIColors.debug("Write lock released for reindex.")
    #         except Timeout:
    #             # Handle timeout
    #             ...
    #         except Exception as e:
    #             # Handle error, release lock
    #             ...