import sqlite3
import json
from pathlib import Path
from typing import Optional
import numpy as np
import hashlib # For basic file hashing

from .core import db
from .indexing import parser, chunking
from .vectorization.manager import VectorizationManager
from ascii_colors import ASCIIColors, LogLevel

class SafeStore:
    """
    Main class for interacting with the SafeStore database.
    """
    DEFAULT_VECTORIZER = "st:all-MiniLM-L6-v2" # Default Sentence Transformer

    def __init__(self, db_path: str | Path = "safestore.db", log_level: LogLevel = LogLevel.INFO):
        """
        Initializes the SafeStore.

        Args:
            db_path: Path to the SQLite database file.
            log_level: Minimum log level for ascii_colors output (default INFO).
        """
        self.db_path = str(db_path)
        ASCIIColors.set_log_level(log_level) # Set global log level for the library's messages
        ASCIIColors.info(f"Initializing SafeStore with database: {self.db_path}")

        self.conn = db.connect_db(self.db_path)
        db.initialize_schema(self.conn)
        self.vectorizer_manager = VectorizationManager()

        # Basic file hashing for change detection (can be improved)
        self._file_hasher = hashlib.sha256

    def close(self):
        """Closes the database connection."""
        if self.conn:
            ASCIIColors.debug("Closing database connection.")
            self.conn.close()
            self.conn = None

    def __enter__(self):
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
                while chunk := f.read(8192): # Read in chunks
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            return "" # Or raise? If we call this, file should exist.
        except Exception as e:
            ASCIIColors.warning(f"Could not generate hash for {file_path}: {e}")
            return ""


    def add_document(
        self,
        file_path: str | Path,
        vectorizer_name: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[dict] = None,
        force_reindex: bool = False # Add option to force reindexing
    ):
        """
        Adds a document to the SafeStore, including parsing, chunking, and vectorization.

        Args:
            file_path: Path to the document file.
            vectorizer_name: Name of the vectorizer to use (e.g., 'st:all-MiniLM-L6-v2').
                               Defaults to SafeStore.DEFAULT_VECTORIZER.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            metadata: Optional dictionary of metadata to store with the document.
            force_reindex: If True, re-indexes the document even if path/hash hasn't changed.
        """
        file_path = Path(file_path)
        vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        abs_file_path = str(file_path.resolve()) # Store absolute path

        ASCIIColors.info(f"Starting indexing process for: {file_path.name}")
        ASCIIColors.debug(f"Params: vectorizer='{vectorizer_name}', chunk_size={chunk_size}, overlap={chunk_overlap}")

        if not file_path.exists():
            ASCIIColors.error(f"File not found: {abs_file_path}")
            return

        # --- Check if document needs indexing ---
        current_hash = self._get_file_hash(file_path)
        existing_doc_id = None
        existing_hash = None

        # Query existing document data
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (abs_file_path,))
            result = cursor.fetchone()
            if result:
                existing_doc_id, existing_hash = result
                ASCIIColors.debug(f"Document '{file_path.name}' found in DB (doc_id={existing_doc_id}). Hash: {existing_hash}")
        except sqlite3.Error as e:
            ASCIIColors.error(f"Error checking existing document '{file_path.name}': {e}")
            return # Abort on DB error


        if existing_doc_id is not None and not force_reindex:
            if existing_hash == current_hash:
                ASCIIColors.info(f"Document '{file_path.name}' is unchanged. Checking vectorization...")
                # Check if this specific vectorization already exists for this doc
                try:
                    # First get the method_id for the requested vectorizer
                    _vec_instance, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, self.conn)

                    # Now check if vectors exist for this doc_id and method_id
                    cursor.execute("""
                        SELECT 1 FROM vectors v
                        JOIN chunks c ON v.chunk_id = c.chunk_id
                        WHERE c.doc_id = ? AND v.method_id = ?
                        LIMIT 1
                    """, (existing_doc_id, method_id))
                    vector_exists = cursor.fetchone() is not None
                    if vector_exists:
                        ASCIIColors.success(f"Vectorization '{vectorizer_name}' already exists for '{file_path.name}'. Skipping.")
                        return # Nothing to do
                    else:
                         ASCIIColors.info(f"Document '{file_path.name}' exists, but needs vectorization '{vectorizer_name}'. Proceeding...")
                         # Need to re-vectorize existing chunks below
                    pass # Fall through to vectorization part for existing doc
                except Exception as e: # Catch potential errors in get_vectorizer or DB query
                    ASCIIColors.error(f"Error checking existing vectors for '{file_path.name}': {e}", exc_info=True)
                    return # Abort
            else:
                ASCIIColors.warning(f"Document '{file_path.name}' has changed (hash mismatch). Re-indexing...")
                # Need to delete old chunks/vectors associated with this doc_id before re-adding
                try:
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    # Vectors are deleted automatically due to CASCADE constraint
                    self.conn.commit()
                    ASCIIColors.debug(f"Deleted old chunks/vectors for changed document doc_id={existing_doc_id}.")
                    # We also need to update the document record itself (hash, maybe metadata)
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?",
                                   (current_hash, None, json.dumps(metadata) if metadata else None, existing_doc_id)) # Clear full_text until reparsed
                    self.conn.commit()

                except sqlite3.Error as e:
                    ASCIIColors.error(f"Failed to delete old data for changed document '{file_path.name}': {e}", exc_info=True)
                    self.conn.rollback()
                    return # Abort
                # Fall through to parsing/chunking/vectorizing


        # --- Parse Document ---
        try:
            full_text = parser.parse_document(file_path)
            ASCIIColors.debug(f"Parsed document '{file_path.name}'. Length: {len(full_text)} chars.")
        except Exception as e:
            ASCIIColors.error(f"Failed to parse {file_path.name}: {e}")
            return # Abort

        # --- Start Transaction ---
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            # --- Add/Update Document Record ---
            if existing_doc_id is None: # It's a completely new document
                 doc_id = db.add_document_record(
                     self.conn, abs_file_path, full_text, current_hash, json.dumps(metadata) if metadata else None
                 )
                 if doc_id is None: # Should have raised in add_document_record
                     raise RuntimeError("Failed to get doc_id for new document.")
            else: # Document existed, potentially updated hash/metadata earlier
                 doc_id = existing_doc_id
                 # Ensure full_text is updated if re-indexing due to hash change or force_reindex
                 cursor.execute("UPDATE documents SET full_text = ?, file_hash = ? WHERE doc_id = ?", (full_text, current_hash, doc_id))


            # --- Chunk Text ---
            chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
            if not chunks_data:
                ASCIIColors.warning(f"No chunks generated for {file_path.name}. Skipping vectorization.")
                self.conn.rollback() # Nothing to add
                return

            ASCIIColors.info(f"Generated {len(chunks_data)} chunks for '{file_path.name}'.")

            # --- Add Chunk Records (if new or re-indexing) ---
            chunk_ids = []
            chunk_texts = []
            if existing_doc_id is None or force_reindex or existing_hash != current_hash:
                ASCIIColors.debug(f"Adding {len(chunks_data)} chunk records to DB...")
                for i, (text, start, end) in enumerate(chunks_data):
                    chunk_id = db.add_chunk_record(self.conn, doc_id, text, start, end, i)
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(text)
            else: # Document existed, hash matched, but vectors were missing for this method_id
                 # We need to retrieve existing chunk_ids and texts for vectorization
                 ASCIIColors.debug(f"Retrieving existing chunks for doc_id={doc_id} to add new vectors...")
                 cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
                 results = cursor.fetchall()
                 if not results:
                      ASCIIColors.error(f"Document {doc_id} found but no chunks retrieved! Inconsistent state.")
                      self.conn.rollback()
                      return
                 chunk_ids = [row[0] for row in results]
                 chunk_texts = [row[1] for row in results]
                 ASCIIColors.debug(f"Retrieved {len(chunk_ids)} existing chunk IDs and texts.")


            # --- Vectorize Chunks ---
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, self.conn)
            ASCIIColors.info(f"Vectorizing {len(chunk_texts)} chunks using '{vectorizer_name}'...")

            # Vectorize in batches for efficiency (Sentence Transformers handles this internally)
            vectors = vectorizer.vectorize(chunk_texts)

            if vectors.shape[0] != len(chunk_ids):
                ASCIIColors.error(f"Mismatch between number of chunks ({len(chunk_ids)}) and generated vectors ({vectors.shape[0]})!")
                raise ValueError("Chunk and vector count mismatch.")

            # --- Add Vector Records ---
            ASCIIColors.debug(f"Adding {len(vectors)} vector records to DB (method_id={method_id})...")
            for chunk_id, vector in zip(chunk_ids, vectors):
                 # Ensure vector is C-contiguous for tobytes() if needed (usually true from encode)
                 vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                 db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)


            # --- Commit Transaction ---
            self.conn.commit()
            ASCIIColors.success(f"Successfully indexed and vectorized '{file_path.name}' with '{vectorizer_name}'.")

        except Exception as e:
            ASCIIColors.error(f"Error during indexing of '{file_path.name}': {e}", exc_info=True)
            if self.conn:
                self.conn.rollback() # Rollback on any error during the process

    # Add query, reindex, etc. methods in later phases