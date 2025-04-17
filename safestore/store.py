# safestore/store.py
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import hashlib # For basic file hashing

from .core import db
from .indexing import parser, chunking
from .search import similarity # Added import
from .vectorization.manager import VectorizationManager
from .vectorization.methods.tfidf import TfidfVectorizerWrapper # Added import
from ascii_colors import ASCIIColors, LogLevel

class SafeStore:
    """
    Main class for interacting with the SafeStore database.
    Manages document indexing, vectorization, and querying.
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
        self.vectorizer_manager.clear_cache() # Clear cache on close

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
            ASCIIColors.error(f"File not found when trying to hash: {file_path}")
            raise # Re-raise file not found
        except Exception as e:
            ASCIIColors.warning(f"Could not generate hash for {file_path}: {e}")
            return "" # Or raise? Let's return empty to signal failure


    def add_document(
        self,
        file_path: str | Path,
        vectorizer_name: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        metadata: Optional[dict] = None,
        force_reindex: bool = False,
        vectorizer_params: Optional[dict] = None # Added for TFIDF etc.
    ):
        """
        Adds or updates a document in the SafeStore, including parsing, chunking,
        and vectorization with the specified method.

        If the document exists and its content hash matches the stored hash,
        it checks if vectors for the *specified* vectorizer_name already exist.
        If they do, it skips processing. If not, it vectorizes the existing chunks.

        If the document exists but the hash differs, or if force_reindex is True,
        it deletes the old chunks and vectors, updates the document record,
        re-parses, re-chunks, and re-vectorizes the content.

        Args:
            file_path: Path to the document file.
            vectorizer_name: Name of the vectorizer to use (e.g., 'st:all-MiniLM-L6-v2', 'tfidf:basic').
                               Defaults to SafeStore.DEFAULT_VECTORIZER.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            metadata: Optional dictionary of metadata to store with the document.
            force_reindex: If True, re-indexes the document even if path/hash hasn't changed.
            vectorizer_params: Optional dictionary of parameters specifically for the
                               vectorizer initialization (e.g., for TF-IDF max_features).
                               These are typically only used when the vectorizer method is *first* created.
        """
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
                    # Needs parsing, chunking, vectorization
                elif existing_hash == current_hash:
                    ASCIIColors.info(f"Document '{file_path.name}' is unchanged.")
                    needs_parsing_chunking = False # Don't need to re-parse/re-chunk
                    # Check if *this specific vectorization* already exists
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
                         # Needs only vectorization of existing chunks
                else:
                    ASCIIColors.warning(f"Document '{file_path.name}' has changed (hash mismatch). Re-indexing...")
                    # Needs parsing, chunking, vectorization. Delete old chunks first.
                    cursor.execute("BEGIN")
                    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
                    # Vectors are deleted automatically due to CASCADE constraint
                    cursor.execute("UPDATE documents SET file_hash = ?, full_text = ?, metadata = ? WHERE doc_id = ?",
                                   (current_hash, None, json.dumps(metadata) if metadata else None, existing_doc_id)) # Clear full_text until reparsed
                    self.conn.commit()
                    ASCIIColors.debug(f"Deleted old chunks/vectors and updated document record for changed doc_id={existing_doc_id}.")
            else:
                # Document is completely new
                 ASCIIColors.info(f"Document '{file_path.name}' is new.")
                 # Needs parsing, chunking, vectorization

        except sqlite3.Error as e:
            ASCIIColors.error(f"Database error checking/updating document '{file_path.name}': {e}", exc_info=True)
            self.conn.rollback() # Rollback if BEGIN was executed
            raise
        except Exception as e: # Catch other errors like get_vectorizer failure
             ASCIIColors.error(f"Error preparing indexing for '{file_path.name}': {e}", exc_info=True)
             raise

        if not needs_parsing_chunking and not needs_vectorization:
             return # Nothing more to do

        # --- Start Transaction for main processing ---
        cursor = self.conn.cursor()
        try:
            cursor.execute("BEGIN")

            doc_id = existing_doc_id
            full_text = None
            chunks_data = [] # List of (text, start, end)
            chunk_ids = [] # List of corresponding chunk_ids
            chunk_texts = [] # List of corresponding chunk texts

            # --- Step 1: Parsing and Chunking (if needed) ---
            if needs_parsing_chunking:
                ASCIIColors.debug(f"Parsing document: {file_path.name}")
                try:
                    full_text = parser.parse_document(file_path)
                    ASCIIColors.debug(f"Parsed document '{file_path.name}'. Length: {len(full_text)} chars.")
                except Exception as e:
                    ASCIIColors.error(f"Failed to parse {file_path.name}: {e}")
                    raise # Re-raise to be caught by outer try/except

                # Add/Update document record
                if doc_id is None: # New document
                     _doc_id = db.add_document_record(
                         self.conn, abs_file_path, full_text, current_hash, json.dumps(metadata) if metadata else None
                     )
                     if _doc_id is None: raise RuntimeError("Failed to get doc_id for new document.")
                     doc_id = _doc_id
                else: # Existing document, update full_text (hash/metadata updated earlier if changed)
                    cursor.execute("UPDATE documents SET full_text = ? WHERE doc_id = ?", (full_text, doc_id))

                # Chunk text
                chunks_data = chunking.chunk_text(full_text, chunk_size, chunk_overlap)
                if not chunks_data:
                    ASCIIColors.warning(f"No chunks generated for {file_path.name}. Skipping vectorization.")
                    self.conn.commit() # Commit document record update even if no chunks
                    return

                ASCIIColors.info(f"Generated {len(chunks_data)} chunks for '{file_path.name}'. Storing chunks...")

                # Add chunk records
                for i, (text, start, end) in enumerate(chunks_data):
                    chunk_id = db.add_chunk_record(self.conn, doc_id, text, start, end, i)
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(text)

            else: # No parsing/chunking needed, retrieve existing chunks for vectorization
                ASCIIColors.debug(f"Retrieving existing chunks for doc_id={doc_id} to add new vectors...")
                cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? ORDER BY chunk_seq", (doc_id,))
                results = cursor.fetchall()
                if not results:
                      ASCIIColors.error(f"Document {doc_id} exists but no chunks found! Inconsistent state.")
                      raise RuntimeError(f"Inconsistent state: No chunks found for existing document ID {doc_id}")
                chunk_ids = [row[0] for row in results]
                chunk_texts = [row[1] for row in results]
                ASCIIColors.debug(f"Retrieved {len(chunk_ids)} existing chunk IDs and texts.")


            # --- Step 2: Vectorization (if needed) ---
            if needs_vectorization:
                if not chunk_ids or not chunk_texts:
                     ASCIIColors.warning(f"No chunks available to vectorize for '{file_path.name}'.")
                     # Commit previous changes (doc record, chunks if added)
                     self.conn.commit()
                     return

                # Get vectorizer (handles TF-IDF state loading/initialization)
                vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn)

                # --- Special Handling: TF-IDF Fitting (if needed) ---
                # TF-IDF needs to be fitted *before* vectorizing.
                # If it's a new TF-IDF method or loaded state wasn't fitted, fit it now.
                # Fit on the chunks of the *current* document. This might not be ideal for global TF-IDF.
                # The 'add_vectorization' method provides a way to fit on the whole corpus.
                if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                     ASCIIColors.warning(f"TF-IDF vectorizer '{_vectorizer_name}' is not fitted. Fitting on chunks from '{file_path.name}' only.")
                     try:
                         vectorizer.fit(chunk_texts)
                         # Update the method in DB with fitted params and dimension
                         new_params = vectorizer.get_params_to_store()
                         self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                     except Exception as e:
                         ASCIIColors.error(f"Failed to fit TF-IDF model '{_vectorizer_name}' on '{file_path.name}': {e}")
                         raise # Re-raise to rollback transaction


                ASCIIColors.info(f"Vectorizing {len(chunk_texts)} chunks using '{_vectorizer_name}' (method_id={method_id})...")

                # Vectorize (Sentence Transformers handles batching internally)
                try:
                     vectors = vectorizer.vectorize(chunk_texts)
                except Exception as e:
                     ASCIIColors.error(f"Vectorization failed for '{_vectorizer_name}': {e}", exc_info=True)
                     raise # Re-raise to rollback transaction


                if vectors.shape[0] != len(chunk_ids):
                    ASCIIColors.error(f"Mismatch between number of chunks ({len(chunk_ids)}) and generated vectors ({vectors.shape[0]})!")
                    raise ValueError("Chunk and vector count mismatch during indexing.")

                # Add Vector Records
                ASCIIColors.debug(f"Adding {len(vectors)} vector records to DB (method_id={method_id})...")
                for chunk_id, vector in zip(chunk_ids, vectors):
                    vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                    db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)

            # --- Commit Transaction ---
            self.conn.commit()
            ASCIIColors.success(f"Successfully processed '{file_path.name}' with vectorizer '{_vectorizer_name}'.")

        except Exception as e:
            ASCIIColors.error(f"Error during indexing of '{file_path.name}': {e}", exc_info=True)
            if self.conn:
                self.conn.rollback() # Rollback on any error during the process
            raise # Re-raise the exception


    def query(
        self,
        query_text: str,
        vectorizer_name: str | None = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries the store for chunks most similar to the query text using the specified vectorizer.

        Args:
            query_text: The text to search for.
            vectorizer_name: The name of the vectorizer method to use for the query
                               (must match a method used during indexing). Defaults to DEFAULT_VECTORIZER.
            top_k: The maximum number of similar chunks to return.

        Returns:
            A list of dictionaries, each representing a relevant chunk, sorted by similarity (descending).
            Each dictionary contains:
            - 'chunk_id': ID of the chunk.
            - 'chunk_text': The text content of the chunk.
            - 'similarity': Cosine similarity score (between 0 and 1).
            - 'doc_id': ID of the source document.
            - 'file_path': Path to the source document file.
            - 'start_pos': Start character offset of the chunk in the document.
            - 'end_pos': End character offset of the chunk in the document.
            - 'chunk_seq': Sequence number of the chunk within the document.
        """
        _vectorizer_name = vectorizer_name or self.DEFAULT_VECTORIZER
        ASCIIColors.info(f"Received query. Searching with '{_vectorizer_name}', top_k={top_k}.")

        results = []
        cursor = self.conn.cursor()

        try:
            # --- 1. Get Vectorizer and Method ID ---
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(_vectorizer_name, self.conn)
            ASCIIColors.debug(f"Using vectorizer '{_vectorizer_name}' (method_id={method_id})")

            # --- 2. Vectorize Query ---
            ASCIIColors.debug(f"Vectorizing query text...")
            query_vector = vectorizer.vectorize([query_text])[0] # Vectorize returns 2D array
            query_vector = np.ascontiguousarray(query_vector, dtype=vectorizer.dtype)
            ASCIIColors.debug(f"Query vector generated. Shape: {query_vector.shape}, Dtype: {query_vector.dtype}")


            # --- 3. Load Candidate Vectors from DB ---
            # **Performance Bottleneck**: Loads ALL vectors for the method.
            # TODO: Explore optimizations like indexing (FAISS, Annoy) or SQL-based VSS if available.
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

            # Reconstruct vectors (need dtype)
            method_details = self.vectorizer_manager._get_method_details_from_db(self.conn, _vectorizer_name)
            if not method_details: # Should exist if get_vectorizer succeeded
                 raise RuntimeError(f"Could not retrieve method details for '{_vectorizer_name}' after getting instance.")
            vector_dtype = method_details['vector_dtype']

            ASCIIColors.debug(f"Reconstructing {len(vector_blobs)} vectors from BLOBs with dtype '{vector_dtype}'...")
            try:
                 candidate_vectors = np.array([db.reconstruct_vector(blob, vector_dtype) for blob in vector_blobs])
            except ValueError as e:
                 ASCIIColors.error(f"Failed to reconstruct one or more vectors: {e}")
                 raise

            ASCIIColors.debug(f"Candidate vectors loaded. Matrix shape: {candidate_vectors.shape}")

            # --- 4. Calculate Similarities ---
            ASCIIColors.debug("Calculating similarity scores...")
            scores = similarity.cosine_similarity(query_vector, candidate_vectors)
            ASCIIColors.debug(f"Similarity scores calculated. Shape: {scores.shape}")


            # --- 5. Get Top-K Results ---
            # Get indices of top-k scores (descending order)
            num_candidates = len(scores)
            k = min(top_k, num_candidates) # Adjust k if fewer candidates than requested
            top_k_indices = np.argsort(scores)[::-1][:k]

            ASCIIColors.debug(f"Identified top {k} indices.")

            # --- 6. Retrieve Chunk Details for Top-K ---
            if k > 0:
                top_chunk_ids = [chunk_ids_ordered[i] for i in top_k_indices]
                top_scores = [scores[i] for i in top_k_indices]

                # Create placeholders for IN clause query
                placeholders = ','.join('?' * len(top_chunk_ids))
                sql_chunk_details = f"""
                    SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.chunk_seq,
                           d.doc_id, d.file_path
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.chunk_id IN ({placeholders})
                """
                cursor.execute(sql_chunk_details, top_chunk_ids)
                chunk_details_list = cursor.fetchall()

                # Create a mapping from chunk_id to its details for easier lookup
                chunk_details_map = {
                     row[0]: {
                        "chunk_id": row[0],
                        "chunk_text": row[1],
                        "start_pos": row[2],
                        "end_pos": row[3],
                        "chunk_seq": row[4],
                        "doc_id": row[5],
                        "file_path": row[6]
                     } for row in chunk_details_list
                }

                # Build final results list, ordered by similarity score
                for chunk_id, score in zip(top_chunk_ids, top_scores):
                    if chunk_id in chunk_details_map:
                        result_item = chunk_details_map[chunk_id].copy()
                        result_item["similarity"] = float(score) # Ensure score is float
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


    def add_vectorization(
        self,
        vectorizer_name: str,
        target_doc_path: Optional[str | Path] = None,
        vectorizer_params: Optional[dict] = None,
        batch_size: int = 64 # Batch size for vectorizing chunks
    ):
        """
        Adds vector embeddings using a new or existing vectorization method
        to documents already present in the store.

        If the vectorizer is TF-IDF and it hasn't been fitted yet, this method
        will fit it on the text chunks of the specified documents (or all documents
        if target_doc_path is None) before generating vectors.

        Args:
            vectorizer_name: The name of the vectorizer method to add
                               (e.g., 'st:paraphrase-mpnet-base-v2', 'tfidf:ngram1-max5k').
            target_doc_path: If specified, only add vectors for this document.
                               If None (default), add vectors for ALL documents in the store.
            vectorizer_params: Optional dictionary of parameters for vectorizer initialization
                               (e.g., TF-IDF settings). Only used if the method is new.
            batch_size: Number of chunks to vectorize in a single batch (if applicable).
        """
        ASCIIColors.info(f"Starting process to add vectorization '{vectorizer_name}'.")
        if target_doc_path:
             target_doc_path = Path(target_doc_path).resolve()
             ASCIIColors.info(f"Targeting specific document: {target_doc_path}")
        else:
             ASCIIColors.info("Targeting all documents in the store.")

        cursor = self.conn.cursor()
        try:
            # --- 1. Get/Register Vectorizer Method ---
            # Use vectorizer_params only if the method is truly new.
            # The manager handles loading existing state.
            vectorizer, method_id = self.vectorizer_manager.get_vectorizer(vectorizer_name, self.conn)

            # --- 2. Handle TF-IDF Fitting (if needed and not already fitted) ---
            if isinstance(vectorizer, TfidfVectorizerWrapper) and not vectorizer._fitted:
                ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' requires fitting.")
                # Fetch chunks to fit on
                fit_sql = "SELECT chunk_text FROM chunks"
                fit_params: List[Any] = []
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
                texts_to_fit = [row[0] for row in cursor.fetchall()]

                if not texts_to_fit:
                     ASCIIColors.warning("No text chunks found to fit the TF-IDF model. Aborting vectorization add.")
                     return

                try:
                    vectorizer.fit(texts_to_fit)
                    # Update the method in DB with fitted params and dimension
                    new_params = vectorizer.get_params_to_store()
                    self.vectorizer_manager.update_method_params(self.conn, method_id, new_params, vectorizer.dim)
                    ASCIIColors.info(f"TF-IDF vectorizer '{vectorizer_name}' fitted successfully.")
                except Exception as e:
                    ASCIIColors.error(f"Failed to fit TF-IDF model '{vectorizer_name}': {e}", exc_info=True)
                    return # Abort


            # --- 3. Fetch Target Chunks ---
            chunks_to_vectorize_sql = f"""
                SELECT c.chunk_id, c.chunk_text
                FROM chunks c
                LEFT JOIN vectors v ON c.chunk_id = v.chunk_id AND v.method_id = ?
                WHERE v.vector_id IS NULL -- Only chunks *without* vectors for this method
            """
            sql_params: List[Any] = [method_id]

            if target_doc_path:
                # We already found target_doc_id during potential TF-IDF fit check
                # Re-fetch just in case or use stored value
                if 'target_doc_id' not in locals(): # Fetch if not found before
                    cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (str(target_doc_path),))
                    target_doc_id_result = cursor.fetchone()
                    if not target_doc_id_result:
                        ASCIIColors.error(f"Target document '{target_doc_path}' not found in the database.")
                        return
                    target_doc_id = target_doc_id_result[0]

                chunks_to_vectorize_sql += " AND c.doc_id = ?"
                sql_params.append(target_doc_id)
                ASCIIColors.info(f"Fetching chunks missing '{vectorizer_name}' vectors for document ID {target_doc_id}...")
            else:
                 ASCIIColors.info(f"Fetching all chunks missing '{vectorizer_name}' vectors...")


            cursor.execute(chunks_to_vectorize_sql, tuple(sql_params))
            chunks_data = cursor.fetchall() # List of (chunk_id, chunk_text)

            if not chunks_data:
                ASCIIColors.success(f"No chunks found needing vectorization for '{vectorizer_name}'. Process complete.")
                return

            total_chunks = len(chunks_data)
            ASCIIColors.info(f"Found {total_chunks} chunks to vectorize.")

            # --- 4. Vectorize in Batches and Store ---
            num_added = 0
            cursor.execute("BEGIN") # Start transaction for adding vectors
            try:
                for i in range(0, total_chunks, batch_size):
                    batch = chunks_data[i : i + batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_texts = [item[1] for item in batch]

                    ASCIIColors.debug(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch_texts)} chunks)")

                    try:
                         vectors = vectorizer.vectorize(batch_texts)
                         if vectors.shape[0] != len(batch_ids):
                              raise ValueError(f"Vectorization output count ({vectors.shape[0]}) doesn't match batch size ({len(batch_ids)}).")
                    except Exception as e:
                         ASCIIColors.error(f"Vectorization failed for batch: {e}", exc_info=True)
                         # Option: Skip batch or abort? Let's abort the transaction.
                         raise RuntimeError("Vectorization failed, aborting add_vectorization.") from e


                    # Add vectors for the batch
                    for chunk_id, vector in zip(batch_ids, vectors):
                         vector_contiguous = np.ascontiguousarray(vector, dtype=vectorizer.dtype)
                         db.add_vector_record(self.conn, chunk_id, method_id, vector_contiguous)
                    num_added += len(batch_ids)
                    ASCIIColors.debug(f"Added {len(batch_ids)} vectors for batch.")

                self.conn.commit() # Commit transaction after all batches
                ASCIIColors.success(f"Successfully added {num_added} vector embeddings using '{vectorizer_name}'.")

            except Exception as e:
                 ASCIIColors.error(f"Error during vectorization/storage: {e}", exc_info=True)
                 self.conn.rollback() # Rollback transaction on error
                 raise # Re-raise

        except sqlite3.Error as e:
             ASCIIColors.error(f"Database error during add_vectorization: {e}", exc_info=True)
             raise
        except Exception as e:
             ASCIIColors.error(f"An unexpected error occurred during add_vectorization: {e}", exc_info=True)
             raise

    def remove_vectorization(self, vectorizer_name: str):
        """
        Removes a vectorization method and all associated vector embeddings from the store.

        Args:
            vectorizer_name: The name of the vectorizer method to remove.
        """
        ASCIIColors.warning(f"Attempting to remove vectorization method '{vectorizer_name}' and all associated vectors.")

        cursor = self.conn.cursor()
        try:
            # --- 1. Find Method ID ---
            cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_name,))
            result = cursor.fetchone()
            if not result:
                ASCIIColors.error(f"Vectorization method '{vectorizer_name}' not found in the database. Cannot remove.")
                return
            method_id = result[0]
            ASCIIColors.debug(f"Found method_id {method_id} for '{vectorizer_name}'.")

            # --- 2. Delete Vectors and Method ---
            cursor.execute("BEGIN")
            # Vectors might be deleted by cascade if ON DELETE CASCADE is reliable,
            # but explicit deletion is safer.
            cursor.execute("DELETE FROM vectors WHERE method_id = ?", (method_id,))
            deleted_vectors = cursor.rowcount
            ASCIIColors.debug(f"Deleted {deleted_vectors} vector records.")

            cursor.execute("DELETE FROM vectorization_methods WHERE method_id = ?", (method_id,))
            deleted_methods = cursor.rowcount
            ASCIIColors.debug(f"Deleted {deleted_methods} vectorization method record.")

            self.conn.commit()

            # Clear from cache if it exists
            if vectorizer_name in self.vectorizer_manager._cache:
                 del self.vectorizer_manager._cache[vectorizer_name]
                 ASCIIColors.debug(f"Removed '{vectorizer_name}' from vectorizer cache.")


            ASCIIColors.success(f"Successfully removed vectorization method '{vectorizer_name}' (ID: {method_id}) and {deleted_vectors} associated vectors.")

        except sqlite3.Error as e:
             ASCIIColors.error(f"Database error during removal of '{vectorizer_name}': {e}", exc_info=True)
             self.conn.rollback()
             raise
        except Exception as e:
             ASCIIColors.error(f"An unexpected error occurred during removal of '{vectorizer_name}': {e}", exc_info=True)
             raise