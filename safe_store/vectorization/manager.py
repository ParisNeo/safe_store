# safe_store/vectorization/manager.py
import sqlite3
import json
from typing import Tuple, Optional, Dict, Any
import numpy as np

from ..core import db
from ..core.exceptions import ConfigurationError, VectorizationError, DatabaseError, SafeStoreError
from .base import BaseVectorizer
from .methods.sentence_transformer import SentenceTransformerVectorizer
from .methods.tfidf import TfidfVectorizerWrapper
from ascii_colors import ASCIIColors

class VectorizationManager:
    """
    Manages available vectorization methods and their instances.

    Handles loading, caching, and retrieving vectorizer instances based on
    their names (e.g., 'st:model-name', 'tfidf:method-name'). Interacts with the
    database to store and retrieve method details, including fitted state for
    methods like TF-IDF.
    """

    def __init__(self):
        # Cache format: name -> (instance, method_id, params_from_db_at_load)
        self._cache: Dict[str, Tuple[BaseVectorizer, int, Optional[Dict[str, Any]]]] = {}

    def get_vectorizer(
        self,
        name: str,
        conn: sqlite3.Connection,
        initial_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[BaseVectorizer, int]:
        """
        Gets or initializes a vectorizer instance and its DB method ID.

        Checks the cache first. If not found, it checks the database for an
        existing method record. If found, it loads the configuration (including
        fitted state for TF-IDF). If not found in the DB, it initializes a new
        vectorizer instance based on the name format, registers it in the DB,
        and caches it.

        Args:
            name: The unique identifier for the vectorizer (e.g.,
                  'st:all-MiniLM-L6-v2', 'tfidf:ngram1-max5k').
            conn: Active database connection.
            initial_params: Optional dictionary of parameters, primarily used
                            for initializing TF-IDF with specific configurations
                            if it's being created for the first time via this call.

        Returns:
            A tuple containing the (potentially cached) vectorizer instance
            and its method_id from the database.

        Raises:
            ConfigurationError: If required dependencies are missing or the name
                                format is unknown.
            VectorizationError: If initializing the vectorizer model fails.
            DatabaseError: If database interaction fails.
            SafeStoreError: For unexpected errors.
        """
        # Check cache first
        if name in self._cache:
            instance, method_id, _ = self._cache[name]
            ASCIIColors.debug(f"Vectorizer '{name}' found in cache (method_id={method_id}).")
            return instance, method_id

        ASCIIColors.info(f"Initializing vectorizer: {name}")

        # --- 1. Check DB for Existing Method ---
        method_details = self._get_method_details_from_db(conn, name)
        vectorizer: Optional[BaseVectorizer] = None
        method_id: Optional[int] = None
        params_from_db: Optional[Dict[str, Any]] = None
        method_type: Optional[str] = None

        if method_details:
            method_id = method_details['method_id']
            method_type = method_details['method_type']
            dim = method_details['vector_dim'] # Dim/dtype primarily for info/verification
            dtype_str = method_details['vector_dtype']
            params_str = method_details['params']
            if params_str:
                try:
                    params_from_db = json.loads(params_str)
                except json.JSONDecodeError:
                    ASCIIColors.warning(f"Could not decode params JSON for existing method '{name}' (ID: {method_id}). Proceeding without loaded params.")
            ASCIIColors.debug(f"Found existing method '{name}' (ID: {method_id}) in DB. Type: {method_type}, Dim: {dim}, Dtype: {dtype_str}")
        else:
             ASCIIColors.debug(f"No existing method found in DB for '{name}'. Will create.")


        # --- 2. Instantiate Vectorizer ---
        try:
            # Determine type from name prefix if not loaded from DB
            if method_type is None:
                 if name.startswith("st:"):
                     method_type = "sentence_transformer"
                 elif name.startswith("tfidf:"):
                     method_type = "tfidf"
                 else:
                     raise ConfigurationError(f"Unknown vectorizer name format: {name}. Must start with 'st:' or 'tfidf:'.")

            # Instantiate based on type
            if method_type == "sentence_transformer":
                model_name_part = name.split(":", 1)[1] if ":" in name else SentenceTransformerVectorizer.DEFAULT_MODEL
                # ST initialization doesn't use params_from_db or initial_params
                vectorizer = SentenceTransformerVectorizer(model_name=model_name_part)

            elif method_type == "tfidf":
                # Use initial_params if creating new, else use loaded params_from_db
                init_or_load_params = initial_params if params_from_db is None else params_from_db.get("sklearn_params", {})
                vectorizer = TfidfVectorizerWrapper(params=init_or_load_params)
                # Attempt to load fitted state ONLY if params were loaded from DB
                if params_from_db and params_from_db.get("fitted"):
                    vectorizer.load_fitted_state(params_from_db)

            else:
                 # This case should be caught earlier by name format check
                 raise ConfigurationError(f"Unsupported vectorizer type '{method_type}' derived from name '{name}'.")

        except ImportError as e:
            # Raised by vectorizer __init__ if dependency missing
            dep_map = {
                "sentence_transformer": "safe_store[sentence-transformers]",
                "tfidf": "safe_store[tfidf]"
            }
            install_cmd = dep_map.get(method_type, f"the required library for '{method_type}'")
            msg = f"Missing dependency for {method_type} vectorizer '{name}'. Please install '{install_cmd}'. Error: {e}"
            ASCIIColors.error(msg)
            raise ConfigurationError(msg) from e
        except Exception as e:
            # Catch other initialization errors (e.g., model loading failed)
            msg = f"Failed to initialize {method_type} vectorizer '{name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e # Use specific error

        if not vectorizer:
            # Should not be reachable if instantiation logic is correct
            raise SafeStoreError(f"Vectorizer instance could not be created for '{name}'.")


        # --- 3. Add/Get Method Record in DB ---
        # Transaction should be handled by the caller (e.g., safe_store._add_document_impl)
        if method_id is None:
            # Registering a new method
            current_params: Dict[str, Any] = {}
            dim_to_store: int
            if isinstance(vectorizer, TfidfVectorizerWrapper):
                # TF-IDF might not be fitted yet, dim might be None. Store initial state.
                current_params = vectorizer.get_params_to_store() # Includes fitted=False initially
                dim_to_store = vectorizer.dim if vectorizer.dim is not None else 0 # Use 0 if dim unknown
            else:
                # For ST, dim is known, params are minimal/none
                current_params = {} # Or potentially {'model_name': vectorizer.model_name}? Keep simple for now.
                dim_to_store = vectorizer.dim

            try:
                 method_id = db.add_or_get_vectorization_method(
                     conn=conn,
                     name=name,
                     type=method_type,
                     dim=dim_to_store,
                     dtype=np.dtype(vectorizer.dtype).name,
                     params=json.dumps(current_params) if current_params else None
                 )
                 ASCIIColors.debug(f"Vectorizer '{name}' registered in DB with method_id {method_id}.")
            except DatabaseError as e:
                 # Handle potential race condition if another process added it between check and insert
                 if "UNIQUE constraint failed" in str(e):
                      ASCIIColors.warning(f"Race condition detected? Method '{name}' registered by another process. Fetching existing ID.")
                      conn.rollback() # Rollback the failed insert attempt
                      method_details = self._get_method_details_from_db(conn, name)
                      if method_details:
                           method_id = method_details['method_id']
                           # Update params_from_db as it might have changed
                           params_str = method_details['params']
                           if params_str:
                                try: params_from_db = json.loads(params_str)
                                except json.JSONDecodeError: pass
                      else:
                           # Should not happen if UNIQUE constraint failed
                           msg = f"UNIQUE constraint failed for '{name}', but cannot fetch existing ID afterwards."
                           ASCIIColors.critical(msg)
                           raise DatabaseError(msg) from e
                 else:
                      raise # Re-raise other DatabaseErrors
        else:
             # Method already existed in DB
             ASCIIColors.debug(f"Using existing method_id {method_id} for '{name}'.")


        # --- 4. Cache Result ---
        if method_id is None: # Defensive check
             raise SafeStoreError(f"Failed to obtain a valid method_id for '{name}'.")

        self._cache[name] = (vectorizer, method_id, params_from_db)
        ASCIIColors.debug(f"Vectorizer '{name}' ready and cached (method_id {method_id}).")
        return vectorizer, method_id

    def _get_method_details_from_db(self, conn: sqlite3.Connection, name: str) -> Optional[Dict[str, Any]]:
        """Fetches method details from the DB by name. Internal use."""
        sql: str = """
        SELECT method_id, method_type, vector_dim, vector_dtype, params
        FROM vectorization_methods
        WHERE method_name = ?
        """
        cursor = conn.cursor()
        try:
            cursor.execute(sql, (name,))
            result = cursor.fetchone()
            if result:
                return {
                    'method_id': result[0],
                    'method_type': result[1],
                    'vector_dim': result[2],
                    'vector_dtype': result[3],
                    'params': result[4] # Keep as string, caller decides decoding
                }
            return None
        except sqlite3.Error as e:
            msg = f"Database error fetching vectorization method details for '{name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise DatabaseError(msg) from e # Re-raise as DatabaseError

    def update_method_params(
        self,
        conn: sqlite3.Connection,
        method_id: int,
        new_params: Dict[str, Any],
        new_dim: Optional[int] = None
    ) -> None:
        """
        Updates the parameters and optionally the dimension of a vectorization method in the DB.

        Typically used after fitting a TF-IDF model to store its vocabulary,
        IDF weights, and calculated dimension. Invalidates the cache entry for
        the updated method.

        Args:
            conn: Active database connection.
            method_id: The ID of the method to update.
            new_params: The new dictionary of parameters to store as JSON.
            new_dim: Optional new dimension value to store.

        Raises:
            DatabaseError: If the update fails.
            SafeStoreError: For unexpected errors.
        """
        sql_parts = ["UPDATE vectorization_methods SET params = ?"]
        # Use list for params, convert dict to JSON string
        params_list: List[Any] = [json.dumps(new_params)]
        if new_dim is not None:
            sql_parts.append(", vector_dim = ?")
            params_list.append(new_dim)

        sql_parts.append("WHERE method_id = ?")
        params_list.append(method_id)

        sql: str = " ".join(sql_parts)

        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params_list))
            # No commit here, assume caller handles transaction
            ASCIIColors.debug(f"Prepared update for params (and dim if provided) for method_id {method_id}.")
            self.remove_from_cache_by_id(method_id, log_reason="param update")

        except sqlite3.Error as e:
            msg = f"Database error updating vectorization method params for ID {method_id}: {e}"
            ASCIIColors.error(msg, exc_info=True)
            # Rollback handled by caller
            raise DatabaseError(msg) from e
        except Exception as e:
            msg = f"Database error updating vectorization method params for ID {method_id}: {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise SafeStoreError(msg) from e

    def remove_from_cache_by_id(self, method_id: int, log_reason: str = "removal") -> None:
        """Removes cache entries associated with a given method ID."""
        # Use list() to avoid modifying dict during iteration if multiple names point to same ID (shouldn't happen)
        cached_names_to_remove = [name for name, (_, mid, _) in self._cache.items() if mid == method_id]
        for name in cached_names_to_remove:
            if name in self._cache:
                del self._cache[name]
                ASCIIColors.debug(f"Invalidated cache for method '{name}' (ID: {method_id}) due to {log_reason}.")

    def clear_cache(self) -> None:
        """Clears the entire internal vectorizer cache."""
        count = len(self._cache)
        self._cache = {}
        if count > 0:
             ASCIIColors.debug(f"Cleared vectorizer manager cache ({count} items).")