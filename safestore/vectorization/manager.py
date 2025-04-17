# safestore/vectorization/manager.py
import sqlite3
import json
from typing import Tuple, Optional
from .base import BaseVectorizer
from .methods.sentence_transformer import SentenceTransformerVectorizer
from .methods.tfidf import TfidfVectorizerWrapper # Added import
from ..core import db # Import db functions
from ascii_colors import ASCIIColors
import numpy as np

class VectorizationManager:
    """Manages available vectorization methods."""

    def __init__(self):
        self._cache: dict[str, Tuple[BaseVectorizer, int, dict]] = {} # Cache: name -> (instance, method_id, params_from_db)

    def get_vectorizer(self, name: str, conn: sqlite3.Connection) -> Tuple[BaseVectorizer, int]:
        """
        Gets or initializes a vectorizer instance and its corresponding DB method ID.
        Handles loading fitted state for methods like TF-IDF if available.

        Args:
            name: The identifier for the vectorizer (e.g., 'st:all-MiniLM-L6-v2', 'tfidf:ngram1-max5k').
            conn: Active database connection.

        Returns:
            A tuple containing the vectorizer instance and its method_id from the DB.
        """
        if name in self._cache:
            ASCIIColors.debug(f"Vectorizer '{name}' found in cache.")
            instance, method_id, _ = self._cache[name]
            return instance, method_id

        ASCIIColors.info(f"Initializing vectorizer: {name}")

        # --- 1. Check if method exists in DB and get details ---
        method_details = self._get_method_details_from_db(conn, name)
        vectorizer = None
        method_id = None
        params_from_db = {}
        method_type = "unknown"
        dim = None
        dtype_str = None

        if method_details:
             method_id = method_details['method_id']
             method_type = method_details['method_type']
             dim = method_details['vector_dim']
             dtype_str = method_details['vector_dtype']
             params_str = method_details['params']
             if params_str:
                 try:
                     params_from_db = json.loads(params_str)
                 except json.JSONDecodeError:
                     ASCIIColors.warning(f"Could not decode params JSON for method '{name}' (ID: {method_id}).")
             ASCIIColors.debug(f"Found existing method '{name}' (ID: {method_id}) in DB. Type: {method_type}, Dim: {dim}, Dtype: {dtype_str}")


        # --- 2. Instantiate based on type ---
        init_params = {} # Params passed only during initial creation, if any
        if name.startswith("st:") or (method_type == "sentence_transformer"):
             method_type = "sentence_transformer" # Ensure type consistency
             model_name_part = name.split(":", 1)[1] if ":" in name else SentenceTransformerVectorizer.DEFAULT_MODEL
             init_params = {"model_name": model_name_part}
             # Always use the name part for ST model, ignore stored params for model choice
             try:
                 vectorizer = SentenceTransformerVectorizer(model_name=model_name_part)
             except ImportError as e:
                 ASCIIColors.error(f"Cannot initialize '{name}'. Missing dependency: {e}")
                 raise
             except Exception as e:
                 ASCIIColors.error(f"Failed to initialize SentenceTransformer '{model_name_part}': {e}")
                 raise

        elif name.startswith("tfidf:") or (method_type == "tfidf"):
             method_type = "tfidf" # Ensure type consistency
             # Extract config from name? e.g., tfidf:max_features=5000,ngram_range=1-2
             # For now, keep it simple: just use 'tfidf' or 'tfidf:custom' as name
             # Store actual TFIDF params (max_features etc) in the params column only.
             # Use params_from_db for instantiation and potential state loading.
             try:
                 vectorizer = TfidfVectorizerWrapper(params=params_from_db.get("sklearn_params", {}))
                 # Attempt to load fitted state if it exists in params_from_db
                 if params_from_db and params_from_db.get("fitted"):
                     vectorizer.load_fitted_state(params_from_db)

             except ImportError as e:
                 ASCIIColors.error(f"Cannot initialize '{name}'. Missing dependency: {e}")
                 raise
             except Exception as e:
                 ASCIIColors.error(f"Failed to initialize TfidfVectorizerWrapper: {e}")
                 raise

        else:
            ASCIIColors.error(f"Unknown vectorizer name format or type: {name}")
            raise ValueError(f"Unknown vectorizer name format or type: {name}")

        if not vectorizer:
            # Should have raised earlier if instantiation failed
            raise RuntimeError(f"Vectorizer instance could not be created for '{name}'.")


        # --- 3. Add/Get method record in DB if not already found ---
        if method_id is None:
            current_params = {}
            # Special handling for TF-IDF: get params *after* potential fitting
            if isinstance(vectorizer, TfidfVectorizerWrapper):
                # TF-IDF needs fitting, which happens later in add_vectorization
                # We register it with initial params and 'fitted': False state initially.
                # The dimension is also unknown until fitting. Use -1 or None? Let's use 0 temporarily.
                 current_params = vectorizer.get_params_to_store() # Gets initial params, fitted=False
                 if vectorizer.dim is None:
                     dim_to_store = 0 # Placeholder until fitted
                 else:
                     dim_to_store = vectorizer.dim
            else:
                # For ST, params are simple init params, dim is known
                current_params = init_params
                dim_to_store = vectorizer.dim

            method_id = db.add_or_get_vectorization_method(
                conn=conn,
                name=name, # Use the full identifier as the unique name
                type=method_type,
                dim=dim_to_store,
                dtype=np.dtype(vectorizer.dtype).name, # Store dtype as string 'float32' etc.
                params=json.dumps(current_params) if current_params else None
            )
            ASCIIColors.debug(f"Vectorizer '{name}' registered in DB with method_id {method_id}.")
        else:
             # If method existed, maybe update params if they changed (e.g., TF-IDF got fitted)
             # This update logic belongs in add_vectorization or fit methods.
             pass


        self._cache[name] = (vectorizer, method_id, params_from_db)
        ASCIIColors.debug(f"Vectorizer '{name}' ready (method_id {method_id}).")
        return vectorizer, method_id

    def _get_method_details_from_db(self, conn: sqlite3.Connection, name: str) -> Optional[dict]:
         """Fetches method details from the DB by name."""
         sql = """
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
                     'params': result[4]
                 }
             return None
         except sqlite3.Error as e:
             ASCIIColors.error(f"Error fetching vectorization method '{name}': {e}")
             raise # Re-raise to signal failure


    def update_method_params(self, conn: sqlite3.Connection, method_id: int, new_params: dict, new_dim: Optional[int] = None):
        """Updates the parameters and optionally dimension of a vectorization method in the DB."""
        sql_parts = ["UPDATE vectorization_methods SET params = ?"]
        params_list = [json.dumps(new_params)] # Use a different name to avoid confusion with module 'params'
        if new_dim is not None:
             # *** FIXED: Added comma before vector_dim ***
             sql_parts.append(", vector_dim = ?")
             params_list.append(new_dim)

        sql_parts.append("WHERE method_id = ?")
        params_list.append(method_id)

        sql = " ".join(sql_parts)

        cursor = conn.cursor()
        try:
            cursor.execute(sql, tuple(params_list))
            conn.commit()
            ASCIIColors.debug(f"Updated params (and dim if provided) for method_id {method_id}.")
            # Invalidate cache for this method as its params/state changed
            # Use list() to avoid modifying dict during iteration
            for name, (_, mid, _) in list(self._cache.items()):
                 if mid == method_id:
                     # Check if name still exists before deleting (might have been removed by another operation)
                     if name in self._cache:
                         del self._cache[name]
                         ASCIIColors.debug(f"Invalidated cache for method '{name}' due to param update.")

        except sqlite3.Error as e:
            ASCIIColors.error(f"Error updating vectorization method params for ID {method_id}: {e}")
            conn.rollback()
            raise

    def clear_cache(self):
         """Clears the internal vectorizer cache."""
         self._cache = {}
         ASCIIColors.debug("Cleared vectorizer manager cache.")