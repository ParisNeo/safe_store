# safe_store/vectorization/manager.py
import sqlite3
import json
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import importlib
from pathlib import Path
import sys 

from ..core import db
from ..core.exceptions import ConfigurationError, VectorizationError, DatabaseError, SafeStoreError
from .base import BaseVectorizer
from ascii_colors import ASCIIColors

class VectorizationManager:
    """
    Manages available vectorization methods and their instances.

    Handles loading, caching, and retrieving vectorizer instances based on
    their name and configuration. Interacts with the database to store and
    retrieve method details, including fitted state for methods like TF-IDF.
    Dynamically loads vectorizer implementations from the 'methods' subdirectory.
    """

    def __init__(self, cache_folder:Optional[str] = None):
        # Cache format: unique_name -> (instance, method_id, params_from_db_at_load)
        if cache_folder:
            self.cache_folder = Path(cache_folder)
            self.cache_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_folder = None
        self._cache: Dict[str, Tuple[BaseVectorizer, int, Optional[Dict[str, Any]]]] = {}

    @staticmethod
    def _create_unique_name(vectorizer_name: str, config: Optional[Dict[str, Any]]) -> str:
        """Creates a unique, deterministic name from the vectorizer name and its config."""
        if not config:
            return vectorizer_name
        # Create a canonical representation of the config
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return f"{vectorizer_name}:{config_str}"

    def get_vectorizer(
        self,
        vectorizer_name: str,
        vectorizer_config: Optional[Dict[str, Any]],
        conn: sqlite3.Connection,
    ) -> Tuple[BaseVectorizer, int]:
        """
        Gets or initializes a vectorizer instance and its DB method ID.

        Checks the cache first. If not found, it checks the database. If not
        found in the DB, it initializes a new vectorizer, registers it in the
        DB, and caches it.

        Args:
            vectorizer_name: The type of the vectorizer (e.g., 'st', 'openai').
            vectorizer_config: A dictionary with the specific configuration for
                               the vectorizer (e.g., `{"model": "name"}`).
            conn: Active database connection.

        Returns:
            A tuple containing the vectorizer instance and its method_id.

        Raises:
            ConfigurationError: If dependencies are missing or config is invalid.
            VectorizationError: If initializing the vectorizer model fails.
            DatabaseError: If database interaction fails.
        """
        unique_name = self._create_unique_name(vectorizer_name, vectorizer_config)

        if unique_name in self._cache:
            instance, method_id, _ = self._cache[unique_name]
            ASCIIColors.debug(f"Vectorizer '{unique_name}' found in cache (method_id={method_id}).")
            return instance, method_id

        ASCIIColors.info(f"Initializing vectorizer: {unique_name}")
        
        # Use a merged config for instantiation, primarily from vectorizer_config
        config_for_init = vectorizer_config or {}

        # Check DB for existing method
        method_details = self._get_method_details_from_db(conn, unique_name)
        method_id: Optional[int] = None
        params_from_db: Optional[Dict[str, Any]] = None

        if method_details:
            method_id = method_details['method_id']
            params_str = method_details['params']
            if params_str:
                try:
                    params_from_db = json.loads(params_str)
                    # For stateful vectorizers like TF-IDF, params from DB are crucial
                    config_for_init.update(params_from_db)
                except json.JSONDecodeError:
                    ASCIIColors.warning(f"Could not decode params JSON for existing method '{unique_name}'.")
            ASCIIColors.debug(f"Found existing method '{unique_name}' (ID: {method_id}) in DB.")
        else:
            ASCIIColors.debug(f"No existing method found in DB for '{unique_name}'.")

        # Instantiate Vectorizer
        try:
            module_name = f"safe_store.vectorization.methods.{vectorizer_name}"
            module = importlib.import_module(module_name)
            VectorizerClass = getattr(module, module.class_name)
            
            if not issubclass(VectorizerClass, BaseVectorizer):
                raise ConfigurationError(f"Class '{module.class_name}' from '{module_name}' does not inherit from BaseVectorizer.")

            # Pass the config dictionary to the vectorizer's init
            vectorizer_instance = VectorizerClass(model_config=config_for_init, cache_folder=self.cache_folder)

        except ImportError as e:
            dep_map = {"st": "safe_store[sentence-transformers]", "tfidf": "safe_store[tfidf]", "ollama": "safe_store[ollama]", "openai": "safe_store[openai]", "lollms": "safe_store[openai]"}
            install_cmd = dep_map.get(vectorizer_name, f"the required library for '{vectorizer_name}'")
            msg = f"Missing dependency for '{vectorizer_name}' vectorizer. Please install with: pip install \"{install_cmd}\". Original error: {e}"
            raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize '{vectorizer_name}' vectorizer with config {config_for_init}: {e}"
            raise VectorizationError(msg) from e

        # Add/Get Method Record in DB
        if method_id is None:
            params_to_store = {}
            if hasattr(vectorizer_instance, "get_params_to_store"):
                params_to_store = vectorizer_instance.get_params_to_store()

            dim_to_store = vectorizer_instance.dim if vectorizer_instance.dim is not None else 0

            try:
                method_id = db.add_or_get_vectorization_method(
                    conn=conn,
                    name=unique_name,
                    type=vectorizer_name,
                    dim=dim_to_store,
                    dtype=np.dtype(vectorizer_instance.dtype).name,
                    params=json.dumps(params_to_store) if params_to_store else '{}'
                )
            except DatabaseError as e:
                raise DatabaseError(f"Failed to register vectorizer '{unique_name}' in database.") from e
        
        if method_id is None:
             raise SafeStoreError(f"Failed to obtain a valid method_id for '{unique_name}'.")

        # Cache Result
        self._cache[unique_name] = (vectorizer_instance, method_id, params_from_db)
        ASCIIColors.debug(f"Vectorizer '{unique_name}' ready and cached (method_id {method_id}).")
        return vectorizer_instance, method_id

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
                    'params': result[4]
                }
            return None
        except sqlite3.Error as e:
            msg = f"Database error fetching vectorization method details for '{name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

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
            ASCIIColors.debug(f"Prepared update for params (and dim if provided) for method_id {method_id}.")
            self.remove_from_cache_by_id(method_id, log_reason="param update")

        except sqlite3.Error as e:
            msg = f"Database error updating vectorization method params for ID {method_id}: {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise DatabaseError(msg) from e
        except Exception as e:
            if not isinstance(e, DatabaseError):
                msg = f"Unexpected error updating vectorization method params for ID {method_id}: {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise SafeStoreError(msg) from e
            else: # Should already be caught by sqlite3.Error or be a DatabaseError
                raise


    def remove_from_cache_by_id(self, method_id: int, log_reason: str = "removal") -> None:
        """Removes cache entries associated with a given method ID."""
        # Use list() to avoid modifying dict during iteration
        cached_names_to_remove = [name for name, (_, mid, _) in list(self._cache.items()) if mid == method_id]
        for name in cached_names_to_remove:
            if name in self._cache:
                del self._cache[name]
                ASCIIColors.debug(f"Invalidated cache for method '{name}' (ID: {method_id}) due to {log_reason}.")

    def clear_cache(self) -> None:
        """Clears the entire internal vectorizer cache."""
        count = len(self._cache)
        self._cache.clear()
        if count > 0:
             ASCIIColors.debug(f"Cleared vectorizer manager cache ({count} items).")