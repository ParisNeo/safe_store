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
    their names (e.g., 'st:model-name', 'tfidf:method-name'). Interacts with the
    database to store and retrieve method details, including fitted state for
    methods like TF-IDF.
    Dynamically loads vectorizer implementations from the 'methods' subdirectory.
    """

    def __init__(self, cache_folder:Optional[str] = None):
        # Cache format: name -> (instance, method_id, params_from_db_at_load)
        
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Tuple[BaseVectorizer, int, Optional[Dict[str, Any]]]] = {}

    @staticmethod
    def _to_pascal_case(text: str) -> str:
        """Converts snake_case or kebab-case text to PascalCase."""
        return text.replace('_', ' ').replace('-', ' ').title().replace(' ', '')

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
                  'st:all-MiniLM-L6-v2', 'tfidf:ngram1-max5k', 'ollama:localhost:11434::llama2').
            conn: Active database connection.
            initial_params: Optional dictionary of parameters, primarily used
                            for initializing stateful vectorizers (like TF-IDF)
                            with specific configurations if it's being created
                            for the first time via this call.

        Returns:
            A tuple containing the (potentially cached) vectorizer instance
            and its method_id from the database.

        Raises:
            ConfigurationError: If required dependencies are missing, the name
                                format is unknown, or module/class loading fails.
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

        # --- 0. Parse Name ---
        if ":" not in name:
            raise ConfigurationError(
                f"Invalid vectorizer name format: '{name}'. Must be 'type_key:identifier_string'."
            )
        parsed_method_type_key, model_identifier_string = name.split(":", 1)

        # --- 1. Check DB for Existing Method ---
        method_details = self._get_method_details_from_db(conn, name)
        vectorizer_instance: Optional[BaseVectorizer] = None
        method_id: Optional[int] = None
        params_from_db: Optional[Dict[str, Any]] = None
        db_method_type: Optional[str] = None

        if method_details:
            method_id = method_details['method_id']
            db_method_type = method_details['method_type']
            params_str = method_details['params']

            if db_method_type != parsed_method_type_key:
                ASCIIColors.warning(
                    f"Method type key from name prefix ('{parsed_method_type_key}') "
                    f"differs from DB method_type ('{db_method_type}') for '{name}'. "
                    f"Using '{parsed_method_type_key}' for loading vectorizer module."
                )

            if params_str:
                try:
                    params_from_db = json.loads(params_str)
                except json.JSONDecodeError:
                    ASCIIColors.warning(f"Could not decode params JSON for existing method '{name}' (ID: {method_id}). Proceeding without loaded params.")
            ASCIIColors.debug(f"Found existing method '{name}' (ID: {method_id}) in DB. Type from DB: {db_method_type}, Parsed type key: {parsed_method_type_key}")
        else:
             ASCIIColors.debug(f"No existing method found in DB for '{name}'. Will create with type key '{parsed_method_type_key}'.")

        # --- 2. Instantiate Vectorizer ---
        try:
            module_name = f"safe_store.vectorization.methods.{parsed_method_type_key}"

            try:
                module = importlib.import_module(module_name)
                
                VectorizerClass = getattr(module, module.class_name)
            except ImportError as e:
                if e.name == module_name or (hasattr(e, 'path') and e.path is not None and parsed_method_type_key in e.path): # type: ignore
                    raise ConfigurationError(f"No vectorizer module found for type '{parsed_method_type_key}' (expected at {module_name}.py). Error: {e}") from e
                else: # ImportError from within the vectorizer's module
                    raise # Let outer ImportError handler below catch this
            except AttributeError:
                raise ConfigurationError(f"Vectorizer class '{module.class_name}' not found in module '{module_name}'. Ensure the class exists and follows the naming convention '{self._to_pascal_case(parsed_method_type_key)}Vectorizer'.")

            if not issubclass(VectorizerClass, BaseVectorizer):
                raise ConfigurationError(f"Class '{module.class_name}' from '{module_name}' does not inherit from BaseVectorizer.")

            vectorizer_instance = VectorizerClass(model_identifier_string=model_identifier_string, cache_folder=self.cache_folder)

            if params_from_db: # Existing method in DB
                if hasattr(vectorizer_instance, "configure_from_db_params"):
                    vectorizer_instance.configure_from_db_params(params_from_db)
                
                if isinstance(params_from_db, dict) and params_from_db.get("fitted") and hasattr(vectorizer_instance, "load_fitted_state"):
                    vectorizer_instance.load_fitted_state(params_from_db)
            
            elif initial_params: # New vectorizer instance, with initial_params
                if hasattr(vectorizer_instance, "configure_from_initial_params"):
                    vectorizer_instance.configure_from_initial_params(initial_params)
                else:
                    ASCIIColors.warning(f"Vectorizer '{name}' of type '{parsed_method_type_key}' received initial_params but does not implement 'configure_from_initial_params'. Params ignored.")
            # If new and no initial_params, the vectorizer's __init__ handles its default setup.

        except ImportError as e:
            dep_map = {
                "st": "safe_store[sentence-transformers]",
                "tfidf": "safe_store[tfidf]",
                "ollama": "safe_store[ollama]", # Example
                "openai": "safe_store[openai]", # Example
            }
            install_cmd = dep_map.get(parsed_method_type_key, f"the required library for '{parsed_method_type_key}' (check vectorizer documentation)")
            msg = f"Missing dependency for '{parsed_method_type_key}' vectorizer '{name}'. Please install with: pip install \"{install_cmd}\". Original error: {e}"
            ASCIIColors.error(msg)
            raise ConfigurationError(msg) from e
        except ConfigurationError: # Re-raise explicitly
            raise
        except Exception as e:
            msg = f"Failed to initialize '{parsed_method_type_key}' vectorizer '{name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

        if not vectorizer_instance:
            raise SafeStoreError(f"Vectorizer instance could not be created for '{name}'. This is an unexpected internal state.")

        # --- 3. Add/Get Method Record in DB ---
        if method_id is None:
            current_params_to_store: Optional[Dict[str, Any]] = None
            if hasattr(vectorizer_instance, "get_params_to_store"):
                 current_params_to_store = vectorizer_instance.get_params_to_store()

            dim_to_store: int
            if vectorizer_instance.dim is None:
                ASCIIColors.debug(f"Vectorizer '{name}' has no dimension (dim is None) after initialization/configuration. Storing 0 for dim.")
                dim_to_store = 0
            else:
                dim_to_store = vectorizer_instance.dim

            try:
                 method_id = db.add_or_get_vectorization_method(
                     conn=conn,
                     name=name,
                     type=parsed_method_type_key,
                     dim=dim_to_store,
                     dtype=np.dtype(vectorizer_instance.dtype).name,
                     params=json.dumps(current_params_to_store) if current_params_to_store else None
                 )
                 ASCIIColors.debug(f"Vectorizer '{name}' (type '{parsed_method_type_key}') registered/retrieved in DB with method_id {method_id}.")
            except DatabaseError as e:
                 if "UNIQUE constraint failed" in str(e):
                      ASCIIColors.warning(f"Race condition? Method '{name}' registered by another process. Fetching existing ID.")
                      conn.rollback()
                      method_details_rc = self._get_method_details_from_db(conn, name)
                      if method_details_rc:
                           method_id = method_details_rc['method_id']
                           rc_params_str = method_details_rc['params']
                           if rc_params_str:
                                try: params_from_db = json.loads(rc_params_str)
                                except json.JSONDecodeError: pass
                      else:
                           msg = f"UNIQUE constraint failed for '{name}', but cannot fetch existing ID afterwards. Critical error."
                           ASCIIColors.critical(msg)
                           raise DatabaseError(msg) from e
                 else:
                      raise
        else:
             ASCIIColors.debug(f"Using existing method_id {method_id} for '{name}'.")

        # --- 4. Cache Result ---
        if method_id is None:
             raise SafeStoreError(f"Failed to obtain a valid method_id for '{name}'.")

        self._cache[name] = (vectorizer_instance, method_id, params_from_db)
        ASCIIColors.debug(f"Vectorizer '{name}' ready and cached (method_id {method_id}).")
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
