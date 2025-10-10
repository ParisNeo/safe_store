# safe_store/vectorization/manager.py
import json
from typing import Tuple, Optional, Dict, Any
import numpy as np
import importlib
from pathlib import Path

from ..core.exceptions import ConfigurationError, VectorizationError
from .base import BaseVectorizer
from ascii_colors import ASCIIColors

class VectorizationManager:
    """
    Manages and creates vectorizer instances.

    Handles loading, caching, and retrieving vectorizer instances based on
    their name and configuration. Dynamically loads vectorizer implementations
    from the 'methods' subdirectory. This class is stateless regarding the database.
    """

    def __init__(self, cache_folder: Optional[str] = None):
        if cache_folder:
            self.cache_folder = Path(cache_folder)
            self.cache_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_folder = None
        # The cache key is the unique name (name + config)
        self._cache: Dict[str, BaseVectorizer] = {}

    @staticmethod
    def _create_unique_name(vectorizer_name: str, config: Optional[Dict[str, Any]]) -> str:
        """Creates a unique, deterministic name from the vectorizer name and its config."""
        if not config:
            return vectorizer_name
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return f"{vectorizer_name}:{config_str}"

    def get_vectorizer(
        self,
        vectorizer_name: str,
        vectorizer_config: Optional[Dict[str, Any]],
    ) -> BaseVectorizer:
        """
        Gets or initializes a vectorizer instance.

        Checks the cache first. If not found, it instantiates a new
        vectorizer and caches it.

        Args:
            vectorizer_name: The type of the vectorizer (e.g., 'st', 'openai').
            vectorizer_config: A dictionary with the specific configuration for
                               the vectorizer (e.g., `{"model": "name"}`).

        Returns:
            The vectorizer instance.

        Raises:
            ConfigurationError: If dependencies are missing or config is invalid.
            VectorizationError: If initializing the vectorizer model fails.
        """
        unique_name = self._create_unique_name(vectorizer_name, vectorizer_config)

        if unique_name in self._cache:
            ASCIIColors.debug(f"Vectorizer '{unique_name}' found in cache.")
            return self._cache[unique_name]

        ASCIIColors.info(f"Initializing vectorizer: {unique_name}")
        
        config_for_init = vectorizer_config or {}

        try:
            module_name = f"safe_store.vectorization.methods.{vectorizer_name}"
            module = importlib.import_module(module_name)
            VectorizerClass = getattr(module, module.class_name)
            
            if not issubclass(VectorizerClass, BaseVectorizer):
                raise ConfigurationError(f"Class '{module.class_name}' from '{module_name}' does not inherit from BaseVectorizer.")

            vectorizer_instance = VectorizerClass(model_config=config_for_init, cache_folder=self.cache_folder)

        except ImportError as e:
            dep_map = {"st": "safe_store[sentence-transformers]", "tfidf": "safe_store[tfidf]", "ollama": "safe_store[ollama]", "openai": "safe_store[openai]", "lollms": "safe_store[openai]"}
            install_cmd = dep_map.get(vectorizer_name, f"the required library for '{vectorizer_name}'")
            msg = f"Missing dependency for '{vectorizer_name}' vectorizer. Please install with: pip install \"{install_cmd}\". Original error: {e}"
            raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize '{vectorizer_name}' vectorizer with config {config_for_init}: {e}"
            raise VectorizationError(msg) from e

        self._cache[unique_name] = vectorizer_instance
        ASCIIColors.debug(f"Vectorizer '{unique_name}' ready and cached.")
        return vectorizer_instance

    def clear_cache(self) -> None:
        """Clears the entire internal vectorizer cache."""
        count = len(self._cache)
        self._cache.clear()
        if count > 0:
             ASCIIColors.debug(f"Cleared vectorizer manager cache ({count} items).")