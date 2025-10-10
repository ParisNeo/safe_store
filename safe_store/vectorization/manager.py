# safe_store/vectorization/manager.py
import json
from typing import Tuple, Optional, Dict, Any
import numpy as np
import importlib
from pathlib import Path

from ..core.exceptions import ConfigurationError, VectorizationError
from .base import BaseVectorizer
from ascii_colors import ASCIIColors
from .utils import load_vectorizer_module

class VectorizationManager:
    """
    Manages and creates vectorizer instances from built-in or custom locations.
    """

    def __init__(self, cache_folder: Optional[str] = None, custom_vectorizers_path: Optional[str] = None):
        self.cache_folder = Path(cache_folder) if cache_folder else None
        if self.cache_folder:
            self.cache_folder.mkdir(parents=True, exist_ok=True)
        
        self.custom_vectorizers_path = custom_vectorizers_path
        self._cache: Dict[str, BaseVectorizer] = {}

    @staticmethod
    def _create_unique_name(vectorizer_name: str, config: Optional[Dict[str, Any]]) -> str:
        if not config:
            return vectorizer_name
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return f"{vectorizer_name}:{config_str}"

    def get_vectorizer(
        self,
        vectorizer_name: str,
        vectorizer_config: Optional[Dict[str, Any]],
    ) -> BaseVectorizer:
        unique_name = self._create_unique_name(vectorizer_name, vectorizer_config)

        if unique_name in self._cache:
            return self._cache[unique_name]

        ASCIIColors.info(f"Initializing vectorizer: {unique_name}")
        config_for_init = vectorizer_config or {}

        try:
            # Use the dynamic loader to find the module
            module = load_vectorizer_module(vectorizer_name, self.custom_vectorizers_path)
            VectorizerClass = getattr(module, module.class_name)
            
            if not issubclass(VectorizerClass, BaseVectorizer):
                raise ConfigurationError(f"Class '{module.class_name}' does not inherit from BaseVectorizer.")

            vectorizer_instance = VectorizerClass(model_config=config_for_init, cache_folder=self.cache_folder)

        except (ImportError, FileNotFoundError) as e:
            raise ConfigurationError(f"Could not find or load vectorizer module for '{vectorizer_name}'.") from e
        except Exception as e:
            raise VectorizationError(f"Failed to initialize '{vectorizer_name}' vectorizer: {e}") from e

        self._cache[unique_name] = vectorizer_instance
        return vectorizer_instance

    def clear_cache(self) -> None:
        self._cache.clear()