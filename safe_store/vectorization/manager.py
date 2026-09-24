# safe_store/vectorization/manager.py
import json
import yaml
import pipmaster as pm
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

from ..core.exceptions import ConfigurationError, VectorizationError
from .base import BaseVectorizer
from ascii_colors import ASCIIColors
from .utils import load_vectorizer_module

class VectorizationManager:
    """
    Manages and creates vectorizer instances from built-in or custom locations.
    Also provides methods to discover available vectorizers and their configurations.
    """

    RUNTIME_TRANSPORT_KEYS = {
        "use_shared_server",
        "shared_server",
        "shared_mode",
        "shared",
        "port",
        "host",
        "idle_timeout",
        "batch_window",
        "max_batch_size",
        "stream_server_logs",
        "reuse_model_in_process",
        "cache_folder",
        "api_key",
        "service_key",
        "verify_ssl_certificate"
    }

    def __init__(self, cache_folder: Optional[str] = None, custom_vectorizers_path: Optional[str] = None):
        pm.ensure_packages(["PyYAML"])
        self.cache_folder = Path(cache_folder) if cache_folder else None
        if self.cache_folder:
            self.cache_folder.mkdir(parents=True, exist_ok=True)

        self.custom_vectorizers_path = custom_vectorizers_path
        self._cache: Dict[str, BaseVectorizer] = {}

    @classmethod
    def _filter_semantic_config(cls, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Strips deployment/transport keys and normalizes equivalent model parameters."""
        if not config:
            return {}
        clean = {k: v for k, v in config.items() if k not in cls.RUNTIME_TRANSPORT_KEYS}
        # Normalize model / model_name equivalence across configs
        model_val = clean.pop("model_name", None) or clean.pop("model", None)
        if model_val is not None:
            clean["model"] = str(model_val)
        return clean

    @classmethod
    def _create_unique_name(cls, vectorizer_name: str, config: Optional[Dict[str, Any]]) -> str:
        # Normalize vectorizer name aliases
        norm_name = "sentense_transformer" if vectorizer_name == "st" else ("tf_idf" if vectorizer_name == "tfidf" else vectorizer_name)
        clean_config = cls._filter_semantic_config(config)
        if not clean_config:
            return norm_name
        config_str = json.dumps(clean_config, sort_keys=True, separators=(',', ':'))
        return f"{norm_name}:{config_str}"


    @staticmethod
    def _create_vectorizer_ascii_infos(vectorizer_name: str, config: Optional[Dict[str, Any]]) -> str:
        lines = []

        lines.append(
            f"{'[bold][white]Name[/bold][/white]'} : "
            f"{vectorizer_name}"
        )

        if config:
            lines.append("")
            lines.append("[bold]Configuration:[/bold]")
            lines.append("[yellow]──────────────[/yellow]")

            pretty_config = json.dumps(config, indent=2, sort_keys=True)
            for line in pretty_config.splitlines():
                lines.append(line)
        else:
            lines.append("No configuration provided.")

        return "\n".join(lines)

    def list_vectorizers(self) -> List[Dict[str, Any]]:
        """Scans for available vectorizers and returns their metadata from description.yaml."""
        vectorizers = []
        
        # Scan built-in methods directory
        methods_path = Path(__file__).parent / "methods"
        for p in methods_path.iterdir():
            if p.is_dir() and (p / "description.yaml").exists():
                with open(p / "description.yaml", 'r', encoding='utf-8') as f:
                    try:
                        data = yaml.safe_load(f)
                        data['name'] = p.name  # Add the folder name as the identifier
                        vectorizers.append(data)
                    except yaml.YAMLError:
                        ASCIIColors.warning(f"Could not parse description.yaml for vectorizer '{p.name}'")
        
        # Scan custom path if provided
        if self.custom_vectorizers_path:
            custom_path = Path(self.custom_vectorizers_path)
            if custom_path.is_dir():
                 for p in custom_path.iterdir():
                    if p.is_dir() and (p / "description.yaml").exists():
                        with open(p / "description.yaml", 'r', encoding='utf-8') as f:
                            try:
                                data = yaml.safe_load(f)
                                data['name'] = p.name
                                data['is_custom'] = True
                                vectorizers.append(data)
                            except yaml.YAMLError:
                                ASCIIColors.warning(f"Could not parse description.yaml for custom vectorizer '{p.name}'")

        return vectorizers

    def get_vectorizer(
        self,
        vectorizer_name: str,
        vectorizer_config: Optional[Dict[str, Any]],
    ) -> BaseVectorizer:
        # Fix: Add aliases for common vectorizer names to their actual folder names.
        # Note: The folder 'sentense_transformer' has a typo and should be 'sentence_transformer'.
        if vectorizer_name == "st":
            vectorizer_name = "sentense_transformer"
        # Alias 'tfidf' to 'tf_idf' to match the folder name
        elif vectorizer_name == "tfidf":
            vectorizer_name = "tf_idf"

        unique_name = self._create_unique_name(vectorizer_name, vectorizer_config)

        if unique_name in self._cache:
            return self._cache[unique_name]
        ASCIIColors.info(f"Initializing vectorizer: {vectorizer_name}")
        ASCIIColors.rich_print("Initializing vectorizer:")
        ASCIIColors.panel(f"{self._create_vectorizer_ascii_infos(vectorizer_name, vectorizer_config)}", "[bold][magenta]VECTORISER INFORMATION[/bold][/magenta]")
        config_for_init = vectorizer_config or {}

        try:
            module = load_vectorizer_module(vectorizer_name, self.custom_vectorizers_path)
            
            # The class name is now fetched from the module itself
            if not hasattr(module, 'class_name'):
                raise ConfigurationError(f"Vectorizer module '{vectorizer_name}' does not define a 'class_name' variable.")
            
            VectorizerClass = getattr(module, module.class_name)
            
            if not issubclass(VectorizerClass, BaseVectorizer):
                raise ConfigurationError(f"Class '{module.class_name}' does not inherit from BaseVectorizer.")

            vectorizer_instance = VectorizerClass(model_config=config_for_init, cache_folder=self.cache_folder)

        except (ImportError, FileNotFoundError) as e:
            raise ConfigurationError(f"Unsupported vectorizer '{vectorizer_name}': Could not find or load vectorizer module.") from e
        except Exception as e:
            raise VectorizationError(f"Failed to initialize '{vectorizer_name}' vectorizer: {e}") from e

        self._cache[unique_name] = vectorizer_instance
        return vectorizer_instance

    def clear_cache(self) -> None:
        """Closes and unloads all cached vectorizers and purges the cache."""
        for unique_name, vec in list(self._cache.items()):
            try:
                if hasattr(vec, "close"):
                    vec.close()
                elif hasattr(vec, "unload"):
                    vec.unload()
            except Exception as e:
                ASCIIColors.warning(f"Error closing vectorizer '{unique_name}': {e}")
        self._cache.clear()
        ASCIIColors.debug("Cleared vectorizer manager cache")