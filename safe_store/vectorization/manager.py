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

    def __init__(self, cache_folder: Optional[str] = None, custom_vectorizers_path: Optional[str] = None):
        pm.ensure_packages(["PyYAML"])
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


    @staticmethod
    def _create_vectorizer_ascii_infos(vectorizer_name: str, config: Optional[Dict[str, Any]]) -> str:
        lines = []

        lines.append(ASCIIColors.bold(ASCIIColors.magenta("════════════════════════════════════════"), emit=False))
        lines.append(ASCIIColors.bold(ASCIIColors.magenta("  VECTORISER INFORMATION"), emit=False))
        lines.append(ASCIIColors.bold(ASCIIColors.magenta("════════════════════════════════════════", emit=False), emit=False))
        lines.append("")

        lines.append(
            f"{ASCIIColors.cyan('Name')} : "
            f"{ASCIIColors.bold(ASCIIColors.green(vectorizer_name, emit=False),emit=False)}"
        )

        if config:
            lines.append("")
            lines.append(ASCIIColors.yellow("Configuration:"))
            lines.append(ASCIIColors.orange("──────────────",emit=False))

            pretty_config = json.dumps(config, indent=2, sort_keys=True)
            for line in pretty_config.splitlines():
                lines.append(ASCIIColors.blue(line,emit=False))
        else:
            lines.append("")
            lines.append(ASCIIColors.red("No configuration provided.",emit=False))

        lines.append("")
        lines.append(ASCIIColors.bold(ASCIIColors.magenta("════════════════════════════════════════",emit=False),emit=False))

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
        # Fix: Add an alias for 'st' to point to the correct folder.
        # Note: The folder 'sentense_transformer' has a typo and should ideally be 'sentence_transformer'.
        if vectorizer_name == "st":
            vectorizer_name = "sentense_transformer"

        unique_name = self._create_unique_name(vectorizer_name, vectorizer_config)

        if unique_name in self._cache:
            return self._cache[unique_name]

        ASCIIColors.info(f"Initializing vectorizer:\n{self._create_vectorizer_ascii_infos(vectorizer_name, vectorizer_config)}")
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
            raise ConfigurationError(f"Could not find or load vectorizer module for '{vectorizer_name}'.") from e
        except Exception as e:
            raise VectorizationError(f"Failed to initialize '{vectorizer_name}' vectorizer: {e}") from e

        self._cache[unique_name] = vectorizer_instance
        return vectorizer_instance

    def clear_cache(self) -> None:
        self._cache.clear()