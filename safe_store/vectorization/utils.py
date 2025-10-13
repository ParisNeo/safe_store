# safe_store/vectorization/utils.py
import importlib.util
from pathlib import Path
from typing import Any
from ..core.exceptions import ConfigurationError

def load_vectorizer_module(vectorizer_name: str, custom_vectorizers_path: str = None) -> Any:
    """Dynamically loads a vectorizer module from built-in methods or a custom path."""
    
    # First, try loading from the custom path if provided
    if custom_vectorizers_path:
        custom_path = Path(custom_vectorizers_path) / vectorizer_name / "__init__.py"
        if custom_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(f"custom_vectorizers.{vectorizer_name}", custom_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            except Exception as e:
                 raise ConfigurationError(f"Failed to load custom vectorizer '{vectorizer_name}' from {custom_path}: {e}") from e
    
    # If not in custom, try built-in methods
    builtin_path = Path(__file__).parent / "methods" / vectorizer_name / "__init__.py"
    if builtin_path.exists():
        try:
            module_name = f"safe_store.vectorization.methods.{vectorizer_name}"
            return importlib.import_module(module_name)
        except Exception as e:
            raise ConfigurationError(f"Failed to load built-in vectorizer '{vectorizer_name}': {e}") from e
            
    raise FileNotFoundError(f"Vectorizer module '{vectorizer_name}' not found in built-in methods or custom path.")