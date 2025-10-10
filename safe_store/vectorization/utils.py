# safe_store/vectorization/utils.py
import importlib.util
from pathlib import Path
from types import ModuleType

def load_vectorizer_module(vectorizer_name: str, custom_path: str = None) -> ModuleType:
    """
    Dynamically loads a vectorizer module.

    It first searches in the provided custom_path, then falls back to the
    built-in methods directory.

    Args:
        vectorizer_name: The name of the vectorizer module (e.g., 'st', 'ollama').
        custom_path: The file path to a directory containing custom vectorizer modules.

    Returns:
        The loaded module.

    Raises:
        FileNotFoundError: If the module cannot be found in either location.
    """
    # 1. Try to load from custom path first
    if custom_path:
        custom_module_path = Path(custom_path) / f"{vectorizer_name}.py"
        if custom_module_path.is_file():
            spec = importlib.util.spec_from_file_location(vectorizer_name, custom_module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    # 2. Fallback to built-in methods
    try:
        module_path = f"safe_store.vectorization.methods.{vectorizer_name}"
        return importlib.import_module(module_path)
    except ImportError as e:
        raise FileNotFoundError(
            f"Vectorizer module '{vectorizer_name}.py' not found in custom path "
            f"'{custom_path}' or as a built-in method."
        ) from e