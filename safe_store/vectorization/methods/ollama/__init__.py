import numpy as np
from typing import List, Optional, Dict, Any
import os
import json
from ...base import BaseVectorizer
from ....core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors, trace_exception

# each vectorizer must have a class name variable to be identified
class_name="OllamaVectorizer"

# Lazy import + lazy install: `ollama` is NOT a base dependency and MUST NOT be
# installed at module-import time — `VectorizationManager.list_vectorizers()`
# imports every method module just to read `class_name`, so a top-level
# `pm.ensure_packages(["ollama"])` would trigger a `pip install ollama` merely
# to LIST the vectorizers. Installed on first real use instead.
import pipmaster as pm

_ollama_module = None
_OLLAMA_DEPS_READY = False


def _ensure_ollama_deps_installed() -> None:
    """Install the `ollama` client via pipmaster on first use. Idempotent."""
    global _OLLAMA_DEPS_READY
    if _OLLAMA_DEPS_READY:
        return
    try:
        import ollama  # noqa: F401 - probe only
        _OLLAMA_DEPS_READY = True
        return
    except (ImportError, ModuleNotFoundError):
        pass
    ASCIIColors.info(
        "OllamaVectorizer: installing 'ollama' package (first use only)..."
    )
    pm.ensure_packages(["ollama"])
    _OLLAMA_DEPS_READY = True


def _ensure_ollama_loaded():
    """Import and return the `ollama` module on first use. Idempotent."""
    global _ollama_module
    if _ollama_module is not None:
        return _ollama_module
    try:
        _ensure_ollama_deps_installed()
        import ollama as _ol
    except Exception as e:
        trace_exception(e)
        raise ConfigurationError(
            "OllamaVectorizer requires 'ollama' (auto-installed via pipmaster "
            "on first use). If this persists, run `pip install ollama`."
        ) from e
    _ollama_module = _ol
    return _ollama_module

def list_available_models(**kwargs) -> List[str]:
    """Dynamically lists models from a running Ollama server."""
    ollama = _ensure_ollama_loaded()
    try:
        response = ollama.list()
        # The structure from ollama.list() is a dict with a 'models' key
        # which is a list of dicts, each with a 'name' key.
        return [model.model for model in response.models]
    except ollama.RequestError as e:
        raise VectorizationError("Could not connect to the Ollama server. Please ensure it is running.") from e
    except Exception as e:
        raise VectorizationError(f"An unexpected error occurred while listing Ollama models: {e}") from e

class OllamaVectorizer(BaseVectorizer):
    """
    Vectorizes text using models hosted by an Ollama instance.
    Requires the `ollama` Python library to be installed.
    """

    def __init__(self,
                 model_config: Dict[str, Any],
                 **kwargs):
        """
        Initializes the OllamaVectorizer.

        Args:
            model_config: A dictionary containing the vectorizer's configuration.
                          - "model" (str): Mandatory. The name of the model to use.
                          - "host" (str): Optional. The URL of the Ollama server.
                            Defaults to http://localhost:11434 or OLLAMA_HOST env var.
                          - "api_key" (str): Optional. API key for authentication.
                            If not provided, OLLAMA_API_KEY environment variable is used.

        Raises:
            ConfigurationError: If 'ollama' is not installed or config is invalid.
            VectorizationError: If connection to Ollama fails or the model is invalid.
        """
        super().__init__(vectorizer_name="ollama")
        ollama = _ensure_ollama_loaded()

        if not isinstance(model_config, dict) or "model" not in model_config:
            raise ConfigurationError("Ollama vectorizer config must be a dictionary with a 'model' key.")

        self.model_name: str = model_config["model"]
        self.host: Optional[str] = model_config.get("host") # Let the client handle default
        
        # API key discovery logic (same pattern as OpenAI/Cohere)
        chosen_api_key: Optional[str] = model_config.get("api_key")
        if chosen_api_key:
            ASCIIColors.info("Using Ollama API key provided in vectorizer_config.")
        else:
            ASCIIColors.info("API key not in config. Checking OLLAMA_API_KEY environment variable.")
            chosen_api_key = os.environ.get("OLLAMA_API_KEY")
        
        self.api_key: Optional[str] = chosen_api_key

        ASCIIColors.info(f"Initializing Ollama client. Model: {self.model_name}, Host: {self.host or 'default'}")
        try:
            # Build headers with optional API key authentication
            headers: Optional[Dict[str, str]] = None
            if self.api_key:
                headers = {'Authorization': f'Bearer {self.api_key}'}
                ASCIIColors.info("Using API key authentication for Ollama.")
            
            self.client: ollama.Client = ollama.Client(host=self.host.strip() if self.host else None, headers=headers)

            # Test connection and get embedding dimension
            test_prompt = "hello world"
            response = self.client.embeddings(model=self.model_name, prompt=test_prompt)
            
            embedding = response.get("embedding")

            if not isinstance(embedding, list) or not embedding:
                raise VectorizationError(f"Ollama model '{self.model_name}' did not return a valid embedding.")

            self._dim = len(embedding)
            if self._dim == 0:
                raise VectorizationError(f"Ollama model '{self.model_name}' returned a zero-dimension embedding.")

            self._dtype = np.dtype(np.float32)
            ASCIIColors.info(f"Ollama model '{self.model_name}' ready. Dimension: {self._dim}")

        except json.JSONDecodeError as e:
            # Handle malformed JSON responses (e.g., proxy servers appending error messages)
            trace_exception(e)
            raise VectorizationError(
                f"Ollama server returned malformed JSON response. "
                f"This often happens when using a proxy server that appends error messages. "
                f"Original error: {e}. "
                f"Check that your Ollama server/proxy at '{self.host or 'default'}' is properly configured."
            ) from e
        except ollama.ResponseError as e:
            trace_exception(e)
            raise VectorizationError(f"Ollama API error for model '{self.model_name}': {e.error}") from e
        except ollama.RequestError as e:
            trace_exception(e)
            raise VectorizationError(f"Ollama request error connecting to host '{self.host or 'default'}': {e}") from e
        except Exception as e:
            trace_exception(e)
            raise VectorizationError(f"Failed to initialize Ollama vectorizer '{self.model_name}': {e}") from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=self.dtype)
        
        ollama = _ensure_ollama_loaded()
        embeddings_list = []
        try:
            for i, text in enumerate(texts):
                if not text.strip():
                    embeddings_list.append(np.zeros(self.dim, dtype=self._dtype))
                    continue

                response = self.client.embeddings(model=self.model_name, prompt=text)
                embedding_vector = response.get("embedding")

                if not isinstance(embedding_vector, list) or len(embedding_vector) != self.dim:
                    raise VectorizationError(f"Ollama model '{self.model_name}' returned an invalid embedding for text at index {i}.")
                embeddings_list.append(embedding_vector)

            embeddings_array = np.array(embeddings_list, dtype=self._dtype)
            
            if embeddings_array.shape != (len(texts), self.dim):
                 raise VectorizationError(f"Ollama vectorization resulted in unexpected shape {embeddings_array.shape}.")

            return embeddings_array

        except (ollama.ResponseError, ollama.RequestError) as e:
            raise VectorizationError(f"Ollama API error during vectorization: {e}") from e
        except Exception as e:
            raise VectorizationError(f"Unexpected error during Ollama vectorization: {e}") from e

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        """Lists available models from the running Ollama instance."""
        ollama = _ensure_ollama_loaded()
        try:
            response = ollama.list()
            # The structure from ollama.list() is a dict with a 'models' key
            # which is a list of dicts, each with a 'name' key.
            return [model.model for model in response.models]
        except ollama.RequestError as e:
            raise VectorizationError("Could not connect to the Ollama server. Please ensure it is running.") from e
        except Exception as e:
            raise VectorizationError(f"An unexpected error occurred while listing Ollama models: {e}") from e
