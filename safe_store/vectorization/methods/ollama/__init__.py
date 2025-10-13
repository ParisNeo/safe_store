import numpy as np
from typing import List, Optional, Dict, Any
from ...base import BaseVectorizer
from ....core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors, trace_exception

# each vectorizer must have a class name variable to be identified
class_name="OllamaVectorizer"

# Attempt import of ollama and related error types, handle gracefully
try:
    import pipmaster as pm
    pm.ensure_packages(["ollama"])
    import ollama
    _OllamaResponseError = ollama.ResponseError
    _OllamaRequestError = ollama.RequestError
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    ollama = None
    class _OllamaResponseError(Exception): pass
    class _OllamaRequestError(Exception): pass

def list_available_models(**kwargs) -> List[str]:
    """Dynamically lists models from a running Ollama server."""
    if not _OLLAMA_AVAILABLE:
        raise ConfigurationError("Ollama support is not installed. Please run: pip install safe_store[ollama]")
    
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

        Raises:
            ConfigurationError: If 'ollama' is not installed or config is invalid.
            VectorizationError: If connection to Ollama fails or the model is invalid.
        """
        super().__init__(vectorizer_name="ollama")
        if not _OLLAMA_AVAILABLE or ollama is None:
            raise ConfigurationError("OllamaVectorizer requires the 'ollama' library. Install with: pip install safe_store[ollama]")

        if not isinstance(model_config, dict) or "model" not in model_config:
            raise ConfigurationError("Ollama vectorizer config must be a dictionary with a 'model' key.")

        self.model_name: str = model_config["model"]
        self.host: Optional[str] = model_config.get("host") # Let the client handle default

        ASCIIColors.info(f"Initializing Ollama client. Model: {self.model_name}, Host: {self.host or 'default'}")
        try:
            self.client: ollama.Client = ollama.Client(host=self.host.strip())

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

        except _OllamaResponseError as e:
            trace_exception(e)
            raise VectorizationError(f"Ollama API error for model '{self.model_name}': {e.error}") from e
        except _OllamaRequestError as e:
            trace_exception(e)
            raise VectorizationError(f"Ollama request error connecting to host '{self.host or 'default'}': {e}") from e
        except Exception as e:
            trace_exception(e)
            raise VectorizationError(f"Failed to initialize Ollama vectorizer '{self.model_name}': {e}") from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=self.dtype)
        
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

        except (_OllamaResponseError, _OllamaRequestError) as e:
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
        try:
            response = ollama.list()
            # The structure from ollama.list() is a dict with a 'models' key
            # which is a list of dicts, each with a 'name' key.
            return [model.model for model in response.models]
        except ollama.RequestError as e:
            raise VectorizationError("Could not connect to the Ollama server. Please ensure it is running.") from e
        except Exception as e:
            raise VectorizationError(f"An unexpected error occurred while listing Ollama models: {e}") from e
