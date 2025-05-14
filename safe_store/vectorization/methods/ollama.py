# safe_store/vectorization/methods/ollama.py
import numpy as np
from typing import List, Optional, Dict, Any
from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors
# each vectorizer must have a class name variable to be identified
class_name="OllamaVectorizer"

# Attempt import of ollama and related error types, handle gracefully
try:
    import pipmaster as pm
    pm.ensure_packages(["ollama"])
    import ollama
    # Specific errors from the ollama library that we might want to catch
    _OllamaResponseError = ollama.ResponseError
    _OllamaRequestError = ollama.RequestError
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    ollama = None  # Set to None if import fails
    _OllamaResponseError = None # Define dummy exception if ollama not present
    _OllamaRequestError = None # Define dummy exception if ollama not present


class OllamaVectorizer(BaseVectorizer):
    """
    Vectorizes text using models hosted by an Ollama instance.

    Requires the `ollama` Python library to be installed (`pip install ollama`).
    An Ollama instance must be running and accessible with the specified model pulled.

    The `model_identifier_string` is used to specify the Ollama host, model name,
    and an optional API key for authenticated instances. The format is:
    `"ollama_host::model_name"` or `"ollama_host::model_name::api_key"`.
    For example: `"http://localhost:11434::nomic-embed-text"`
    or `"https://api.example.com::custom-embed-model::your-secret-key"`.

    Attributes:
        host (str): The host URL of the Ollama instance.
        model_name (str): The name of the Ollama model to use for embeddings.
        api_key (Optional[str]): The API key, if provided, for authentication.
        client (ollama.Client): The Ollama client instance.
    """

    def __init__(self, model_identifier_string: str, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the OllamaVectorizer.

        Parses the model identifier string, connects to the Ollama instance,
        and verifies the model by fetching a test embedding to determine its dimension.

        Args:
            model_identifier_string: A string identifying the Ollama host, model,
                                     and optionally an API key.
                                     Format: "model_name::[host=http://localhost:11434]::[api_key]".

        Raises:
            ConfigurationError: If the 'ollama' library is not installed, or if
                                the `model_identifier_string` is malformed.
            VectorizationError: If connection to Ollama fails, the model cannot be
                                reached, or the model doesn't return valid embeddings.
        """
        super().__init__(
            vectorizer_name = "ollama"
        )
        if not _OLLAMA_AVAILABLE or ollama is None:
            msg = ("OllamaVectorizer requires the 'ollama' library. "
                   "Install with: pip install safe_store[ollama] (or pip install ollama)")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        if not model_identifier_string or not isinstance(model_identifier_string, str):
            msg = "Ollama `model_identifier_string` must be a non-empty string."
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        parts = model_identifier_string.split('::')
        if not parts[0]:
            msg = (f"Invalid Ollama model identifier string: '{model_identifier_string}'. "
                   "Expected format 'host::model_name' or 'host::model_name::api_key'.")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        self.host: str = parts[1]  if len(parts) > 1 and parts[1] else "http://localhost:11434"
        self.model_name: str = parts[0]
        self.api_key: Optional[str] = parts[2] if len(parts) > 2 and parts[2] else None

        client_headers: Optional[Dict[str, str]] = None
        if self.api_key:
            client_headers = {'Authorization': f'Bearer {self.api_key}'}
            ASCIIColors.info(f"Using API key for Ollama client at {self.host}")

        ASCIIColors.info(f"Initializing Ollama client. Host: {self.host}, Model: {self.model_name}")
        try:
            # Instantiate the Ollama client
            self.client: ollama.Client = ollama.Client(host=self.host, headers=client_headers)

            # Test connection and get embedding dimension by sending a dummy prompt
            ASCIIColors.debug(f"Testing Ollama model '{self.model_name}' and retrieving dimension...")
            test_prompt = "hello world" # A generic, simple prompt
            response = self.client.embeddings(model=self.model_name, prompt=test_prompt)
            
            embedding = response.get("embedding")

            if not isinstance(embedding, list) or not embedding:
                msg = (f"Ollama model '{self.model_name}' at host '{self.host}' did not return a valid embedding list "
                       f"for test prompt. Response: {response}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dim = len(embedding)
            if self._dim == 0:
                msg = (f"Ollama model '{self.model_name}' at host '{self.host}' returned a zero-dimension embedding.")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            # Ollama embeddings are lists of floats; np.float32 is a common choice for storage/performance.
            self._dtype = np.dtype(np.float32)
            ASCIIColors.info(f"Ollama model '{self.model_name}' ready. Dimension: {self._dim}, Dtype: {self._dtype.name}")

        except _OllamaResponseError as e: # Catch specific Ollama API errors
            msg = (f"Ollama API error for model '{self.model_name}' at '{self.host}': "
                   f"Status {e.status_code} - {e.error if hasattr(e, 'error') else str(e)}") # e.error might not exist on all versions
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except _OllamaRequestError as e: # Catch specific Ollama client-side/connection errors
            msg = f"Ollama request error connecting to '{self.host}' for model '{self.model_name}': {e}"
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except Exception as e: # Catch other unexpected errors (e.g., network issues not caught by RequestError)
            msg = f"Failed to initialize Ollama vectorizer or test model '{self.model_name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts using the configured Ollama model.

        Args:
            texts: A list of text strings to vectorize.

        Returns:
            A 2D NumPy array of shape (len(texts), self.dim) containing the
            vector embeddings, with dtype self.dtype.

        Raises:
            VectorizationError: If the embedding process fails for any text.
        """
        if not texts:
            ASCIIColors.debug("Received empty list for Ollama vectorization, returning empty array.")
            return np.empty((0, self.dim), dtype=self.dtype)

        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using Ollama model '{self.model_name}' at '{self.host}'...")
        
        embeddings_list = []
        try:
            for i, text in enumerate(texts):
                if not text.strip(): # Handle empty or whitespace-only strings if Ollama can't
                    ASCIIColors.warning(f"Text at index {i} is empty or whitespace. Generating zero vector of dim {self.dim}.")
                    embeddings_list.append(np.zeros(self.dim, dtype=self._dtype))
                    continue

                response = self.client.embeddings(model=self.model_name, prompt=text)
                embedding_vector = response.get("embedding")

                if not isinstance(embedding_vector, list) or len(embedding_vector) != self.dim:
                    msg = (f"Ollama model '{self.model_name}' returned an invalid embedding for text at index {i}. "
                           f"Expected dimension {self.dim}, got {len(embedding_vector) if isinstance(embedding_vector, list) else 'None/Invalid'}. Text: '{text[:50]}...'")
                    ASCIIColors.error(msg)
                    # Option: raise error immediately, or collect errors and raise at end, or fill with zeros.
                    # For now, raising immediately for critical failure.
                    raise VectorizationError(msg)
                embeddings_list.append(embedding_vector)

            # Convert the list of lists into a 2D NumPy array
            embeddings_array = np.array(embeddings_list, dtype=self._dtype)

            # Verify shape (should be guaranteed by checks above and np.array behavior)
            if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(texts) or embeddings_array.shape[1] != self.dim:
                 msg = (f"Ollama vectorization resulted in unexpected shape {embeddings_array.shape}. "
                        f"Expected ({len(texts)}, {self.dim}).")
                 ASCIIColors.error(msg)
                 raise VectorizationError(msg)

            ASCIIColors.debug(f"Ollama vectorization complete. Output shape: {embeddings_array.shape}")
            return embeddings_array

        except (_OllamaResponseError, _OllamaRequestError) as e: # Catch Ollama-specific errors during batch
            failed_text_preview = text[:50] + '...' if 'text' in locals() and text else "N/A"
            err_type = "API error" if isinstance(e, _OllamaResponseError) else "Request error"
            status_info = f"Status {e.status_code} - {e.error if hasattr(e, 'error') else str(e)}" if isinstance(e, _OllamaResponseError) else str(e)
            msg = (f"Ollama {err_type} during vectorization with model '{self.model_name}' "
                   f"(text: '{failed_text_preview}'): {status_info}")
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e
        except Exception as e:
            # Catch any other unexpected errors during the embedding process
            failed_text_preview = text[:50] + '...' if 'text' in locals() and text else "N/A"
            msg = f"Unexpected error during Ollama vectorization with '{self.model_name}' (text: '{failed_text_preview}'): {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

    @property
    def dim(self) -> int:
        """Returns the dimension of the vectors produced by this vectorizer."""
        # _dim is guaranteed to be set in __init__ if successful
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        """Returns the numpy dtype of the vectors (typically np.float32)."""
        return self._dtype