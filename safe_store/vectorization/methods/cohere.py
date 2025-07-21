# safe_store/vectorization/methods/cohere.py
import numpy as np
from typing import List, Optional, Dict, Any
import os

from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors

# each vectorizer must have a class name variable to be identified
class_name = "CohereVectorizer"

# Attempt import of cohere and related error types, handle gracefully
try:
    import pipmaster as pm
    pm.ensure_packages(["cohere"]) # Ensure cohere is installed
    import cohere
    # Specific errors from the cohere library
    _CohereAPIError = cohere.CohereAPIError
    _CohereConnectionError = cohere.CohereConnectionError
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False
    cohere = None  # Set to None if import fails
    # Define dummy exceptions if cohere not present
    class _CohereAPIError(Exception): pass
    class _CohereConnectionError(Exception): pass


class CohereVectorizer(BaseVectorizer):
    """
    Vectorizes text using models from Cohere via their API.

    Requires the `cohere` Python library to be installed (`pip install cohere`).
    A Cohere API key is required.

    The API key can be provided directly in the `model_identifier_string`
    (e.g., "model_name::api_key"). **Warning:** The `model_identifier_string`
    (including the API key) is stored in a database,
    the API key will be exposed there. It is generally recommended to use
    environment variables for API keys.

    If an API key is not found in the `model_identifier_string`, the vectorizer
    will attempt to use the `COHERE_API_KEY` environment variable.

    The `model_identifier_string` format: `"model_name[::api_key]"`.
    Example: `"embed-english-v3.0"` (uses COHERE_API_KEY env var)
    or `"embed-english-v3.0::yourSecretCohereKey"` (uses key from string).

    Additional parameters for the Cohere API like `input_type` and `truncate`
    can be passed via the `params` argument during initialization.

    Attributes:
        model_name (str): The name of the Cohere model to use for embeddings.
        api_key (str): The Cohere API key being used.
        client (cohere.Client): The Cohere client instance.
        input_type (str): The input type for the Cohere embedding model.
        truncate (str): The truncation strategy for the Cohere embedding model.
    """
    DEFAULT_INPUT_TYPE = "search_document"
    DEFAULT_TRUNCATE = "END" # Alternatives: "NONE", "START"

    def __init__(self,
                 model_identifier_string: str,
                 params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initializes the CohereVectorizer.

        Args:
            model_identifier_string: A string identifying the Cohere model,
                                     and optionally an API key.
                                     Format: "model_name[::api_key]".
                                     If api_key is provided here, it will be used.
                                     Otherwise, COHERE_API_KEY env var is checked.
                                     **Warning:** Providing the API key in this string means
                                     it might be stored if this string is persisted.
            params: Optional dictionary of additional parameters for Cohere API.
                    Supported keys:
                    - "input_type" (str): E.g., "search_document", "search_query", "classification", "clustering".
                                          Defaults to "search_document".
                    - "truncate" (str): E.g., "NONE", "START", "END". Defaults to "END".

        Raises:
            ConfigurationError: If 'cohere' library not installed, `model_identifier_string` malformed,
                                API key missing/invalid, or model not found/accessible.
            VectorizationError: If connection to Cohere API fails or model returns invalid embeddings.
        """
        super().__init__(
            vectorizer_name="cohere"
        )
        if not _COHERE_AVAILABLE or cohere is None:
            msg = ("CohereVectorizer requires the 'cohere' library. "
                   "Install with: pip install safe_store[cohere] (or pip install cohere)")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        if not model_identifier_string or not isinstance(model_identifier_string, str):
            msg = "Cohere `model_identifier_string` must be a non-empty string."
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        parts = model_identifier_string.split('::')
        if not parts[0].strip():
            msg = (f"Invalid Cohere model identifier string: '{model_identifier_string}'. "
                   "Model name (first part) cannot be empty. "
                   "Expected format 'model_name' or 'model_name::api_key'.")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        self.model_name: str = parts[0].strip()
        
        _api_key_from_string: Optional[str] = None
        if len(parts) > 1 and parts[1].strip():
            _api_key_from_string = parts[1].strip()

        chosen_api_key: Optional[str] = None

        if _api_key_from_string:
            chosen_api_key = _api_key_from_string
            ASCIIColors.warning("Using Cohere API key provided directly in model_identifier_string. "
                                "Note: If this identifier string is stored, the API key will also be stored. "
                                "Consider using the COHERE_API_KEY environment variable for better security.")
        else:
            _api_key_from_env = os.environ.get("COHERE_API_KEY")
            if _api_key_from_env:
                chosen_api_key = _api_key_from_env
                ASCIIColors.info("Using COHERE_API_KEY environment variable for Cohere client.")
        
        if not chosen_api_key:
            msg = ("Cohere API key not found. Provide it in the model_identifier_string "
                   "(e.g., 'embed-english-v3.0::YOUR_KEY') or set the COHERE_API_KEY environment variable.")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)
        
        self.api_key: str = chosen_api_key

        _params = params or {}
        self.input_type: str = _params.get("input_type", self.DEFAULT_INPUT_TYPE)
        self.truncate: str = _params.get("truncate", self.DEFAULT_TRUNCATE)
        
        valid_input_types = ["search_document", "search_query", "classification", "clustering", "rerank"]
        if self.input_type not in valid_input_types:
             ASCIIColors.warning(f"Invalid input_type '{self.input_type}' provided. Using default '{self.DEFAULT_INPUT_TYPE}'. Valid types: {valid_input_types}")
             self.input_type = self.DEFAULT_INPUT_TYPE

        valid_truncate_types = ["NONE", "START", "END"]
        if self.truncate not in valid_truncate_types:
             ASCIIColors.warning(f"Invalid truncate type '{self.truncate}' provided. Using default '{self.DEFAULT_TRUNCATE}'. Valid types: {valid_truncate_types}")
             self.truncate = self.DEFAULT_TRUNCATE


        ASCIIColors.info(f"Initializing Cohere client. Model: {self.model_name}, Input Type: {self.input_type}, Truncate: {self.truncate}")
        try:
            self.client: cohere.Client = cohere.Client(api_key=self.api_key)

            ASCIIColors.debug(f"Testing Cohere model '{self.model_name}' and retrieving dimension...")
            test_prompt = "hello world"
            
            response = self.client.embed(
                texts=[test_prompt],
                model=self.model_name,
                input_type=self.input_type,
                truncate=self.truncate
            )

            if not hasattr(response, 'embeddings') or not response.embeddings or not response.embeddings[0]:
                msg = (f"Cohere model '{self.model_name}' did not return a valid embedding list "
                       f"for test prompt. Response: {response}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)
            
            embedding = response.embeddings[0]

            if not isinstance(embedding, list) or not embedding:
                msg = (f"Cohere model '{self.model_name}' returned an invalid embedding structure "
                       f"for test prompt. Embedding: {embedding}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dim = len(embedding)
            if self._dim == 0:
                msg = (f"Cohere model '{self.model_name}' returned a zero-dimension embedding.")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dtype = np.dtype(np.float32) # Cohere embeddings are float, float32 is standard.
            ASCIIColors.info(f"Cohere model '{self.model_name}' ready. Dimension: {self._dim}, Dtype: {self._dtype.name}")

        except _CohereAPIError as e:
            msg = (f"Cohere API error for model '{self.model_name}': {e}. "
                   f"HTTP Status: {e.http_status if hasattr(e, 'http_status') else 'N/A'}. "
                   "Check API key, model name, and Cohere service status.")
            ASCIIColors.error(msg)
            if hasattr(e, 'http_status'):
                if e.http_status in [401, 403]: # Unauthorized or Forbidden
                     raise ConfigurationError(msg) from e
                if e.http_status == 404: # Not Found (e.g. model name)
                     raise ConfigurationError(msg) from e
            raise VectorizationError(msg) from e
        except _CohereConnectionError as e:
            msg = f"Cohere connection error for model '{self.model_name}': {e}. Check network and Cohere API status."
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize Cohere vectorizer or test model '{self.model_name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts using the configured Cohere model.

        Args:
            texts: A list of text strings to vectorize. Empty or whitespace-only
                   strings will result in zero vectors of the model's dimension.

        Returns:
            A 2D NumPy array of shape (len(texts), self.dim) containing the
            vector embeddings, with dtype self.dtype.

        Raises:
            VectorizationError: If the embedding process fails for any text.
        """
        if not texts:
            ASCIIColors.debug("Received empty list for Cohere vectorization, returning empty array.")
            return np.empty((0, self.dim), dtype=self.dtype)

        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using Cohere model '{self.model_name}'...")
        
        embeddings_results = [None] * len(texts)
        actual_texts_to_embed: List[str] = []
        original_indices_for_api_texts: List[int] = []

        for i, text in enumerate(texts):
            stripped_text = text.strip()
            if not stripped_text:
                ASCIIColors.warning(f"Text at index {i} is empty or whitespace. Generating zero vector of dim {self.dim}.")
                embeddings_results[i] = np.zeros(self.dim, dtype=self._dtype)
            else:
                actual_texts_to_embed.append(stripped_text)
                original_indices_for_api_texts.append(i)
        
        if actual_texts_to_embed:
            try:
                # Cohere's embed endpoint can take up to 96 texts at a time (for most models).
                # This implementation will send all at once.
                # If this limit is often hit by users, batching logic should be added here.
                if len(actual_texts_to_embed) > 96:
                     ASCIIColors.warning(f"Attempting to vectorize {len(actual_texts_to_embed)} texts with Cohere, "
                                         f"which exceeds the typical batch limit of 96 for `embed` endpoint. This may fail or be slow. "
                                         f"Consider splitting input texts into smaller batches.")


                response = self.client.embed(
                    texts=actual_texts_to_embed,
                    model=self.model_name,
                    input_type=self.input_type,
                    truncate=self.truncate
                )

                if not hasattr(response, 'embeddings') or len(response.embeddings) != len(actual_texts_to_embed):
                    msg = (f"Cohere API returned {len(response.embeddings) if hasattr(response, 'embeddings') else 'N/A'} "
                           f"embeddings for {len(actual_texts_to_embed)} inputs. This indicates a mismatch.")
                    ASCIIColors.error(msg)
                    raise VectorizationError(msg)

                for i, embedding_vector in enumerate(response.embeddings):
                    original_idx = original_indices_for_api_texts[i]
                    
                    if not isinstance(embedding_vector, list) or len(embedding_vector) != self.dim:
                        failed_text_preview = actual_texts_to_embed[i][:50] + '...'
                        msg = (f"Cohere model '{self.model_name}' returned an invalid embedding for text at original index {original_idx} "
                               f"(API batch index {i}). Expected dimension {self.dim}, "
                               f"got {len(embedding_vector) if isinstance(embedding_vector, list) else 'None/Invalid'}. Text: '{failed_text_preview}'")
                        ASCIIColors.error(msg)
                        raise VectorizationError(msg)
                    embeddings_results[original_idx] = embedding_vector

            except (_CohereAPIError, _CohereConnectionError) as e:
                err_type = e.__class__.__name__
                status_info = (f"HTTP Status: {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {str(e)}")
                failed_text_preview = "N/A (batch operation)"

                msg = (f"Cohere {err_type} during vectorization with model '{self.model_name}' "
                       f"(text hint: '{failed_text_preview}'): {status_info}")
                ASCIIColors.error(msg, exc_info=True)
                raise VectorizationError(msg) from e
            except Exception as e:
                failed_text_preview = "N/A (batch operation)"
                msg = f"Unexpected error during Cohere vectorization with '{self.model_name}' (text hint: '{failed_text_preview}'): {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise VectorizationError(msg) from e

        embeddings_array = np.array(embeddings_results, dtype=self._dtype)

        if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(texts) or embeddings_array.shape[1] != self.dim:
             msg = (f"Cohere vectorization resulted in unexpected shape {embeddings_array.shape}. "
                    f"Expected ({len(texts)}, {self.dim}). This indicates an internal logic error.")
             ASCIIColors.error(msg)
             raise VectorizationError(msg)

        ASCIIColors.debug(f"Cohere vectorization complete. Output shape: {embeddings_array.shape}")
        return embeddings_array

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype
