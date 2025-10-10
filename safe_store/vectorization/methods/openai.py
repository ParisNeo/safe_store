# safe_store/vectorization/methods/openai.py
import numpy as np
from typing import List, Optional, Dict, Any
import os # For accessing environment variables like OPENAI_API_KEY

from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors

# each vectorizer must have a class name variable to be identified
class_name = "OpenAIVectorizer"

# Attempt import of openai and related error types, handle gracefully
import pipmaster as pm
pm.ensure_packages(["openai"]) # Ensure openai is installed
import openai
# Specific errors from the openai library (v1.x+)
_OpenAIAPIError = openai.APIError
_OpenAIAuthenticationError = openai.AuthenticationError
_OpenAINotFoundError = openai.NotFoundError
_OpenAIRateLimitError = openai.RateLimitError
_OpenAIBadRequestError = openai.BadRequestError
_OpenAIAPIConnectionError = openai.APIConnectionError
_OPENAI_AVAILABLE = True

class OpenAIVectorizer(BaseVectorizer):
    """
    Vectorizes text using models from OpenAI via their API.

    Requires the `openai` Python library to be installed (`pip install openai`).
    An OpenAI API key is required, which can be provided via the
    `model_config` or the `OPENAI_API_KEY` environment variable.

    The `model_config` dictionary specifies the OpenAI model and connection
    parameters.
    Example:
    `{"model": "text-embedding-3-small", "api_key": "sk-...", "base_url": "..."}`

    If the API key is not provided in the config, the client will attempt to
    use the `OPENAI_API_KEY` environment variable.

    Attributes:
        model_name (str): The name of the OpenAI model to use for embeddings.
        api_key (Optional[str]): The OpenAI API key, if provided directly.
        client (openai.OpenAI): The OpenAI client instance.
    """

    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """
        Initializes the OpenAIVectorizer.

        Parses the model config, sets up the OpenAI client, and verifies the
        model by fetching a test embedding to determine its dimension.

        Args:
            model_config: A dictionary containing the vectorizer's configuration.
                          Must contain a "model" key. Can also contain "api_key"
                          and "base_url".
                          Example: `{"model": "text-embedding-ada-002"}`

        Raises:
            ConfigurationError: If the 'openai' library is not installed, if
                                the `model_config` is invalid, if the API key
                                is missing or invalid, or if the model is not found.
            VectorizationError: If connection to OpenAI API fails during the test,
                                or the model doesn't return valid embeddings.
        """
        super().__init__(
            vectorizer_name="openai"
        )
        if not _OPENAI_AVAILABLE or openai is None:
            msg = ("OpenAIVectorizer requires the 'openai' library. "
                   "Install with: pip install safe_store[openai] (or pip install openai)")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        if not isinstance(model_config, dict) or "model" not in model_config:
            msg = "OpenAI vectorizer config must be a dictionary with a 'model' key."
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        self.model_name: str = model_config["model"]
        self.api_key: Optional[str] = model_config.get("api_key")
        self.base_url: Optional[str] = model_config.get("base_url")

        if self.api_key:
            ASCIIColors.info("Using API key provided in vectorizer_config for OpenAI client.")
        else:
            ASCIIColors.info("API key not in config. OpenAI client will use OPENAI_API_KEY env var if set.")

        ASCIIColors.info(f"Initializing OpenAI client. Model: {self.model_name}")
        try:
            # Instantiate the OpenAI client
            self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

            # Test connection and get embedding dimension by sending a dummy prompt
            ASCIIColors.debug(f"Testing OpenAI model '{self.model_name}' and retrieving dimension...")
            test_prompt = "hello world"
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[test_prompt]
            )

            if not response.data or not response.data[0].embedding:
                msg = (f"OpenAI model '{self.model_name}' did not return a valid embedding list "
                       f"for test prompt. Response: {response}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)
            
            embedding = response.data[0].embedding

            if not isinstance(embedding, list) or not embedding:
                msg = (f"OpenAI model '{self.model_name}' returned an invalid embedding structure "
                       f"for test prompt. Embedding: {embedding}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dim = len(embedding)
            if self._dim == 0:
                msg = f"OpenAI model '{self.model_name}' returned a zero-dimension embedding."
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dtype = np.dtype(np.float32)
            ASCIIColors.info(f"OpenAI model '{self.model_name}' ready. Dimension: {self._dim}, Dtype: {self._dtype.name}")

        except _OpenAIAuthenticationError as e:
            msg = (f"OpenAI API authentication error for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}. "
                   "Please check your API key (from vectorizer_config or OPENAI_API_KEY env var).")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg) from e
        except _OpenAINotFoundError as e:
            msg = (f"OpenAI model '{self.model_name}' not found: "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}. "
                   "Ensure the model name is correct and available to your API key.")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg) from e
        except (_OpenAIBadRequestError, _OpenAIRateLimitError, _OpenAIAPIConnectionError, _OpenAIAPIError) as e:
            msg = (f"OpenAI API error for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}.")
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize OpenAI vectorizer or test model '{self.model_name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts using the configured OpenAI model.

        Args:
            texts: A list of text strings to vectorize. Empty or whitespace-only
                   strings will result in zero vectors of the model's dimension.

        Returns:
            A 2D NumPy array of shape (len(texts), self.dim) containing the
            vector embeddings, with dtype self.dtype.

        Raises:
            VectorizationError: If the embedding process fails for any text due to API
                                errors, rate limits, or unexpected responses.
        """
        if not texts:
            ASCIIColors.debug("Received empty list for OpenAI vectorization, returning empty array.")
            return np.empty((0, self.dim), dtype=self.dtype)

        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using OpenAI model '{self.model_name}'...")
        
        # Prepare lists for batching, handling empty/whitespace strings
        embeddings_results = [None] * len(texts)
        actual_texts_to_embed: List[str] = []
        original_indices_for_api_texts: List[int] = []

        for i, text in enumerate(texts):
            if not text.strip():
                ASCIIColors.warning(f"Text at index {i} is empty or whitespace. Generating zero vector of dim {self.dim}.")
                embeddings_results[i] = np.zeros(self.dim, dtype=self._dtype)
            else:
                actual_texts_to_embed.append(text)
                original_indices_for_api_texts.append(i)
        
        # Make API call only if there are non-empty texts
        if actual_texts_to_embed:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=actual_texts_to_embed
                )

                if len(response.data) != len(actual_texts_to_embed):
                    msg = (f"OpenAI API returned {len(response.data)} embeddings for {len(actual_texts_to_embed)} inputs. "
                           "This indicates a mismatch.")
                    ASCIIColors.error(msg)
                    raise VectorizationError(msg)

                for i, embedding_data in enumerate(response.data):
                    original_idx = original_indices_for_api_texts[i]
                    
                    if embedding_data.index != i :
                         ASCIIColors.warning(f"OpenAI embedding index mismatch: expected {i}, got {embedding_data.index}. Relying on list order.")

                    embedding_vector = embedding_data.embedding
                    if not isinstance(embedding_vector, list) or len(embedding_vector) != self.dim:
                        failed_text_preview = actual_texts_to_embed[i][:50] + '...'
                        msg = (f"OpenAI model '{self.model_name}' returned an invalid embedding for text at original index {original_idx} "
                               f"(API batch index {i}). Expected dimension {self.dim}, "
                               f"got {len(embedding_vector) if isinstance(embedding_vector, list) else 'None/Invalid'}. Text: '{failed_text_preview}'")
                        ASCIIColors.error(msg)
                        raise VectorizationError(msg)
                    embeddings_results[original_idx] = embedding_vector

            except (_OpenAIBadRequestError, _OpenAIRateLimitError, _OpenAIAPIConnectionError, _OpenAIAPIError) as e:
                err_type = e.__class__.__name__
                status_info = (f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - "
                               f"{e.code if hasattr(e, 'code') else 'N/A'} - {e.message if hasattr(e, 'message') else str(e)}")
                
                failed_text_preview = "N/A (batch operation)"
                msg = (f"OpenAI {err_type} during vectorization with model '{self.model_name}' "
                       f"(problematic text hint: '{failed_text_preview}'): {status_info}")
                ASCIIColors.error(msg, exc_info=True)
                raise VectorizationError(msg) from e
            except Exception as e:
                failed_text_preview = "N/A (batch operation)"
                msg = f"Unexpected error during OpenAI vectorization with '{self.model_name}' (text hint: '{failed_text_preview}'): {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise VectorizationError(msg) from e

        embeddings_array = np.array(embeddings_results, dtype=self._dtype)

        if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(texts) or embeddings_array.shape[1] != self.dim:
             msg = (f"OpenAI vectorization resulted in unexpected shape {embeddings_array.shape}. "
                    f"Expected ({len(texts)}, {self.dim}). This indicates an internal logic error.")
             ASCIIColors.error(msg)
             raise VectorizationError(msg)

        ASCIIColors.debug(f"OpenAI vectorization complete. Output shape: {embeddings_array.shape}")
        return embeddings_array

    @property
    def dim(self) -> int:
        """Returns the dimension of the vectors produced by this vectorizer."""
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        """Returns the numpy dtype of the vectors (typically np.float32)."""
        return self._dtype