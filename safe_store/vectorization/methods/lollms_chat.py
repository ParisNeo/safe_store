# safe_store/vectorization/methods/openai.py
import numpy as np
from typing import List, Optional, Dict, Any
import os # For accessing environment variables like OPENAI_API_KEY

from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors

# each vectorizer must have a class name variable to be identified
class_name = "LollmsChatVectorizer"

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

class LollmsChatVectorizer(BaseVectorizer):
    """
    Vectorizes text using models from OpenAI via their API.

    Requires the `openai` Python library to be installed (`pip install openai`).
    An OpenAI API key is required, which can be provided via the
    `model_identifier_string` or the `OPENAI_API_KEY` environment variable.

    The `model_identifier_string` is used to specify the OpenAI model name
    and an optional API key. The format is:
    `"model_name"` or `"model_name::api_key"`.
    For example: `"text-embedding-ada-002"`
    or `"text-embedding-3-small::sk-yourSecretOpenAIKey"`.

    If the API key is not provided in the `model_identifier_string`,
    the client will attempt to use the `OPENAI_API_KEY` environment variable.

    Attributes:
        model_name (str): The name of the OpenAI model to use for embeddings
                          (e.g., "text-embedding-ada-002").
        api_key (Optional[str]): The OpenAI API key, if provided directly.
        client (openai.OpenAI): The OpenAI client instance.
    """

    def __init__(self,
                 model_identifier_string: str,
                 params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initializes the LollmsChatVectorizer.

        Parses the model identifier string, sets up the OpenAI client,
        and verifies the model by fetching a test embedding to determine its dimension.

        Args:
            model_identifier_string: A string identifying the OpenAI model,
                                     and optionally an API key.
                                     Format: "model_name[::api_key][::host_address]".
                                     Example: "text-embedding-ada-002" or
                                              "text-embedding-ada-002::YOUR_API_KEY" or
                                              "text-embedding-ada-002::YOUR_API_KEY::http://localhost:8080".
            params: Optional dictionary of additional parameters. Not currently used by this vectorizer
                    but included for interface consistency.

        Raises:
            ConfigurationError: If the 'openai' library is not installed, if
                                the `model_identifier_string` is malformed,
                                if the API key is missing or invalid, or if the
                                specified model is not found.
            VectorizationError: If connection to OpenAI API fails during the test,
                                or the model doesn't return valid embeddings.
        """
        super().__init__(
            vectorizer_name="lollms_chat"
        )
        if not _OPENAI_AVAILABLE or openai is None:
            msg = ("LollmsChatVectorizer requires the 'openai' library. "
                   "Install with: pip install safe_store[openai] (or pip install openai)")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        if not model_identifier_string or not isinstance(model_identifier_string, str):
            msg = "OpenAI `model_identifier_string` must be a non-empty string."
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        parts = model_identifier_string.split('::')
        if not parts[0].strip():
            msg = (f"Invalid OpenAI model identifier string: '{model_identifier_string}'. "
                   "Model name (first part) cannot be empty. "
                   "Expected format 'model_name' or 'model_name::api_key'.")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        self.model_name: str = parts[0].strip()
        self.api_key: Optional[str] = None
        if len(parts) > 1 and parts[1].strip():
            self.api_key = parts[1].strip()
            ASCIIColors.info(f"Using API key provided in model_identifier_string for OpenAI client.")
        else:
            ASCIIColors.info(f"API key not in model_identifier_string. OpenAI client will use OPENAI_API_KEY env var if set.")

        self.base_url: Optional[str] = None
        if len(parts) > 2 and parts[2].strip():
            self.base_url = parts[2].strip()
        ASCIIColors.info(f"Initializing OpenAI client. Model: {self.model_name}")
        try:
            # Instantiate the OpenAI client
            # If self.api_key is None, OpenAI client will look for OPENAI_API_KEY env var.
            self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key,base_url=self.base_url)

            # Test connection and get embedding dimension by sending a dummy prompt
            ASCIIColors.debug(f"Testing OpenAI model '{self.model_name}' and retrieving dimension...")
            test_prompt = "hello world"  # A generic, simple prompt
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[test_prompt] # API expects a list of strings
            )

            if not response.data or not response.data[0].embedding:
                msg = (f"OpenAI model '{self.model_name}' did not return a valid embedding list "
                       f"for test prompt. Response: {response}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)
            
            embedding = response.data[0].embedding

            if not isinstance(embedding, list) or not embedding: # Should be caught by above, but defensive
                msg = (f"OpenAI model '{self.model_name}' returned an invalid embedding structure "
                       f"for test prompt. Embedding: {embedding}")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dim = len(embedding)
            if self._dim == 0:
                msg = (f"OpenAI model '{self.model_name}' returned a zero-dimension embedding.")
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            self._dtype = np.dtype(np.float32) # OpenAI embeddings are float, float32 is standard.
            ASCIIColors.info(f"OpenAI model '{self.model_name}' ready. Dimension: {self._dim}, Dtype: {self._dtype.name}")

        except _OpenAIAuthenticationError as e:
            msg = (f"OpenAI API authentication error for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}. "
                   "Please check your API key (from model_identifier_string or OPENAI_API_KEY env var).")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg) from e
        except _OpenAINotFoundError as e:
            msg = (f"OpenAI model '{self.model_name}' not found: "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}. "
                   "Ensure the model name is correct and available to your API key.")
            ASCIIColors.error(msg)
            raise ConfigurationError(msg) from e
        except _OpenAIBadRequestError as e: # e.g. if test prompt was empty, or other model config issue
             msg = (f"OpenAI API bad request during test for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}.")
             ASCIIColors.error(msg)
             raise VectorizationError(msg) from e
        except _OpenAIRateLimitError as e:
            msg = (f"OpenAI API rate limit hit during test for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}.")
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except _OpenAIAPIConnectionError as e:
            msg = f"OpenAI API connection error for model '{self.model_name}': {e}. Check network and OpenAI API status."
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except _OpenAIAPIError as e: # Catch other OpenAI API errors
            msg = (f"OpenAI API error for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}.")
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from e
        except Exception as e: # Catch other unexpected errors
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
                    # The OpenAI API documentation states that the order of embeddings in the 'data' list
                    # corresponds to the order of the input texts.
                    # The `embedding_data.index` field also confirms this.
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
                
                # Try to identify which text might have caused it, if possible from error (often not for batch)
                # For OpenAI, error messages sometimes contain `param` like `input.[ problematic_index ]`
                failed_text_preview = "N/A (batch operation)" 
                if hasattr(e, 'body') and e.body and 'param' in e.body: # type: ignore
                    param_info = e.body['param'] # type: ignore
                    if 'input.' in param_info: # type: ignore
                        try:
                            idx_str = param_info.split('.')[1].strip('[]') # type: ignore
                            problematic_api_idx = int(idx_str)
                            if 0 <= problematic_api_idx < len(actual_texts_to_embed):
                                failed_text_preview = actual_texts_to_embed[problematic_api_idx][:50] + '...'
                        except Exception:
                            pass # Keep generic preview

                msg = (f"OpenAI {err_type} during vectorization with model '{self.model_name}' "
                       f"(problematic text hint: '{failed_text_preview}'): {status_info}")
                ASCIIColors.error(msg, exc_info=True)
                raise VectorizationError(msg) from e
            except Exception as e:
                failed_text_preview = "N/A (batch operation)" # In general exceptions, hard to know
                msg = f"Unexpected error during OpenAI vectorization with '{self.model_name}' (text hint: '{failed_text_preview}'): {e}"
                ASCIIColors.error(msg, exc_info=True)
                raise VectorizationError(msg) from e

        # Convert the list of lists/np.zeros into a 2D NumPy array
        embeddings_array = np.array(embeddings_results, dtype=self._dtype)

        if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(texts) or embeddings_array.shape[1] != self.dim:
             msg = (f"OpenAI vectorization resulted in unexpected shape {embeddings_array.shape}. "
                    f"Expected ({len(texts)}, {self.dim}). This indicates an internal logic error.")
             ASCIIColors.error(msg)
             raise VectorizationError(msg) # Should not happen if logic is correct

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
