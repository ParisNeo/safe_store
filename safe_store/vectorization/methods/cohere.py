import numpy as np
from typing import List, Optional, Dict, Any
import os

from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors

# each vectorizer must have a class name variable to be identified
class_name = "CohereVectorizer"

def list_available_models(**kwargs) -> List[str]:
    """
    Returns a static list of common Cohere embedding models.
    """
    return ["embed-english-v3.0", "embed-english-light-v3.0", "embed-multilingual-v3.0"]

# Attempt import of cohere and related error types, handle gracefully
try:
    import pipmaster as pm
    pm.ensure_packages(["cohere"]) # Ensure cohere is installed
    import cohere
    _CohereAPIError = cohere.APIError
    _CohereConnectionError = cohere.ConnectionError
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False
    cohere = None
    class _CohereAPIError(Exception): pass
    class _CohereConnectionError(Exception): pass


class CohereVectorizer(BaseVectorizer):
    """
    Vectorizes text using models from Cohere via their API.

    Requires the `cohere` Python library and a Cohere API key. The key can be
    provided in the `model_config` dictionary or via the `COHERE_API_KEY`
    environment variable.

    Attributes:
        model_name (str): The name of the Cohere model to use.
        api_key (str): The Cohere API key being used.
        client (cohere.Client): The Cohere client instance.
        input_type (str): The input type for the embedding model.
        truncate (str): The truncation strategy for the model.
    """
    DEFAULT_INPUT_TYPE = "search_document"
    DEFAULT_TRUNCATE = "END"

    def __init__(self,
                 model_config: Dict[str, Any],
                 **kwargs):
        """
        Initializes the CohereVectorizer.

        Args:
            model_config: A dictionary containing the vectorizer's configuration.
                          - "model" (str): Mandatory. The name of the model to use.
                          - "api_key" (str): Optional. Your Cohere API key. If not
                            provided, the COHERE_API_KEY environment variable is used.
                          - "input_type" (str): Optional. E.g., "search_document".
                          - "truncate" (str): Optional. E.g., "END".

        Raises:
            ConfigurationError: If 'cohere' is not installed, config is invalid,
                                or the API key is missing.
            VectorizationError: If connection to Cohere fails or the model is invalid.
        """
        super().__init__(vectorizer_name="cohere")
        if not _COHERE_AVAILABLE or cohere is None:
            raise ConfigurationError("CohereVectorizer requires the 'cohere' library. Install with: pip install safe_store[cohere]")

        if not isinstance(model_config, dict) or "model" not in model_config:
            raise ConfigurationError("Cohere vectorizer config must be a dictionary with a 'model' key.")

        self.model_name: str = model_config["model"]
        
        # API key discovery logic
        chosen_api_key: Optional[str] = model_config.get("api_key")
        if chosen_api_key:
            ASCIIColors.info("Using Cohere API key provided in vectorizer_config.")
        else:
            ASCIIColors.info("API key not in config. Checking COHERE_API_KEY environment variable.")
            chosen_api_key = os.environ.get("COHERE_API_KEY")
        
        if not chosen_api_key:
            raise ConfigurationError("Cohere API key not found. Provide it in the 'api_key' field of vectorizer_config or set the COHERE_API_KEY environment variable.")
        
        self.api_key: str = chosen_api_key

        # Get additional parameters from config
        self.input_type: str = model_config.get("input_type", self.DEFAULT_INPUT_TYPE)
        self.truncate: str = model_config.get("truncate", self.DEFAULT_TRUNCATE)
        
        # Parameter validation
        valid_input_types = ["search_document", "search_query", "classification", "clustering", "rerank"]
        if self.input_type not in valid_input_types:
             ASCIIColors.warning(f"Invalid input_type '{self.input_type}'. Defaulting to '{self.DEFAULT_INPUT_TYPE}'.")
             self.input_type = self.DEFAULT_INPUT_TYPE

        valid_truncate_types = ["NONE", "START", "END"]
        if self.truncate not in valid_truncate_types:
             ASCIIColors.warning(f"Invalid truncate type '{self.truncate}'. Defaulting to '{self.DEFAULT_TRUNCATE}'.")
             self.truncate = self.DEFAULT_TRUNCATE

        ASCIIColors.info(f"Initializing Cohere client. Model: {self.model_name}, Input Type: {self.input_type}")
        try:
            self.client: cohere.Client = cohere.Client(api_key=self.api_key)

            # Test connection and get embedding dimension
            test_prompt = "hello world"
            response = self.client.embed(
                texts=[test_prompt],
                model=self.model_name,
                input_type=self.input_type,
                truncate=self.truncate
            )

            if not hasattr(response, 'embeddings') or not response.embeddings or not response.embeddings[0]:
                raise VectorizationError(f"Cohere model '{self.model_name}' did not return valid embeddings.")
            
            embedding = response.embeddings[0]

            self._dim = len(embedding)
            if self._dim == 0:
                raise VectorizationError(f"Cohere model '{self.model_name}' returned a zero-dimension embedding.")

            self._dtype = np.dtype(np.float32)
            ASCIIColors.info(f"Cohere model '{self.model_name}' ready. Dimension: {self._dim}")

        except _CohereAPIError as e:
            msg = f"Cohere API error for model '{self.model_name}': {e}. Check API key and model name."
            if hasattr(e, 'http_status') and e.http_status in [401, 403, 404]:
                 raise ConfigurationError(msg) from e
            raise VectorizationError(msg) from e
        except _CohereConnectionError as e:
            raise VectorizationError(f"Cohere connection error for model '{self.model_name}': {e}.") from e
        except Exception as e:
            raise VectorizationError(f"Failed to initialize Cohere vectorizer '{self.model_name}': {e}") from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=self.dtype)

        embeddings_results = [None] * len(texts)
        actual_texts_to_embed: List[str] = []
        original_indices_for_api_texts: List[int] = []

        for i, text in enumerate(texts):
            stripped_text = text.strip()
            if not stripped_text:
                embeddings_results[i] = np.zeros(self.dim, dtype=self._dtype)
            else:
                actual_texts_to_embed.append(stripped_text)
                original_indices_for_api_texts.append(i)
        
        if actual_texts_to_embed:
            try:
                if len(actual_texts_to_embed) > 96:
                     ASCIIColors.warning(f"Attempting to vectorize {len(actual_texts_to_embed)} texts with Cohere, which may exceed batch limits.")

                response = self.client.embed(
                    texts=actual_texts_to_embed,
                    model=self.model_name,
                    input_type=self.input_type,
                    truncate=self.truncate
                )

                if not hasattr(response, 'embeddings') or len(response.embeddings) != len(actual_texts_to_embed):
                    raise VectorizationError("Cohere API returned a mismatched number of embeddings.")

                for i, embedding_vector in enumerate(response.embeddings):
                    original_idx = original_indices_for_api_texts[i]
                    if not isinstance(embedding_vector, list) or len(embedding_vector) != self.dim:
                        raise VectorizationError(f"Cohere model returned an invalid embedding for text at index {original_idx}.")
                    embeddings_results[original_idx] = embedding_vector

            except (_CohereAPIError, _CohereConnectionError) as e:
                raise VectorizationError(f"Cohere API error during vectorization: {e}") from e
            except Exception as e:
                raise VectorizationError(f"Unexpected error during Cohere vectorization: {e}") from e

        embeddings_array = np.array(embeddings_results, dtype=self._dtype)

        if embeddings_array.shape != (len(texts), self.dim):
             raise VectorizationError(f"Cohere vectorization resulted in unexpected shape {embeddings_array.shape}.")

        return embeddings_array

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype