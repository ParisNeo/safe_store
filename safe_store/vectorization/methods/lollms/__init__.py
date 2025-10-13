# safe_store/vectorization/methods/lollms.py
import numpy as np
from typing import List, Optional, Dict, Any
import os

from ...base import BaseVectorizer
from ....core.exceptions import ConfigurationError, VectorizationError
from ascii_colors import ASCIIColors

class_name = "LollmsVectorizer"

import pipmaster as pm
pm.ensure_packages(["openai"])
import openai

_OpenAIAPIError = openai.APIError
_OpenAIAuthenticationError = openai.AuthenticationError
_OpenAINotFoundError = openai.NotFoundError
_OpenAIRateLimitError = openai.RateLimitError
_OpenAIBadRequestError = openai.BadRequestError
_OpenAIAPIConnectionError = openai.APIConnectionError
_OPENAI_AVAILABLE = True

def list_available_models(**kwargs) -> List[str]:
    """
    Dynamically lists models from a running Lollms (OpenAI-compatible) server.
    """
    if not _OPENAI_AVAILABLE:
        raise ConfigurationError("Lollms support requires 'openai'. Please run: pip install safe_store[openai]")
    
    base_url = kwargs.get("base_url", "http://localhost:9600")
    api_key = kwargs.get("api_key", "not_needed")
    
    try:
        client = openai.Client(base_url=base_url, api_key=api_key)
        models = client.models.list()
        # The response is an object with a 'data' attribute which is a list of model objects
        return [model.id for model in models.data]
    except openai.APIConnectionError as e:
        raise VectorizationError(f"Could not connect to Lollms server at '{base_url}'. Please ensure it is running.") from e
    except Exception as e:
        raise VectorizationError(f"An unexpected error occurred while listing Lollms models: {e}") from e

class LollmsVectorizer(BaseVectorizer):
    """
    Vectorizes text using an OpenAI-compatible API, such as a local Lollms instance.

    Requires the `openai` Python library. The `model_config` dictionary specifies
    the model name and connection parameters.
    Example:
    `{"model": "nomic-embed-text", "base_url": "http://localhost:9600", "api_key": "..."}`

    Attributes:
        model_name (str): The name of the model to use for embeddings.
        api_key (Optional[str]): The API key for the service.
        base_url (Optional[str]): The base URL of the OpenAI-compatible API endpoint.
        client (openai.OpenAI): The OpenAI client instance.
    """

    def __init__(self, model_config: Dict[str, Any], **kwargs):
        """
        Initializes the LollmsVectorizer.

        Args:
            model_config: A dictionary with configuration. Must contain "model".
                          Can also contain "api_key" and "base_url".

        Raises:
            ConfigurationError: If 'openai' is not installed or config is invalid.
            VectorizationError: If connection or model test fails.
        """
        super().__init__(
            vectorizer_name="lollms"
        )
        if not _OPENAI_AVAILABLE or openai is None:
            msg = "LollmsVectorizer requires the 'openai' library. Install with: pip install safe_store[openai]"
            raise ConfigurationError(msg)

        if not isinstance(model_config, dict) or "model" not in model_config:
            msg = "Lollms vectorizer config must be a dictionary with a 'model' key."
            raise ConfigurationError(msg)

        self.model_name: str = model_config["model"]
        self.api_key: Optional[str] = model_config.get("api_key", "not_needed")
        self.base_url: Optional[str] = model_config.get("base_url")

        ASCIIColors.info(f"Initializing Lollms (OpenAI-compatible) client. Model: {self.model_name}, Base URL: {self.base_url}")
        try:
            self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

            ASCIIColors.debug(f"Testing model '{self.model_name}' and retrieving dimension...")
            test_prompt = "hello world"
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[test_prompt]
            )

            if not response.data or not response.data[0].embedding:
                raise VectorizationError(f"Model '{self.model_name}' did not return a valid embedding. Response: {response}")
            
            embedding = response.data[0].embedding

            if not isinstance(embedding, list) or not embedding:
                raise VectorizationError(f"Model '{self.model_name}' returned an invalid embedding structure. Embedding: {embedding}")

            self._dim = len(embedding)
            if self._dim == 0:
                raise VectorizationError(f"Model '{self.model_name}' returned a zero-dimension embedding.")

            self._dtype = np.dtype(np.float32)
            ASCIIColors.info(f"Model '{self.model_name}' ready. Dimension: {self._dim}, Dtype: {self._dtype.name}")

        except (_OpenAIAuthenticationError, _OpenAINotFoundError, _OpenAIBadRequestError, _OpenAIRateLimitError, _OpenAIAPIConnectionError, _OpenAIAPIError) as e:
            msg = (f"API error for model '{self.model_name}': "
                   f"HTTP {e.http_status if hasattr(e, 'http_status') else 'N/A'} - {e.code if hasattr(e, 'code') else 'N/A'} - {e.message}.")
            raise VectorizationError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize Lollms vectorizer or test model '{self.model_name}': {e}"
            raise VectorizationError(msg) from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts.

        Args:
            texts: A list of text strings to vectorize.

        Returns:
            A 2D NumPy array of embeddings.

        Raises:
            VectorizationError: If the embedding process fails.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=self.dtype)

        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using Lollms model '{self.model_name}'...")
        
        embeddings_results = [None] * len(texts)
        actual_texts_to_embed: List[str] = []
        original_indices_for_api_texts: List[int] = []

        for i, text in enumerate(texts):
            if not text.strip():
                embeddings_results[i] = np.zeros(self.dim, dtype=self._dtype)
            else:
                actual_texts_to_embed.append(text)
                original_indices_for_api_texts.append(i)
        
        if actual_texts_to_embed:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=actual_texts_to_embed
                )

                if len(response.data) != len(actual_texts_to_embed):
                    raise VectorizationError(f"API returned {len(response.data)} embeddings for {len(actual_texts_to_embed)} inputs.")

                for i, embedding_data in enumerate(response.data):
                    original_idx = original_indices_for_api_texts[i]
                    embedding_vector = embedding_data.embedding
                    if not isinstance(embedding_vector, list) or len(embedding_vector) != self.dim:
                        raise VectorizationError(f"Model '{self.model_name}' returned an invalid embedding for text at original index {original_idx}.")
                    embeddings_results[original_idx] = embedding_vector

            except (_OpenAIBadRequestError, _OpenAIRateLimitError, _OpenAIAPIConnectionError, _OpenAIAPIError) as e:
                msg = f"Lollms API error during vectorization: {e.message if hasattr(e, 'message') else str(e)}"
                raise VectorizationError(msg) from e
            except Exception as e:
                msg = f"Unexpected error during Lollms vectorization: {e}"
                raise VectorizationError(msg) from e

        embeddings_array = np.array(embeddings_results, dtype=self._dtype)

        if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(texts) or embeddings_array.shape[1] != self.dim:
             raise VectorizationError(f"Vectorization resulted in unexpected shape {embeddings_array.shape}. Expected ({len(texts)}, {self.dim}).")

        ASCIIColors.debug(f"Lollms vectorization complete. Output shape: {embeddings_array.shape}")
        return embeddings_array

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype
    
    @staticmethod
    def list_models(**kwargs) -> List[str]:
        """Listing models is dependent on the lollms binding and not exposed via client yet."""
        return []