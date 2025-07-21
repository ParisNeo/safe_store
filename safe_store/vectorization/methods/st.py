# safe_store/vectorization/methods/sentence_transformer.py
import numpy as np
from typing import List, Optional, Dict, Any # Added Optional
from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError # Import custom exceptions
from ascii_colors import ASCIIColors, trace_exception

# each vectorizer must have a class name variable to be identified
class_name="STVectorizer"

# Attempt import, handle gracefully
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None # Set to None if import fails


class STVectorizer(BaseVectorizer):
    """
    Vectorizes text using models from the sentence-transformers library.

    Requires `sentence-transformers` to be installed (`pip install safe_store[sentence-transformers]`).

    Attributes:
        model_name (str): The name of the sentence-transformer model being used.
        model (SentenceTransformer): The loaded sentence-transformer model instance.
    """

    DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

    def __init__(self, model_identifier_string: Optional[str] = None, params: Optional[Dict[str, Any]] = None, cache_folder: Optional[str] = None, **kwargs):
        """
        Initializes the SentenceTransformerVectorizer.

        Loads the specified sentence-transformer model.

        Args:
            model_identifier_string: The name of the model to load from the
                        sentence-transformers library (e.g., 'all-MiniLM-L6-v2').
                        Defaults to `SentenceTransformerVectorizer.DEFAULT_MODEL`.

        Raises:
            ConfigurationError: If the 'sentence-transformers' library is not installed.
            VectorizationError: If the specified model cannot be loaded.
        """
        super().__init__(
            vectorizer_name = "sentence_transformer"
        )

        if not _SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            msg = "SentenceTransformerVectorizer requires 'sentence-transformers' library. Install with: pip install safe_store[sentence-transformers]"
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        self.model_name: str = model_identifier_string or self.DEFAULT_MODEL
        ASCIIColors.info(f"Loading Sentence Transformer model: {self.model_name}")
        try:
            # Instantiate the model
            self.model: SentenceTransformer = SentenceTransformer(self.model_name, cache_folder=cache_folder)

            # Get dimension and dtype AFTER successful loading
            self._dim: int = self.model.get_sentence_embedding_dimension()
            # Sentence Transformers typically output float32
            self._dtype: np.dtype = np.dtype(np.float32)
            ASCIIColors.info(f"Model '{self.model_name}' loaded. Dimension: {self._dim}, Dtype: {self._dtype.name}")

        except Exception as e:
            # Catch errors during model loading (e.g., model not found, network issue)
            msg = f"Failed to load Sentence Transformer model '{self.model_name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            # Wrap in VectorizationError as it's a model loading issue
            raise VectorizationError(msg) from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts using the loaded model.

        Args:
            texts: A list of text strings to vectorize.

        Returns:
            A 2D NumPy array of shape (len(texts), self.dim) containing the
            vector embeddings, with dtype self.dtype.

        Raises:
            VectorizationError: If the encoding process fails.
        """
        if not texts:
            ASCIIColors.debug("Received empty list for vectorization, returning empty array.")
            # Return empty array with correct shape (0, dim) and dtype
            return np.empty((0, self.dim), dtype=self.dtype)

        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using '{self.model_name}'...")
        try:
            # Perform encoding
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False # Keep progress bar off for library use
            )
            # Ensure the output is a numpy array (should be by default)
            if not isinstance(embeddings, np.ndarray):
                 # This shouldn't happen with convert_to_numpy=True, but check defensively
                 msg = f"SentenceTransformer model '{self.model_name}' did not return a NumPy array."
                 ASCIIColors.error(msg)
                 raise VectorizationError(msg)

            # Ensure the output has the expected dtype (usually float32)
            if embeddings.dtype != self._dtype:
                ASCIIColors.warning(f"SentenceTransformer output dtype ({embeddings.dtype}) differs from expected ({self._dtype}). Casting...")
                try:
                     embeddings = embeddings.astype(self._dtype)
                except Exception as cast_err:
                     msg = f"Failed to cast SentenceTransformer output to {self._dtype}: {cast_err}"
                     ASCIIColors.error(msg)
                     raise VectorizationError(msg) from cast_err

            # Verify shape
            if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
                 msg = f"SentenceTransformer output shape ({embeddings.shape}) is not compatible with expected dimension ({self.dim})."
                 ASCIIColors.error(msg)
                 raise VectorizationError(msg)

            ASCIIColors.debug(f"Vectorization complete. Output shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            trace_exception(e)
            # Catch any unexpected errors during the encode process
            msg = f"Error during sentence-transformer encoding with '{self.model_name}': {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

    @property
    def dim(self) -> int:
        """Returns the dimension of the vectors produced by this vectorizer."""
        # Dim is guaranteed to be set in __init__ if successful
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        """Returns the numpy dtype of the vectors (np.float32)."""
        return self._dtype
