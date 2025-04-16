import numpy as np
from typing import List
from ..base import BaseVectorizer
from ascii_colors import ASCIIColors

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    ASCIIColors.warning(
        "SentenceTransformerVectorizer requires 'sentence-transformers'. "
        "Install with: pip install safestore[sentence-transformers]"
    )
    SentenceTransformer = None # Set to None if import fails


class SentenceTransformerVectorizer(BaseVectorizer):
    """Vectorizes text using models from the sentence-transformers library."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers library is not installed.")

        self.model_name = model_name or self.DEFAULT_MODEL
        ASCIIColors.info(f"Loading Sentence Transformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
            # Sentence Transformers typically use float32
            self._dtype = np.float32
            ASCIIColors.info(f"Model {self.model_name} loaded. Dimension: {self._dim}, Dtype: {self._dtype}")
        except Exception as e:
            ASCIIColors.error(f"Failed to load Sentence Transformer model '{self.model_name}': {e}", exc_info=True)
            raise

    def vectorize(self, texts: List[str]) -> np.ndarray:
        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using {self.model_name}...")
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False) # Consider progress bar for large batches
            # Ensure the output is the expected dtype
            if embeddings.dtype != self._dtype:
                 embeddings = embeddings.astype(self._dtype)
            ASCIIColors.debug(f"Vectorization complete. Output shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            ASCIIColors.error(f"Error during sentence-transformer encoding: {e}", exc_info=True)
            raise

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
         return self._dtype