# safe_store/vectorization/methods/st.py
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from safe_store.vectorization.base import BaseVectorizer
from safe_store.core.exceptions import ConfigurationError, VectorizationError
from safe_store.processing.tokenizers import HuggingFaceTokenizerWrapper
from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

class_name="STVectorizer"

try:
    pm.ensure_packages(["torch", "sentence-transformers"])
    # Proactively test if torchvision is present but broken; mask it out so transformers stays in text-only mode
    try:
        import torchvision
    except Exception:
        import sys
        sys.modules["torchvision"] = None

    from sentence_transformers import SentenceTransformer
except Exception as e:
    trace_exception(e)
    SentenceTransformer = None


def list_available_models(**kwargs) -> List[str]:
    """
    Returns a curated list of popular and effective Sentence Transformer models.
    This list is static as querying the Hugging Face Hub dynamically is not practical.
    """
    return [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "paraphrase-albert-small-v2",
        "LaBSE"
    ]

class STVectorizer(BaseVectorizer):
    """Vectorizes text using models from the sentence-transformers library."""

    DEFAULT_MODEL: str = "all-MiniLM-L6-v2"

    def __init__(self, model_config: Dict[str, Any], cache_folder: Optional[str] = None, **kwargs):
        super().__init__(vectorizer_name="st")

        if SentenceTransformer is None:
            raise ConfigurationError("STVectorizer requires 'sentence-transformers'. Install with: pip install safe_store[sentence-transformers]")

        self.model_name: str = model_config.get("model", self.DEFAULT_MODEL)
        if not self.model_name:
             raise ConfigurationError("STVectorizer config must include a 'model' key.")

        try:
            ASCIIColors.info(f"Loading Sentence Transformer model: {self.model_name}")
            st_kwargs = {}
            if cache_folder is not None:
                st_kwargs["cache_folder"] = cache_folder
            self.model: SentenceTransformer = SentenceTransformer(self.model_name, **st_kwargs)
            self._dim: int = self.model.get_sentence_embedding_dimension()
            self._dtype: np.dtype = np.dtype(np.float32)
            ASCIIColors.info(f"Model '{self.model_name}' loaded. Dimension: {self._dim}")
        except Exception as e:
            raise VectorizationError(f"Failed to load Sentence Transformer model '{self.model_name}': {e}") from e

    def get_tokenizer(self) -> Optional[HuggingFaceTokenizerWrapper]:
        """Returns the tokenizer from the loaded SentenceTransformer model, wrapped."""
        if hasattr(self.model, 'tokenizer'):
            return HuggingFaceTokenizerWrapper(self.model.tokenizer)
        return None

    def supports_late_chunking(self) -> bool:
        return self.model is not None

    def late_chunk_embed(self, text: str, chunk_spans: List[Tuple[int, int]]) -> np.ndarray:
        """
        Late Chunking: Encodes the full document through the transformer to obtain
        full-context token embeddings, then mean-pools tokens across each chunk boundary.
        """
        if not self.model or not chunk_spans:
            return np.empty((0, self.dim), dtype=self.dtype)

        try:
            import torch
            tokenizer = getattr(self.model, 'tokenizer', None)
            if tokenizer is None:
                chunk_texts = [text[s:e] for s, e in chunk_spans]
                return self.vectorize(chunk_texts)

            encoded = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=8192
            )

            offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
            device = self.model.device
            inputs = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                transformer_module = self.model[0]
                outputs = transformer_module.auto_model(**inputs)
                token_embeddings = outputs.last_hidden_state[0]

            vectors = []
            for start_char, end_char in chunk_spans:
                token_indices = []
                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == 0 and tok_end == 0:
                        continue
                    if tok_start < end_char and tok_end > start_char:
                        token_indices.append(tok_idx)

                if token_indices:
                    span_tokens = token_embeddings[token_indices]
                    chunk_vec = span_tokens.mean(dim=0).cpu().numpy()
                    norm = np.linalg.norm(chunk_vec)
                    if norm > 0:
                        chunk_vec = chunk_vec / norm
                else:
                    chunk_vec = self.vectorize([text[start_char:end_char]])[0]

                vectors.append(chunk_vec)

            return np.array(vectors, dtype=self.dtype)

        except Exception as e:
            ASCIIColors.warning(f"Late chunking fallback: {e}")
            chunk_texts = [text[s:e] for s, e in chunk_spans]
            return self.vectorize(chunk_texts)

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=self.dtype)
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if not isinstance(embeddings, np.ndarray):
                 raise VectorizationError("SentenceTransformer model did not return a NumPy array.")
            if embeddings.dtype != self._dtype:
                embeddings = embeddings.astype(self._dtype)
            return embeddings
        except Exception as e:
            raise VectorizationError(f"Error during sentence-transformer encoding: {e}") from e

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        """
        Returns a list of popular SentenceTransformer models.
        This is not an exhaustive list from an API but a curated selection.
        """
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased-v1",
            "all-roberta-large-v1"
        ]