# safe_store/vectorization/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Any, Tuple

class BaseVectorizer(ABC):
    """
    Abstract base class for all vectorizer implementations within safe_store.
    """

    def __init__(self, vectorizer_name:str="unknown"):
        self.vectorizer_name = vectorizer_name

    @abstractmethod
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Converts a list of text documents into a NumPy array of vector embeddings."""
        pass

    @property
    @abstractmethod
    def dim(self) -> Optional[int]:
        """The dimension of the vectors produced by this vectorizer."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The NumPy data type of the vector embeddings."""
        pass

    def get_tokenizer(self) -> Optional[Any]:
        """
        Returns the tokenizer associated with the vectorizer, if available.
        """
        return None

    def supports_late_chunking(self) -> bool:
        """Returns True if this vectorizer supports full-document late chunking."""
        return False

    def late_chunk_embed(self, text: str, chunk_spans: List[Tuple[int, int]]) -> np.ndarray:
        """
        Embeds the full document through the model first, then pools token embeddings
        over the provided chunk character spans. Falls back to standard chunk encoding.
        """
        chunk_texts = [text[start:end] for start, end in chunk_spans]
        return self.vectorize(chunk_texts)

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        """
        Lists the available models for this vectorizer.
        """
        return []