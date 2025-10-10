# safe_store/vectorization/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Any

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

        The returned tokenizer should have `encode` and `decode` methods
        compatible with libraries like Hugging Face's tokenizers.

        Returns:
            A tokenizer object or None if no tokenizer is available client-side.
        """
        return None