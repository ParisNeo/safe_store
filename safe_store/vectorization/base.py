# safe_store/vectorization/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

class BaseVectorizer(ABC):
    """
    Abstract base class for all vectorizer implementations within safe_store.

    Defines the common interface for converting text into numerical vectors.
    """

    @abstractmethod
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of text documents into a NumPy array of vector embeddings.

        Args:
            texts: A list of strings, where each string is a document or chunk.

        Returns:
            A 2D NumPy array where each row corresponds to the vector embedding
            of the input text at the same index. Shape: (len(texts), self.dim).
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> Optional[int]: # Allow dim to be None before fitting (e.g., TF-IDF)
        """
        Returns the dimension (number of features) of the vectors produced by
        this vectorizer. Can be None if the dimension is not known until fitting
        (e.g., TF-IDF).
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns the NumPy data type (e.g., np.float32, np.float64) of the
        vector embeddings produced by this vectorizer.
        """
        pass