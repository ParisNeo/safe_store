from abc import ABC, abstractmethod
import numpy as np
from typing import List

class BaseVectorizer(ABC):
    """Abstract base class for all vectorizers."""

    @abstractmethod
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Converts a list of texts into a numpy array of vectors."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Returns the dimension of the vectors produced by this vectorizer."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Returns the numpy dtype of the vectors (e.g., np.float32)."""
        pass