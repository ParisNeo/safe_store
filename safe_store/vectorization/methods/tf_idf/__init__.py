# safe_store/vectorization/methods/tf_idf/__init__.py
import numpy as np
from typing import List, Optional, Dict, Any
import pickle
from safe_store.vectorization.base import BaseVectorizer
from safe_store.core.exceptions import ConfigurationError, VectorizationError
import pipmaster as pm

class_name = "TfIdfVectorizer"

class TfIdfVectorizer(BaseVectorizer):
    def __init__(self, model_config: Dict[str, Any], cache_folder: Optional[str] = None):
        super().__init__("tfidf")
        pm.ensure_packages(["scikit-learn"])
        from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
        
        self.vectorizer = SklearnTfidfVectorizer()
        self._fitted = False
        self._dim = None

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        self._dim = len(self.vectorizer.get_feature_names_out())

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise VectorizationError("TF-IDF vectorizer must be fitted before vectorizing.")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return np.float32

    def get_params_to_store(self) -> Dict[str, Any]:
        return {"vectorizer_pickle": pickle.dumps(self.vectorizer)}

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        """TF-IDF is a data-dependent model, not a pre-trained one. It has one 'model' type."""
        return ["tfidf"]