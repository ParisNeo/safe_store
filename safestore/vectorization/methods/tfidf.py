# safestore/vectorization/methods/tfidf.py
import numpy as np
from typing import List, Optional, Dict, Any
import json
from ..base import BaseVectorizer
from ascii_colors import ASCIIColors

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import NotFittedError
except ImportError:
    ASCIIColors.warning(
        "TfidfVectorizerWrapper requires 'scikit-learn'. "
        "Install with: pip install safestore[tfidf]"
    )
    TfidfVectorizer = None
    NotFittedError = None # Define dummy exception if sklearn not present

class TfidfVectorizerWrapper(BaseVectorizer):
    """
    Wraps scikit-learn's TfidfVectorizer. Requires fitting on a corpus.
    Manages storing/loading fitted state (vocabulary, idf weights).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn library is not installed.")

        self.initial_params = params or {} # Store params passed during init
        # Extract relevant sklearn params, provide defaults if not specified
        # Use only params relevant to TfidfVectorizer constructor
        allowed_sklearn_params = {
            'input', 'encoding', 'decode_error', 'strip_accents', 'lowercase',
            'preprocessor', 'tokenizer', 'analyzer', 'stop_words', 'token_pattern',
            'ngram_range', 'max_df', 'min_df', 'max_features', 'vocabulary',
            'binary', 'dtype', 'norm', 'use_idf', 'smooth_idf', 'sublinear_tf'
        }
        tfidf_constructor_params = {
            k: v for k, v in self.initial_params.items() if k in allowed_sklearn_params
        }

        # Set defaults if not provided in params
        tfidf_constructor_params.setdefault('ngram_range', (1, 1))
        tfidf_constructor_params.setdefault('stop_words', 'english')
        # Note: max_features default is handled by sklearn

        ASCIIColors.info(f"Initializing TfidfVectorizer with constructor params: {tfidf_constructor_params}")
        self.vectorizer = TfidfVectorizer(**tfidf_constructor_params)
        self._dim: Optional[int] = None
        self._dtype = np.float64 # Default TF-IDF dtype in sklearn
        if 'dtype' in tfidf_constructor_params: # Allow override via params
            try:
                self._dtype = np.dtype(tfidf_constructor_params['dtype'])
            except TypeError:
                ASCIIColors.warning(f"Invalid dtype specified in TF-IDF params: {tfidf_constructor_params['dtype']}. Using default {self._dtype}.")


        self._fitted = False

    def fit(self, texts: List[str]):
        """Fits the TF-IDF vectorizer to the provided texts."""
        if not texts:
            ASCIIColors.warning("Cannot fit TF-IDF vectorizer on empty text list.")
            return

        ASCIIColors.info(f"Fitting TfidfVectorizer on {len(texts)} documents...")
        try:
            self.vectorizer.fit(texts)
            # Dimension is the size of the vocabulary
            self._dim = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
            self._fitted = True
            # Ensure dtype matches the internal representation after fit (might change based on data/params)
            if hasattr(self.vectorizer, 'idf_'):
                self._dtype = self.vectorizer.idf_.dtype
            ASCIIColors.info(f"TF-IDF fitting complete. Vocabulary size (dimension): {self._dim}, Dtype: {self._dtype}")
        except Exception as e:
            ASCIIColors.error(f"Error during TF-IDF fitting: {e}", exc_info=True)
            self._fitted = False # Ensure fitted is false on error
            raise

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Vectorizes texts using the fitted TF-IDF model."""
        if not self._fitted:
            # Raise error if not fitted. Fitting should be handled by SafeStore logic explicitly.
            ASCIIColors.error("TF-IDF vectorizer must be fitted before vectorizing.")
            raise RuntimeError("TF-IDF vectorizer must be fitted before vectorizing.")


        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using fitted TF-IDF...")
        try:
            # transform returns a sparse matrix, convert to dense for consistency
            vectors_sparse = self.vectorizer.transform(texts)
            vectors_dense = vectors_sparse.toarray().astype(self._dtype) # Ensure correct dtype
            ASCIIColors.debug(f"TF-IDF vectorization complete. Output shape: {vectors_dense.shape}")

            # Check dimension consistency
            if self._dim is None: # Should not happen if fitted is True
                self._dim = vectors_dense.shape[1]
                ASCIIColors.warning(f"TF-IDF dimension was None after fit? Setting to output dim: {self._dim}")
            elif vectors_dense.shape[1] != self._dim:
                 # This could happen if transform encounters terms not in the vocabulary
                 # The shape should still match the vocabulary size. If not, something is wrong.
                 ASCIIColors.error(f"TF-IDF output dimension {vectors_dense.shape[1]} differs from fitted vocabulary size {self._dim}. This indicates an issue.")
                 # Option 1: Raise Error (Safer)
                 raise ValueError("TF-IDF output dimension mismatch after transform.")
                 # Option 2: Warn and continue (might lead to downstream errors)
                 # ASCIIColors.warning(f"TF-IDF output dimension {vectors_dense.shape[1]} differs from expected {self._dim}. Using output dim.")
                 # self._dim = vectors_dense.shape[1]

            return vectors_dense
        except NotFittedError:
            # This check should be redundant due to the check at the start.
            ASCIIColors.error("Attempted to vectorize with an unfitted TF-IDF model (internal error).")
            raise RuntimeError("TF-IDF vectorizer must be fitted before vectorizing.")
        except Exception as e:
            ASCIIColors.error(f"Error during TF-IDF transform: {e}", exc_info=True)
            raise

    @property
    def dim(self) -> Optional[int]:
        # Dimension is only known after fitting. Can be None before fitting.
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def get_params_to_store(self) -> Dict[str, Any]:
        """Returns parameters needed to reconstruct the state, including fitted state if applicable."""
        # Start with the initial constructor parameters
        params_to_store = {"sklearn_params": self.initial_params.copy()}

        if not self._fitted or not hasattr(self.vectorizer, 'vocabulary_') or not hasattr(self.vectorizer, 'idf_'):
            params_to_store["fitted"] = False
            return params_to_store

        # Store fitted state: vocabulary and IDF weights
        params_to_store["fitted"] = True
        params_to_store["vocabulary"] = self.vectorizer.vocabulary_
        params_to_store["idf"] = self.vectorizer.idf_.tolist() # Convert ndarray to list for JSON

        # Also store the learned dimension and dtype, in case they differ from initial assumptions
        params_to_store["learned_dim"] = self.dim
        params_to_store["learned_dtype"] = np.dtype(self.dtype).name

        return params_to_store

    def load_fitted_state(self, stored_params: Dict[str, Any]):
         """Loads a previously fitted state from stored parameters."""
         if not stored_params or not stored_params.get("fitted"):
             ASCIIColors.warning("Attempted to load non-fitted or empty TF-IDF state. Vectorizer remains unfitted.")
             self._fitted = False
             self._dim = None
             return

         ASCIIColors.debug(f"Loading fitted TF-IDF state from stored params.")
         try:
             vocab = stored_params.get("vocabulary")
             idf = stored_params.get("idf")
             learned_dim = stored_params.get("learned_dim")
             learned_dtype_str = stored_params.get("learned_dtype")
             # Re-use initial constructor params stored in 'sklearn_params'
             sklearn_constructor_params = stored_params.get("sklearn_params", {})

             if vocab and idf is not None and learned_dim is not None:
                 # Re-initialize with original constructor params
                 self.vectorizer = TfidfVectorizer(**sklearn_constructor_params)
                 # Set the learned attributes
                 self.vectorizer.vocabulary_ = vocab
                 self.vectorizer.idf_ = np.array(idf) # Convert list back to ndarray
                 # Set internal state
                 self._dim = learned_dim
                 self._fitted = True
                 # Set dtype based on loaded state or default
                 try:
                     self._dtype = np.dtype(learned_dtype_str) if learned_dtype_str else np.float64
                 except TypeError:
                      ASCIIColors.warning(f"Invalid learned dtype in stored TF-IDF params: {learned_dtype_str}. Using default {self._dtype}.")
                      self._dtype = np.float64
                 # We need to set the dtype on the internal vectorizer as well if possible
                 # This requires setting the private _tfidf attribute after fit usually.
                 # Let's try setting it after setting vocab/idf. Might be fragile.
                 try:
                     self.vectorizer._tfidf._idf_diag.dtype = self._dtype
                 except AttributeError:
                      ASCIIColors.debug("Could not directly set dtype on internal TF-IDF matrix after loading state.")
                 # Ensure the vectorizer's internal dtype is set correctly for future transforms
                 self.vectorizer.dtype = self._dtype


                 ASCIIColors.info(f"Successfully loaded fitted TF-IDF state. Vocab size: {self._dim}, Dtype: {self._dtype}")
             else:
                 ASCIIColors.warning("Stored TF-IDF state missing vocabulary, idf weights, or dimension. Loading failed.")
                 self._fitted = False
                 self._dim = None
         except Exception as e:
             ASCIIColors.error(f"Error loading fitted TF-IDF state: {e}", exc_info=True)
             # Leave state as potentially unfitted
             self._fitted = False
             self._dim = None