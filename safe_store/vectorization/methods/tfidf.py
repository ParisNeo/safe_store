# safe_store/vectorization/methods/tfidf.py
import numpy as np
from typing import List, Optional, Dict, Any
import json
from ..base import BaseVectorizer
from ...core.exceptions import ConfigurationError, VectorizationError # Import custom exceptions
from ascii_colors import ASCIIColors

# Attempt import, handle gracefully
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import NotFittedError
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    NotFittedError = None # Define dummy exception if sklearn not present


class TfidfVectorizerWrapper(BaseVectorizer):
    """
    Wraps scikit-learn's TfidfVectorizer for use within safe_store.

    This vectorizer requires fitting on a corpus before it can transform text.
    The fitting process determines the vocabulary and Inverse Document Frequency (IDF)
    weights. The fitted state (vocabulary, IDF weights, dimension, dtype, original
    parameters) is managed and can be stored/loaded via the `get_params_to_store`
    and `load_fitted_state` methods, enabling persistence in the database.

    Requires `scikit-learn` to be installed (`pip install safe_store[tfidf]`).

    Attributes:
        vectorizer (TfidfVectorizer): The underlying scikit-learn TfidfVectorizer instance.
        initial_params (Dict[str, Any]): Parameters used during initialization.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the TfidfVectorizerWrapper.

        Args:
            params: Optional dictionary of parameters to pass to the underlying
                    scikit-learn TfidfVectorizer constructor (e.g., `ngram_range`,
                    `max_features`, `stop_words`, `dtype`). See scikit-learn
                    documentation for available parameters. Defaults are applied
                    if not specified (e.g., `stop_words='english'`).

        Raises:
            ConfigurationError: If the 'scikit-learn' library is not installed.
            VectorizationError: If provided parameters are invalid for TfidfVectorizer.
        """
        if not _SKLEARN_AVAILABLE or TfidfVectorizer is None:
            msg = "TfidfVectorizerWrapper requires 'scikit-learn' library. Install with: pip install safe_store[tfidf]"
            ASCIIColors.error(msg)
            raise ConfigurationError(msg)

        self.initial_params = params or {} # Store params passed during init
        # Extract relevant sklearn params, provide defaults if not specified
        allowed_sklearn_params = {
            'input', 'encoding', 'decode_error', 'strip_accents', 'lowercase',
            'preprocessor', 'tokenizer', 'analyzer', 'stop_words', 'token_pattern',
            'ngram_range', 'max_df', 'min_df', 'max_features', 'vocabulary',
            'binary', 'dtype', 'norm', 'use_idf', 'smooth_idf', 'sublinear_tf'
        }
        # Filter initial_params to only include those valid for TfidfVectorizer constructor
        tfidf_constructor_params = {
            k: v for k, v in self.initial_params.items() if k in allowed_sklearn_params
        }

        # Set defaults if not provided in params (can be overridden by user params)
        tfidf_constructor_params.setdefault('ngram_range', (1, 1))
        tfidf_constructor_params.setdefault('stop_words', 'english')
        # Default dtype for TF-IDF is usually float64, but allow user override
        default_dtype = np.float64
        try:
             # Use user-specified dtype if valid
             user_dtype = tfidf_constructor_params.get('dtype')
             if user_dtype:
                  self._dtype = np.dtype(user_dtype)
             else:
                  self._dtype = np.dtype(default_dtype)
                  tfidf_constructor_params['dtype'] = self._dtype # Ensure constructor gets it
        except TypeError:
             ASCIIColors.warning(f"Invalid dtype specified in TF-IDF params: {tfidf_constructor_params.get('dtype')}. Using default {default_dtype.name}.")
             self._dtype = np.dtype(default_dtype)
             tfidf_constructor_params['dtype'] = self._dtype


        ASCIIColors.info(f"Initializing TfidfVectorizer with constructor params: {tfidf_constructor_params}")
        try:
            self.vectorizer: TfidfVectorizer = TfidfVectorizer(**tfidf_constructor_params)
        except Exception as e:
             # Catch errors during sklearn TfidfVectorizer initialization (e.g., bad params)
             msg = f"Failed to initialize scikit-learn TfidfVectorizer: {e}"
             ASCIIColors.error(msg, exc_info=True)
             raise VectorizationError(msg) from e

        self._dim: Optional[int] = None # Dimension (vocab size) is unknown until fit
        self._fitted: bool = False

    def fit(self, texts: List[str]) -> None:
        """
        Fits the internal TF-IDF vectorizer to the provided text corpus.

        This determines the vocabulary and calculates IDF weights. Updates the
        internal `_dim`, `_dtype`, and `_fitted` state.

        Args:
            texts: A list of strings representing the documents to fit on.

        Raises:
            VectorizationError: If fitting fails.
        """
        if not texts:
            ASCIIColors.warning("Cannot fit TF-IDF vectorizer on empty text list.")
            # Should we raise an error or allow fitting on empty? Allow for now.
            # Set state appropriately if allowed.
            self._fitted = False # Cannot be fitted on empty
            self._dim = 0
            self.vectorizer.vocabulary_ = {}
            self.vectorizer.idf_ = np.array([], dtype=self.dtype)
            return

        ASCIIColors.info(f"Fitting TfidfVectorizer on {len(texts)} documents...")
        try:
            self.vectorizer.fit(texts)
            # Dimension is the size of the learned vocabulary
            # Use getattr for safety, although fit() should create these attributes
            self._dim = len(getattr(self.vectorizer, 'vocabulary_', {}))
            self._fitted = True
            # Update dtype based on the fitted vectorizer's internal state (e.g., idf_)
            # This ensures consistency if sklearn uses a different internal precision
            if hasattr(self.vectorizer, 'idf_') and self.vectorizer.idf_ is not None:
                actual_dtype = self.vectorizer.idf_.dtype
                if actual_dtype != self._dtype:
                     ASCIIColors.debug(f"TF-IDF internal dtype ({actual_dtype.name}) differs from initial/param dtype ({self._dtype.name}). Updating wrapper dtype.")
                     self._dtype = actual_dtype

            # Check if dim is 0 after fitting on non-empty text (e.g., all stop words)
            if self._dim == 0 and texts:
                 ASCIIColors.warning(f"TF-IDF fitting resulted in an empty vocabulary (dimension 0) for {len(texts)} non-empty documents. Check input text and TF-IDF parameters (e.g., stop words, min_df).")
            else:
                 ASCIIColors.info(f"TF-IDF fitting complete. Vocabulary size (dimension): {self._dim}, Dtype: {self._dtype.name}")

        except Exception as e:
            # Catch errors during the sklearn fit process
            msg = f"Error during TF-IDF fitting: {e}"
            ASCIIColors.error(msg, exc_info=True)
            self._fitted = False # Ensure fitted is false on error
            self._dim = None
            raise VectorizationError(msg) from e

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """
        Transforms text documents into TF-IDF vectors using the fitted model.

        Args:
            texts: A list of text strings to vectorize.

        Returns:
            A 2D NumPy array of shape (len(texts), self.dim) containing the
            TF-IDF vectors (dense format), with dtype self.dtype.

        Raises:
            VectorizationError: If the vectorizer has not been fitted yet, or if
                                the transformation process fails.
        """
        if not self._fitted or NotFittedError is None: # Check flag and availability of exception
            # Raise error if not fitted. Fitting should be handled explicitly by safe_store logic.
            msg = "TF-IDF vectorizer must be fitted before vectorizing."
            ASCIIColors.error(msg)
            raise VectorizationError(msg) # Use custom error

        if not texts:
             ASCIIColors.debug("Received empty list for TF-IDF vectorization, returning empty array.")
             # Return empty array with correct shape (0, dim) and dtype
             # Use dim=0 if it's None (e.g., fitted on empty)
             dim = self._dim if self._dim is not None else 0
             return np.empty((0, dim), dtype=self.dtype)

        ASCIIColors.debug(f"Vectorizing {len(texts)} texts using fitted TF-IDF (dim={self._dim})...")
        try:
            # Call sklearn's transform method
            vectors_sparse = self.vectorizer.transform(texts)
            # Convert the sparse matrix output to a dense NumPy array
            # Ensure the output dense array has the expected dtype
            vectors_dense = vectors_sparse.toarray().astype(self._dtype)
            ASCIIColors.debug(f"TF-IDF vectorization complete. Output shape: {vectors_dense.shape}")

            # --- Dimension Consistency Check ---
            output_dim = vectors_dense.shape[1]
            # Check if dim was set during fit
            if self._dim is None:
                # This case implies fitting didn't set dim correctly, which shouldn't happen if fitted=True
                ASCIIColors.error(f"Internal state error: TF-IDF dimension is None after fitting was marked True. Using output dim: {output_dim}")
                self._dim = output_dim # Try to recover, but indicates a problem
            elif output_dim != self._dim:
                # The output dimension *must* match the vocabulary size determined during fit.
                # If it doesn't, something is fundamentally wrong in sklearn's behavior or our state management.
                msg = f"TF-IDF output dimension {output_dim} differs from fitted vocabulary size {self._dim}. This indicates a critical internal inconsistency."
                ASCIIColors.critical(msg) # Log as critical because it breaks assumptions
                raise VectorizationError(msg)

            return vectors_dense

        except NotFittedError: # Catch sklearn's specific error
            # This check should be redundant due to the self._fitted check at the start,
            # but included for robustness.
            msg = "Attempted to vectorize with an unfitted TF-IDF model (NotFittedError)."
            ASCIIColors.error(msg)
            raise VectorizationError(msg) from NotFittedError
        except Exception as e:
            # Catch other unexpected errors during transform
            msg = f"Error during TF-IDF transform: {e}"
            ASCIIColors.error(msg, exc_info=True)
            raise VectorizationError(msg) from e

    @property
    def dim(self) -> Optional[int]:
        """
        Returns the dimension (vocabulary size) of the vectors.
        Returns None if the vectorizer has not been fitted yet.
        """
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        """Returns the numpy dtype of the TF-IDF vectors (e.g., np.float64)."""
        return self._dtype

    def get_params_to_store(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the state needed to reconstruct this vectorizer.

        Includes the initial constructor parameters and, if fitted, the learned
        vocabulary, IDF weights, dimension, and dtype. Designed to be JSON serializable.

        Returns:
            A dictionary containing the vectorizer's configuration and fitted state.
        """
        # Start with the initial constructor parameters under a specific key
        params_to_store = {"sklearn_params": self.initial_params.copy()}
        params_to_store["fitted"] = self._fitted

        if not self._fitted or not hasattr(self.vectorizer, 'vocabulary_') or not hasattr(self.vectorizer, 'idf_'):
            ASCIIColors.debug("TF-IDF not fitted, storing only initial params and fitted=False.")
            return params_to_store

        # Store fitted state if available
        try:
             # Ensure attributes exist before accessing
             vocab = getattr(self.vectorizer, 'vocabulary_', None)
             idf = getattr(self.vectorizer, 'idf_', None)

             if vocab is not None and idf is not None:
                 params_to_store["vocabulary"] = vocab # Store vocabulary dict
                 params_to_store["idf"] = idf.tolist() # Convert ndarray to list for JSON

                 # Also store the learned dimension and dtype for verification during load
                 params_to_store["learned_dim"] = self.dim
                 params_to_store["learned_dtype"] = self.dtype.name # Store dtype name string
                 ASCIIColors.debug(f"Storing fitted TF-IDF state: vocab_size={len(vocab)}, dim={self.dim}, dtype={self.dtype.name}")
             else:
                  # This case means fitted=True, but attributes are missing - internal inconsistency
                  ASCIIColors.warning("TF-IDF marked as fitted, but vocabulary or idf_ attribute is missing. Storing minimal state.")
                  params_to_store["fitted"] = False # Mark as not fitted in stored params

        except Exception as e:
             # Catch unexpected errors during state retrieval
             ASCIIColors.error(f"Error retrieving fitted state for TF-IDF: {e}", exc_info=True)
             # Fallback to storing as not fitted
             params_to_store["fitted"] = False
             if "vocabulary" in params_to_store: del params_to_store["vocabulary"]
             if "idf" in params_to_store: del params_to_store["idf"]
             if "learned_dim" in params_to_store: del params_to_store["learned_dim"]
             if "learned_dtype" in params_to_store: del params_to_store["learned_dtype"]

        return params_to_store

    def load_fitted_state(self, stored_params: Dict[str, Any]) -> None:
        """
        Loads a previously fitted state into the vectorizer instance.

        Reconstructs the internal state (vocabulary, IDF weights) from the
        provided dictionary, which should have been produced by `get_params_to_store`.

        Args:
            stored_params: A dictionary containing the previously saved state.

        Raises:
            VectorizationError: If the stored state is incomplete, invalid, or
                                if reconstructing the state fails.
        """
        if not stored_params or not stored_params.get("fitted"):
            ASCIIColors.warning("Attempted to load non-fitted or empty TF-IDF state. Vectorizer remains unfitted.")
            self._fitted = False
            self._dim = None
            return

        ASCIIColors.debug("Loading fitted TF-IDF state from stored params.")
        try:
            # --- Retrieve essential components from stored state ---
            vocab = stored_params.get("vocabulary")
            idf_list = stored_params.get("idf")
            learned_dim = stored_params.get("learned_dim")
            learned_dtype_str = stored_params.get("learned_dtype")
            # Get original constructor params used when this state was created
            sklearn_constructor_params = stored_params.get("sklearn_params", {})

            # --- Validate retrieved components ---
            if vocab is None or idf_list is None or learned_dim is None:
                msg = "Stored TF-IDF state is incomplete: missing vocabulary, idf weights, or dimension. Cannot load."
                ASCIIColors.error(msg)
                raise VectorizationError(msg)

            if not isinstance(vocab, dict):
                 raise VectorizationError(f"Invalid vocabulary format in stored state (expected dict, got {type(vocab)}).")
            if not isinstance(idf_list, list):
                 raise VectorizationError(f"Invalid idf format in stored state (expected list, got {type(idf_list)}).")
            if not isinstance(learned_dim, int) or learned_dim < 0:
                 raise VectorizationError(f"Invalid learned dimension in stored state: {learned_dim}.")
            if len(vocab) != learned_dim or len(idf_list) != learned_dim:
                 raise VectorizationError(f"Dimension mismatch in stored state: vocab({len(vocab)}), idf({len(idf_list)}), learned_dim({learned_dim}).")

            # --- Re-initialize and Set State ---
            # Determine the dtype to use for reconstruction
            try:
                 target_dtype = np.dtype(learned_dtype_str) if learned_dtype_str else np.float64
            except TypeError:
                 ASCIIColors.warning(f"Invalid learned dtype in stored TF-IDF params: '{learned_dtype_str}'. Using default np.float64.")
                 target_dtype = np.dtype(np.float64)

            # Ensure the constructor gets the correct dtype for internal consistency
            sklearn_constructor_params['dtype'] = target_dtype

            # Re-initialize the underlying vectorizer with original params + correct dtype
            self.vectorizer = TfidfVectorizer(**sklearn_constructor_params)

            # Set the learned attributes directly
            self.vectorizer.vocabulary_ = vocab
            # Convert IDF list back to NumPy array with the target dtype
            self.vectorizer.idf_ = np.array(idf_list, dtype=target_dtype)

            # Manually trigger the internal calculation that normally happens after fit
            # This step is crucial for the transform method to work correctly after loading state.
            # It involves setting the private `_tfidf._idf_diag` sparse matrix.
            if hasattr(self.vectorizer, '_check_vocabulary') and callable(self.vectorizer._check_vocabulary):
                 self.vectorizer._check_vocabulary() # Checks vocab consistency
            if hasattr(self.vectorizer, '_tfidf') and hasattr(self.vectorizer._tfidf, '_set_idf_diag') and callable(self.vectorizer._tfidf._set_idf_diag):
                 # This uses internal sklearn details - potentially fragile across versions
                 from scipy.sparse import diags
                 self.vectorizer._tfidf._set_idf_diag(diags(self.vectorizer.idf_, offsets=0,
                                                         shape=(learned_dim, learned_dim),
                                                         format='csr',
                                                         dtype=target_dtype))
                 ASCIIColors.debug("Successfully set internal IDF diagonal matrix.")
            else:
                  # Fallback or warning if internal structure changes
                  ASCIIColors.warning("Could not directly set internal TF-IDF state (_tfidf._idf_diag). Transform might fail if internal sklearn API changed.")


            # Update wrapper state
            self._dim = learned_dim
            self._dtype = target_dtype
            self._fitted = True

            ASCIIColors.info(f"Successfully loaded fitted TF-IDF state. Vocab size: {self._dim}, Dtype: {self._dtype.name}")

        except (ValueError, TypeError, KeyError, AttributeError) as e:
             # Catch errors during state reconstruction or validation
             msg = f"Error loading fitted TF-IDF state: {e}"
             ASCIIColors.error(msg, exc_info=True)
             # Reset state to unfitted on failure
             self._fitted = False
             self._dim = None
             # Re-initialize vectorizer to a clean state? Or leave as is?
             # Re-initializing might be safer.
             try:
                  self.vectorizer = TfidfVectorizer(**self.initial_params)
             except Exception: pass # Ignore errors during cleanup re-init
             raise VectorizationError(msg) from e
        except Exception as e:
             # Catch other unexpected errors
             msg = f"Unexpected error loading TF-IDF state: {e}"
             ASCIIColors.error(msg, exc_info=True)
             self._fitted = False
             self._dim = None
             raise VectorizationError(msg) from e