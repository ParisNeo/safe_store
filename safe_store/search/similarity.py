# safe_store/search/similarity.py
import numpy as np
from ascii_colors import ASCIIColors
from typing import Union

# Type hint for vectors
VectorInput = Union[np.ndarray, list[float]] # Allow lists as input for query? No, enforce ndarray.
Vector1D = np.ndarray # Shape (D,)
Matrix2D = np.ndarray # Shape (N, D)

def cosine_similarity(query_vector: Vector1D, vectors: Matrix2D) -> np.ndarray:
    """
    Calculates cosine similarity between a single query vector and a matrix of vectors.

    Handles normalization and potential zero vectors gracefully.

    Args:
        query_vector: A 1D NumPy array representing the query vector (shape D).
        vectors: A 2D NumPy array where each row is a vector to compare against
                 (shape N, D). Can also handle the case where vectors is 1D
                 (shape D) representing a single comparison vector, by reshaping it.

    Returns:
        A 1D NumPy array of shape (N,) containing the cosine similarity scores,
        where each score is between -1.0 and 1.0.

    Raises:
        TypeError: If inputs are not NumPy arrays.
        ValueError: If input shapes are incompatible (e.g., query is not 1D,
                    matrix is not 1D or 2D, or dimensions mismatch).
    """
    if not isinstance(query_vector, np.ndarray) or not isinstance(vectors, np.ndarray):
        raise TypeError("Input query_vector and vectors must be NumPy arrays.")

    # Validate query_vector shape
    if query_vector.ndim != 1:
        raise ValueError(f"Query vector must be 1D, but got shape {query_vector.shape}")

    # Validate and potentially reshape vectors matrix
    if vectors.ndim == 1:
        # Allow comparing query to a single vector passed as 1D array
        if query_vector.shape[0] == vectors.shape[0]:
            vectors = vectors.reshape(1, -1) # Reshape to (1, D)
            ASCIIColors.debug("Reshaped 1D input 'vectors' to 2D for single vector comparison.")
        else:
            raise ValueError(
                f"If 'vectors' is 1D, its dimension ({vectors.shape[0]}) must match "
                f"query_vector dimension ({query_vector.shape[0]})"
            )
    elif vectors.ndim != 2:
        raise ValueError(f"Input 'vectors' must be a 1D or 2D array, but got shape {vectors.shape}")

    # Dimension compatibility check
    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Query vector dimension ({query_vector.shape[0]}) must match "
            f"the dimension of vectors in the matrix ({vectors.shape[1]})"
        )

    num_vectors = vectors.shape[0]
    if num_vectors == 0:
         ASCIIColors.debug("Input 'vectors' matrix is empty, returning empty similarity array.")
         return np.array([], dtype=query_vector.dtype) # Return empty array of appropriate type

    ASCIIColors.debug(f"Calculating cosine similarity: query_shape={query_vector.shape}, matrix_shape={vectors.shape}")

    # Calculate norms, adding epsilon for numerical stability and avoiding zero division
    epsilon = np.finfo(query_vector.dtype).eps # Use machine epsilon for the data type
    query_norm = np.linalg.norm(query_vector)
    vectors_norm = np.linalg.norm(vectors, axis=1) # Norm of each row vector

    # Handle potential zero vectors by replacing norm with epsilon
    query_norm_safe = query_norm if query_norm > epsilon else epsilon
    vectors_norm_safe = np.where(vectors_norm > epsilon, vectors_norm, epsilon)

    # Normalize vectors
    # Using np.divide with 'out' and 'where' could be slightly more robust, but direct division is common
    norm_query = query_vector / query_norm_safe
    # Use broadcasting for matrix normalization: vectors_norm_safe[:, np.newaxis] ensures (N, 1) shape
    norm_vectors = vectors / vectors_norm_safe[:, np.newaxis]

    # Calculate dot product between normalized matrix rows and the normalized query vector
    # Result is (N, D) dot (D,) -> (N,)
    similarity_scores = np.dot(norm_vectors, norm_query)

    # Clip scores to be strictly within [-1, 1] due to potential floating point inaccuracies
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)

    ASCIIColors.debug(f"Similarity calculation complete. Output shape: {similarity_scores.shape}")
    return similarity_scores