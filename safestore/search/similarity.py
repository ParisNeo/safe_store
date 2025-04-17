# safestore/search/similarity.py
import numpy as np
from ascii_colors import ASCIIColors

def cosine_similarity(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between a query vector and a matrix of vectors.

    Args:
        query_vector: A 1D numpy array representing the query vector.
        vectors: A 2D numpy array where each row is a vector to compare against.

    Returns:
        A 1D numpy array containing the cosine similarity scores.
    """
    if not isinstance(query_vector, np.ndarray) or not isinstance(vectors, np.ndarray):
        raise TypeError("Input vectors must be numpy arrays.")

    if query_vector.ndim != 1:
        raise ValueError(f"Query vector must be 1D, but got shape {query_vector.shape}")
    if vectors.ndim != 2:
        # Handle case where only one vector is provided in the matrix
        if vectors.ndim == 1 and query_vector.shape[0] == vectors.shape[0]:
            vectors = vectors.reshape(1, -1) # Reshape to 2D
            ASCIIColors.debug("Reshaped 1D vectors matrix to 2D for similarity calculation.")
        else:
            raise ValueError(f"Vectors matrix must be 2D, but got shape {vectors.shape}")

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Query vector dimension ({query_vector.shape[0]}) must match "
            f"matrix vectors dimension ({vectors.shape[1]})"
        )

    ASCIIColors.debug(f"Calculating cosine similarity: query_shape={query_vector.shape}, matrix_shape={vectors.shape}")

    # Normalize the query vector and the matrix vectors
    # Add small epsilon to avoid division by zero for zero vectors
    epsilon = 1e-9
    query_norm = np.linalg.norm(query_vector)
    vectors_norm = np.linalg.norm(vectors, axis=1) # Norm of each row vector

    # Handle potential zero vectors
    query_norm_safe = query_norm if query_norm > epsilon else epsilon
    vectors_norm_safe = np.where(vectors_norm > epsilon, vectors_norm, epsilon)

    norm_query = query_vector / query_norm_safe
    norm_vectors = vectors / vectors_norm_safe[:, np.newaxis] # Use broadcasting

    # Calculate dot product
    similarity_scores = np.dot(norm_vectors, norm_query)

    # Clip scores to be within [-1, 1] due to potential floating point inaccuracies
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)

    ASCIIColors.debug(f"Similarity calculation complete. Output shape: {similarity_scores.shape}")
    return similarity_scores