# safestore/core/exceptions.py

class SafeStoreError(Exception):
    """Base class for all SafeStore specific errors."""
    pass

class DatabaseError(SafeStoreError):
    """Errors related to database operations (connection, schema, query, transaction)."""
    pass

class FileHandlingError(SafeStoreError):
    """Errors related to file system operations (reading, writing, hashing, not found)."""
    pass

class ParsingError(FileHandlingError):
    """Errors occurring during document parsing (subclass of FileHandlingError)."""
    pass

class ConfigurationError(SafeStoreError):
    """Errors related to invalid configuration, setup, or missing optional dependencies."""
    pass

class IndexingError(SafeStoreError):
    """Errors specifically within the document indexing pipeline (chunking, storage logic)."""
    # Note: ParsingError, VectorizationError cover sub-steps. This is for orchestration.
    pass

class VectorizationError(SafeStoreError):
    """Errors related to vectorization processes (model loading, encoding, fitting)."""
    pass

class QueryError(SafeStoreError):
    """Errors occurring during query execution (similarity calculation, result fetching)."""
    pass

class ConcurrencyError(SafeStoreError):
    """Errors related to file locking or concurrent access issues (e.g., timeouts)."""
    pass

# Optional: Add EncryptionError later if needed
# class EncryptionError(SafeStoreError):
#    """Errors related to data encryption or decryption."""
#    pass