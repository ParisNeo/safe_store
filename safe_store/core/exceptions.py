# safe_store/core/exceptions.py

class safe_storeError(Exception):
    """Base class for all safe_store specific errors."""
    pass

class DatabaseError(safe_storeError):
    """Errors related to database operations (connection, schema, query, transaction)."""
    pass

class FileHandlingError(safe_storeError):
    """Errors related to file system operations (reading, writing, hashing, not found)."""
    pass

class ParsingError(FileHandlingError):
    """Errors occurring during document parsing (subclass of FileHandlingError)."""
    pass

class ConfigurationError(safe_storeError):
    """Errors related to invalid configuration, setup, or missing optional dependencies."""
    pass

class IndexingError(safe_storeError):
    """Errors specifically within the document indexing pipeline (chunking, storage logic)."""
    # Note: ParsingError, VectorizationError cover sub-steps. This is for orchestration.
    pass

class VectorizationError(safe_storeError):
    """Errors related to vectorization processes (model loading, encoding, fitting)."""
    pass

class QueryError(safe_storeError):
    """Errors occurring during query execution (similarity calculation, result fetching)."""
    pass

class ConcurrencyError(safe_storeError):
    """Errors related to file locking or concurrent access issues (e.g., timeouts)."""
    pass

# Optional: Add EncryptionError later if needed
# class EncryptionError(safe_storeError):
#    """Errors related to data encryption or decryption."""
#    pass