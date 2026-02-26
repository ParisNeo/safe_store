# safe_store/core/exceptions.py

class SafeStoreError(Exception):
    """Base class for all safe_store specific errors."""
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

class EncryptionError(SafeStoreError):
   """Errors related to data encryption or decryption."""
   pass

# --- New Graph-related Exceptions ---
class GraphError(SafeStoreError):
    """Base class for graph-specific errors."""
    pass

class GraphDBError(DatabaseError, GraphError): # Inherits from DatabaseError and GraphError
    """Errors related to graph database operations."""
    pass

class GraphProcessingError(GraphError):
    """Errors occurring during the processing of text to extract graph elements."""
    pass

class LLMCallbackError(GraphProcessingError):
    """Errors related to the LLM processing callback function."""
    pass

class NodeNotFoundError(GraphError):
    """Errors occurring during the processing of text to extract graph elements."""
    pass
class RelationshipNotFoundError(GraphError):
    """Errors occurring during the processing of text to extract graph elements."""
    pass
class DocumentNotFoundError(GraphError):
    """Errors occurring during the processing of text or file."""
    pass

class GraphEntityFusionError(GraphProcessingError):
    """Errors related to the entity fusion process, including LLM decisions."""
    pass

# --- PageIndex Exceptions ---
class PageIndexError(SafeStoreError):
    """Base class for page indexing errors."""
    pass

class PageNotFoundError(PageIndexError):
    """Raised when a specific page ID cannot be found."""
    pass