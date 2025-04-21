# safe_store/core/db.py
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict, Union # Added Union
import json
from ascii_colors import ASCIIColors
from .exceptions import DatabaseError

# --- Type Hinting ---
SQLQuery = str
SQLParams = Union[Tuple[Any, ...], Dict[str, Any]] # For parameters

# --- Adapters remain the same ---
def adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    """Converts a NumPy array to SQLite Binary data."""
    return sqlite3.Binary(arr.tobytes())

sqlite3.register_adapter(np.ndarray, adapt_array)

def reconstruct_vector(blob: bytes, dtype_str: str) -> np.ndarray:
    """
    Safely reconstructs a NumPy array from SQLite blob data and dtype string.

    Args:
        blob: The byte data retrieved from the database.
        dtype_str: The string representation of the NumPy dtype (e.g., 'float32').

    Returns:
        The reconstructed NumPy array.

    Raises:
        DatabaseError: If the dtype string is invalid or reconstruction fails.
    """
    try:
        # Basic check for potentially unsafe dtype strings (though np.dtype handles most)
        if any(char in dtype_str for char in ';()[]{}'):
             raise ValueError(f"Invalid characters found in dtype string: '{dtype_str}'")
        dtype = np.dtype(dtype_str)
        # Check for excessive size to prevent potential memory issues (optional)
        # expected_itemsize = dtype.itemsize
        # if len(blob) % expected_itemsize != 0:
        #    raise ValueError("Blob size is not a multiple of the dtype item size.")
        # max_elements = 1_000_000 # Example limit
        # if len(blob) // expected_itemsize > max_elements:
        #    raise ValueError("Reconstructed vector exceeds maximum allowed size.")

        return np.frombuffer(blob, dtype=dtype)
    except (TypeError, ValueError) as e:
        msg = f"Failed to reconstruct vector: invalid or unsafe dtype '{dtype_str}' or blob data mismatch. Error: {e}"
        ASCIIColors.error(msg)
        raise DatabaseError(msg) from e # Use custom error, chain original exception

def connect_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.

    Enables WAL mode for better concurrency. Creates parent directories if needed.

    Args:
        db_path: The path to the SQLite database file.

    Returns:
        An active sqlite3.Connection object.

    Raises:
        DatabaseError: If the connection fails.
    """
    db_path_obj = Path(db_path).resolve()
    try:
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        # PARSE_DECLTYPES allows automatic conversion using registered converters
        conn = sqlite3.connect(
            str(db_path_obj),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False # Recommended for use with external locking like FileLock
        )
        # Enable Write-Ahead Logging for better read/write concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        # Enable foreign key constraints enforcement
        conn.execute("PRAGMA foreign_keys = ON;")
        ASCIIColors.debug(f"Connected to database: {db_path_obj} (WAL enabled)")
        return conn
    except sqlite3.Error as e:
        msg = f"Database connection error to {db_path_obj}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e # Use custom error

def initialize_schema(conn: sqlite3.Connection) -> None:
    """
    Initializes the database schema if tables don't exist.

    Creates tables for documents, vectorization methods, chunks, and vectors
    with appropriate relationships and indexes. Uses foreign key constraints
    with cascading deletes where appropriate.

    Args:
        conn: An active sqlite3.Connection object.

    Raises:
        DatabaseError: If schema initialization fails.
    """
    cursor = conn.cursor()
    try:
        # Documents Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_hash TEXT, -- For detecting changes
            full_text TEXT, -- Store full text for re-chunking/re-indexing
            metadata TEXT, -- JSON stored as TEXT
            added_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_file_path ON documents (file_path);")

        # Vectorization Methods Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectorization_methods (
            method_id INTEGER PRIMARY KEY AUTOINCREMENT,
            method_name TEXT UNIQUE NOT NULL,
            method_type TEXT NOT NULL, -- e.g., 'sentence_transformer', 'tfidf'
            vector_dim INTEGER NOT NULL,
            vector_dtype TEXT NOT NULL, -- e.g., 'float32', 'float64'
            params TEXT -- JSON storing config (model name, TF-IDF params, fitted state)
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_method_name ON vectorization_methods (method_name);")

        # Chunks Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            start_pos INTEGER NOT NULL,
            end_pos INTEGER NOT NULL,
            chunk_seq INTEGER NOT NULL, -- Order within the document
            tags TEXT, -- JSON stored as TEXT, nullable
            is_encrypted INTEGER DEFAULT 0 NOT NULL, -- Boolean (0 or 1)
            encryption_metadata BLOB, -- Nullable, stores details like IV/salt
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
        );
        """)
        # Index for efficient lookup of chunks by document
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON chunks (doc_id);")

        # Vectors Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            method_id INTEGER NOT NULL,
            vector_data BLOB NOT NULL, -- Store numpy array as bytes using adapter
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            FOREIGN KEY (method_id) REFERENCES vectorization_methods (method_id) ON DELETE CASCADE,
            UNIQUE (chunk_id, method_id) -- Ensure only one vector per chunk per method
        );
        """)
        # Index for joining chunks and methods
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunk_method ON vectors (chunk_id, method_id);")
        # Index for efficient lookup/deletion by method_id
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_method_id ON vectors (method_id);")

        conn.commit()
        ASCIIColors.debug("Database schema initialized or verified.")
    except sqlite3.Error as e:
        msg = f"Schema initialization error: {e}"
        ASCIIColors.error(msg, exc_info=True)
        conn.rollback()
        raise DatabaseError(msg) from e # Use custom error

# --- CRUD Operations ---

def add_document_record(
    conn: sqlite3.Connection,
    file_path: str,
    full_text: str,
    file_hash: Optional[str] = None,
    metadata: Optional[str] = None
) -> int:
    """
    Adds a document record to the 'documents' table.

    Args:
        conn: Active database connection.
        file_path: Absolute path to the document file.
        full_text: The full text content of the document.
        file_hash: Optional SHA256 hash of the file content.
        metadata: Optional JSON string representing document metadata.

    Returns:
        The integer ID (doc_id) of the inserted document.

    Raises:
        DatabaseError: If insertion fails or the document path already exists
                       and its ID cannot be retrieved consistently.
    """
    sql: SQLQuery = """
    INSERT INTO documents (file_path, file_hash, full_text, metadata)
    VALUES (?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path, file_hash, full_text, metadata))
        # No commit here, assume caller handles transaction
        doc_id = cursor.lastrowid
        if doc_id is None:
             # Should not happen with AUTOINCREMENT but safety check
             raise DatabaseError(f"Failed to get lastrowid after inserting document '{file_path}'.")
        ASCIIColors.debug(f"Prepared insertion for document record '{Path(file_path).name}', doc_id={doc_id}")
        return doc_id
    except sqlite3.IntegrityError as e:
        # This indicates the file_path already exists (UNIQUE constraint)
        ASCIIColors.warning(f"Document path '{file_path}' already exists. IntegrityError: {e}")
        # Rollback the failed INSERT attempt before trying to fetch existing ID
        conn.rollback()
        existing_id = get_document_id_by_path(conn, file_path)
        if existing_id is not None:
            ASCIIColors.debug(f"Retrieved existing doc_id {existing_id} for path '{file_path}'.")
            return existing_id
        else:
            # This suggests an inconsistent state (IntegrityError but no existing ID found)
            msg = f"IntegrityError adding '{file_path}', but could not retrieve existing ID. DB state might be inconsistent."
            ASCIIColors.critical(msg) # Use critical as this is unexpected
            raise DatabaseError(msg) from e
    except sqlite3.Error as e:
        # Catch other potential database errors during insert
        msg = f"Error preparing document insertion for '{file_path}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        # Rollback is handled by caller's transaction management
        raise DatabaseError(msg) from e # Use custom error


def get_document_id_by_path(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    """Retrieves the doc_id for a given file path."""
    sql: SQLQuery = "SELECT doc_id FROM documents WHERE file_path = ?"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        # Log error, but return None to indicate not found/error
        ASCIIColors.error(f"Error fetching document ID for '{file_path}': {e}", exc_info=True)
        # Raising here might mask the original IntegrityError in add_document_record
        # Let the caller handle the None return value
        return None

def add_or_get_vectorization_method(
    conn: sqlite3.Connection,
    name: str,
    type: str,
    dim: int,
    dtype: str,
    params: Optional[str] = None
) -> int:
    """
    Adds a vectorization method record if it doesn't exist, or returns the existing ID.

    Args:
        conn: Active database connection.
        name: Unique name for the vectorization method (e.g., 'st:model-name').
        type: Type of the vectorizer (e.g., 'sentence_transformer', 'tfidf').
        dim: Dimension of the vectors produced.
        dtype: NumPy dtype of the vectors (e.g., 'float32').
        params: Optional JSON string storing method configuration (model name, etc.).

    Returns:
        The integer ID (method_id) of the method.

    Raises:
        DatabaseError: If adding or retrieving the method fails.
    """
    sql_select: SQLQuery = "SELECT method_id FROM vectorization_methods WHERE method_name = ?"
    sql_insert: SQLQuery = """
    INSERT INTO vectorization_methods (method_name, method_type, vector_dim, vector_dtype, params)
    VALUES (?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        # 1. Check if method exists
        cursor.execute(sql_select, (name,))
        result = cursor.fetchone()
        if result:
            method_id = result[0]
            ASCIIColors.debug(f"Vectorization method '{name}' already exists with ID {method_id}.")
            return method_id
        else:
            # 2. If not, insert it
            ASCIIColors.debug(f"Adding new vectorization method '{name}' (Type: {type}, Dim: {dim}, Dtype: {dtype}).")
            params_to_store = params if params is not None else '{}'
            cursor.execute(sql_insert, (name, type, dim, dtype, params_to_store))
            method_id = cursor.lastrowid
            if method_id is None: raise DatabaseError(f"Failed to get lastrowid after inserting vectorization method '{name}'.")
            ASCIIColors.debug(f"Prepared insertion for vectorization method '{name}', method_id={method_id}.")
            return method_id
    except sqlite3.Error as e:
        msg = f"Error adding/getting vectorization method '{name}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        # Rollback handled by caller's transaction
        raise DatabaseError(msg) from e # Use custom error

def add_chunk_record(
    conn: sqlite3.Connection,
    doc_id: int,
    text: str,
    start: int,
    end: int,
    seq: int,
    tags: Optional[str] = None,
    is_encrypted: bool = False, # Added encryption fields
    encryption_metadata: Optional[bytes] = None
) -> int:
    """
    Adds a chunk record to the 'chunks' table.

    Args:
        conn: Active database connection.
        doc_id: ID of the parent document.
        text: The text content of the chunk.
        start: Start character offset in the original document.
        end: End character offset in the original document.
        seq: Sequence number of the chunk within the document.
        tags: Optional JSON string of tags associated with the chunk.
        is_encrypted: Flag indicating if chunk_text is encrypted.
        encryption_metadata: Metadata needed for decryption (e.g., IV, salt).

    Returns:
        The integer ID (chunk_id) of the inserted chunk.

    Raises:
        DatabaseError: If the insertion fails.
    """
    sql: SQLQuery = """
    INSERT INTO chunks (doc_id, chunk_text, start_pos, end_pos, chunk_seq, tags, is_encrypted, encryption_metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        # Convert boolean to integer for SQLite
        encrypted_flag = 1 if is_encrypted else 0
        cursor.execute(sql, (doc_id, text, start, end, seq, tags, encrypted_flag, encryption_metadata))
        # No commit here, assume caller handles transaction
        chunk_id = cursor.lastrowid
        if chunk_id is None:
            raise DatabaseError(f"Failed to get lastrowid after inserting chunk (doc={doc_id}, seq={seq}).")
        # ASCIIColors.debug(f"Prepared insertion for chunk record (doc={doc_id}, seq={seq}), chunk_id={chunk_id}") # Optional: too verbose?
        return chunk_id
    except sqlite3.Error as e:
        msg = f"Error preparing chunk insertion (doc={doc_id}, seq={seq}): {e}"
        ASCIIColors.error(msg, exc_info=True)
        # Rollback handled by caller's transaction
        raise DatabaseError(msg) from e # Use custom error

def add_vector_record(
    conn: sqlite3.Connection,
    chunk_id: int,
    method_id: int,
    vector: np.ndarray
) -> None:
    """
    Adds a vector record to the 'vectors' table.

    Uses `ON CONFLICT DO NOTHING` to silently handle cases where a vector for
    the given chunk_id and method_id already exists.

    Args:
        conn: Active database connection.
        chunk_id: ID of the associated chunk.
        method_id: ID of the associated vectorization method.
        vector: The NumPy vector array to store (will be adapted to BLOB).

    Raises:
        DatabaseError: If the insertion fails for reasons other than conflict.
    """
    sql: SQLQuery = """
    INSERT INTO vectors (chunk_id, method_id, vector_data)
    VALUES (?, ?, ?)
    ON CONFLICT(chunk_id, method_id) DO NOTHING;
    """
    cursor = conn.cursor()
    try:
        # The adapt_array function handles conversion to BLOB
        cursor.execute(sql, (chunk_id, method_id, vector))
        # No commit here, assume caller handles transaction
        # ASCIIColors.debug(f"Prepared insertion/ignore for vector record (chunk={chunk_id}, method={method_id})") # Optional: too verbose?
    except sqlite3.Error as e:
        msg = f"Error preparing vector insertion (chunk={chunk_id}, method={method_id}): {e}"
        ASCIIColors.error(msg, exc_info=True)
        # Rollback handled by caller's transaction
        raise DatabaseError(msg) from e # Use custom error