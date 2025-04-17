# safestore/core/db.py
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict
import json # Needed for tags
from ascii_colors import ASCIIColors

# --- Adapters (remain the same) ---
def adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    return sqlite3.Binary(arr.tobytes())

sqlite3.register_adapter(np.ndarray, adapt_array)
# NB: We will NOT register a global converter. Conversion will happen dynamically.


def reconstruct_vector(blob: bytes, dtype_str: str) -> np.ndarray:
    """Safely reconstructs a numpy array from blob data and dtype string."""
    try:
        dtype = np.dtype(dtype_str)
        return np.frombuffer(blob, dtype=dtype)
    except (TypeError, ValueError) as e:
        ASCIIColors.error(f"Failed to reconstruct vector: invalid dtype '{dtype_str}'? Error: {e}")
        # Return an empty array of a default type or raise? Raising might be safer.
        raise ValueError(f"Could not reconstruct vector with dtype '{dtype_str}'") from e

def connect_db(db_path: str | Path) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # isolation_level=None for autocommit mode might be simpler for now,
        # but explicit transactions are better for atomicity.
        conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        # Enable WAL mode for better concurrency later
        conn.execute("PRAGMA journal_mode=WAL;")
        ASCIIColors.debug(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        ASCIIColors.error(f"Database connection error to {db_path}: {e}", exc_info=True)
        raise

def initialize_schema(conn: sqlite3.Connection):
    """Initializes the database schema if tables don't exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_hash TEXT, -- For detecting changes
            full_text TEXT, -- Store full text for re-chunking
            metadata TEXT, -- JSON stored as TEXT
            added_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectorization_methods (
            method_id INTEGER PRIMARY KEY AUTOINCREMENT,
            method_name TEXT UNIQUE NOT NULL,
            method_type TEXT NOT NULL, -- e.g., 'sentence_transformer', 'tfidf'
            vector_dim INTEGER NOT NULL,
            vector_dtype TEXT NOT NULL, -- e.g., 'float32', 'float64'
            params TEXT -- JSON storing model name, etc.
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            start_pos INTEGER NOT NULL,
            end_pos INTEGER NOT NULL,
            chunk_seq INTEGER NOT NULL, -- Order within the document
            tags TEXT, -- JSON stored as TEXT, nullable
            is_encrypted INTEGER DEFAULT 0, -- Boolean (0 or 1)
            encryption_metadata BLOB, -- Nullable
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
        );
        """)
        # Index for faster lookup when joining
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON chunks (doc_id);")


        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            method_id INTEGER NOT NULL,
            vector_data BLOB NOT NULL, -- Store numpy array as bytes
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            FOREIGN KEY (method_id) REFERENCES vectorization_methods (method_id) ON DELETE CASCADE,
            UNIQUE (chunk_id, method_id) -- Ensure only one vector per chunk per method
        );
        """)
        # Index for faster vector retrieval by method/chunk
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunk_method ON vectors (chunk_id, method_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_method_id ON vectors (method_id);")


        conn.commit()
        ASCIIColors.debug("Database schema initialized or verified.")
    except sqlite3.Error as e:
        ASCIIColors.error(f"Schema initialization error: {e}", exc_info=True)
        conn.rollback()
        raise

# --- CRUD Operations ---

def add_document_record(conn: sqlite3.Connection, file_path: str, full_text: str, file_hash: Optional[str] = None, metadata: Optional[str] = None) -> int:
    """Adds a document record and returns its ID."""
    sql = """
    INSERT INTO documents (file_path, file_hash, full_text, metadata)
    VALUES (?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path, file_hash, full_text, metadata))
        conn.commit()
        doc_id = cursor.lastrowid
        ASCIIColors.debug(f"Added document record for '{Path(file_path).name}', doc_id={doc_id}")
        return doc_id
    except sqlite3.IntegrityError:
        ASCIIColors.warning(f"Document path '{file_path}' already exists in the database.")
        conn.rollback()
        # Optionally retrieve and return the existing doc_id
        existing_id = get_document_id_by_path(conn, file_path)
        if existing_id:
            return existing_id
        else:
            # This case should ideally not happen if IntegrityError was due to UNIQUE constraint
            ASCIIColors.error(f"IntegrityError for '{file_path}', but couldn't retrieve existing ID.")
            raise
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error adding document '{file_path}': {e}", exc_info=True)
        conn.rollback()
        raise

def get_document_id_by_path(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    """Retrieves the doc_id for a given file path."""
    sql = "SELECT doc_id FROM documents WHERE file_path = ?"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error fetching document ID for '{file_path}': {e}", exc_info=True)
        return None

def add_or_get_vectorization_method(conn: sqlite3.Connection, name: str, type: str, dim: int, dtype: str, params: Optional[str] = None) -> int:
    """Adds a vectorization method if it doesn't exist, returns its ID."""
    sql_select = "SELECT method_id FROM vectorization_methods WHERE method_name = ?"
    sql_insert = """
    INSERT INTO vectorization_methods (method_name, method_type, vector_dim, vector_dtype, params)
    VALUES (?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_select, (name,))
        result = cursor.fetchone()
        if result:
            ASCIIColors.debug(f"Vectorization method '{name}' already exists with ID {result[0]}.")
            return result[0]
        else:
            cursor.execute(sql_insert, (name, type, dim, dtype, params))
            conn.commit()
            method_id = cursor.lastrowid
            ASCIIColors.debug(f"Added vectorization method '{name}' with ID {method_id}.")
            return method_id
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error adding/getting vectorization method '{name}': {e}", exc_info=True)
        conn.rollback()
        raise

def add_chunk_record(conn: sqlite3.Connection, doc_id: int, text: str, start: int, end: int, seq: int, tags: Optional[str] = None) -> int:
    """Adds a chunk record and returns its ID."""
    sql = """
    INSERT INTO chunks (doc_id, chunk_text, start_pos, end_pos, chunk_seq, tags)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (doc_id, text, start, end, seq, tags))
        # Don't commit here, commit after all chunks and vectors for the doc are added
        chunk_id = cursor.lastrowid
        # ASCIIColors.debug(f"Prepared chunk record seq {seq} for doc_id {doc_id}, chunk_id={chunk_id}")
        return chunk_id
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error adding chunk record (doc={doc_id}, seq={seq}): {e}", exc_info=True)
        conn.rollback() # Rollback the whole transaction if one chunk fails
        raise

def add_vector_record(conn: sqlite3.Connection, chunk_id: int, method_id: int, vector: np.ndarray):
    """Adds a vector record."""
    sql = """
    INSERT INTO vectors (chunk_id, method_id, vector_data)
    VALUES (?, ?, ?)
    ON CONFLICT(chunk_id, method_id) DO NOTHING; -- Or UPDATE if re-vectorization is needed
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (chunk_id, method_id, vector))
        # ASCIIColors.debug(f"Prepared vector record for chunk_id {chunk_id}, method_id {method_id}")
        # Don't commit here, commit after all chunks and vectors for the doc are added
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error adding vector record (chunk={chunk_id}, method={method_id}): {e}", exc_info=True)
        conn.rollback() # Rollback the whole transaction
        raise