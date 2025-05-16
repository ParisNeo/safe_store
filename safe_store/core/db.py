# safe_store/core/db.py
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict, Union # Added Union
import json
from ascii_colors import ASCIIColors
from .exceptions import DatabaseError, GraphDBError # Added GraphDBError

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
        return np.frombuffer(blob, dtype=dtype)
    except (TypeError, ValueError) as e:
        msg = f"Failed to reconstruct vector: invalid or unsafe dtype '{dtype_str}' or blob data mismatch. Error: {e}"
        ASCIIColors.error(msg)
        raise DatabaseError(msg) from e

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
        conn = sqlite3.connect(
            str(db_path_obj),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        ASCIIColors.debug(f"Connected to database: {db_path_obj} (WAL enabled)")
        return conn
    except sqlite3.Error as e:
        msg = f"Database connection error to {db_path_obj}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def initialize_schema(conn: sqlite3.Connection) -> None:
    """
    Initializes the database schema if tables don't exist.

    Creates tables for documents, vectorization methods, chunks, vectors,
    and graph-related data (nodes, relationships, links, metadata).

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
            file_hash TEXT,
            full_text TEXT,
            metadata TEXT,
            added_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_file_path ON documents (file_path);")

        # Vectorization Methods Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectorization_methods (
            method_id INTEGER PRIMARY KEY AUTOINCREMENT,
            method_name TEXT UNIQUE NOT NULL,
            method_type TEXT NOT NULL,
            vector_dim INTEGER NOT NULL,
            vector_dtype TEXT NOT NULL,
            params TEXT
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
            chunk_seq INTEGER NOT NULL,
            tags TEXT,
            is_encrypted INTEGER DEFAULT 0 NOT NULL,
            encryption_metadata BLOB,
            graph_processed_at DATETIME, -- New column for tracking graph processing
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON chunks (doc_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_graph_processed_at ON chunks (graph_processed_at);") # Index for faster unprocessed chunk lookup


        # Vectors Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL,
            method_id INTEGER NOT NULL,
            vector_data BLOB NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            FOREIGN KEY (method_id) REFERENCES vectorization_methods (method_id) ON DELETE CASCADE,
            UNIQUE (chunk_id, method_id)
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunk_method ON vectors (chunk_id, method_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_method_id ON vectors (method_id);")

        # --- Graph Schema ---
        # Store Metadata Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS store_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """)

        # Graph Nodes Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_label TEXT NOT NULL,
            node_properties TEXT, 
            unique_signature TEXT UNIQUE 
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_label ON graph_nodes (node_label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_signature ON graph_nodes (unique_signature);")


        # Graph Relationships Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_relationships (
            relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_node_id INTEGER NOT NULL,
            target_node_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            relationship_properties TEXT, 
            FOREIGN KEY (source_node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (target_node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_source_type ON graph_relationships (source_node_id, relationship_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_target_type ON graph_relationships (target_node_id, relationship_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_type ON graph_relationships (relationship_type);")


        # Node-Chunk Links Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_chunk_links (
            node_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            FOREIGN KEY (node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            PRIMARY KEY (node_id, chunk_id)
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ncl_node_id ON node_chunk_links (node_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ncl_chunk_id ON node_chunk_links (chunk_id);")


        conn.commit()
        ASCIIColors.debug("Database schema (including graph tables and 'chunks.graph_processed_at') initialized or verified.")
    except sqlite3.Error as e:
        msg = f"Schema initialization error: {e}"
        ASCIIColors.error(msg, exc_info=True)
        conn.rollback()
        raise DatabaseError(msg) from e

# --- CRUD Operations (existing ones remain, new graph ones below) ---

def add_document_record(
    conn: sqlite3.Connection,
    file_path: str,
    full_text: str,
    file_hash: Optional[str] = None,
    metadata: Optional[str] = None
) -> int:
    sql: SQLQuery = """
    INSERT INTO documents (file_path, file_hash, full_text, metadata)
    VALUES (?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path, file_hash, full_text, metadata))
        doc_id = cursor.lastrowid
        if doc_id is None:
             raise DatabaseError(f"Failed to get lastrowid after inserting document '{file_path}'.")
        ASCIIColors.debug(f"Prepared insertion for document record '{Path(file_path).name}', doc_id={doc_id}")
        return doc_id
    except sqlite3.IntegrityError as e:
        conn.rollback()
        existing_id = get_document_id_by_path(conn, file_path)
        if existing_id is not None:
            ASCIIColors.debug(f"Retrieved existing doc_id {existing_id} for path '{file_path}'.")
            return existing_id
        else:
            msg = f"IntegrityError adding '{file_path}', but could not retrieve existing ID. DB state might be inconsistent."
            ASCIIColors.critical(msg)
            raise DatabaseError(msg) from e
    except sqlite3.Error as e:
        msg = f"Error preparing document insertion for '{file_path}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e


def get_document_id_by_path(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    sql: SQLQuery = "SELECT doc_id FROM documents WHERE file_path = ?"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error fetching document ID for '{file_path}': {e}", exc_info=True)
        return None

def add_or_get_vectorization_method(
    conn: sqlite3.Connection,
    name: str,
    type: str,
    dim: int,
    dtype: str,
    params: Optional[str] = None
) -> int:
    sql_select: SQLQuery = "SELECT method_id FROM vectorization_methods WHERE method_name = ?"
    sql_insert: SQLQuery = """
    INSERT INTO vectorization_methods (method_name, method_type, vector_dim, vector_dtype, params)
    VALUES (?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_select, (name,))
        result = cursor.fetchone()
        if result:
            method_id = result[0]
            ASCIIColors.debug(f"Vectorization method '{name}' already exists with ID {method_id}.")
            return method_id
        else:
            params_to_store = params if params is not None else '{}'
            cursor.execute(sql_insert, (name, type, dim, dtype, params_to_store))
            method_id = cursor.lastrowid
            if method_id is None: raise DatabaseError(f"Failed to get lastrowid after inserting vectorization method '{name}'.")
            ASCIIColors.debug(f"Prepared insertion for vectorization method '{name}', method_id={method_id}.")
            return method_id
    except sqlite3.Error as e:
        msg = f"Error adding/getting vectorization method '{name}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def add_chunk_record(
    conn: sqlite3.Connection,
    doc_id: int,
    text: str,
    start: int,
    end: int,
    seq: int,
    tags: Optional[str] = None,
    is_encrypted: bool = False,
    encryption_metadata: Optional[bytes] = None
) -> int:
    sql: SQLQuery = """
    INSERT INTO chunks (doc_id, chunk_text, start_pos, end_pos, chunk_seq, tags, is_encrypted, encryption_metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        encrypted_flag = 1 if is_encrypted else 0
        cursor.execute(sql, (doc_id, text, start, end, seq, tags, encrypted_flag, encryption_metadata))
        chunk_id = cursor.lastrowid
        if chunk_id is None:
            raise DatabaseError(f"Failed to get lastrowid after inserting chunk (doc={doc_id}, seq={seq}).")
        return chunk_id
    except sqlite3.Error as e:
        msg = f"Error preparing chunk insertion (doc={doc_id}, seq={seq}): {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def add_vector_record(
    conn: sqlite3.Connection,
    chunk_id: int,
    method_id: int,
    vector: np.ndarray
) -> None:
    sql: SQLQuery = """
    INSERT INTO vectors (chunk_id, method_id, vector_data)
    VALUES (?, ?, ?)
    ON CONFLICT(chunk_id, method_id) DO NOTHING;
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (chunk_id, method_id, vector))
    except sqlite3.Error as e:
        msg = f"Error preparing vector insertion (chunk={chunk_id}, method={method_id}): {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

# --- New Graph DB Functions ---

def set_store_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Sets a key-value pair in the store_metadata table."""
    sql: SQLQuery = "INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (key, value))
        # No commit here, assume caller handles transaction
        ASCIIColors.debug(f"Prepared set/update for store_metadata: {key} = {value}")
    except sqlite3.Error as e:
        msg = f"Error setting store metadata '{key}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def get_store_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Gets a value from the store_metadata table by key."""
    sql: SQLQuery = "SELECT value FROM store_metadata WHERE key = ?"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (key,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        msg = f"Error getting store metadata for key '{key}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def add_graph_node(
    conn: sqlite3.Connection,
    label: str,
    properties_json: Optional[str] = None,
    unique_signature: Optional[str] = None
) -> int:
    """
    Adds a graph node if it doesn't exist (based on unique_signature), or returns existing node_id.
    If unique_signature is None, it always inserts a new node.

    Args:
        conn: Active database connection.
        label: Label for the node (e.g., "Person", "Location").
        properties_json: JSON string of node properties.
        unique_signature: An optional unique string to identify this node.

    Returns:
        The node_id of the added or existing node.

    Raises:
        GraphDBError: If database interaction fails.
    """
    cursor = conn.cursor()
    if unique_signature:
        existing_node_id = get_graph_node_by_signature(conn, unique_signature)
        if existing_node_id is not None:
            ASCIIColors.debug(f"Node with signature '{unique_signature}' already exists (ID: {existing_node_id}).")
            return existing_node_id

    sql_insert: SQLQuery = """
    INSERT INTO graph_nodes (node_label, node_properties, unique_signature)
    VALUES (?, ?, ?)
    """
    try:
        cursor.execute(sql_insert, (label, properties_json, unique_signature))
        node_id = cursor.lastrowid
        if node_id is None:
            raise GraphDBError(f"Failed to get lastrowid after inserting graph node (label='{label}', signature='{unique_signature}').")
        ASCIIColors.debug(f"Prepared insertion for graph node '{label}' (Sig: {unique_signature}), node_id={node_id}")
        return node_id
    except sqlite3.IntegrityError as e: # Should be caught by get_graph_node_by_signature if signature is NOT NULL and UNIQUE
        conn.rollback() # Rollback the failed insert
        if unique_signature: # This implies a race condition if the earlier check passed
            ASCIIColors.warning(f"Race condition? IntegrityError for node signature '{unique_signature}'. Re-fetching.")
            existing_node_id = get_graph_node_by_signature(conn, unique_signature)
            if existing_node_id is not None: return existing_node_id
        msg = f"IntegrityError adding graph node (label='{label}', signature='{unique_signature}'): {e}"
        ASCIIColors.error(msg)
        raise GraphDBError(msg) from e
    except sqlite3.Error as e:
        msg = f"Error preparing graph node insertion (label='{label}', signature='{unique_signature}'): {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise GraphDBError(msg) from e

def get_graph_node_by_signature(conn: sqlite3.Connection, signature: str) -> Optional[int]:
    """Retrieves a graph node_id by its unique_signature."""
    sql: SQLQuery = "SELECT node_id FROM graph_nodes WHERE unique_signature = ?"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (signature,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        ASCIIColors.error(f"Error fetching graph node by signature '{signature}': {e}", exc_info=True)
        # Let caller handle None, or raise GraphDBError if preferred
        return None


def add_graph_relationship(
    conn: sqlite3.Connection,
    source_node_id: int,
    target_node_id: int,
    type: str,
    properties_json: Optional[str] = None
) -> int:
    """
    Adds a relationship between two nodes.

    Args:
        conn: Active database connection.
        source_node_id: ID of the source node.
        target_node_id: ID of the target node.
        type: Type of the relationship (e.g., "WORKS_AT").
        properties_json: JSON string of relationship properties.

    Returns:
        The relationship_id of the added relationship.

    Raises:
        GraphDBError: If database interaction fails.
    """
    sql_insert: SQLQuery = """
    INSERT INTO graph_relationships (source_node_id, target_node_id, relationship_type, relationship_properties)
    VALUES (?, ?, ?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_insert, (source_node_id, target_node_id, type, properties_json))
        rel_id = cursor.lastrowid
        if rel_id is None:
            raise GraphDBError(f"Failed to get lastrowid after inserting graph relationship (source={source_node_id}, target={target_node_id}, type='{type}').")
        ASCIIColors.debug(f"Prepared insertion for graph relationship '{type}' (Source:{source_node_id} -> Target:{target_node_id}), rel_id={rel_id}")
        return rel_id
    except sqlite3.Error as e:
        msg = f"Error preparing graph relationship insertion (source={source_node_id}, target={target_node_id}, type='{type}'): {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise GraphDBError(msg) from e

def link_node_to_chunk(conn: sqlite3.Connection, node_id: int, chunk_id: int) -> None:
    """
    Links a graph node to a text chunk. Ignores if the link already exists.

    Args:
        conn: Active database connection.
        node_id: ID of the graph node.
        chunk_id: ID of the text chunk.

    Raises:
        GraphDBError: If database interaction fails.
    """
    sql: SQLQuery = """
    INSERT OR IGNORE INTO node_chunk_links (node_id, chunk_id)
    VALUES (?, ?)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (node_id, chunk_id))
        # ASCIIColors.debug(f"Prepared link for node {node_id} to chunk {chunk_id}") # Can be verbose
    except sqlite3.Error as e:
        msg = f"Error linking node {node_id} to chunk {chunk_id}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise GraphDBError(msg) from e

def mark_chunks_graph_processed(conn: sqlite3.Connection, chunk_ids: List[int]) -> None:
    """Marks a list of chunks as graph processed by setting graph_processed_at."""
    if not chunk_ids:
        return
    sql: SQLQuery = "UPDATE chunks SET graph_processed_at = CURRENT_TIMESTAMP WHERE chunk_id IN ({})".format(
        ",".join("?" * len(chunk_ids))
    )
    cursor = conn.cursor()
    try:
        cursor.execute(sql, tuple(chunk_ids))
        ASCIIColors.debug(f"Marked {cursor.rowcount} chunks as graph processed.")
    except sqlite3.Error as e:
        msg = f"Error marking chunks as graph processed: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise GraphDBError(msg) from e
    

def add_or_update_graph_node(
    conn: sqlite3.Connection,
    label: str,
    properties: Dict[str, Any], # Pass properties as dict
    unique_signature: Optional[str] = None,
    property_merge_strategy: str = "merge_overwrite_new" # e.g., new values for same keys win
) -> int:
    """
    Adds a graph node if it doesn't exist (based on unique_signature),
    or updates its properties if it does exist.

    Returns:
        The node_id of the added or existing/updated node.
    """
    cursor = conn.cursor()
    node_id: Optional[int] = None

    if unique_signature:
        cursor.execute("SELECT node_id, node_properties FROM graph_nodes WHERE unique_signature = ?", (unique_signature,))
        row = cursor.fetchone()
        if row:
            node_id = row[0]
            existing_properties_json = row[1]
            existing_properties = json.loads(existing_properties_json) if existing_properties_json else {}
            
            if property_merge_strategy == "merge_overwrite_new":
                merged_properties = {**existing_properties, **properties}
            elif property_merge_strategy == "merge_prefer_existing":
                merged_properties = {**properties, **existing_properties}
            elif property_merge_strategy == "overwrite_all": # New properties completely replace
                merged_properties = properties
            else: # Default to merge_overwrite_new
                merged_properties = {**existing_properties, **properties}

            new_properties_json = json.dumps(merged_properties)
            if new_properties_json != existing_properties_json: # Only update if there's a change
                try:
                    cursor.execute("UPDATE graph_nodes SET node_properties = ? WHERE node_id = ?", 
                                   (new_properties_json, node_id))
                    ASCIIColors.debug(f"Updated properties for existing node ID {node_id} (Sig: {unique_signature}).")
                except sqlite3.Error as e:
                    msg = f"Error updating properties for node ID {node_id}: {e}"
                    ASCIIColors.error(msg)
                    raise GraphDBError(msg) from e
            else:
                ASCIIColors.debug(f"Node ID {node_id} (Sig: {unique_signature}) found, properties unchanged.")
            return node_id

    # If node_id is still None, it means it's a new node or no signature was provided
    properties_json_to_insert = json.dumps(properties)
    sql_insert: SQLQuery = """
    INSERT INTO graph_nodes (node_label, node_properties, unique_signature)
    VALUES (?, ?, ?)
    """
    try:
        cursor.execute(sql_insert, (label, properties_json_to_insert, unique_signature))
        node_id = cursor.lastrowid
        if node_id is None:
            raise GraphDBError(f"Failed to get lastrowid after inserting graph node (label='{label}', signature='{unique_signature}').")
        ASCIIColors.debug(f"Inserted new graph node '{label}' (Sig: {unique_signature}), node_id={node_id}")
        return node_id
    except sqlite3.IntegrityError as e: 
        # This case should ideally only be hit if unique_signature is None and some other constraint fails,
        # or if there was a race condition not caught by the initial select.
        conn.rollback() 
        if unique_signature: 
            ASCIIColors.warning(f"Race condition or unexpected IntegrityError for node signature '{unique_signature}'. Re-fetching.")
            # Re-fetch to be safe, though the earlier select should have caught it.
            cursor.execute("SELECT node_id FROM graph_nodes WHERE unique_signature = ?", (unique_signature,))
            refetched_row = cursor.fetchone()
            if refetched_row: return refetched_row[0]
        msg = f"IntegrityError adding/updating graph node (label='{label}', signature='{unique_signature}'): {e}"
        ASCIIColors.error(msg)
        raise GraphDBError(msg) from e
    except sqlite3.Error as e:
        msg = f"Error inserting/updating graph node (label='{label}', signature='{unique_signature}'): {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise GraphDBError(msg) from e
    
# --- New Graph Query DB Functions (Phase 3) ---

def get_graph_node_by_label_and_property(
    conn: sqlite3.Connection,
    label: str,
    prop_key: str,
    prop_value: Any
) -> List[Dict[str, Any]]:
    """
    Finds graph nodes by label and a specific property key-value pair.
    Uses json_extract for querying JSON properties. Returns a list of matching nodes.
    Note: SQLite's json_extract returns JSON values as strings, numbers, or null.
          Comparisons should be mindful of types if prop_value is not a string.
          For simplicity, this example assumes prop_value will be matched as text if it's not numeric.
    """
    # Ensure prop_key is safe for SQL injection if it were directly in query string (it's not here)
    # For json_extract, the path `$.prop_key` is safe.
    sql: SQLQuery = f"""
    SELECT node_id, node_label, node_properties, unique_signature
    FROM graph_nodes
    WHERE node_label = ? AND json_extract(node_properties, '$.{prop_key}') = ?;
    """
    # If prop_value is numeric, SQLite's json_extract might compare it as a number.
    # If it's a string, it should be compared as a string.
    # json_extract typically returns text for string values in JSON.
    params = (label, str(prop_value)) # Cast prop_value to string for robust comparison with json_extract text output
    
    cursor = conn.cursor()
    nodes_found: List[Dict[str, Any]] = []
    try:
        cursor.execute(sql, params)
        for row in cursor.fetchall():
            properties = json.loads(row[2]) if row[2] else {}
            nodes_found.append({
                "node_id": row[0],
                "label": row[1],
                "properties": properties,
                "unique_signature": row[3]
            })
        return nodes_found
    except sqlite3.Error as e:
        msg = f"DB error finding node by label '{label}' and prop '{prop_key}={prop_value}': {e}"
        ASCIIColors.error(msg)
        raise GraphDBError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"JSON decode error for properties while finding node by label '{label}' and prop '{prop_key}={prop_value}': {e}"
        ASCIIColors.error(msg)
        raise GraphDBError(msg) from e


def get_node_details_db(conn: sqlite3.Connection, node_id: int) -> Optional[Dict[str, Any]]:
    """Fetches details (label, properties, signature) for a single node by its ID."""
    sql: SQLQuery = "SELECT node_label, node_properties, unique_signature FROM graph_nodes WHERE node_id = ?"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (node_id,))
        row = cursor.fetchone()
        if row:
            properties = json.loads(row[1]) if row[1] else {}
            return {
                "node_id": node_id,
                "label": row[0],
                "properties": properties,
                "unique_signature": row[2]
            }
        return None
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error fetching details for node_id {node_id}: {e}") from e
    except json.JSONDecodeError as e:
        raise GraphDBError(f"JSON decode error for properties of node_id {node_id}: {e}") from e

def get_relationship_details_db(conn: sqlite3.Connection, relationship_id: int) -> Optional[Dict[str, Any]]:
    """Fetches details for a single relationship by its ID."""
    sql: SQLQuery = """
    SELECT source_node_id, target_node_id, relationship_type, relationship_properties
    FROM graph_relationships WHERE relationship_id = ?
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (relationship_id,))
        row = cursor.fetchone()
        if row:
            properties = json.loads(row[3]) if row[3] else {}
            return {
                "relationship_id": relationship_id,
                "source_node_id": row[0],
                "target_node_id": row[1],
                "type": row[2],
                "properties": properties
            }
        return None
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error fetching details for relationship_id {relationship_id}: {e}") from e
    except json.JSONDecodeError as e:
        raise GraphDBError(f"JSON decode error for properties of relationship_id {relationship_id}: {e}") from e

def get_relationships_for_node_db(
    conn: sqlite3.Connection,
    node_id: int,
    relationship_type: Optional[str] = None,
    direction: str = "any", # "outgoing", "incoming", "any"
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Fetches relationships connected to a given node.
    Returns a list of relationship detail dictionaries.
    """
    conditions = []
    params: List[Any] = []

    if direction == "outgoing":
        conditions.append("r.source_node_id = ?")
        params.append(node_id)
    elif direction == "incoming":
        conditions.append("r.target_node_id = ?")
        params.append(node_id)
    elif direction == "any":
        conditions.append("(r.source_node_id = ? OR r.target_node_id = ?)")
        params.extend([node_id, node_id])
    else:
        raise ValueError("Invalid direction. Must be 'outgoing', 'incoming', or 'any'.")

    if relationship_type:
        conditions.append("r.relationship_type = ?")
        params.append(relationship_type)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    sql: SQLQuery = f"""
    SELECT r.relationship_id, r.source_node_id, r.target_node_id, r.relationship_type, r.relationship_properties,
           s_node.node_label as source_label, s_node.node_properties as source_properties,
           t_node.node_label as target_label, t_node.node_properties as target_properties
    FROM graph_relationships r
    JOIN graph_nodes s_node ON r.source_node_id = s_node.node_id
    JOIN graph_nodes t_node ON r.target_node_id = t_node.node_id
    WHERE {where_clause}
    LIMIT ?;
    """
    params.append(limit)
    
    cursor = conn.cursor()
    relationships: List[Dict[str, Any]] = []
    try:
        cursor.execute(sql, tuple(params))
        for row in cursor.fetchall():
            rel_props = json.loads(row[4]) if row[4] else {}
            src_props = json.loads(row[6]) if row[6] else {}
            tgt_props = json.loads(row[8]) if row[8] else {}
            relationships.append({
                "relationship_id": row[0],
                "source_node_id": row[1],
                "target_node_id": row[2],
                "type": row[3],
                "properties": rel_props,
                "source_node": {"node_id": row[1], "label": row[5], "properties": src_props},
                "target_node": {"node_id": row[2], "label": row[7], "properties": tgt_props}
            })
        return relationships
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error fetching relationships for node_id {node_id}: {e}") from e
    except json.JSONDecodeError as e:
        raise GraphDBError(f"JSON decode error for properties while fetching relationships for node_id {node_id}: {e}") from e


def get_chunk_ids_for_nodes_db(conn: sqlite3.Connection, node_ids: List[int]) -> Dict[int, List[int]]:
    """
    Given a list of node IDs, returns a dictionary mapping each node ID
    to a list of chunk IDs it's linked to.
    """
    if not node_ids:
        return {}
    
    placeholders = ",".join("?" * len(node_ids))
    sql: SQLQuery = f"SELECT node_id, chunk_id FROM node_chunk_links WHERE node_id IN ({placeholders})"
    
    results: Dict[int, List[int]] = {node_id: [] for node_id in node_ids}
    cursor = conn.cursor()
    try:
        cursor.execute(sql, tuple(node_ids))
        for row in cursor.fetchall():
            node_id, chunk_id = row
            if node_id in results: # Should always be true due to IN clause
                results[node_id].append(chunk_id)
        return results
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error fetching chunk links for nodes: {e}") from e

def get_chunk_details_db(
    conn: sqlite3.Connection,
    chunk_ids: List[int],
    encryptor: Optional[Any] = None # Pass Encryptor instance if decryption needed
) -> List[Dict[str, Any]]:
    """
    Fetches full details for a list of chunk IDs, including document path and metadata.
    Handles decryption if an encryptor is provided and chunks are encrypted.
    """
    if not chunk_ids:
        return []

    placeholders = ",".join("?" * len(chunk_ids))
    sql: SQLQuery = f"""
    SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.chunk_seq, c.is_encrypted,
           d.doc_id, d.file_path, d.metadata
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    WHERE c.chunk_id IN ({placeholders})
    """
    
    chunk_details_list: List[Dict[str, Any]] = []
    cursor = conn.cursor()
    try:
        # Store original text_factory and set to bytes for reading chunk_text
        original_text_factory = conn.text_factory
        conn.text_factory = bytes

        cursor.execute(sql, tuple(chunk_ids))
        rows = cursor.fetchall()
        
        conn.text_factory = original_text_factory # Reset text_factory

        for row_bytes in rows:
            # Decode other text fields manually if read as bytes due to global text_factory change
            chunk_id, chunk_text_data, start_pos, end_pos, chunk_seq, is_encrypted_flag, \
            doc_id, file_path_bytes, metadata_json_bytes = row_bytes

            file_path = file_path_bytes.decode('utf-8') if isinstance(file_path_bytes, bytes) else str(file_path_bytes)
            metadata_json = metadata_json_bytes.decode('utf-8') if isinstance(metadata_json_bytes, bytes) else str(metadata_json_bytes)
            
            chunk_text_final: str
            if is_encrypted_flag:
                if encryptor and encryptor.is_enabled:
                    if not isinstance(chunk_text_data, bytes):
                        chunk_text_final = "[Encrypted - Decryption Failed: Invalid Type]"
                        ASCIIColors.error(f"Cannot decrypt chunk {chunk_id}: data type {type(chunk_text_data)}.")
                    else:
                        try:
                            chunk_text_final = encryptor.decrypt(chunk_text_data)
                        except Exception as e: # Catch specific EncryptionError from encryptor
                            chunk_text_final = "[Encrypted - Decryption Failed]"
                            ASCIIColors.error(f"Failed to decrypt chunk {chunk_id}: {e}")
                else:
                    chunk_text_final = "[Encrypted - Key Unavailable]"
            else: # Not encrypted
                if isinstance(chunk_text_data, bytes):
                    try: chunk_text_final = chunk_text_data.decode('utf-8')
                    except UnicodeDecodeError: chunk_text_final = "[Data Decode Error]"
                elif isinstance(chunk_text_data, str): chunk_text_final = chunk_text_data # Should not happen if text_factory was bytes
                else: chunk_text_final = str(chunk_text_data)


            metadata_dict: Optional[Dict[str, Any]] = None
            if metadata_json:
                try: metadata_dict = json.loads(metadata_json)
                except json.JSONDecodeError: metadata_dict = {"error": "invalid JSON"}
            
            chunk_details_list.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text_final,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "chunk_seq": chunk_seq,
                "is_encrypted": bool(is_encrypted_flag),
                "doc_id": doc_id,
                "file_path": file_path,
                "metadata": metadata_dict
            })
        return chunk_details_list
    except sqlite3.Error as e:
        if 'original_text_factory' in locals(): conn.text_factory = original_text_factory # Ensure reset on error
        raise GraphDBError(f"DB error fetching chunk details: {e}") from e
    except Exception as e: # Catch other errors like decryption issues if not handled by encryptor
        if 'original_text_factory' in locals(): conn.text_factory = original_text_factory
        raise GraphDBError(f"Unexpected error fetching chunk details: {e}") from e

# --- Functions for updating graph elements (for curation app - Phase 3.5 / 4) ---
def update_graph_node_properties_db(
    conn: sqlite3.Connection,
    node_id: int,
    new_properties: Dict[str, Any],
    merge_strategy: str = "merge_overwrite_new"
) -> bool:
    """
    Updates the properties of an existing graph node.
    Returns True if update occurred, False otherwise (e.g. node not found, no change).
    """
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT node_properties FROM graph_nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        if not row:
            ASCIIColors.warning(f"Node ID {node_id} not found for property update.")
            return False

        existing_properties_json = row[0]
        existing_properties = json.loads(existing_properties_json) if existing_properties_json else {}
        
        merged_properties: Dict[str, Any]
        if merge_strategy == "merge_overwrite_new":
            merged_properties = {**existing_properties, **new_properties}
        elif merge_strategy == "merge_prefer_existing":
            merged_properties = {**new_properties, **existing_properties}
        elif merge_strategy == "overwrite_all":
            merged_properties = new_properties.copy()
        else:
            ASCIIColors.warning(f"Unknown property_merge_strategy '{merge_strategy}'. Defaulting to 'merge_overwrite_new'.")
            merged_properties = {**existing_properties, **new_properties}

        updated_properties_json = json.dumps(merged_properties)

        if updated_properties_json == existing_properties_json:
            ASCIIColors.debug(f"Node ID {node_id} properties unchanged after merge. No update needed.")
            return False # No actual change

        cursor.execute("UPDATE graph_nodes SET node_properties = ? WHERE node_id = ?", (updated_properties_json, node_id))
        ASCIIColors.debug(f"Successfully updated properties for node ID {node_id}.")
        return True # Update occurred
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error updating properties for node ID {node_id}: {e}") from e
    except json.JSONDecodeError as e:
        raise GraphDBError(f"JSON error processing properties for node ID {node_id}: {e}") from e