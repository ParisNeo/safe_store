# [FINAL & COMPLETE] safe_store/core/db.py
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict, Union
import json
from ascii_colors import ASCIIColors
from .exceptions import DatabaseError, GraphDBError

# --- Type Hinting ---
SQLQuery = str
SQLParams = Union[Tuple[Any, ...], Dict[str, Any]]

# --- Adapters for NumPy arrays ---
def adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    """Converts a NumPy array to SQLite Binary data."""
    return sqlite3.Binary(arr.tobytes())

sqlite3.register_adapter(np.ndarray, adapt_array)

def reconstruct_vector(blob: bytes, dtype_str: str) -> np.ndarray:
    """Safely reconstructs a NumPy array from SQLite blob data and dtype string."""
    try:
        if any(char in dtype_str for char in ';()[]{}'):
            raise ValueError(f"Invalid characters found in dtype string: '{dtype_str}'")
        dtype = np.dtype(dtype_str)
        return np.frombuffer(blob, dtype=dtype)
    except (TypeError, ValueError) as e:
        msg = f"Failed to reconstruct vector: invalid or unsafe dtype '{dtype_str}' or blob data mismatch. Error: {e}"
        ASCIIColors.error(msg)
        raise DatabaseError(msg) from e

# --- Core DB Functions ---
def connect_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database, enabling WAL mode and foreign keys."""
    db_path_str = str(db_path)
    try:
        if db_path_str.lower() == ":memory:":
            conn = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        else:
            db_path_obj = Path(db_path_str).resolve()
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path_obj), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
        
        conn.execute("PRAGMA foreign_keys = ON;")
        ASCIIColors.debug(f"Connected to database: {db_path_str} (WAL enabled, Foreign Keys ON)")
        return conn
    except (sqlite3.Error, OSError) as e:
        msg = f"Database connection error to {db_path_str}: {e}"
        raise DatabaseError(msg) from e

def initialize_schema(conn: sqlite3.Connection) -> None:
    """Initializes the database schema including all tables for documents, vectors, and the knowledge graph."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT UNIQUE NOT NULL, file_hash TEXT,
            full_text TEXT, metadata TEXT, added_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_file_path ON documents (file_path);")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectorization_methods (
            method_id INTEGER PRIMARY KEY AUTOINCREMENT, method_name TEXT UNIQUE NOT NULL, method_type TEXT NOT NULL,
            vector_dim INTEGER NOT NULL, vector_dtype TEXT NOT NULL, params TEXT
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_method_name ON vectorization_methods (method_name);")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL, chunk_text BLOB NOT NULL,
            start_pos INTEGER NOT NULL, end_pos INTEGER NOT NULL, chunk_seq INTEGER NOT NULL, tags TEXT,
            is_encrypted INTEGER DEFAULT 0 NOT NULL, encryption_metadata BLOB, graph_processed_at DATETIME,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON chunks (doc_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_graph_processed_at ON chunks (graph_processed_at);")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id INTEGER NOT NULL, method_id INTEGER NOT NULL,
            vector_data BLOB NOT NULL, FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            FOREIGN KEY (method_id) REFERENCES vectorization_methods (method_id) ON DELETE CASCADE,
            UNIQUE (chunk_id, method_id)
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_method_id ON vectors (method_id);")
        
        cursor.execute("CREATE TABLE IF NOT EXISTS store_metadata (key TEXT PRIMARY KEY, value TEXT);")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT, node_label TEXT NOT NULL, node_properties TEXT,
            unique_signature TEXT UNIQUE, node_vector BLOB
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_label ON graph_nodes (node_label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_signature ON graph_nodes (unique_signature);")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_relationships (
            relationship_id INTEGER PRIMARY KEY AUTOINCREMENT, source_node_id INTEGER NOT NULL,
            target_node_id INTEGER NOT NULL, relationship_type TEXT NOT NULL, relationship_properties TEXT,
            FOREIGN KEY (source_node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (target_node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE
        );""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_source_type ON graph_relationships (source_node_id, relationship_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_target_type ON graph_relationships (target_node_id, relationship_type);")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_chunk_links (
            node_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL,
            FOREIGN KEY (node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            PRIMARY KEY (node_id, chunk_id)
        );""")

        conn.commit()
        ASCIIColors.debug("Database schema verified/initialized successfully.")
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Schema initialization error: {e}") from e

# --- Original SafeStore CRUD Functions ---
def add_document_record(conn: sqlite3.Connection, file_path: str, full_text: str, file_hash: Optional[str] = None, metadata: Optional[str] = None) -> int:
    sql = "INSERT INTO documents (file_path, file_hash, full_text, metadata) VALUES (?, ?, ?, ?)"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (file_path, file_hash, full_text, metadata))
        doc_id = cursor.lastrowid
        if doc_id is None: raise DatabaseError(f"Failed to get lastrowid for document '{file_path}'.")
        return doc_id
    except sqlite3.IntegrityError as e:
        existing_id = get_document_id_by_path(conn, file_path)
        if existing_id is not None: return existing_id
        raise DatabaseError(f"IntegrityError for '{file_path}', but could not retrieve existing ID.") from e
    except sqlite3.Error as e:
        raise DatabaseError(f"Error inserting document '{file_path}': {e}") from e

def get_document_id_by_path(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (file_path,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching doc_id for '{file_path}': {e}") from e

def add_or_get_vectorization_method(conn: sqlite3.Connection, name: str, type: str, dim: int, dtype: str, params: Optional[str] = None) -> int:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (name,))
        result = cursor.fetchone()
        if result: return result[0]
        cursor.execute("INSERT INTO vectorization_methods (method_name, method_type, vector_dim, vector_dtype, params) VALUES (?, ?, ?, ?, ?)",
                       (name, type, dim, dtype, params or '{}'))
        method_id = cursor.lastrowid
        if method_id is None: raise DatabaseError(f"Failed to get lastrowid for vectorizer '{name}'.")
        return method_id
    except sqlite3.Error as e:
        raise DatabaseError(f"Error adding/getting vectorizer '{name}': {e}") from e

def add_chunk_record(conn: sqlite3.Connection, doc_id: int, text: Union[str, bytes], start: int, end: int, seq: int, tags: Optional[str] = None, is_encrypted: bool = False, encryption_metadata: Optional[bytes] = None) -> int:
    sql = "INSERT INTO chunks (doc_id, chunk_text, start_pos, end_pos, chunk_seq, tags, is_encrypted, encryption_metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (doc_id, text, start, end, seq, tags, 1 if is_encrypted else 0, encryption_metadata))
        chunk_id = cursor.lastrowid
        if chunk_id is None: raise DatabaseError(f"Failed to get lastrowid for chunk (doc={doc_id}, seq={seq}).")
        return chunk_id
    except sqlite3.Error as e:
        raise DatabaseError(f"Error inserting chunk (doc={doc_id}, seq={seq}): {e}") from e

def add_vector_record(conn: sqlite3.Connection, chunk_id: int, method_id: int, vector: np.ndarray) -> None:
    sql = "INSERT OR IGNORE INTO vectors (chunk_id, method_id, vector_data) VALUES (?, ?, ?)"
    try:
        conn.execute(sql, (chunk_id, method_id, vector))
    except sqlite3.Error as e:
        raise DatabaseError(f"Error inserting vector (chunk={chunk_id}, method={method_id}): {e}") from e

# --- Metadata Functions ---
def set_store_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    try:
        conn.execute("INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)", (key, value))
    except sqlite3.Error as e:
        raise DatabaseError(f"Error setting store metadata '{key}': {e}") from e

def get_store_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        cursor = conn.execute("SELECT value FROM store_metadata WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        raise DatabaseError(f"Error getting store metadata for key '{key}': {e}") from e

# --- Graph Node Functions ---
def add_or_update_graph_node(conn: sqlite3.Connection, label: str, properties: Dict[str, Any], unique_signature: str) -> int:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT node_id, node_properties FROM graph_nodes WHERE unique_signature = ?", (unique_signature,))
        row = cursor.fetchone()
        if row:
            node_id, existing_props_json = row
            existing_props = json.loads(existing_props_json) if existing_props_json else {}
            merged_props = {**existing_props, **properties}
            if json.dumps(merged_props) != existing_props_json:
                cursor.execute("UPDATE graph_nodes SET node_properties = ? WHERE node_id = ?", (json.dumps(merged_props), node_id))
            return node_id
        else:
            props_json = json.dumps(properties)
            cursor.execute("INSERT INTO graph_nodes (node_label, node_properties, unique_signature) VALUES (?, ?, ?)", (label, props_json, unique_signature))
            new_id = cursor.lastrowid
            if new_id is None: raise GraphDBError("Failed to get lastrowid for new node.")
            return new_id
    except sqlite3.Error as e:
        raise GraphDBError(f"Error adding/updating graph node with signature '{unique_signature}': {e}") from e

def get_graph_node_by_signature(conn: sqlite3.Connection, signature: str) -> Optional[int]:
    try:
        cursor = conn.execute("SELECT node_id FROM graph_nodes WHERE unique_signature = ?", (signature,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        raise GraphDBError(f"Error fetching node by signature '{signature}': {e}") from e

def find_similar_nodes_by_property(conn, label, prop_key, prop_value, limit=5) -> List[Dict[str, Any]]:
    safe_prop_key = ''.join(c for c in prop_key if c.isalnum() or c in '_')
    if safe_prop_key != prop_key: raise GraphDBError(f"Invalid characters in property key: {prop_key}")
    sql = f"SELECT node_id, node_properties FROM graph_nodes WHERE node_label = ? AND json_extract(node_properties, '$.{safe_prop_key}') LIKE ? LIMIT ?"
    nodes = []
    try:
        cursor = conn.execute(sql, (label, f"%{prop_value}%", limit))
        for row in cursor.fetchall():
            nodes.append({"node_id": row[0], "properties": json.loads(row[1]) if row[1] else {}})
        return nodes
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error finding similar nodes for prop '{prop_key}': {e}") from e

def find_node_by_label_and_property_value(conn: sqlite3.Connection, label: str, value: str, limit: int = 1) -> List[Dict[str, Any]]:
    like_pattern = f"%{value}%"
    sql = "SELECT node_id, node_label, node_properties, unique_signature FROM graph_nodes WHERE node_label = ? AND (json_extract(node_properties, '$.name') LIKE ? OR json_extract(node_properties, '$.title') LIKE ?) LIMIT ?;"
    nodes_found = []
    try:
        cursor = conn.execute(sql, (label, like_pattern, like_pattern, limit))
        for row in cursor.fetchall():
            nodes_found.append({"node_id": row[0], "label": row[1], "properties": json.loads(row[2]) if row[2] else {}, "unique_signature": row[3]})
        return nodes_found
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error finding node by multi-property value LIKE '{value}': {e}") from e

def get_node_details_db(conn: sqlite3.Connection, node_id: int) -> Optional[Dict[str, Any]]:
    try:
        cursor = conn.execute("SELECT node_id, node_label, node_properties, unique_signature FROM graph_nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        if row:
            return {"node_id": row[0], "label": row[1], "properties": json.loads(row[2]) if row[2] else {}, "unique_signature": row[3]}
        return None
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error getting details for node {node_id}: {e}") from e

def get_nodes_by_label_db(conn: sqlite3.Connection, label: str, limit: int = 100) -> List[Dict[str, Any]]:
    nodes_found = []
    try:
        cursor = conn.execute("SELECT node_id, node_label, node_properties, unique_signature FROM graph_nodes WHERE node_label = ? LIMIT ?", (label, limit))
        for row in cursor.fetchall():
            nodes_found.append({"node_id": row[0], "label": row[1], "properties": json.loads(row[2]) if row[2] else {}, "unique_signature": row[3]})
        return nodes_found
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error getting nodes by label '{label}': {e}") from e

def update_graph_node_label_db(conn: sqlite3.Connection, node_id: int, new_label: str):
    try:
        conn.execute("UPDATE graph_nodes SET node_label = ? WHERE node_id = ?", (new_label, node_id))
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error updating label for node {node_id}: {e}") from e

def update_graph_node_properties_db(conn: sqlite3.Connection, node_id: int, new_properties: Dict, merge_strategy: str) -> bool:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT node_properties FROM graph_nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        if not row: return False
        
        existing_props = json.loads(row[0]) if row[0] else {}
        if merge_strategy == "overwrite_all": merged_props = new_properties
        else: merged_props = {**existing_props, **new_properties}
        
        props_json = json.dumps(merged_props)
        cursor.execute("UPDATE graph_nodes SET node_properties = ? WHERE node_id = ?", (props_json, node_id))
        return cursor.rowcount > 0
    except (sqlite3.Error, json.JSONDecodeError) as e:
        raise GraphDBError(f"DB error updating properties for node {node_id}: {e}") from e

def delete_graph_node_and_relationships_db(conn: sqlite3.Connection, node_id: int) -> int:
    try:
        cursor = conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
        return cursor.rowcount
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error deleting node {node_id}: {e}") from e

# --- Graph Relationship Functions ---
def add_graph_relationship(conn: sqlite3.Connection, source_node_id: int, target_node_id: int, rel_type: str, properties_json: str) -> int:
    try:
        cursor = conn.execute("INSERT INTO graph_relationships (source_node_id, target_node_id, relationship_type, relationship_properties) VALUES (?, ?, ?, ?)",
                              (source_node_id, target_node_id, rel_type, properties_json))
        new_id = cursor.lastrowid
        if new_id is None: raise GraphDBError("Failed to get lastrowid for new relationship.")
        return new_id
    except sqlite3.Error as e:
        raise GraphDBError(f"Error adding relationship '{rel_type}': {e}") from e

def get_relationships_for_node_db(conn, node_id, relationship_type, direction, limit) -> List[Dict[str, Any]]:
    conditions, params = [], []
    if direction == "any":
        conditions.append("(r.source_node_id = ? OR r.target_node_id = ?)")
        params.extend([node_id, node_id])
    elif direction == "outgoing":
        conditions.append("r.source_node_id = ?"); params.append(node_id)
    elif direction == "incoming":
        conditions.append("r.target_node_id = ?"); params.append(node_id)
    if relationship_type:
        conditions.append("r.relationship_type = ?"); params.append(relationship_type)
    
    where_clause = " AND ".join(conditions)
    sql = f"""
    SELECT r.relationship_id, r.source_node_id, r.target_node_id, r.relationship_type, r.relationship_properties,
           s.node_label as source_label, s.node_properties as source_properties,
           t.node_label as target_label, t.node_properties as target_properties
    FROM graph_relationships r JOIN graph_nodes s ON r.source_node_id = s.node_id JOIN graph_nodes t ON r.target_node_id = t.node_id
    WHERE {where_clause} LIMIT ?;
    """
    params.append(limit)
    relationships = []
    try:
        cursor = conn.execute(sql, tuple(params))
        for row in cursor.fetchall():
            relationships.append({
                "relationship_id": row[0], "source_node_id": row[1], "target_node_id": row[2], "type": row[3],
                "properties": json.loads(row[4]) if row[4] else {},
                "source_node": {"node_id": row[1], "label": row[5], "properties": json.loads(row[6]) if row[6] else {}},
                "target_node": {"node_id": row[2], "label": row[7], "properties": json.loads(row[8]) if row[8] else {}}
            })
        return relationships
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error fetching relationships for node {node_id}: {e}") from e

def delete_graph_relationship_db(conn: sqlite3.Connection, relationship_id: int) -> int:
    try:
        cursor = conn.execute("DELETE FROM graph_relationships WHERE relationship_id = ?", (relationship_id,))
        return cursor.rowcount
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error deleting relationship {relationship_id}: {e}") from e

# --- Graph Vector Functions ---
def enable_vector_search_on_graph_nodes(conn: sqlite3.Connection, vector_dim: int) -> None:
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(graph_nodes);")
        if 'node_vector' not in [col[1] for col in cursor.fetchall()]:
            cursor.execute("ALTER TABLE graph_nodes ADD COLUMN node_vector BLOB;")
    except sqlite3.Error as e:
        raise GraphDBError(f"Failed to enable vector search on graph nodes: {e}") from e

def update_node_vector(conn: sqlite3.Connection, node_id: int, vector: np.ndarray) -> None:
    try:
        conn.execute("UPDATE graph_nodes SET node_vector = ? WHERE node_id = ?", (vector, node_id))
    except sqlite3.Error as e:
        raise GraphDBError(f"Failed to update vector for node {node_id}: {e}") from e

def search_graph_nodes_by_vector(conn: sqlite3.Connection, query_vector: np.ndarray, top_k: int) -> List[int]:
    try:
        cursor = conn.execute("SELECT node_id, node_vector FROM graph_nodes WHERE node_vector IS NOT NULL")
        all_nodes = cursor.fetchall()
        if not all_nodes: return []
        node_ids, vectors_blob = zip(*all_nodes)
        dtype_str = str(query_vector.dtype)
        candidate_vectors = np.array([reconstruct_vector(blob, dtype_str) for blob in vectors_blob])
        norm_query = query_vector / np.linalg.norm(query_vector)
        norm_candidates = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        scores = np.dot(norm_candidates, norm_query)
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        return [node_ids[i] for i in top_indices]
    except (sqlite3.Error, DatabaseError) as e:
        raise GraphDBError(f"Failed to search graph nodes by vector: {e}") from e

# --- Chunk and Link Functions ---
def link_node_to_chunk(conn: sqlite3.Connection, node_id: int, chunk_id: int) -> None:
    try:
        conn.execute("INSERT OR IGNORE INTO node_chunk_links (node_id, chunk_id) VALUES (?, ?)", (node_id, chunk_id))
    except sqlite3.Error as e:
        raise GraphDBError(f"Error linking node {node_id} to chunk {chunk_id}: {e}") from e

def get_chunk_ids_for_nodes_db(conn: sqlite3.Connection, node_ids: List[int]) -> Dict[int, List[int]]:
    if not node_ids: return {}
    placeholders = ",".join("?" * len(node_ids))
    sql = f"SELECT node_id, chunk_id FROM node_chunk_links WHERE node_id IN ({placeholders})"
    results: Dict[int, List[int]] = {node_id: [] for node_id in node_ids}
    try:
        cursor = conn.execute(sql, tuple(node_ids))
        for row in cursor.fetchall():
            results[row[0]].append(row[1])
        return results
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error fetching chunk links for nodes: {e}") from e

def get_chunk_details_db(conn, chunk_ids, encryptor):
    if not chunk_ids: return []
    placeholders = ",".join("?" * len(chunk_ids))
    sql = f"SELECT c.chunk_id, c.chunk_text, c.start_pos, c.end_pos, c.is_encrypted, d.file_path FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE c.chunk_id IN ({placeholders})"
    details = []
    try:
        original_factory = conn.text_factory
        conn.text_factory = bytes
        cursor = conn.execute(sql, tuple(chunk_ids))
        for row in cursor.fetchall():
            text_data = row[1]
            if row[4] and encryptor and encryptor.is_enabled:
                text = encryptor.decrypt(text_data)
            else:
                text = text_data.decode('utf-8')
            details.append({"chunk_id": row[0], "chunk_text": text, "start_pos": row[2], "end_pos": row[3], "file_path": row[5].decode('utf-8')})
        conn.text_factory = original_factory
        return details
    except sqlite3.Error as e:
        if 'original_factory' in locals(): conn.text_factory = original_factory
        raise GraphDBError(f"DB error fetching chunk details: {e}") from e

def mark_chunks_graph_processed(conn: sqlite3.Connection, chunk_ids: List[int]) -> None:
    if not chunk_ids: return
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(f"UPDATE chunks SET graph_processed_at = CURRENT_TIMESTAMP WHERE chunk_id IN ({placeholders})", tuple(chunk_ids))
    except sqlite3.Error as e:
        raise GraphDBError(f"Error marking chunks as graph processed: {e}") from e

# --- Complex Graph Operations ---
def merge_nodes_db(conn, source_node_id, target_node_id):
    if source_node_id == target_node_id: return
    try:
        conn.execute("UPDATE graph_relationships SET target_node_id = ? WHERE target_node_id = ?", (target_node_id, source_node_id))
        conn.execute("UPDATE graph_relationships SET source_node_id = ? WHERE source_node_id = ?", (target_node_id, source_node_id))
        conn.execute("INSERT OR IGNORE INTO node_chunk_links (node_id, chunk_id) SELECT ?, chunk_id FROM node_chunk_links WHERE node_id = ?", (target_node_id, source_node_id))
        conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (source_node_id,))
    except sqlite3.Error as e:
        raise GraphDBError(f"DB error merging node {source_node_id} into {target_node_id}: {e}") from e