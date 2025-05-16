# migrate_v1_to_v2.py
import sqlite3
from pathlib import Path
from typing import Union, Optional, Any # Added for connect_db
from ascii_colors import ASCIIColors # Assuming this is in your environment

# --- Copied from your db.py for standalone script execution ---
class DatabaseError(Exception):
    pass

def connect_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.
    (Identical to your provided connect_db)
    """
    db_path_obj = Path(db_path).resolve()
    try:
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(db_path_obj),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False # Important for potential multi-threaded access
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        ASCIIColors.debug(f"Connected to database: {db_path_obj} (WAL enabled)")
        return conn
    except sqlite3.Error as e:
        msg = f"Database connection error to {db_path_obj}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def set_store_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Sets a key-value pair in the store_metadata table."""
    sql = "INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)"
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (key, value))
        # Commit is handled by the main migration transaction
        ASCIIColors.debug(f"Set store_metadata: {key} = {value}")
    except sqlite3.Error as e:
        msg = f"Error setting store metadata '{key}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e # Re-raise to be caught by migration transaction

def get_store_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Gets a value from the store_metadata table by key."""
    # Check if table exists first, as this might be called before migration
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='store_metadata';")
        if not cursor.fetchone():
            return None # Table doesn't exist yet

        sql = "SELECT value FROM store_metadata WHERE key = ?"
        cursor.execute(sql, (key,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        # If table doesn't exist yet, this might error.
        # It's safer to assume None if there's an issue before full migration.
        ASCIIColors.warning(f"Could not get store metadata for key '{key}' (may not exist yet): {e}")
        return None


def migrate_v1_to_v2(db_path: Union[str, Path]):
    """
    Migrates the SafeStore database from v1.0 schema to v2.0 schema.
    Adds graph-related tables and columns.
    """
    ASCIIColors.info(f"Attempting migration for database: {db_path}")
    ASCIIColors.warning("IMPORTANT: Please backup your database file before proceeding!")
    # input("Press Enter to continue or Ctrl+C to abort...") # Optional interactive step

    conn = None
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        # --- Version Check ---
        # Heuristic: if 'graph_nodes' table exists and 'chunks' has 'graph_processed_at',
        # it's likely v2 or newer.
        # More robust: check a version marker in store_metadata.
        
        current_version = get_store_metadata(conn, 'schema_version')
        if current_version == '2.0':
            ASCIIColors.success(f"Database '{db_path}' is already at schema version 2.0. No migration needed.")
            return
        elif current_version:
            ASCIIColors.warning(f"Database '{db_path}' has an unknown schema version: '{current_version}'. Migration for v1->v2 might not be appropriate.")
            if input("Continue anyway? (yes/NO): ").lower() != 'yes':
                ASCIIColors.info("Migration aborted by user.")
                return

        ASCIIColors.info("Proceeding with v1.0 to v2.0 migration...")

        # --- Begin Transaction ---
        # Note: Some DDL like ALTER TABLE might commit implicitly in SQLite,
        # but it's good practice to group changes.
        # We will explicitly commit at the end or rollback on error.

        # 1. Add 'graph_processed_at' column and index to 'chunks' table
        ASCIIColors.info("Checking 'chunks' table for 'graph_processed_at' column...")
        cursor.execute("PRAGMA table_info(chunks);")
        columns_in_chunks = [info[1] for info in cursor.fetchall()]

        if 'graph_processed_at' not in columns_in_chunks:
            ASCIIColors.info("Adding 'graph_processed_at' DATETIME column to 'chunks' table.")
            cursor.execute("ALTER TABLE chunks ADD COLUMN graph_processed_at DATETIME;")
            ASCIIColors.info("Creating index 'idx_chunk_graph_processed_at' on 'chunks(graph_processed_at)'.")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_graph_processed_at ON chunks (graph_processed_at);")
            ASCIIColors.green("Successfully updated 'chunks' table.")
        else:
            ASCIIColors.info("'graph_processed_at' column already exists in 'chunks'. Ensuring index exists...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_graph_processed_at ON chunks (graph_processed_at);")
            ASCIIColors.green("'chunks' table structure seems up-to-date for this part.")


        # 2. Create 'store_metadata' table
        ASCIIColors.info("Creating 'store_metadata' table if it doesn't exist...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS store_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """)
        ASCIIColors.green("'store_metadata' table created or already exists.")

        # 3. Create 'graph_nodes' table
        ASCIIColors.info("Creating 'graph_nodes' table if it doesn't exist...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_label TEXT NOT NULL,
            node_properties TEXT, 
            unique_signature TEXT UNIQUE 
        );
        """)
        ASCIIColors.info("Creating index 'idx_graph_node_label' on 'graph_nodes(node_label)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_label ON graph_nodes (node_label);")
        ASCIIColors.info("Creating index 'idx_graph_node_signature' on 'graph_nodes(unique_signature)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_signature ON graph_nodes (unique_signature);")
        ASCIIColors.green("'graph_nodes' table and indexes created or already exist.")

        # 4. Create 'graph_relationships' table
        ASCIIColors.info("Creating 'graph_relationships' table if it doesn't exist...")
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
        ASCIIColors.info("Creating index 'idx_graph_rel_source_type' on 'graph_relationships(source_node_id, relationship_type)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_source_type ON graph_relationships (source_node_id, relationship_type);")
        ASCIIColors.info("Creating index 'idx_graph_rel_target_type' on 'graph_relationships(target_node_id, relationship_type)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_target_type ON graph_relationships (target_node_id, relationship_type);")
        ASCIIColors.info("Creating index 'idx_graph_rel_type' on 'graph_relationships(relationship_type)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_type ON graph_relationships (relationship_type);")
        ASCIIColors.green("'graph_relationships' table and indexes created or already exist.")

        # 5. Create 'node_chunk_links' table
        ASCIIColors.info("Creating 'node_chunk_links' table if it doesn't exist...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_chunk_links (
            node_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            FOREIGN KEY (node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            PRIMARY KEY (node_id, chunk_id)
        );
        """)
        ASCIIColors.info("Creating index 'idx_ncl_node_id' on 'node_chunk_links(node_id)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ncl_node_id ON node_chunk_links (node_id);")
        ASCIIColors.info("Creating index 'idx_ncl_chunk_id' on 'node_chunk_links(chunk_id)'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ncl_chunk_id ON node_chunk_links (chunk_id);")
        ASCIIColors.green("'node_chunk_links' table and indexes created or already exist.")

        # 6. Update schema version in store_metadata
        ASCIIColors.info("Updating schema version to 2.0 in 'store_metadata'.")
        set_store_metadata(conn, 'schema_version', '2.0') # Uses the helper

        # --- Commit Transaction ---
        conn.commit()
        ASCIIColors.success(f"Database migration to v2.0 completed successfully for: {db_path}")

    except sqlite3.Error as e:
        ASCIIColors.error(f"SQLite error during migration: {e}")
        if conn:
            ASCIIColors.warning("Rolling back changes due to error.")
            conn.rollback()
        raise DatabaseError(f"Migration failed: {e}") from e
    except DatabaseError as e: # Catch custom DatabaseError from helpers
        ASCIIColors.error(f"Database operation error during migration: {e}")
        if conn:
            ASCIIColors.warning("Rolling back changes due to error.")
            conn.rollback()
        raise # Re-raise after rollback
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred during migration: {e}", exc_info=True)
        if conn:
            ASCIIColors.warning("Rolling back changes due to error.")
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            ASCIIColors.debug("Database connection closed.")

if __name__ == "__main__":
    # --- Configuration ---
    # !!! IMPORTANT: REPLACE WITH THE ACTUAL PATH TO YOUR DATABASE !!!
    DATABASE_FILE_PATH = "your_safestore.db" 
    # Example: DATABASE_FILE_PATH = "/path/to/your/safe_store.db"
    # Example: DATABASE_FILE_PATH = Path.home() / ".safe_store" / "default.db"


    if DATABASE_FILE_PATH == "your_safestore.db":
        ASCIIColors.critical("Please update 'DATABASE_FILE_PATH' in the script with the actual path to your database.")
    else:
        db_path = Path(DATABASE_FILE_PATH)
        if not db_path.exists():
            ASCIIColors.warning(f"Database file {db_path} does not exist. "
                                "If this is a new setup, the main application will initialize it to v2.0.")
            ASCIIColors.info("This migration script is intended for existing v1.0 databases.")
        else:
            try:
                migrate_v1_to_v2(db_path)
            except Exception as e:
                ASCIIColors.critical(f"Migration process encountered an unrecoverable error: {e}")
