# migrate_v1_to_v2_argparse.py
import sqlite3
from pathlib import Path
from typing import Union, Optional, Any # Added for connect_db
import argparse # New import
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
        ASCIIColors.debug(f"Set store_metadata: {key} = {value}")
    except sqlite3.Error as e:
        msg = f"Error setting store metadata '{key}': {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise DatabaseError(msg) from e

def get_store_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Gets a value from the store_metadata table by key."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='store_metadata';")
        if not cursor.fetchone():
            return None 

        sql = "SELECT value FROM store_metadata WHERE key = ?"
        cursor.execute(sql, (key,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        ASCIIColors.warning(f"Could not get store metadata for key '{key}' (may not exist yet): {e}")
        return None


def migrate_v1_to_v2(db_path: Path, auto_yes: bool = False):
    """
    Migrates the SafeStore database from v1.0 schema to v2.0 schema.
    Adds graph-related tables and columns.

    Args:
        db_path: Path object to the database file.
        auto_yes: If True, skips interactive prompts.
    """
    ASCIIColors.info(f"Attempting migration for database: {db_path}")
    
    if not db_path.exists():
        ASCIIColors.error(f"Database file {db_path} does not exist. Cannot migrate.")
        ASCIIColors.info("If this is a new setup, the main application will initialize it to v2.0.")
        return False # Indicate failure

    if not auto_yes:
        ASCIIColors.warning("IMPORTANT: Please backup your database file before proceeding!")
        try:
            # Check if running in a non-interactive environment
            if not Path("/dev/tty").is_char_device(): # Crude check for tty
                 ASCIIColors.info("Non-interactive environment detected, proceeding without prompt.")
            elif input("Press Enter to continue or Ctrl+C to abort..."): # Anything entered aborts
                ASCIIColors.info("Migration aborted by user input.")
                return False
        except (EOFError, KeyboardInterrupt): # Handle cases where input() is not available or Ctrl+C
            ASCIIColors.info("Migration aborted.")
            return False
        except Exception: # Catch other potential issues with input() in some environments
            ASCIIColors.info("Could not get interactive input, proceeding with caution. Use --yes to bypass.")


    conn = None
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        current_version = get_store_metadata(conn, 'schema_version')
        if current_version == '2.0':
            ASCIIColors.success(f"Database '{db_path}' is already at schema version 2.0. No migration needed.")
            return True # Indicate success (no action needed)
        elif current_version:
            ASCIIColors.warning(f"Database '{db_path}' has an existing schema version: '{current_version}'.")
            ASCIIColors.warning("This script is designed for v1.0 (no version) to v2.0 migration.")
            if not auto_yes:
                if input(f"Continue migration from '{current_version}' to '2.0'? (yes/NO): ").lower() != 'yes':
                    ASCIIColors.info("Migration aborted by user.")
                    return False
            else:
                ASCIIColors.info(f"Auto-proceeding with migration from '{current_version}' to '2.0'.")


        ASCIIColors.info("Proceeding with v1.0 to v2.0 migration...")

        # --- Begin Transaction ---
        # Explicitly begin transaction for DDL (though some might auto-commit)
        # For SQLite, it's generally better to execute DDL outside explicit transactions if possible,
        # or commit after each significant DDL if needed. Here, we group them.
        # However, ALTER TABLE ADD COLUMN is transactional.
        # For simplicity, we'll use one commit/rollback block.

        cursor.execute("PRAGMA foreign_keys=OFF;") # Disable FKs during schema changes

        # 1. Add 'graph_processed_at' column and index to 'chunks' table
        ASCIIColors.info("Updating 'chunks' table...")
        cursor.execute("PRAGMA table_info(chunks);")
        columns_in_chunks = [info[1] for info in cursor.fetchall()]
        if 'graph_processed_at' not in columns_in_chunks:
            cursor.execute("ALTER TABLE chunks ADD COLUMN graph_processed_at DATETIME;")
            ASCIIColors.info("Added 'graph_processed_at' column to 'chunks'.")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_graph_processed_at ON chunks (graph_processed_at);")
        ASCIIColors.green("'chunks' table updated and indexed.")

        # 2. Create 'store_metadata' table
        ASCIIColors.info("Creating 'store_metadata' table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS store_metadata (key TEXT PRIMARY KEY, value TEXT);
        """)
        ASCIIColors.green("'store_metadata' table ensured.")

        # 3. Create 'graph_nodes' table and indexes
        ASCIIColors.info("Creating 'graph_nodes' table and indexes...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT, node_label TEXT NOT NULL,
            node_properties TEXT, unique_signature TEXT UNIQUE);
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_label ON graph_nodes (node_label);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_signature ON graph_nodes (unique_signature);")
        ASCIIColors.green("'graph_nodes' table and indexes ensured.")

        # 4. Create 'graph_relationships' table and indexes
        ASCIIColors.info("Creating 'graph_relationships' table and indexes...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_relationships (
            relationship_id INTEGER PRIMARY KEY AUTOINCREMENT, source_node_id INTEGER NOT NULL,
            target_node_id INTEGER NOT NULL, relationship_type TEXT NOT NULL,
            relationship_properties TEXT,
            FOREIGN KEY (source_node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (target_node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE);
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_source_type ON graph_relationships (source_node_id, relationship_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_target_type ON graph_relationships (target_node_id, relationship_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_type ON graph_relationships (relationship_type);")
        ASCIIColors.green("'graph_relationships' table and indexes ensured.")

        # 5. Create 'node_chunk_links' table and indexes
        ASCIIColors.info("Creating 'node_chunk_links' table and indexes...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_chunk_links (
            node_id INTEGER NOT NULL, chunk_id INTEGER NOT NULL,
            FOREIGN KEY (node_id) REFERENCES graph_nodes (node_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            PRIMARY KEY (node_id, chunk_id));
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ncl_node_id ON node_chunk_links (node_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ncl_chunk_id ON node_chunk_links (chunk_id);")
        ASCIIColors.green("'node_chunk_links' table and indexes ensured.")

        # 6. Update schema version in store_metadata
        ASCIIColors.info("Updating schema version to 2.0 in 'store_metadata'.")
        # Use direct execute here as set_store_metadata might be part of the transaction
        cursor.execute("INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)", ('schema_version', '2.0'))

        cursor.execute("PRAGMA foreign_keys=ON;") # Re-enable FKs

        conn.commit()
        ASCIIColors.success(f"Database migration to v2.0 completed successfully for: {db_path}")
        return True # Indicate success

    except sqlite3.Error as e:
        ASCIIColors.error(f"SQLite error during migration: {e}")
        if conn:
            ASCIIColors.warning("Rolling back changes due to error.")
            conn.rollback()
        return False # Indicate failure
    except DatabaseError as e:
        ASCIIColors.error(f"Database operation error during migration: {e}")
        if conn:
            ASCIIColors.warning("Rolling back changes due to error.")
            conn.rollback()
        return False # Indicate failure
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred during migration: {e}", exc_info=True)
        if conn:
            ASCIIColors.warning("Rolling back changes due to error.")
            conn.rollback()
        return False # Indicate failure
    finally:
        if conn:
            conn.close()
            ASCIIColors.debug("Database connection closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SafeStore SQLite database from v1.0 schema to v2.0 schema.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  python %(prog)s /path/to/your/safestore.db
  python %(prog)s my_database.sqlite --yes

This script adds new tables and columns for graph database functionality.
It is designed to be run on a database created with a pre-graph version of SafeStore.
Ensure you have a backup of your database before running this script.
"""
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the SQLite database file to migrate."
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically answer 'yes' to confirmation prompts (use with caution)."
    )
    # You could add a --force flag here if needed, but it's generally risky for migrations
    # parser.add_argument(
    #     "--force",
    #     action="store_true",
    #     help="Force migration even if schema version is unexpected (EXTREMELY DANGEROUS)."
    # )

    args = parser.parse_args()

    if migrate_v1_to_v2(args.db_path, auto_yes=args.yes):
        ASCIIColors.highlight("Migration process finished.")
    else:
        ASCIIColors.critical("Migration process failed or was aborted. Please check the logs.")
        exit(1) # Exit with error code if migration failed

if __name__ == "__main__":
    main()
