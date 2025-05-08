# examples/custom_logging.py
"""
Demonstrates how to configure ascii_colors globally to customize
safe_store's logging output (and any other ascii_colors usage).
"""
import safe_store
from ascii_colors import ASCIIColors, LogLevel, FileHandler, Formatter
from pathlib import Path
import shutil

# --- Configuration ---
DB_FILE = "custom_log_store.db"
LOG_FILE = "safe_store_custom.log"
DOC_DIR = Path("temp_docs_custom_log")

# --- Helper Functions ---
def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def cleanup():
    print_header("Cleaning Up")
    db_path = Path(DB_FILE)
    log_path = Path(LOG_FILE)
    lock_path = Path(f"{DB_FILE}.lock")
    wal_path = Path(f"{DB_FILE}-wal")
    shm_path = Path(f"{DB_FILE}-shm")

    if DOC_DIR.exists(): shutil.rmtree(DOC_DIR)
    if db_path.exists(): db_path.unlink()
    if log_path.exists(): log_path.unlink()
    if lock_path.exists(): lock_path.unlink(missing_ok=True)
    if wal_path.exists(): wal_path.unlink(missing_ok=True)
    if shm_path.exists(): shm_path.unlink(missing_ok=True)
    print("- Cleanup complete.")

# --- Main Script ---
if __name__ == "__main__":
    cleanup() # Start fresh

    print_header("Configuring Global Logging")

    # 1. Set the global minimum log level (e.g., show DEBUG messages)
    ASCIIColors.set_log_level(LogLevel.DEBUG)
    print(f"- Global log level set to: {LogLevel.DEBUG.name}")

    # 2. Create a file handler to log messages to a file
    file_handler = FileHandler(LOG_FILE, encoding='utf-8')
    print(f"- Configured file logging to: {LOG_FILE}")

    # 3. Define a format for the file logger
    # Example format: Timestamp - Level Name - Message
    file_formatter = Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    print(f"- Set custom format for file logger.")

    # 4. Add the configured file handler to ascii_colors
    ASCIIColors.add_handler(file_handler)
    print(f"- Added file handler globally.")

    # Optional: Remove the default console handler if you *only* want file logging
    # default_console_handler = ASCIIColors.get_default_handler()
    # if default_console_handler:
    #    ASCIIColors.remove_handler(default_console_handler)
    #    print("- Removed default console handler.")
    # else:
    #    print("- Default console handler not found or already removed.")
    print("- Default console handler remains active (logs will go to console AND file).")


    # --- Initialize and use safe_store ---
    # It will now use the global logging configuration we just set.
    print_header("Initializing and Using safe_store")
    print("safe_store actions will now be logged according to the global settings.")
    print(f"Check the console output AND the '{LOG_FILE}' file.")

    try:
        store = safe_store.SafeStore(DB_FILE) # Uses global log level (DEBUG)

        # Prepare a sample document
        DOC_DIR.mkdir(exist_ok=True)
        doc_path = DOC_DIR / "logging_test.txt"
        doc_path.write_text("This is a test document for custom logging.", encoding='utf-8')

        with store:
            # Add the document - DEBUG messages should appear in the log file
            store.add_document(doc_path, vectorizer_name="st:all-MiniLM-L6-v2")

            # Perform a query
            results = store.query("custom logging test")
            print("\n--- Query Results ---")
            if results:
                print(f"Found {len(results)} result(s).")
            else:
                print("No results found.")

    except safe_store.ConfigurationError as e:
         print(f"\n[ERROR] Missing dependency: {e}")
         print("Please install required extras (e.g., pip install safe_store[sentence-transformers])")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e.__class__.__name__}: {e}")
    finally:
         print("\n--- End of Script ---")
         print(f"Review console output and '{LOG_FILE}' for detailed logs.")

