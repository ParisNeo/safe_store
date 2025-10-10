# examples/dynamic_model_selection.py
"""
This example demonstrates how to use the `list_available_models` class method
to dynamically discover and select a model from a running Ollama instance,
and then use it to create and query a SafeStore.
"""
import safe_store
from pathlib import Path
import shutil

# --- Configuration ---
DB_FILE = "dynamic_ollama_store.db"
# This example assumes an Ollama server is running at the default host.
# If your Ollama server is elsewhere, you can specify it:
# OLLAMA_HOST = "http://192.168.1.10:11434"
OLLAMA_HOST = "http://localhost:11434"

def cleanup():
    """Removes the database file from previous runs."""
    Path(DB_FILE).unlink(missing_ok=True)
    Path(f"{DB_FILE}.lock").unlink(missing_ok=True)
    print(f"--- Cleaned up old database file: {DB_FILE} ---")

if __name__ == "__main__":
    cleanup()
    
    # --- 1. Discover available Ollama models ---
    print(f"\n--- Step 1: Discovering models from Ollama at {OLLAMA_HOST} ---")
    try:
        # Use the class method to get a list of models from the Ollama server
        available_models = safe_store.SafeStore.list_available_models(
            vectorizer_name="ollama",
            host=OLLAMA_HOST # Pass the host to the method
        )
        
        if not available_models:
            print("\n[ERROR] No models found on the Ollama server.")
            print("Please make sure Ollama is running and you have pulled at least one model, for example:")
            print("  ollama pull nomic-embed-text")
            exit()

        print("Found available models:")
        for model in available_models:
            print(f"  - {model}")

    except safe_store.VectorizationError as e:
        print(f"\n[ERROR] Could not connect to the Ollama server: {e}")
        print("Please ensure your Ollama server is running and accessible.")
        exit()
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        exit()

    # --- 2. Select a model and configure the store ---
    print("\n--- Step 2: Selecting a model ---")
    # For this example, we'll just pick the first model from the list.
    # In a real application, you might let the user choose.
    selected_model = available_models[0]
    print(f"Selected model: {selected_model}")

    # Prepare the configuration for the SafeStore instance
    vectorizer_name = "ollama"
    vectorizer_config = {
        "model": selected_model,
        "host": OLLAMA_HOST
    }

    # --- 3. Initialize SafeStore with the selected model ---
    print("\n--- Step 3: Initializing SafeStore ---")
    store = safe_store.SafeStore(
        db_path=DB_FILE,
        vectorizer_name=vectorizer_name,
        vectorizer_config=vectorizer_config,
        log_level=safe_store.LogLevel.INFO
    )
    print("SafeStore initialized successfully.")

    # --- 4. Use the store to add and query text ---
    print("\n--- Step 4: Adding text and querying ---")
    with store:
        # Add some sample text
        store.add_text(
            unique_id="tech-report-01",
            text="The new quantum processor shows a 200% performance increase in benchmark tests."
        )
        store.add_text(
            unique_id="finance-summary-01",
            text="Quarterly earnings are up by 15%, driven by the new hardware division."
        )
        print("Added two text entries to the store.")

        # Perform a query
        query_text = "What were the results of the processor benchmarks?"
        print(f"\nQuerying for: '{query_text}'")
        results = store.query(query_text, top_k=1)

        if results:
            result = results[0]
            print(f"Found a relevant chunk with {result['similarity_percent']:.2f}% similarity:")
            print(f" -> Text: '{result['chunk_text']}'")
        else:
            print("No relevant results found for the query.")
            
    print("\n--- Example Finished ---")