from safe_store import SafeStore
from pathlib import Path

# --- Cleanup ---
# Ensure the database from previous runs is removed for a clean start
db_file = Path("basic_usage_store.db")
db_file.unlink(missing_ok=True)
Path(f"{db_file}.lock").unlink(missing_ok=True)


# --- 1. Initialize the store with a fixed configuration ---
# All indexing parameters (vectorizer, chunking, cleaning) are now defined
# when the SafeStore instance is created.
print("--- Initializing SafeStore with a fixed configuration ---")
ss = SafeStore(
    db_path=db_file,
    name="my_database",
    description="A cool database demonstrating fixed configuration",
    
    # Vectorizer Configuration
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    
    # Chunking and Processing Configuration
    chunk_size=10,             # Small chunk size for demonstration (in tokens)
    chunk_overlap=2,           # Small overlap (in tokens)
    chunking_strategy='token', # Use the model's tokenizer for chunking
    expand_before=5,           # Add 5 tokens of context before the vectorized chunk
    expand_after=5,            # Add 5 tokens of context after the vectorized chunk
    text_cleaner='basic'       # Use the built-in basic text cleaner
)

# --- 2. Add content ---
# The add_text method is now much simpler. It uses the configuration
# provided when the store was created.
print("\n--- Adding content to the store ---")
text_to_add = "The quick brown fox jumps over the lazy dog. This sentence is used to demonstrate all letters of the alphabet. It is a classic pangram."
ss.add_text(
    unique_id="pangram_text",
    text=text_to_add
)
print(f"Added text with ID 'pangram_text'.")


# --- 3. Query the store ---
# The query method also uses the instance's configured vectorizer automatically.
print("\n--- Querying the store ---")
query = "a speedy fox"
results = ss.query(query)

print(f"Query: '{query}'")
for r in results:
    print("-" * 20)
    print(f"Similarity: {r['similarity_percent']:.2f}%")
    # The 'chunk_text' returned is the EXPANDED text for better context.
    print(f"Stored (expanded) chunk: '{r['chunk_text']}'")

# --- 4. Vectorize text directly (optional) ---
# This method uses the instance's configured vectorizer.
print("\n--- Vectorizing a new sentence directly ---")
v1 = ss.vectorize_text("Hello there")
print(f"Successfully vectorized a new sentence. Vector dimension: {v1.shape}")

# The store is automatically closed if used in a 'with' block,
# or you can call ss.close() manually.
ss.close()
print("\n--- Example finished ---")