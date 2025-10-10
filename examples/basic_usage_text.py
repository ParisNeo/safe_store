from safe_store import SafeStore

# Define the vectorizer name and its configuration
vectorizer_name = "st"
# vectorizer_config = {"model": "all-MiniLM-L6-v2"}
# vectorizer_config = {"model": "all-MiniLM-L12-v2"}
vectorizer_config = {"model": "LaBSE"}

# Initialize the store
ss = SafeStore("basic_usage_store.db", "my database", "a cool database")

# Add text with the specified vectorizer
i=0
ss.add_text(f"{i}", "Hello there", vectorizer_name=vectorizer_name, vectorizer_config=vectorizer_config)
i += 1
ss.add_text(f"{i}", "What time is it?", vectorizer_name=vectorizer_name, vectorizer_config=vectorizer_config)

# Query the store
results = ss.query("hi", vectorizer_name=vectorizer_name, vectorizer_config=vectorizer_config)
for r in results:
    print(f"{r['chunk_text']}:{r['similarity_percent']}%")

# You can still compare similarity percentages
print(results[0]['similarity_percent']-results[1]['similarity_percent'])

# Vectorize text directly
v1 = ss.vectorize_text("Hello there", vectorizer_name=vectorizer_name, vectorizer_config=vectorizer_config)
v2 = ss.vectorize_text("What time is it?", vectorizer_name=vectorizer_name, vectorizer_config=vectorizer_config)