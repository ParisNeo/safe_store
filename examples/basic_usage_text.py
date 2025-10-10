from safe_store import SafeStore

# Define the vectorizer and its config at initialization
ss = SafeStore(
    "basic_usage_store.db",
    vectorizer_name="st",
    vectorizer_config={"model": "LaBSE"},
    name="my_database",
    description="a cool database"
)

# add_text no longer needs vectorizer info
i = 0
ss.add_text(f"{i}", "Hello there")
i += 1
ss.add_text(f"{i}", "What time is it?")

# query is also simpler
results = ss.query("hi")
for r in results:
    print(f"{r['chunk_text']}:{r['similarity_percent']}%")

print(results[0]['similarity_percent'] - results[1]['similarity_percent'])

# vectorize_text uses the instance's vectorizer
v1 = ss.vectorize_text("Hello there")
v2 = ss.vectorize_text("What time is it?")