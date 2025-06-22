from safe_store import SafeStore
#vectorizer = "st:all-MiniLM-L6-v2"
#vectorizer = "st:all-MiniLM-L12-v2"
vectorizer = "st:LaBSE"
ss = SafeStore("basic_usage_store.db", "my database", "a cool database")
i=0
ss.add_text(f"{i}","Hello there", vectorizer)
i += 1
ss.add_text(f"{i}","What time is it?", vectorizer)
results = ss.query("hi",vectorizer)
for r in results:
    print(f"{r['chunk_text']}:{r['similarity_percent']}%")
print(results[0]['similarity_percent']-results[1]['similarity_percent'])

v1 = ss.vectorize_text("Hello there", vectorizer)
v2 = ss.vectorize_text("What time is it?", vectorizer)
