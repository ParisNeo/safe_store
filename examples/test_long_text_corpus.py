from safe_store import TextVectorizer

from safe_store import TextVectorizer, VectorizationMethod

from pathlib import Path
# Create an instance of TextVectorizer
vectorizer = TextVectorizer(
    vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,
    database_path="database.json",
    save_db=False
)

# Add a document for vectorization
documents = ["llm","space","submarines","new york"]
for doc in documents:
    document_name = Path(__file__).parent/f"{doc}.txt"
    with open(document_name, 'r', encoding='utf-8') as file:
        text = file.read()
    vectorizer.add_document(document_name, text, chunk_size=100, overlap_size=20, force_vectorize=False, add_as_a_bloc=False)

# Index the documents (perform vectorization)
vectorizer.index()

# Embed a query and retrieve similar documents
query_text = "what is space"
query_embedding = vectorizer.embed_query(query_text)
similar_texts, _ = vectorizer.recover_text(query_embedding, top_k=3)

vectorizer.show_document(show_interactive_form=True)

print("Similar Documents:")
for i, text in enumerate(similar_texts):
    print(f"{i + 1}: {text}")
