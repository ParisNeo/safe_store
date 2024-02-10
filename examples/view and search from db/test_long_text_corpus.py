from safe_store import TextVectorizer
from safe_store import TextVectorizer, VectorizationMethod, VisualizationMethod
from safe_store.generic_data_loader import GenericDataLoader
from pathlib import Path
# Create an instance of TextVectorizer
vectorizer = TextVectorizer(
    vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,#=VectorizationMethod.BM25_VECTORIZER,
    database_path=Path(__file__).parent.parent/"vectorized_dbs"/"database.json",
    data_visualization_method=VisualizationMethod.TSNE,#VisualizationMethod.PCA,
    save_db=False
)

# # Add a document for vectorization
database_path = Path(__file__).parent.parent/"test_database"
documents = [d for d in database_path.iterdir()]
for doc in documents:
    text = GenericDataLoader.read_file(doc)
    vectorizer.add_document(str(doc), text, chunk_size=100, overlap_size=20, force_vectorize=False, add_as_a_bloc=False)

# Index the documents (perform vectorization)
vectorizer.index()

# Embed a query and retrieve similar documents
query_text = "What are future space technologies"
similar_texts, _, _ = vectorizer.recover_text(query_text, top_k=3)

vectorizer.show_document(query_text,show_interactive_form=True)


print("Similar Documents:")
for i, text in enumerate(similar_texts):
    print(f"{i + 1}: {text}")
