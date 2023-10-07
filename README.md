# Text Vectorizer Library

Text Vectorizer is a Python library that facilitates text indexing and vectorization using various methods, including TF-IDF Vectorization and Model Embeddings. This library empowers you to efficiently vectorize and analyze text documents, making it suitable for a wide range of applications such as text search, content recommendation, and text similarity analysis.

![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-brightgreen.svg)
![PyPI Version](https://img.shields.io/pypi/v/safe-store.svg)
![License](https://img.shields.io/github/license/ParisNeo/safe_store.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/safe-store.svg)

## Features

- Supports two primary vectorization methods: **TF-IDF Vectorization** and **Model Embeddings**.
- Visualize text embeddings in 2D using **PCA** (Principal Component Analysis) or **t-SNE** (t-distributed Stochastic Neighbor Embedding).
- Search for similar text documents or chunks within your corpus.
- Load and save vectorized documents for future use.

## Installation

You can easily install the Text Vectorizer library using `pip`:

```bash
pip install safe_store
```

## Getting Started

### Initializing the Text Vectorizer

To start using the Text Vectorizer, you'll need to initialize it with the desired vectorization method, provide an optional model (in case of using embeddings), and specify a path for the database to save your vectorized documents.

```python
from text_vectorizer import TextVectorizer, VectorizationMethod

# Initialize the Text Vectorizer
vectorizer = TextVectorizer(
    vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,  # Choose your preferred method
    model=None,  # Provide your model (if using model embeddings)
    database_path="text_db.json",  # Specify the path for the database
    save_db=False,  # Set to True to save the database
    data_visualization_method="PCA"  # Choose your visualization method (PCA or t-SNE)
)
```

### Adding Documents

You can add documents to the Text Vectorizer using the `add_document` method. Specify the document name, the text content, chunk size, and overlap size.

```python
# Add a document
vectorizer.add_document(
    document_name="example.txt",
    text="This is an example document. It can be longer and contain multiple paragraphs.",
    chunk_size=100,  # Set the chunk size for text decomposition
    overlap_size=20  # Set the overlap size between chunks
)
```

### Indexing Documents

To enable searching and analysis, you need to index your documents using the `index` method.

```python
# Index the documents
vectorizer.index()
```

### Visualizing Text Embeddings

You can visualize the text embeddings in 2D using PCA or t-SNE with the `show_document` method. Pass a query text (optional), specify a path to save the visualization (optional), and set `show_interactive_form` to `True` if you want to display an interactive plot.

```python
# Visualize the embeddings
vectorizer.show_document(
    query_text="Query text (optional)",
    save_fig_path="scatter_plot.png",  # Specify the path to save the visualization
    show_interactive_form=True  # Set to True to display an interactive plot
)
```

### Searching for Similar Text

You can retrieve similar text documents to a query using the `embed_query` and `recover_text` methods. Provide a query text, and the library will return similar text chunks based on embeddings.

```python
# Embed the query text
query_embedding = vectorizer.embed_query("Query text")

# Retrieve similar text documents (top_k specifies the number of similar documents to retrieve)
similar_texts, similarities = vectorizer.recover_text(query_embedding, top_k=3)
```

### Clearing the Database

If needed, you can clear the database using the `clear_database` method. This removes all indexed documents and resets the Text Vectorizer.

```python
# Clear the database
vectorizer.clear_database()
```

## Author

- ParisNeo

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

For more detailed usage and options, refer to the [documentation](link-to-documentation).
