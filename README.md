Certainly! Here's a comprehensive project description for the `safe_store` library:

---

# safe_store

`safe_store` is an open-source Python library that provides essential tools for text data management, vectorization, and document retrieval. It empowers users to work with text documents efficiently and effortlessly.

## Key Features:

### 1. Text Vectorizer

- **Versatile Vectorization:** Choose between TF-IDF vectorization and model-based embeddings to convert text documents into numerical representations.
- **Document Similarity:** Find documents similar to a given query text, making it ideal for document retrieval tasks.
- **Interactive Visualization:** Visualize document embeddings in a scatter plot to gain insights into document relationships.
- **No Authentication Required:** Use the library without the need for API keys or authentication, making it accessible for everyone.
- **Commercially Usable:** `safe_store` is 100% open-source and free to use, even for commercial purposes, under the Apache 2.0 License.

### 2. Generic Data Loader

- **Multi-format Support:** Read various file formats, including PDF, DOCX, JSON, HTML, and more.
- **Simplified Text Extraction:** Convert file content to plain text or data structures with ease.
- **Efficient and Time-Saving:** Streamline data loading and processing tasks, reducing the need for manual extraction.

## What Can You Use `safe_store` For?

- **Text Document Analysis:** Analyze and understand the content of text documents quickly and efficiently.
- **Document Retrieval:** Retrieve documents similar to a given query text, facilitating content recommendation and search tasks.
- **Text Data Preprocessing:** Prepare text data for natural language processing (NLP) tasks, such as sentiment analysis and text classification.
- **Data Loading:** Streamline the process of reading and extracting content from various file formats.

`safe_store` is designed to be accessible, versatile, and free for all users. It's an ideal choice for developers, data scientists, and researchers who want a user-friendly and open-source solution for working with text data.

## License

`safe_store` is licensed under the Apache 2.0 License, allowing you to use it freely, even for commercial purposes, without any signup or authorization keys.

---

Explore the world of text data management and analysis with `safe_store` today!

# Table of Contents
- [safe_store](#safe_store)
  - [Text Vectorizer](#text-vectorizer)
    - [Features](#features)
    - [Installation](#installation)
    - [Getting Started](#getting-started)
      - [Initializing the Text Vectorizer](#initializing-the-text-vectorizer)
      - [Adding and Indexing Documents](#adding-and-indexing-documents)
      - [Embedding a Query and Retrieving Similar Documents](#embedding-a-query-and-retrieving-similar-documents)
  - [Generic Data Loader](#generic-data-loader)
    - [Features](#features-1)
    - [Usage](#usage)
    - [Supported File Types](#supported-file-types)
  - [Author](#author)
  - [License](#license)

---

## Text Vectorizer

### Features

- Vectorize and index text documents.
- Retrieve similar documents based on a query.
- Supports both TF-IDF vectorization and model-based embeddings.
- Interactive visualization of document embeddings.
- No authentication or API keys required.

### Installation

To install `safe_store`, you can use `pip`:

```bash
pip install safe_store
```

### Getting Started

#### Initializing the Text Vectorizer

```python
from safe_store import TextVectorizer, VectorizationMethod
from pathlib import Path

# Create an instance of TextVectorizer
vectorizer = TextVectorizer(
    vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,
    database_path="database.json",
    save_db=False
)
```

#### Adding and Indexing Documents

```python
# Add documents for vectorization
documents = ["llm", "space", "submarines", "new york"]
for doc in documents:
    document_name = Path(__file__).parent / f"{doc}.txt"
    with open(document_name, 'r', encoding='utf-8') as file:
        text = file.read()
    vectorizer.add_document(document_name, text, chunk_size=100, overlap_size=20, force_vectorize=False, add_as_a_bloc=False)

# Index the documents (perform vectorization)
vectorizer.index()
```

#### Embedding a Query and Retrieving Similar Documents

```python
# Embed a query and retrieve similar documents
query_text = "what is space"
query_embedding = vectorizer.embed_query(query_text)
similar_texts, _ = vectorizer.recover_text(query_embedding, top_k=3)

# Show the interactive document visualization
vectorizer.show_document(show_interactive_form=True)

print("Similar Documents:")
for i, text in enumerate(similar_texts):
    print(f"{i + 1}: {text}")
```

---

## Generic Data Loader

### Features

- Read various file formats including PDF, DOCX, JSON, HTML, and more.
- Convert file content to text or data structures.

### Usage

To read a file using `GenericDataLoader`, you can use the `read_file` method and provide the file path:

```python
from safe_store import GenericDataLoader
from pathlib import Path

file_path = Path("example.pdf")
file_content = GenericDataLoader.read_file(file_path)
```

### Supported File Types

- PDF
- DOCX
- JSON
- HTML
- PPTX
- TXT
- RTF
- MD
- LOG
- CPP
- Java
- JS
- Python
- Ruby
- Shell Script
- SQL
- CSS
- PHP
- XML
- YAML
- INI
- INF
- MAP
- BAT
```

Feel free to replace `"example.pdf"` with the path to your specific file.

---

## Author

- ParisNeo

## License

This project is licensed under the Apache 2.0 License.
