# examples/grepper_usage.py
"""
Example demonstrating the Grepper vectorizer for keyword-based search.

The Grepper vectorizer is a lightweight, non-ML alternative that:
- Parses markdown documents into header hierarchies (trees)
- Builds an inverted index for ultra-fast exact/phrase matching
- Returns breadcrumb paths showing where matches occurred
- Requires no model downloads or GPU resources

Ideal for:
- Documentation search
- Code repository search
- Any scenario where exact keyword/phrase matching is preferred over semantic similarity
"""

import safe_store
from pathlib import Path
import shutil


def print_header(title):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def cleanup_db_files(db_file):
    """Cleans up the database and its associated files."""
    db_path = Path(db_file)
    paths_to_delete = [
        db_path,
        Path(f"{db_path}.lock"),
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm")
    ]
    for p in paths_to_delete:
        p.unlink(missing_ok=True)
    print(f"- Cleaned up database artifacts for {db_file}")


def prepare_markdown_documents(doc_dir="temp_docs_grepper"):
    """Creates sample markdown documents with rich header hierarchies."""
    DOC_DIR = Path(doc_dir)
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
    print_header("Preparing Sample Markdown Documents")
    DOC_DIR.mkdir(exist_ok=True)

    # Document 1: Project documentation with deep header hierarchy
    (DOC_DIR / "project_guide.md").write_text("""# SafeStore Project

## Overview
SafeStore is a Python library for local vector storage and retrieval.

## Installation

### Prerequisites
You need Python 3.8 or higher installed on your system.

### pip Install
Run the following command to install:
```bash
pip install safe-store
```

## Usage

### Basic Example
Here is a basic example of how to use the library.

### Advanced Configuration
For advanced features, you can customize the chunking strategy.

## API Reference

### SafeStore Class
The main class for managing your vector store.

### Vectorizers
Choose from multiple vectorization backends.

## Troubleshooting

### Common Issues
If you encounter installation problems, check your Python version.

### Getting Help
Visit our GitHub repository for support.
""", encoding='utf-8')

    # Document 2: Different topic with overlapping terms
    (DOC_DIR / "cooking_guide.md").write_text("""# Cooking Basics

## Kitchen Setup
Before you start cooking, organize your kitchen workspace.

## Ingredients

### Fresh Produce
Always select fresh vegetables and fruits for the best flavor.

### Spices and Herbs
Basil, oregano, and thyme are essential Italian herbs.

## Recipes

### Pasta Dishes
Learn to make authentic Italian pasta from scratch.

### Baking
Bread baking requires precise temperature control.

## Troubleshooting

### Burnt Food
If your dish is burnt, reduce the heat and use a timer.

### Substitutions
Missing an ingredient? Find suitable replacements here.
""", encoding='utf-8')

    # Document 3: Technical reference with code
    (DOC_DIR / "api_reference.md").write_text("""# API Documentation

## Authentication
All API requests require a valid authentication token.

## Endpoints

### GET /documents
Retrieve a list of all documents in the store.

### POST /documents
Add a new document to the store with metadata.

### DELETE /documents/{id}
Remove a document by its unique identifier.

## Error Handling

### 400 Bad Request
The request was malformed or missing required parameters.

### 404 Not Found
The requested resource does not exist.

### 500 Internal Server Error
An unexpected error occurred on the server.

## Rate Limiting
API calls are limited to 100 requests per minute per API key.
""", encoding='utf-8')

    print(f"- Created 3 markdown documents in: {DOC_DIR.resolve()}")
    return DOC_DIR


def main():
    db_file = "grepper_store.db"
    print_header("Grepper Vectorizer Example")
    cleanup_db_files(db_file)

    DOC_DIR = prepare_markdown_documents()

    try:
        store = safe_store.SafeStore(
            db_path=db_file,
            vectorizer_name="grepper",
            chunk_size=500,
            chunk_overlap=0,
            log_level=safe_store.LogLevel.INFO
        )

        with store:
            print_header("Indexing Documents")
            for md_file in DOC_DIR.glob("*.md"):
                store.add_document(
                    md_file,
                    metadata={"source": "documentation", "filename": md_file.name}
                )
                print(f"  Indexed: {md_file.name}")

            print_header("Query 1: Single Term Search ('installation')")
            results = store.query("installation", top_k=5)
            print(f"  Found {len(results)} results")
            for i, r in enumerate(results, 1):
                breadcrumb = r.get("header_breadcrumbs", "")
                print(f"  Result {i}:")
                print(f"    Score: {r['similarity_percent']:.2f}%")
                print(f"    Document: {r['file_path']}")
                print(f"    Breadcrumb: {breadcrumb if breadcrumb else '(root level)'}")
                text_preview = r['chunk_text'][:120].replace('\n', ' ')
                print(f"    Text: {text_preview}...")
                print()

            print_header("Query 2: Phrase Search ('pip install safe-store')")
            phrase_results = store.query("pip install safe-store", top_k=3)
            print(f"  Found {len(phrase_results)} results")
            for i, r in enumerate(phrase_results, 1):
                breadcrumb = r.get("header_breadcrumbs", "")
                print(f"  Result {i}:")
                print(f"    Score: {r['similarity_percent']:.2f}%")
                print(f"    Document: {r['file_path']}")
                print(f"    Breadcrumb: {breadcrumb if breadcrumb else '(root level)'}")
                text_preview = r['chunk_text'][:120].replace('\n', ' ')
                print(f"    Text: {text_preview}...")
                matched = r.get("matched_terms", [])
                print(f"    Matched terms: {', '.join(matched)}")
                print()

            print_header("Query 3: Cross-Document Term ('troubleshooting')")
            cross_results = store.query("troubleshooting", top_k=10)
            print(f"  Found {len(cross_results)} results across documents")
            doc_hits = {}
            for r in cross_results:
                doc = r['file_path']
                doc_hits[doc] = doc_hits.get(doc, 0) + 1
            for doc, count in doc_hits.items():
                print(f"    {Path(doc).name}: {count} matching chunks")
            for i, r in enumerate(cross_results[:3], 1):
                breadcrumb = r.get("header_breadcrumbs", "")
                print(f"  Top result {i} from {Path(r['file_path']).name}:")
                print(f"    Breadcrumb: {breadcrumb if breadcrumb else '(root level)'}")
                text_preview = r['chunk_text'][:100].replace('\n', ' ')
                print(f"    Text: {text_preview}...")
                print()

            print_header("Query 4: Header-Specific Term ('Authentication')")
            header_results = store.query("Authentication", top_k=3)
            print(f"  Found {len(header_results)} results")
            for i, r in enumerate(header_results, 1):
                breadcrumb = r.get("header_breadcrumbs", "")
                is_header = "header" in r.get("header_breadcrumbs", "").lower() or r.get("similarity_percent", 0) > 50
                print(f"  Result {i}:")
                print(f"    Score: {r['similarity_percent']:.2f}%")
                print(f"    Document: {r['file_path']}")
                print(f"    Breadcrumb: {breadcrumb if breadcrumb else '(root level)'}")
                text_preview = r['chunk_text'][:120].replace('\n', ' ')
                print(f"    Text: {text_preview}...")
                print()

            print_header("Query 5: No Results ('xyznonexistent')")
            empty_results = store.query("xyznonexistent", top_k=5)
            print(f"  Results: {empty_results}")
            print("  (Empty list as expected for non-matching query)")

            print_header("Query 6: Term with Multiple Occurrences ('API')")
            multi_results = store.query("API", top_k=5)
            print(f"  Found {len(multi_results)} results")
            for i, r in enumerate(multi_results, 1):
                breadcrumb = r.get("header_breadcrumbs", "")
                print(f"  Result {i}:")
                print(f"    Score: {r['similarity_percent']:.2f}%")
                print(f"    Document: {Path(r['file_path']).name}")
                print(f"    Breadcrumb: {breadcrumb if breadcrumb else '(root level)'}")
                text_preview = r['chunk_text'][:120].replace('\n', ' ')
                print(f"    Text: {text_preview}...")
                print()

            print_header("Adding Text Directly")
            store.add_text(
                unique_id="inline_notes.md",
                text="""# Developer Notes

## Quick Tips
Always test your code before committing.
Use grepper for fast documentation search.

## Reminders
Check the documentation for latest API changes.
The grepper vectorizer is great for keyword matching.
""",
                metadata={"type": "notes", "priority": "high"}
            )
            print("  Added inline text document 'inline_notes.md'")

            notes_results = store.query("quick tips", top_k=2)
            print(f"  Query 'quick tips' found {len(notes_results)} results")
            for i, r in enumerate(notes_results, 1):
                breadcrumb = r.get("header_breadcrumbs", "")
                print(f"  Result {i}:")
                print(f"    Score: {r['similarity_percent']:.2f}%")
                print(f"    Breadcrumb: {breadcrumb if breadcrumb else '(root level)'}")
                text_preview = r['chunk_text'][:120].replace('\n', ' ')
                print(f"    Text: {text_preview}...")
                print()

    except safe_store.ConfigurationError as e:
        print(f"  [SKIP] Configuration error: {e}")
    except Exception as e:
        print(f"  [ERROR] An unexpected error occurred: {e}")
        raise

    print_header("Final Cleanup")
    cleanup_db_files(db_file)
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
        print(f"- Removed directory: {DOC_DIR}")

    print("\n--- End of Grepper Example ---")


if __name__ == "__main__":
    main()