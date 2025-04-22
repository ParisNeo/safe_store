==========
Quick Start
==========

Here's a basic example demonstrating indexing and querying:

.. code-block:: python

    import safe_store
    from pathlib import Path
    import time # For demonstrating concurrency

    # --- 1. Prepare Sample Documents ---
    doc_dir = Path("my_docs")
    doc_dir.mkdir(exist_ok=True)
    doc1_path = doc_dir / "doc1.txt"
    doc1_path.write_text("safe_store makes local vector storage simple and efficient.", encoding='utf-8')
    doc2_path = doc_dir / "doc2.html"
    doc2_path.write_text("<html><body><p>HTML content can also be indexed.</p></body></html>", encoding='utf-8')

    print(f"Created sample files in: {doc_dir.resolve()}")

    # --- 2. Initialize safe_store ---
    # Use DEBUG level for more verbose output, adjust lock timeout if needed
    # Add encryption_key="your-secret-password" to enable encryption
    store = safe_store.safe_store(
        "my_vector_store.db",
        log_level=safe_store.LogLevel.DEBUG,
        lock_timeout=10 # Wait up to 10s for write lock
        # encryption_key="your-secret-password" # Uncomment to enable
    )

    # Best practice: Use safe_store as a context manager
    try:
        with store:
            # --- 3. Add Documents (acquires write lock) ---
            print("\n--- Indexing Documents ---")
            # Requires safe_store[sentence-transformers]
            store.add_document(doc1_path, vectorizer_name="st:all-MiniLM-L6-v2", chunk_size=50, chunk_overlap=10)

            # Requires safe_store[parsing] for HTML
            store.add_document(doc2_path, vectorizer_name="st:all-MiniLM-L6-v2")

            # Add TF-IDF vectors as well (requires safe_store[tfidf])
            # This will fit TF-IDF on all documents
            print("\n--- Adding TF-IDF Vectorization ---")
            store.add_vectorization("tfidf:my_analysis")

            # --- 4. Query (read operation, concurrent with WAL) ---
            print("\n--- Querying using Sentence Transformer ---")
            query_st = "simple storage"
            results_st = store.query(query_st, vectorizer_name="st:all-MiniLM-L6-v2", top_k=2)
            for i, res in enumerate(results_st):
                print(f"ST Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")

            print("\n--- Querying using TF-IDF ---")
            query_tfidf = "html index"
            results_tfidf = store.query(query_tfidf, vectorizer_name="tfidf:my_analysis", top_k=2)
            for i, res in enumerate(results_tfidf):
                print(f"TFIDF Result {i+1}: Score={res['similarity']:.4f}, Path='{Path(res['file_path']).name}', Text='{res['chunk_text'][:60]}...'")

            # --- 5. List Methods ---
            print("\n--- Listing Vectorization Methods ---")
            methods = store.list_vectorization_methods()
            for method in methods:
                print(f"- ID: {method['method_id']}, Name: {method['method_name']}, Type: {method['method_type']}, Dim: {method['vector_dim']}")

    except safe_store.ConfigurationError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Please install the required extras (e.g., pip install safe_store[all])")
    except safe_store.ConcurrencyError as e:
        print(f"\n[ERROR] Lock timeout or concurrency issue: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
    finally:
        # Connection is closed automatically by the 'with' statement exit
        print("\n--- Store context closed ---")
        # Cleanup (optional)
        # import shutil
        # shutil.rmtree(doc_dir)
        # Path("my_vector_store.db").unlink(missing_ok=True)
        # Path("my_vector_store.db.lock").unlink(missing_ok=True)

    print("\nCheck 'my_vector_store.db' and console logs.")

