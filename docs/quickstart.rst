==========
Quick Start
==========

Here's a comprehensive example demonstrating document indexing, diagnostics, hybrid retrieval, and knowledge graph querying with ``safe_store``:

.. code-block:: python

    import safe_store
    from pathlib import Path

    # --- 1. Prepare Sample Documents ---
    doc_dir = Path("my_docs")
    doc_dir.mkdir(exist_ok=True)
    doc1_path = doc_dir / "doc1.txt"
    doc1_path.write_text(
        "SafeStore makes local vector storage, BM25 lexical search, and SPARQL knowledge graphs simple and efficient.",
        encoding='utf-8'
    )
    doc2_path = doc_dir / "incident.txt"
    doc2_path.write_text(
        "Production incident: telemetry supervisor daemon failed with error code ERR-8092. Memory buffer exhausted.",
        encoding='utf-8'
    )

    # --- 2. Initialize safe_store ---
    store = safe_store.SafeStore(
        db_path="my_knowledge_store.db",
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=100,
        chunk_overlap=15,
        log_level=safe_store.LogLevel.INFO
    )

    with store:
        # --- 3. Add Documents ---
        print("\n--- Indexing Documents ---")
        store.add_document(doc1_path, metadata={"category": "Overview"})
        store.add_document(doc2_path, metadata={"category": "Incidents", "severity": "High"})

        # --- 4. Database Diagnostics & Summary ---
        print("\n--- Database Diagnostics ---")
        store.info() # Prints complete diagnostic summary panel

        # --- 5. Dense Vector Query ---
        print("\n--- Dense Semantic Query ---")
        dense_hits = store.query("efficient local vector storage", top_k=1, min_relevance_percent=30.0)
        for h in dense_hits:
            print(f"Dense Hit: {h['file_path']} (Score: {h['relevance_score']:.1f}%)")

        # --- 6. Tri-Modal Hybrid Query (Dense + BM25 Lexical) ---
        print("\n--- Hybrid Query (Exact Code + Meaning) ---")
        hybrid_hits = store.hybrid_query(
            "telemetry daemon memory error ERR-8092",
            top_k=2,
            dense_weight=0.5,
            bm25_weight=0.5,
            min_relevance_percent=35.0
        )
        for h in hybrid_hits:
            print(f"Hybrid Hit: {h['file_path']} (Score: {h['relevance_score']:.1f}%) -> {h['chunk_text']}")

        # --- 7. Full Document Querying ---
        print("\n--- Full Document Retrieval ---")
        full_docs = store.query_full_documents("telemetry supervisor incident", top_k_docs=1, min_relevance_percent=40.0)
        if full_docs:
            print(f"Top Doc: {full_docs[0]['document_title']}\nFull Text:\n{full_docs[0]['full_text']}")