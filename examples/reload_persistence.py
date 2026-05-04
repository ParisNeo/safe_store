# examples/reload_persistence.py
"""
Example demonstrating SafeStore's configuration persistence and reload capability.

This example:
1. Creates a SafeStore with explicit non-default configuration
2. Adds documents and performs queries
3. Closes the store completely
4. Reopens the store using ONLY the db_path (no other parameters)
5. Verifies that all configuration was automatically restored
6. Verifies that existing data is still queryable
7. Verifies that new documents can be added with the restored configuration
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


def prepare_documents(doc_dir="temp_docs_persistence"):
    """Creates sample documents for testing persistence."""
    DOC_DIR = Path(doc_dir)
    if DOC_DIR.exists():
        shutil.rmtree(DOC_DIR)
    print_header("Preparing Sample Documents")
    DOC_DIR.mkdir(exist_ok=True)

    (DOC_DIR / "config_test.txt").write_text(
        "SafeStore configuration persistence allows databases to be reopened "
        "with all settings automatically restored from the database itself. "
        "This makes it easy to share databases between applications without "
        "worrying about keeping configuration in sync.",
        encoding='utf-8'
    )

    (DOC_DIR / "additional_doc.txt").write_text(
        "This is an additional document for testing that new documents can be "
        "added after reloading a persisted database. The chunking and vectorization "
        "settings should remain consistent with the original configuration.",
        encoding='utf-8'
    )

    print(f"- Created test documents in: {DOC_DIR.resolve()}")
    return DOC_DIR


def main():
    db_file = "persistence_test.db"
    doc_dir = "temp_docs_persistence"

    print_header("SafeStore Persistence & Reload Example")
    cleanup_db_files(db_file)

    DOC_DIR = prepare_documents()

    # === PHASE 1: Create store with explicit non-default configuration ===
    print_header("Phase 1: Creating Store with Custom Configuration")

    store = safe_store.SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=50,
        chunk_overlap=10,
        chunking_strategy="character",
        expand_before=5,
        expand_after=5,
        text_cleaner="basic",
        name="persistence_test_store",
        description="Testing configuration persistence across sessions",
        metadata={"version": "1.0", "test": True},
        log_level=safe_store.LogLevel.INFO
    )

    print(f"- Created store with custom configuration")
    print(f"  - vectorizer_name: {store.vectorizer_name}")
    print(f"  - chunk_size: {store.chunk_size}")
    print(f"  - chunk_overlap: {store.chunk_overlap}")
    print(f"  - chunking_strategy: {store.chunking_strategy}")
    print(f"  - expand_before: {store.expand_before}")
    print(f"  - expand_after: {store.expand_after}")
    print(f"  - text_cleaner: {store.text_cleaner_name}")

    with store:
        # Add initial document
        store.add_document(
            DOC_DIR / "config_test.txt",
            metadata={"topic": "persistence", "phase": 1}
        )
        print("- Added initial document")

        # Query to verify it works
        results = store.query("configuration persistence", top_k=2)
        print(f"- Query returned {len(results)} results")
        if results:
            print(f"  Top result similarity: {results[0]['similarity_percent']:.2f}%")

    store.close()
    print("- Store closed successfully")

    # === PHASE 2: Reopen with ONLY db_path ===
    print_header("Phase 2: Reopening Store with Only db_path")

    reloaded_store = safe_store.SafeStore(db_path=db_file)

    print(f"- Reloaded store (no config parameters provided)")
    print(f"  - vectorizer_name: {reloaded_store.vectorizer_name}")
    print(f"  - chunk_size: {reloaded_store.chunk_size}")
    print(f"  - chunk_overlap: {reloaded_store.chunk_overlap}")
    print(f"  - chunking_strategy: {reloaded_store.chunking_strategy}")
    print(f"  - expand_before: {reloaded_store.expand_before}")
    print(f"  - expand_after: {reloaded_store.expand_after}")
    print(f"  - text_cleaner: {reloaded_store.text_cleaner_name}")

    # Verify configuration matches original
    assert reloaded_store.vectorizer_name == "st", f"Expected vectorizer_name='st', got '{reloaded_store.vectorizer_name}'"
    assert reloaded_store.chunk_size == 50, f"Expected chunk_size=50, got {reloaded_store.chunk_size}"
    assert reloaded_store.chunk_overlap == 10, f"Expected chunk_overlap=10, got {reloaded_store.chunk_overlap}"
    assert reloaded_store.chunking_strategy == "character", f"Expected chunking_strategy='character', got '{reloaded_store.chunking_strategy}'"
    assert reloaded_store.expand_before == 5, f"Expected expand_before=5, got {reloaded_store.expand_before}"
    assert reloaded_store.expand_after == 5, f"Expected expand_after=5, got {reloaded_store.expand_after}"
    assert reloaded_store.text_cleaner_name == "basic", f"Expected text_cleaner='basic', got '{reloaded_store.text_cleaner_name}'"
    print("- All configuration values verified successfully!")

    with reloaded_store:
        # Verify existing data is queryable
        print_header("Phase 3: Querying Existing Data from Reloaded Store")
        results = reloaded_store.query("configuration persistence", top_k=2)
        print(f"- Query returned {len(results)} results")
        assert len(results) > 0, "Expected to find existing data after reload"
        print(f"  Top result similarity: {results[0]['similarity_percent']:.2f}%")
        print(f"  Top result text: {results[0]['chunk_text'][:80]}...")

        # Verify we can add new documents
        print_header("Phase 4: Adding New Document to Reloaded Store")
        result = reloaded_store.add_document(
            DOC_DIR / "additional_doc.txt",
            metadata={"topic": "persistence", "phase": 2}
        )
        print(f"- Added new document: {result['num_chunks_added']} chunks added")

        # Query across all documents
        all_results = reloaded_store.query("database reload test", top_k=3)
        print(f"- Cross-document query returned {len(all_results)} results")

        # List documents to verify both are present
        docs = reloaded_store.list_documents()
        print(f"- Store contains {len(docs)} documents:")
        for doc in docs:
            print(f"  - {Path(doc['file_path']).name} (id={doc['doc_id']})")

        assert len(docs) == 2, f"Expected 2 documents, found {len(docs)}"

    reloaded_store.close()
    print("- Reloaded store closed successfully")

    # === PHASE 3: Verify using from_db() class method ===
    print_header("Phase 5: Testing from_db() Class Method")

    from_db_store = safe_store.SafeStore.from_db(db_file)

    assert from_db_store.vectorizer_name == "st", f"from_db: Expected vectorizer_name='st', got '{from_db_store.vectorizer_name}'"
    assert from_db_store.chunk_size == 50, f"from_db: Expected chunk_size=50, got {from_db_store.chunk_size}"
    print("- from_db() class method also restored configuration correctly")

    with from_db_store:
        results = from_db_store.query("persistence", top_k=1)
        assert len(results) > 0, "from_db: Expected query to work"
        print(f"- from_db store query works: {len(results)} results")

    from_db_store.close()

    # === Cleanup ===
    print_header("Final Cleanup")
    cleanup_db_files(db_file)
    if Path(doc_dir).exists():
        shutil.rmtree(doc_dir)
        print(f"- Removed directory: {doc_dir}")

    print("\n" + "=" * 10 + " SUCCESS: All persistence tests passed! " + "=" * 10)


if __name__ == "__main__":
    main()