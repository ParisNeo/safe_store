"""
Example demonstrating SafeStore's Full Document & Context Window Retrieval:
1. Querying entire documents based on aggregated chunk relevance.
2. Querying document content windows around matching chunks.
3. Paginated chunk inspection of stored documents.
"""

from pathlib import Path
from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    import gc
    gc.collect()
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "full_doc_query_demo.db"
    cleanup_db(db_file)

    print("=" * 75)
    print(" SafeStore Full Document & Context Window Retrieval Engine Demo ")
    print("=" * 75)

    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=40,
        chunk_overlap=5,
        log_level=LogLevel.INFO
    )

    with store:
        print("\n[Step 1] Ingesting Technical Whitepapers...")
        
        doc_raft = (
            "Section 1: Consensus Algorithms.\n"
            "The Raft consensus algorithm is designed to be understandable and modular.\n"
            "Leader election uses randomized timers to prevent split votes.\n"
            "Log replication ensures state machine consistency across cluster quorum.\n"
            "Safety invariant guarantees committed entries are permanent."
        )
        doc_storage = (
            "Section 2: Log Structured Merge Trees.\n"
            "LSM Trees buffer incoming writes in an in-memory MemTable.\n"
            "When MemTables reach capacity, they flush to immutable SSTables on disk.\n"
            "Compaction merges overlapping key ranges in background threads.\n"
            "Bloom filters minimize unnecessary disk I/O reads for absent keys."
        )

        store.add_text("raft_consensus_whitepaper", doc_raft, metadata={"topic": "Consensus", "system": "Distributed"})
        store.add_text("lsm_storage_whitepaper", doc_storage, metadata={"topic": "Storage", "system": "Databases"})

        # ---------------------------------------------------------------------
        # 1. Full Document Retrieval
        # ---------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("[Query 1: Full Document Retrieval]")
        print("Query: 'randomized timers leader election quorum'")
        print("-" * 75)

        full_docs = store.query_full_documents(
            query_text="randomized timers leader election quorum",
            top_k_docs=1,
            search_mode='hybrid'
        )

        for doc in full_docs:
            print(f"• Document: {doc['document_title']} (Aggregate Score: {doc['aggregate_score']:.4f})")
            print(f"  Hit Chunks: {doc['hit_chunk_count']} / Total Chunks: {doc['total_chunk_count']}")
            print(f"  Metadata: {doc['metadata']}")
            print("\n--- Complete Document Content ---")
            print(doc["full_text"])

        # ---------------------------------------------------------------------
        # 2. Context Window Retrieval (Surrounding Chunks)
        # ---------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("[Query 2: Context Window Around Matching Chunk]")
        print("Query: 'MemTables flush immutable SSTables'")
        print("-" * 75)

        windows = store.query_document_content_window(
            query_text="MemTables flush immutable SSTables",
            top_k_hits=1,
            window_before=1,
            window_after=1,
            search_mode='hybrid'
        )

        for w in windows:
            print(f"• Document: {w['document_title']} (Target Chunk #{w['target_chunk_seq']})")
            print(f"  Hit Score: {w['hit_score']:.4f}")
            print("\n--- Stitched Continuous Context Window ---")
            print(w["stitched_window_text"])

        # ---------------------------------------------------------------------
        # 3. Document Chunk Pagination
        # ---------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("[Query 3: Document Chunk Pagination]")
        print("-" * 75)

        page_data = store.get_document_content_paginated("lsm_storage_whitepaper", page=1, page_size=2)
        print(f"Document: {page_data['document_title']} (Page {page_data['page']}/{page_data['total_pages']})")
        for c in page_data["chunks"]:
            print(f"  [Chunk Seq {c['chunk_seq']}] {c['chunk_text'][:65]}...")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 75)
    print(" Full document and window retrieval demo completed successfully. ")
    print("=" * 75)


if __name__ == "__main__":
    main()