"""
SafeStore Document-Level & Neighborhood Windowing Retrieval Demo.

Demonstrates:
1. Ingesting multi-section technical documents with metadata and optional encryption.
2. Full Document Querying (`query_full_documents`): aggregating chunk hits into ranked, complete documents.
3. Context Window Expansion (`query_document_content_window`): stitching adjacent preceding/succeeding chunks.
4. Document Chunk Pagination (`get_document_content_paginated`): browsing document content page-by-page.
5. Direct Document Reconstruction (`reconstruct_document_text`).
"""

from pathlib import Path
import json
import time

from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    """Cleans up database and lock artifacts."""
    for ext in ["", ".lock", "-wal", "-shm"]:
        p = Path(f"{db_file}{ext}")
        p.unlink(missing_ok=True)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)


def main():
    db_file = "doc_query_demo.db"
    cleanup_db(db_file)

    print_section("SafeStore Document-Level Retrieval & Window Expansion Engine")

    # 1. Initialize SafeStore (with automatic fallback to TF-IDF if Sentence-Transformers is unavailable)
    try:
        store = SafeStore(
            db_path=db_file,
            vectorizer_name="st",
            vectorizer_config={"model": "all-MiniLM-L6-v2"},
            chunk_size=35,
            chunk_overlap=5,
            log_level=LogLevel.INFO
        )
    except Exception as e:
        print(f"[!] Falling back to 'tfidf' vectorizer: {e}")
        store = SafeStore(
            db_path=db_file,
            vectorizer_name="tfidf",
            chunk_size=35,
            chunk_overlap=5,
            log_level=LogLevel.INFO
        )

    with store:
        # ---------------------------------------------------------------------
        # Step 1: Ingest Structured Technical Documents
        # ---------------------------------------------------------------------
        print("\n[Step 1] Ingesting Multi-Chunk Technical Manuals...")

        # Document 1: Distributed Raft Consensus Whitepaper (Multi-chunk)
        doc_raft = (
            "Chapter 1: Distributed State Machine Replication.\n"
            "Consensus algorithms ensure that a cluster of computing nodes agrees on values.\n"
            "Leader election uses randomized heartbeat timers to avoid split-vote deadlock situations.\n"
            "Log replication ensures state machine consistency across all cluster quorum members.\n"
            "Safety invariant guarantees that committed log entries are immutable and permanent."
        )

        # Document 2: Storage Engine Architecture (Multi-chunk)
        doc_lsm = (
            "Chapter 2: Log-Structured Merge Storage Engines.\n"
            "Incoming write requests are appended to a Write-Ahead Log (WAL) for crash durability.\n"
            "Writes are simultaneously inserted into an in-memory sorted MemTable structure.\n"
            "When the MemTable reaches its capacity threshold, it flushes to immutable SSTables on disk.\n"
            "Background compaction threads merge overlapping key ranges to reclaim disk space."
        )

        # Document 3: Network Telemetry & Incident Runbook (Multi-chunk)
        doc_incident = (
            "Chapter 3: Production Diagnostics & Incident Troubleshooting.\n"
            "Watchdog monitors reported error code ERR-8042 in the distributed proxy gateway.\n"
            "The supervisor daemon triggered an automatic failover to the secondary replica.\n"
            "To remediate error code ERR-8042, flush the connection pool and cycle the ingress gateway."
        )

        store.add_text("whitepaper_raft", doc_raft, metadata={"category": "Distributed Systems", "priority": "High"})
        store.add_text("whitepaper_lsm", doc_lsm, metadata={"category": "Storage Engines", "priority": "Medium"})
        store.add_text("runbook_incident", doc_incident, metadata={"category": "Operations", "priority": "Critical"})

        print(f" Ingested 3 documents into vector and FTS5 indices.\n")

        # ---------------------------------------------------------------------
        # Step 2: Full Document Retrieval with Threshold (`query_full_documents`)
        # ---------------------------------------------------------------------
        print_section("Scenario 1: Query Full Documents (0-100 Grade & Thresholding)")
        query_1 = "randomized heartbeat timers leader election split vote"
        print(f"Query: '{query_1}' (Threshold: min_relevance_percent=40.0%)\n")

        full_docs = store.query_full_documents(
            query_text=query_1,
            top_k_docs=2,
            search_mode='hybrid',
            min_relevance_percent=40.0,
            include_hit_chunks=True
        )


        for rank, doc in enumerate(full_docs, 1):
            print(f"Rank {rank} | Document: {doc['document_title']}")
            print(f"  • Aggregate Document Score : {doc['aggregate_score']:.4f}")
            print(f"  • Top Chunk Hit Score       : {doc['top_chunk_score']:.4f}")
            print(f"  • Hit Chunks / Total Chunks : {doc['hit_chunk_count']} / {doc['total_chunk_count']}")
            print(f"  • Metadata                  : {doc['metadata']}")
            print(f"\n  [Reconstructed Full Document Content]:")
            print("  " + "-" * 60)
            for line in doc['full_text'].split('\n'):
                print(f"    {line}")
            print("  " + "-" * 60 + "\n")

        # Demonstrate threshold rejecting irrelevant query
        irrelevant_query = "ancient Roman pottery techniques"
        print(f"Testing Irrelevant Query with Threshold: '{irrelevant_query}' (min_relevance_percent=50.0%)")
        empty_docs = store.query_full_documents(
            query_text=irrelevant_query,
            top_k_docs=2,
            search_mode='hybrid',
            min_relevance_percent=50.0
        )
        print(f"  • Results Returned: {len(empty_docs)} (Clean empty list as expected, zero noise pulled!)\n")

        # ---------------------------------------------------------------------
        # Step 3: Context Window Expansion (`query_document_content_window`)
        # ---------------------------------------------------------------------
        print_section("Scenario 2: Context Window Expansion Around Matching Chunks")
        query_2 = "MemTable reaches capacity threshold immutable SSTables"
        print(f"Query: '{query_2}'")
        print("Fetching hit chunk + 1 preceding chunk + 1 succeeding chunk...\n")

        windows = store.query_document_content_window(
            query_text=query_2,
            top_k_hits=1,
            window_before=1,
            window_after=1,
            search_mode='hybrid'
        )

        for w in windows:
            print(f"• Document: {w['document_title']} (Target Hit Chunk Seq: #{w['target_chunk_seq']})")
            print(f"• Hit Score: {w['hit_score']:.4f}")
            print(f"• Window Span: {len(w['surrounding_chunks'])} sequential chunks")
            print("\n--- Stitched Continuous Context Window ---")
            print(w['stitched_window_text'])
            print("------------------------------------------\n")

            print("Individual Chunks in Sequence Window:")
            for chunk in w['surrounding_chunks']:
                marker = "🎯 [TARGET HIT]" if chunk['is_target_hit'] else "  [NEIGHBOR]   "
                print(f"  {marker} Seq #{chunk['chunk_seq']} (ID: {chunk['chunk_id']}): \"{chunk['chunk_text'][:60]}...\"")

        # ---------------------------------------------------------------------
        # Step 4: Chunk-Level Document Pagination (`get_document_content_paginated`)
        # ---------------------------------------------------------------------
        print_section("Scenario 3: Document Chunk-Level Pagination")
        doc_key = "whitepaper_raft"
        print(f"Paginating through '{doc_key}' with page_size=2 chunks per page:\n")

        # Fetch Page 1
        page_1 = store.get_document_content_paginated(doc_key, page=1, page_size=2)
        print(f"📄 Page {page_1['page']} / {page_1['total_pages']} (Total Document Chunks: {page_1['total_chunks']}):")
        print(f"  • Has Next Page: {page_1['has_next_page']} | Has Previous Page: {page_1['has_previous_page']}")
        for c in page_1['chunks']:
            print(f"    [Chunk Seq {c['chunk_seq']}] {c['chunk_text']}")

        # Fetch Page 2
        if page_1['has_next_page']:
            page_2 = store.get_document_content_paginated(doc_key, page=2, page_size=2)
            print(f"\n📄 Page {page_2['page']} / {page_2['total_pages']}:")
            print(f"  • Has Next Page: {page_2['has_next_page']} | Has Previous Page: {page_2['has_previous_page']}")
            for c in page_2['chunks']:
                print(f"    [Chunk Seq {c['chunk_seq']}] {c['chunk_text']}")

        # ---------------------------------------------------------------------
        # Step 5: Direct Full Document Reconstruction
        # ---------------------------------------------------------------------
        print_section("Scenario 4: Direct Full Text Reconstruction")
        reconstructed = store.reconstruct_document_text("runbook_incident")
        print(f"Reconstructed 'runbook_incident':\n{reconstructed}\n")

    store.close()
    cleanup_db(db_file)
    print_section("Demo Completed Successfully")


if __name__ == "__main__":
    main()