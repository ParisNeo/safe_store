"""
Example demonstrating SafeStore's Overlapping Chunk Reconstruction Engine:
1. Ingests a document partitioned into small chunks with overlapping boundaries.
2. Performs a query that retrieves multiple adjacent chunks out of order.
3. Compares the raw fragmented, duplicated chunks with the reconstructed continuous document text.
4. Demonstrates non-contiguous gap bridging ('...') and single metadata context blocks.
"""

from pathlib import Path
from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "reconstruction_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore Overlapping Chunk Reconstruction & Chronological Fusion Demo ")
    print("=" * 80)

    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=40,
        chunk_overlap=12,
        chunking_strategy="character",
        log_level=LogLevel.INFO
    )

    with store:
        # Ingest multi-paragraph manual
        tutorial_text = (
            "Section 1: Initializing the Cluster.\n"
            "To deploy the distributed storage cluster, configure all node IP addresses in the yaml file.\n"
            "Section 2: Security and Encryption.\n"
            "Ensure mutual TLS authentication is enabled across all gossip RPC ports before starting the daemons.\n"
            "Section 3: Storage Engine Verification.\n"
            "Start the supervisor daemon and monitor the Write-Ahead Log sync metrics in the telemetry dashboard.\n"
            "Section 4: Production Failover Procedures.\n"
            "In case of node isolation, trigger quorum election and verify standby promotion within 5 seconds."
        )

        store.add_text(
            unique_id="cluster_operations_guide",
            text=tutorial_text,
            metadata={"guide": "Cluster Operations", "version": "4.2", "classification": "Internal"}
        )

        query = "supervisor daemon Write-Ahead Log sync metrics"

        # ---------------------------------------------------------------------
        # 1. Standard Query (Raw Fragmented & Overlapped Chunks)
        # ---------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[1. Standard Query: Raw Fragmented Chunks]")
        print(f"Query: '{query}'")
        print("-" * 80)

        raw_chunks = store.query(query, top_k=3, reconstruct_overlapping_chunks=False)
        for i, c in enumerate(raw_chunks, 1):
            print(f"\nChunk #{i} (Chunk ID: {c['chunk_id']} | Seq: {c.get('chunk_seq', 'N/A')} | Score: {c['similarity_percent']:.1f}%):")
            print(c['chunk_text'])

        # ---------------------------------------------------------------------
        # 2. Reconstructed Query (Seamless Contiguous Document Flow)
        # ---------------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[2. Reconstructed Query: Chronologically Fused & Overlap Deduplicated]")
        print("reconstruct_overlapping_chunks=True")
        print("-" * 80)

        fused_docs = store.query(query, top_k=3, reconstruct_overlapping_chunks=True)
        for i, doc in enumerate(fused_docs, 1):
            print(f"\nDocument #{i} [{doc['document_title']}] (Peak Relevance: {doc['relevance_score']:.1f}%):")
            print(f"• Fused Chunk IDs : {doc['fused_chunk_ids']}")
            print(f"• Chunk Seqs      : {doc['chunk_seqs']}")
            print(f"• Clusters Formed : {doc['cluster_count']}")
            print("\n[Reconstructed Document Context]:")
            print("=" * 60)
            print(doc['chunk_text'])
            print("=" * 60)

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" Reconstruction demo finished successfully. ")
    print("=" * 80)


if __name__ == "__main__":
    main()