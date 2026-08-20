"""
Example demonstrating SafeStore's Tri-Modal Hybrid Retrieval Engine:
1. Pure Dense Vector Search (Sentence-Transformers semantic embeddings)
2. Pure Sparse Lexical Search (SQLite FTS5 BM25 exact keyword matching)
3. Tri-Modal Hybrid Retrieval (Reciprocal Rank Fusion - RRF combining Dense + BM25)
"""

from pathlib import Path
import shutil
from safe_store import SafeStore, BM25Retriever, LogLevel


def cleanup_db(db_file: str):
    """Cleans up database and lock artifacts with Windows lock safety."""
    import gc
    import time
    gc.collect()
    for ext in ["", ".lock", "-wal", "-shm"]:
        p = Path(f"{db_file}{ext}")
        for _ in range(5):
            try:
                p.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05)


def main():
    db_file = "hybrid_bm25_demo.db"
    cleanup_db(db_file)

    print("=" * 70)
    print(" SafeStore Tri-Modal Hybrid Retrieval (Dense + BM25 Lexical + RRF) ")
    print("=" * 70)

    # 1. Initialize SafeStore with Sentence Transformers vectorizer (or fallback)
    try:
        store = SafeStore(
            db_path=db_file,
            vectorizer_name="st",
            vectorizer_config={"model": "all-MiniLM-L6-v2"},
            chunk_size=80,
            chunk_overlap=10,
            log_level=LogLevel.INFO
        )
    except Exception as e:
        print(f"\n[!] Notice: Sentence-Transformers initialization failed: {e}")
        print("[!] If you see 'operator torchvision::nms does not exist', run: pip uninstall -y torchvision")
        print("[!] Falling back to 'tfidf' vectorizer for this demonstration...\n")
        store = SafeStore(
            db_path=db_file,
            vectorizer_name="tfidf",
            chunk_size=80,
            chunk_overlap=10,
            log_level=LogLevel.INFO
        )

    with store:
        print("\n[Step 1] Ingesting Technical Documentation...")
        # Document 1: Incident report with specific technical error code
        store.add_text(
            unique_id="k8s_incident_report",
            text="Production cluster incident report: Node-04 crashed due to an OOMKilled condition in supervisor daemon. "
                 "Specific telemetry error identifier ERR-9021 was emitted by the kubelet watchdog service.",
            metadata={"system": "Kubernetes", "severity": "Critical", "doc_type": "Incident"}
        )

        # Document 2: Unrelated database failover documentation
        store.add_text(
            unique_id="postgres_failover_guide",
            text="PostgreSQL primary database failover protocol: In the event of a network partition, "
                 "the standby replica node promotes itself to primary using distributed Raft consensus quorum.",
            metadata={"system": "PostgreSQL", "severity": "High", "doc_type": "Architecture"}
        )

        # Document 3: Runbook with the exact resolution for ERR-9021
        store.add_text(
            unique_id="runbook_memory_troubleshooting",
            text="Troubleshooting Runbook for Memory Exhaustion: When encountering error code ERR-9021, increase container "
                 "memory limits in the helm deployment manifest and analyze JVM heap dump profiles.",
            metadata={"system": "Runbook", "topic": "Memory", "doc_type": "Resolution"}
        )

        # Document 4: General conceptual article about memory leaks
        store.add_text(
            unique_id="article_memory_management",
            text="Best practices for cloud native memory management: Profiling resident memory and garbage collection "
                 "cycles prevents unexpected application crashes and out of memory terminations.",
            metadata={"system": "General", "topic": "Optimization", "doc_type": "Article"}
        )

        print(" Ingested 4 technical documents into vector and FTS5 indices.\n")

        # ---------------------------------------------------------------------
        # Comparison 1: Pure Semantic Dense Search
        # ---------------------------------------------------------------------
        semantic_query = "how to fix container out of memory termination"
        print("-" * 70)
        print(f"[Query A - Conceptual / Semantic]: '{semantic_query}'")
        print("-" * 70)
        dense_results = store.query(semantic_query, top_k=2)
        for i, r in enumerate(dense_results, 1):
            print(f"  Rank {i} | Similarity: {r['similarity_percent']:.2f}% | Source: {r['file_path']}")
            print(f"  Preview: {r['chunk_text'][:95]}...\n")

        # ---------------------------------------------------------------------
        # Comparison 2: Pure BM25 Lexical Search (Exact Token Matching)
        # ---------------------------------------------------------------------
        exact_code_query = "ERR-9021"
        print("-" * 70)
        print(f"[Query B - Exact Technical Identifier]: '{exact_code_query}'")
        print("-" * 70)
        bm25_retriever = BM25Retriever(store.conn)
        bm25_results = bm25_retriever.search(exact_code_query, top_k=2)
        for i, r in enumerate(bm25_results, 1):
            print(f"  Rank {i} | BM25 Score: {r['score']:.4f} | Source: {r['file_path']}")
            print(f"  Preview: {r['chunk_text'][:95]}...\n")

        # ---------------------------------------------------------------------
        # Comparison 3: Tri-Modal Hybrid Query (Dense + BM25 via RRF)
        # ---------------------------------------------------------------------
        hybrid_query_text = "troubleshooting kubelet supervisor crash ERR-9021 memory limits"
        print("-" * 70)
        print(f"[Query C - Hybrid (Dense + Sparse Fusion)]: '{hybrid_query_text}'")
        print("-" * 70)
        hybrid_results = store.hybrid_query(
            query_text=hybrid_query_text,
            top_k=3,
            dense_weight=0.5,
            bm25_weight=0.5,
            rrf_k=60
        )
        for i, r in enumerate(hybrid_results, 1):
            print(f"  Rank {i} | Fused RRF Score: {r['fused_score']:.5f} | Source: {r['file_path']}")
            print(f"  Preview: {r['chunk_text'][:95]}...\n")

    # Cleanup artifacts after demo
    store.close()
    cleanup_db(db_file)
    print("=" * 70)
    print(" Hybrid retrieval demonstration completed successfully. ")
    print("=" * 70)


if __name__ == "__main__":
    main()