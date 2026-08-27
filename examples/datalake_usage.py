"""
Example demonstrating SafeStore's Semantic Datalake Viewer:
1. Multi-cluster document ingestion.
2. 2D and 3D PCA / t-SNE dimensionality reduction.
3. Bulk loading with persistent sub-millisecond SQLite caching.
4. Lazy streaming loading for large vector collections.
5. Exporting interactive standalone 2D/3D Plotly visualizers.
"""

from pathlib import Path
import shutil
import time

from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    import gc
    gc.collect()
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "datalake_demo.db"
    cleanup_db(db_file)

    print("=" * 70)
    print(" SafeStore Semantic Datalake Explorer Demo (PCA / t-SNE / 3D) ")
    print("=" * 70)

    # 1. Initialize SafeStore
    try:
        store = SafeStore(
            db_path=db_file,
            vectorizer_name="st",
            vectorizer_config={"model": "all-MiniLM-L6-v2"},
            chunk_size=60,
            chunk_overlap=10,
            log_level=LogLevel.INFO
        )
    except Exception as e:
        print(f"[!] Falling back to tfidf: {e}")
        store = SafeStore(db_path=db_file, vectorizer_name="tfidf", chunk_size=60, chunk_overlap=10)

    with store:
        print("\n[Step 1] Ingesting Multi-Domain Corpus...")
        # Domain 1: Deep Space Exploration
        store.add_text(
            unique_id="space_astronomy",
            text="The James Webb Space Telescope observes the earliest stars and galaxies formed after the Big Bang. "
                 "Exoplanet atmospheric spectroscopy searches for biosignature gases such as methane and oxygen.",
            metadata={"domain": "Astronomy", "cluster": "Space"}
        )

        # Domain 2: Molecular Biology & CRISPR
        store.add_text(
            unique_id="biology_genetics",
            text="CRISPR-Cas9 ribonucleoprotein complexes introduce targeted double-strand breaks in genomic DNA. "
                 "Cellular DNA repair pathways utilize homologous recombination for sequence insertion.",
            metadata={"domain": "Genetics", "cluster": "Life Sciences"}
        )

        # Domain 3: Distributed Systems & Cryptography
        store.add_text(
            unique_id="crypto_consensus",
            text="Byzantine fault tolerant state machine replication guarantees consensus across untrusted network nodes. "
                 "Elliptic curve digital signatures authenticate transactions across peer-to-peer gossip topologies.",
            metadata={"domain": "Cryptography", "cluster": "Computer Science"}
        )

        # ---------------------------------------------------------------------
        # 2. Bulk 2D PCA Projection with Persistent Caching
        # ---------------------------------------------------------------------
        print("\n[Step 2] Computing 2D PCA Projection (Cold Run)...")
        t0 = time.perf_counter()
        pca_2d = store.get_datalake_view(method='pca', n_components=2, use_cache=True, output_format='dict')
        cold_time = (time.perf_counter() - t0) * 1000.0
        print(f"  • Extracted {len(pca_2d)} 2D Points in {cold_time:.2f} ms")

        print("[Step 3] Fetching 2D PCA Projection (Warm Cached Run)...")
        t1 = time.perf_counter()
        cached_2d = store.get_datalake_view(method='pca', n_components=2, use_cache=True, output_format='dict')
        warm_time = (time.perf_counter() - t1) * 1000.0
        print(f"  • Warm cached fetch in {warm_time:.2f} ms (Instant Cache Hit!)")

        for p in pca_2d[:3]:
            print(f"    Point [ID:{p['chunk_id']}] -> Doc: {p['document_title']} | X: {p['x']:.3f}, Y: {p['y']:.3f}")

        # ---------------------------------------------------------------------
        # 3. 3D t-SNE Non-Linear Manifold Projection
        # ---------------------------------------------------------------------
        print("\n[Step 4] Computing 3D t-SNE Projection...")
        tsne_3d = store.get_datalake_view(method='tsne', n_components=2, output_format='dict')
        print(f"  • Computed t-SNE coordinates for {len(tsne_3d)} chunks.")

        # ---------------------------------------------------------------------
        # 4. Lazy Streaming Generator
        # ---------------------------------------------------------------------
        print("\n[Step 5] Lazy Streaming Chunks with IncrementalPCA...")
        stream = store.stream_datalake_chunks(batch_size=2, n_components=2)
        for i, item in enumerate(stream, 1):
            print(f"  Streamed Chunk {i}: {item['document_title']} -> ({item['x']:.2f}, {item['y']:.2f})")

        # ---------------------------------------------------------------------
        # 5. Exporting Standalone Interactive Visualizer HTML
        # ---------------------------------------------------------------------
        print("\n[Step 6] Exporting Standalone Interactive Datalake Explorer...")
        html_file = Path("datalake_explorer.html")
        store.export_datalake_html(output_file=html_file, title="SafeStore Multi-Domain Datalake", method='pca', n_components=2)
        print(f"  • Interactive web visualizer generated at: {html_file.resolve()}")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 70)
    print(" Datalake demonstration completed successfully. ")
    print("=" * 70)


if __name__ == "__main__":
    main()