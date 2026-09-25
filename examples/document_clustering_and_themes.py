"""
Example demonstrating SafeStore's Document Clustering and Thematic Synthesis:

1. Ingests a diverse multi-domain corpus (Physics/Astronomy, Genetics/CRISPR, Cryptography/Security).
2. Computes normalized document-level semantic centroids.
3. Automatically clusters documents into cohesive thematic groups.
4. Generates conceptual theme titles, descriptions, and topic tags (using custom LLM callable or c-TF-IDF lexical fallback).
5. Inspects member documents and executes theme-targeted semantic queries.
"""

from pathlib import Path
import json
from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "clustering_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore Document Clustering & Thematic Grouping Engine ")
    print("=" * 80)

    # 1. Initialize SafeStore
    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.INFO
    )

    with store:
        print("\n[Step 1] Ingesting Multi-Domain Corpus (6 Documents across 3 Domains)...")

        # Domain A: Space Exploration & Astronomy
        store.add_text(
            unique_id="hubble_telescope_deep_field",
            text="The Hubble Space Telescope captured deep field images of galaxies formed shortly after the Big Bang. "
                 "Astronomers observe redshifted spectral lines to measure the accelerating cosmic expansion.",
            metadata={"domain": "Astronomy", "platform": "Hubble"}
        )
        store.add_text(
            unique_id="james_webb_spectroscopy",
            text="The James Webb Space Telescope uses infrared instruments to analyze exoplanet atmospheres. "
                 "Transmission spectroscopy detects water vapor, carbon dioxide, and biosignature candidate molecules.",
            metadata={"domain": "Astronomy", "platform": "JWST"}
        )

        # Domain B: Molecular Biology & Genetics
        store.add_text(
            unique_id="crispr_gene_editing_mechanisms",
            text="CRISPR-Cas9 endonuclease complexes introduce targeted double-strand breaks in genomic DNA sequences. "
                 "Cellular homology-directed repair pathways facilitate precision nucleotide modifications and gene knockouts.",
            metadata={"domain": "Genetics", "tool": "CRISPR"}
        )
        store.add_text(
            unique_id="messenger_rna_vaccine_synthesis",
            text="Synthetic mRNA lipid nanoparticles deliver antigen blueprints directly to the cellular cytoplasm. "
                 "Ribosomes translate the modified ribonucleotide sequences, eliciting robust humoral immune responses.",
            metadata={"domain": "Genetics", "technology": "mRNA"}
        )

        # Domain C: Computer Systems & Cryptography
        store.add_text(
            unique_id="asymmetric_cryptography_rsa",
            text="Public key asymmetric cryptography relies on mathematically intractable prime factorization problems. "
                 "Diffie-Hellman key exchanges enable confidential symmetric session keys across unencrypted channels.",
            metadata={"domain": "Cryptography", "primitive": "RSA"}
        )
        store.add_text(
            unique_id="zero_knowledge_succinct_proofs",
            text="Zero-knowledge SNARK proofs allow a prover to verify valid state execution without disclosing underlying witnesses. "
                 "Arithmetic circuits compile computational logic into polynomial constraints verified on blockchain ledgers.",
            metadata={"domain": "Cryptography", "primitive": "zk-SNARK"}
        )

        print("✓ Ingested 6 multi-paragraph documents into vector and SQLite store.")

        # ---------------------------------------------------------------------
        # 2. Cluster Documents with Automatic Optimal k
        # ---------------------------------------------------------------------
        print("\n[Step 2] Executing Semantic Clustering with Auto-K Selection...")
        clusters = store.cluster_documents(
            n_clusters='auto',       # Automatically estimates optimal k=3
            method='kmeans',         # Fast K-Means over normalized centroids
            generate_themes=True,    # Generates titles, descriptions, and tags
            save_to_store=True       # Persists in SQLite for instant cache hits
        )

        print(f"\nDiscovered {len(clusters)} Thematic Clusters:\n")
        for c in clusters:
            print("=" * 70)
            print(f"🎨 Theme #{c['cluster_id'] + 1}: {c['theme_title']}")
            print(f"   Summary: {c['theme_description']}")
            print(f"   Topic Tags: {', '.join(c['key_topics'])}")
            print(f"   Member Documents ({c['document_count']}):")
            for doc in c['documents']:
                print(f"     • {doc['document_title']} (ID #{doc['doc_id']})")
            print()

        # ---------------------------------------------------------------------
        # 3. Retrieve Cached Clusters
        # ---------------------------------------------------------------------
        print("\n[Step 3] Verifying Persistent Retrieval from Database Cache...")
        cached_clusters = store.get_document_clusters(use_cache=True)
        assert len(cached_clusters) == len(clusters)
        print(f"✓ Retrieved {len(cached_clusters)} clusters instantly from SQLite cache!")

        # ---------------------------------------------------------------------
        # 4. Filtered Querying using Theme Insights
        # ---------------------------------------------------------------------
        first_theme = clusters[0]
        query_text = first_theme["theme_title"]
        print(f"\n[Step 4] Querying SafeStore with Discovered Theme: '{query_text}'...")
        results = store.query(query_text, top_k=2)
        for r in results:
            print(f"  Rank: {r['document_title']} (Relevance: {r['relevance_score']:.1f}%)")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" Document clustering and thematic synthesis demo complete! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()