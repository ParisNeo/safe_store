import pytest
import sqlite3
import json
from pathlib import Path
import numpy as np

from safe_store import SafeStore, LogLevel, DocumentClusterer


@pytest.fixture
def clustered_store(tmp_path: Path) -> SafeStore:
    """Sets up a SafeStore populated with documents across 3 distinct semantic domains."""
    db_path = tmp_path / "test_clustering.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=40,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )

    # Domain 1: Deep Space Astronomy
    store.add_text("astronomy_telescope", "The Hubble space telescope and James Webb observe redshifted galaxies and cosmic microwave radiation.")
    store.add_text("astronomy_exoplanets", "Exoplanet atmospheres contain biosignature indicators including methane and water vapor.")

    # Domain 2: Molecular Genetics
    store.add_text("genetics_crispr", "CRISPR-Cas9 endonuclease creates targeted double-strand breaks for precise nucleotide sequence gene editing.")
    store.add_text("genetics_mrna", "Messenger RNA lipid nanoparticles deliver genetic instructions to cellular ribosomes.")

    # Domain 3: Cryptography
    store.add_text("crypto_asymmetric", "Public key asymmetric cryptography uses prime factorization and Diffie-Hellman discrete logarithms.")
    store.add_text("crypto_zkp", "Zero-knowledge succinct proofs allow verifying computations without disclosing confidential witness data.")

    return store


class TestDocumentClusterer:

    def test_compute_document_centroids(self, clustered_store: SafeStore):
        """Test document centroid computation and normalization."""
        clusterer = DocumentClusterer(clustered_store)
        doc_ids, file_paths, centroids, doc_info_map = clusterer.compute_document_centroids()

        assert len(doc_ids) == 6
        assert len(file_paths) == 6
        assert centroids.shape == (6, 384)
        # Verify unit normalization: norm ≈ 1.0
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_cluster_documents_auto_k(self, clustered_store: SafeStore):
        """Test automatic k-estimation and document grouping."""
        clusters = clustered_store.cluster_documents(n_clusters='auto', method='kmeans', generate_themes=True)

        assert len(clusters) >= 2
        total_docs = sum(c["document_count"] for c in clusters)
        assert total_docs == 6

        # Check theme structure
        for c in clusters:
            assert "cluster_id" in c
            assert "theme_title" in c
            assert len(c["theme_title"]) > 0
            assert "theme_description" in c
            assert "key_topics" in c
            assert "documents" in c
            assert c["document_count"] == len(c["documents"])

    def test_cluster_documents_agglomerative(self, clustered_store: SafeStore):
        """Test hierarchical agglomerative clustering method."""
        clusters = clustered_store.cluster_documents(n_clusters=3, method='agglomerative', generate_themes=False)

        assert len(clusters) == 3
        total_docs = sum(c["document_count"] for c in clusters)
        assert total_docs == 6

    def test_theme_generation_with_llm_callable(self, tmp_path: Path):
        """Test that a custom LLM generator callable is used to synthesize rich theme titles."""
        mock_generator_calls = []

        def mock_llm(prompt: str, system_prompt: str = None, json_mode: bool = False, **kwargs) -> str:
            mock_generator_calls.append(prompt)
            return json.dumps({
                "theme_title": "Quantum Cryptography & Zero-Knowledge Systems",
                "theme_description": "Documents focusing on mathematical proof protocols and asymmetric encryption.",
                "key_topics": ["cryptography", "zero-knowledge", "security", "snarks"]
            })

        db_path = tmp_path / "test_llm_themes.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            llm_generator=mock_llm
        )

        store.add_text("doc_a", "Zero-knowledge proofs and arithmetic circuits.")
        store.add_text("doc_b", "Diffie-Hellman asymmetric key exchange protocols.")

        clusters = store.cluster_documents(n_clusters=1, generate_themes=True)
        assert len(clusters) == 1
        assert len(mock_generator_calls) >= 1

        theme = clusters[0]
        assert theme["theme_title"] == "Quantum Cryptography & Zero-Knowledge Systems"
        assert "mathematical proof protocols" in theme["theme_description"]
        assert "zero-knowledge" in theme["key_topics"]

        store.close()

    def test_cluster_caching_and_invalidation(self, clustered_store: SafeStore):
        """Test that clusters persist in SQLite cache and invalidate when new documents are added."""
        # 1. Compute and save
        clusters_1 = clustered_store.cluster_documents(n_clusters=3, save_to_store=True)
        assert len(clusters_1) == 3

        # 2. Retrieve from cache
        cached = clustered_store.get_document_clusters(use_cache=True)
        assert cached is not None
        assert len(cached) == 3
        assert cached[0]["theme_title"] == clusters_1[0]["theme_title"]

        # 3. Add a new document -> Must invalidate cache
        clustered_store.add_text("new_ai_doc", "Transformer neural networks and self-attention mechanisms.")

        # Cache was cleared in db.clear_projection_cache
        fresh_cache = clustered_store.clusterer.get_cached_clusters()
        assert fresh_cache is None

        # Re-clustering includes the new document (total 7 docs)
        clusters_2 = clustered_store.cluster_documents(n_clusters='auto')
        total_docs_after = sum(c["document_count"] for c in clusters_2)
        assert total_docs_after == 7