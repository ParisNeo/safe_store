import pytest
import sqlite3
import json
from pathlib import Path
import numpy as np

from safe_store import SafeStore, LogLevel
from safe_store.datalake.viewer import DatalakeViewer


@pytest.fixture
def populated_datalake_store(tmp_path: Path) -> SafeStore:
    """Provides a store populated with diverse semantic topic clusters."""
    db_path = tmp_path / "test_datalake.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=40,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )

    # Document Cluster 1: Astronomy & Physics
    doc1 = (
        "Astronomy is the scientific study of celestial objects and phenomena. "
        "The Hubble space telescope provided deep field views of distant galaxies. "
        "Gravitational waves ripple across spacetime when neutron stars merge."
    )
    # Document Cluster 2: Biology & Genetics
    doc2 = (
        "DNA transcription and translation govern protein synthesis in living cells. "
        "Mitochondria generate cellular energy through oxidative phosphorylation. "
        "CRISPR gene editing enables precise modifications of nucleotide sequences."
    )
    # Document Cluster 3: Computer Science & Cryptography
    doc3 = (
        "Asymmetric cryptography uses public and private key pairs for secure encryption. "
        "Zero-knowledge proofs allow verification of statements without revealing information. "
        "Distributed hash tables provide decentralized key-value storage across peers."
    )

    store.add_text("astronomy_doc", doc1, metadata={"topic": "Physics", "domain": "Space"})
    store.add_text("biology_doc", doc2, metadata={"topic": "Biology", "domain": "Genetics"})
    store.add_text("crypto_doc", doc3, metadata={"topic": "Security", "domain": "Cryptography"})

    return store


class TestDatalakeViewer:

    def test_pca_2d_projection_dict(self, populated_datalake_store: SafeStore):
        """Test standard PCA 2D point cloud generation."""
        data = populated_datalake_store.get_datalake_view(method='pca', n_components=2, output_format='dict')
        
        assert isinstance(data, list)
        assert len(data) >= 3
        
        first = data[0]
        assert "chunk_id" in first
        assert "document_title" in first
        assert "x" in first
        assert "y" in first
        assert "z" not in first
        assert "metadata" in first
        assert isinstance(first["x"], float)
        assert isinstance(first["y"], float)

    def test_pca_3d_projection(self, populated_datalake_store: SafeStore):
        """Test 3D PCA projection coordinates."""
        data = populated_datalake_store.get_datalake_view(method='pca', n_components=3, output_format='dict')
        
        assert isinstance(data, list)
        assert len(data) >= 3
        first = data[0]
        assert "x" in first
        assert "y" in first
        assert "z" in first
        assert isinstance(first["z"], float)

    def test_tsne_2d_projection(self, populated_datalake_store: SafeStore):
        """Test t-SNE 2D clustering projection."""
        data = populated_datalake_store.get_datalake_view(method='tsne', n_components=2, output_format='dict')
        
        assert isinstance(data, list)
        assert len(data) >= 3
        first = data[0]
        assert "x" in first
        assert "y" in first
        assert isinstance(first["x"], float)

    def test_projection_caching_and_invalidation(self, populated_datalake_store: SafeStore):
        """Test that projection caches work and invalidate upon document additions."""
        # 1. First run computes and caches
        res1 = populated_datalake_store.get_datalake_view(method='pca', n_components=2, use_cache=True, output_format='dict')
        
        # 2. Second run reads from SQLite store_metadata cache
        res2 = populated_datalake_store.get_datalake_view(method='pca', n_components=2, use_cache=True, output_format='dict')
        assert len(res1) == len(res2)
        assert res1[0]["x"] == res2[0]["x"]

        # 3. Add new document -> Cache must be cleared automatically
        populated_datalake_store.add_text("new_doc", "Additional content on quantum chemistry.")
        res3 = populated_datalake_store.get_datalake_view(method='pca', n_components=2, use_cache=True, output_format='dict')
        assert len(res3) > len(res1)

    def test_streaming_lazy_chunks(self, populated_datalake_store: SafeStore):
        """Test stream_datalake_chunks generator for memory-efficient loading."""
        stream = populated_datalake_store.stream_datalake_chunks(batch_size=2, n_components=2)
        
        items = list(stream)
        assert len(items) >= 3
        assert "x" in items[0]
        assert "y" in items[0]
        assert "document_title" in items[0]

    def test_export_datalake_html(self, populated_datalake_store: SafeStore, tmp_path: Path):
        """Test interactive standalone HTML datalake visualizer generation."""
        out_html = tmp_path / "datalake_export.html"
        generated_path = populated_datalake_store.export_datalake_html(
            output_file=out_html,
            title="Custom Datalake Visualizer",
            method='pca',
            n_components=2
        )

        assert generated_path.exists()
        content = generated_path.read_text(encoding='utf-8')
        assert "Custom Datalake Visualizer" in content
        assert "Plotly.newPlot" in content
        assert "astronomy_doc" in content

    def test_output_formats(self, populated_datalake_store: SafeStore):
        """Test output formatting: json_str, csv, and dict."""
        json_out = populated_datalake_store.get_datalake_view(output_format='json_str')
        assert isinstance(json_out, str)
        parsed = json.loads(json_out)
        assert len(parsed) >= 3

        csv_out = populated_datalake_store.get_datalake_view(output_format='csv')
        assert isinstance(csv_out, str)
        assert "chunk_id" in csv_out
        assert "document_title" in csv_out