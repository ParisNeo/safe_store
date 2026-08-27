import pytest
from pathlib import Path
from safe_store import SafeStore, LogLevel, BM25Retriever


@pytest.fixture
def scored_store(tmp_path: Path) -> SafeStore:
    """Sets up a SafeStore with sample docs for relevance grade and threshold testing."""
    db_path = tmp_path / "test_relevance_scores.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=40,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )

    doc_networking = "Subsystem Alpha: BGP border routing protocol and TCP packet stream telemetry."
    doc_security = "Subsystem Gamma: Zero-knowledge proof authentication and TLS cryptographic handshakes."

    store.add_text("net_doc", doc_networking, metadata={"topic": "Networking"})
    store.add_text("sec_doc", doc_security, metadata={"topic": "Security"})

    return store


class TestRelevanceGradingAndThresholding:

    def test_dense_query_relevance_grade_and_threshold(self, scored_store: SafeStore):
        """Test dense vector query applies 0-100 grade and filters out non-matching queries."""
        # Relevant query
        hits = scored_store.query("BGP routing telemetry", top_k=2, min_relevance_percent=40.0)
        assert len(hits) > 0
        first = hits[0]
        assert "relevance_score" in first
        assert "similarity_percent" in first
        assert 0.0 <= first["relevance_score"] <= 100.0
        assert first["relevance_score"] >= 40.0

        # Irrelevant non-existent query with strict threshold returns empty list
        empty_hits = scored_store.query("extraterrestrial agrarian archaeology", top_k=5, min_relevance_percent=95.0)
        assert empty_hits == []

    def test_bm25_relevance_grade_and_threshold(self, scored_store: SafeStore):
        """Test BM25 search applies 0-100 grade and thresholding."""
        bm25 = BM25Retriever(scored_store.conn)
        
        # Exact matching query
        exact_hits = bm25.search("BGP", top_k=2, min_relevance_percent=10.0)
        assert len(exact_hits) > 0
        assert 0.0 <= exact_hits[0]["relevance_score"] <= 100.0

        # Non-matching query returns empty list
        no_hits = bm25.search("nonexistentxyz123", top_k=5, min_relevance_percent=10.0)
        assert no_hits == []

    def test_hybrid_query_thresholding(self, scored_store: SafeStore):
        """Test hybrid query filters out results when below threshold."""
        # Valid query
        results = scored_store.hybrid_query(
            "cryptographic handshakes TLS",
            top_k=2,
            min_relevance_percent=30.0
        )
        assert len(results) > 0
        assert 0.0 <= results[0]["relevance_score"] <= 100.0
        assert results[0]["relevance_score"] >= 30.0

        # High threshold on unrelated query
        no_results = scored_store.hybrid_query(
            "baking chocolate soufflé pastry recipe",
            top_k=3,
            min_relevance_percent=95.0
        )
        assert no_results == []

    def test_full_document_query_thresholding(self, scored_store: SafeStore):
        """Test query_full_documents excludes non-matching documents under threshold."""
        # Matching query
        docs = scored_store.query_full_documents(
            "Zero-knowledge proof authentication",
            top_k_docs=1,
            min_relevance_percent=30.0
        )
        assert len(docs) == 1
        assert docs[0]["file_path"] == "sec_doc"
        assert 0.0 <= docs[0]["relevance_score"] <= 100.0

        # High threshold on unrelated query
        no_docs = scored_store.query_full_documents(
            "hydroponic greenhouse agriculture tomato farming",
            top_k_docs=2,
            min_relevance_percent=90.0
        )
        assert no_docs == []