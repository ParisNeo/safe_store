import pytest
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

from safe_store import SafeStore, LogLevel
from safe_store.search.fusion import reciprocal_rank_fusion, weighted_score_fusion
from safe_store.search.bm25 import BM25Retriever


@pytest.fixture
def hybrid_store(tmp_path: Path) -> SafeStore:
    """Sets up a SafeStore configured for hybrid search evaluation."""
    db_path = tmp_path / "test_hybrid.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )
    return store


@pytest.fixture
def populated_hybrid_store(hybrid_store: SafeStore) -> SafeStore:
    """Adds documents designed to test complementary dense and lexical (BM25) search."""
    doc1 = (
        "Project Apollo: Internal mission documentation. "
        "Error code ERR-4091 occurred in subsystem telemetry. "
        "Telemetry module failed due to buffer overflow."
    )
    doc2 = (
        "Project Artemis: Deep space lunar exploration mission. "
        "The spacecraft was designed to carry crew to the lunar surface. "
        "Propulsion systems utilize cryogenic liquid hydrogen."
    )
    doc3 = (
        "Technical troubleshooting manual. "
        "For resolving error code ERR-4091, replace the memory buffer chip and restart."
    )

    hybrid_store.add_text("apollo_doc", doc1, metadata={"project": "Apollo"})
    hybrid_store.add_text("artemis_doc", doc2, metadata={"project": "Artemis"})
    hybrid_store.add_text("manual_doc", doc3, metadata={"type": "Manual"})

    return hybrid_store


class TestBM25Retriever:

    def test_bm25_exact_code_matching(self, populated_hybrid_store: SafeStore):
        """Test that BM25 retrieves exact technical tokens like error codes with top rank."""
        bm25 = BM25Retriever(conn=populated_hybrid_store.conn)
        results = bm25.search("ERR-4091", top_k=5)

        assert len(results) >= 2
        # Both apollo_doc and manual_doc contain ERR-4091
        top_paths = [r["file_path"] for r in results]
        assert "apollo_doc" in top_paths
        assert "manual_doc" in top_paths
        assert "artemis_doc" not in top_paths[:2]

    def test_bm25_empty_query(self, populated_hybrid_store: SafeStore):
        """Test BM25 search with empty string."""
        bm25 = BM25Retriever(conn=populated_hybrid_store.conn)
        results = bm25.search("", top_k=5)
        assert results == []


class TestRankFusionAlgorithms:

    def test_reciprocal_rank_fusion_logic(self):
        """Test RRF mathematical formula: RRF(d) = sum(w / (k + rank))."""
        list_a = [
            {"chunk_id": 1, "score": 0.95},
            {"chunk_id": 2, "score": 0.85},
            {"chunk_id": 3, "score": 0.70},
        ]
        list_b = [
            {"chunk_id": 2, "score": 12.5},
            {"chunk_id": 1, "score": 8.0},
            {"chunk_id": 4, "score": 5.0},
        ]

        fused = reciprocal_rank_fusion(
            ranked_lists=[list_a, list_b],
            weights=[1.0, 1.0],
            k=60,
            top_k=5
        )

        assert len(fused) == 4
        # Chunk 1: rank 1 in list_a (1/(60+1)), rank 2 in list_b (1/(60+2))
        # Chunk 2: rank 2 in list_a (1/(60+2)), rank 1 in list_b (1/(60+1))
        # Chunk 1 and Chunk 2 should have identical fused score
        assert abs(fused[0]["fused_score"] - fused[1]["fused_score"]) < 1e-6
        assert {fused[0]["chunk_id"], fused[1]["chunk_id"]} == {1, 2}

    def test_weighted_score_fusion_logic(self):
        """Test normalized convex combination of heterogeneous score lists."""
        dense_results = [
            {"chunk_id": 10, "score": 0.9},
            {"chunk_id": 20, "score": 0.3},
        ]
        bm25_results = [
            {"chunk_id": 20, "score": 10.0},
            {"chunk_id": 10, "score": 2.0},
        ]

        fused = weighted_score_fusion(
            scored_lists=[dense_results, bm25_results],
            weights=[0.5, 0.5],
            top_k=2
        )
        assert len(fused) == 2
        assert "fused_score" in fused[0]


class TestEndToEndHybridQuery:

    def test_hybrid_query_combines_vector_and_bm25(self, populated_hybrid_store: SafeStore):
        """
        Test calling store.hybrid_query(...) combining dense semantic search
        and sparse BM25 search.
        """
        with populated_hybrid_store:
            # Query combining a semantic description with an exact error token
            query = "telemetry failure troubleshooting ERR-4091"
            results = populated_hybrid_store.hybrid_query(
                query_text=query,
                top_k=3,
                dense_weight=0.5,
                bm25_weight=0.5,
                rrf_k=60
            )

            assert len(results) > 0
            first_result = results[0]
            assert "chunk_text" in first_result
            assert "fused_score" in first_result
            assert "file_path" in first_result

            # The top results should be apollo_doc or manual_doc
            assert first_result["file_path"] in ["apollo_doc", "manual_doc"]