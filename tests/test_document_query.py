import pytest
from pathlib import Path
from safe_store import SafeStore, LogLevel


@pytest.fixture
def document_query_store(tmp_path: Path) -> SafeStore:
    """Provides a populated SafeStore for document-level and windowing queries."""
    db_path = tmp_path / "test_doc_query.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=30,
        chunk_overlap=5,
        log_level=LogLevel.DEBUG
    )

    doc1_content = (
        "Chapter 1: Network Protocols.\n"
        "Transmission Control Protocol guarantees reliable packet delivery.\n"
        "User Datagram Protocol provides low latency connectionless transport.\n"
        "Border Gateway Protocol coordinates routing between autonomous systems."
    )
    doc2_content = (
        "Chapter 2: Database Storage Engines.\n"
        "Write Ahead Logging guarantees atomic transactions and durability.\n"
        "B-Tree indexes enable logarithmic search complexity.\n"
        "LSM Trees optimize write-heavy workloads via append-only SSTables."
    )
    doc3_content = (
        "Chapter 3: Memory Architectures.\n"
        "CPU cache hierarchies reduce memory latency.\n"
        "Non-Uniform Memory Access clusters multi-socket systems."
    )

    store.add_text("protocols_doc", doc1_content, metadata={"chapter": 1, "topic": "Networking"})
    store.add_text("databases_doc", doc2_content, metadata={"chapter": 2, "topic": "Storage"})
    store.add_text("memory_doc", doc3_content, metadata={"chapter": 3, "topic": "Hardware"})

    return store


class TestFullDocumentAndWindowQuery:

    def test_query_full_documents_returns_entire_document(self, document_query_store: SafeStore):
        """Test query_full_documents returns complete reconstructed text and document ranking."""
        results = document_query_store.query_full_documents(
            query_text="Write Ahead Logging durability transactions",
            top_k_docs=2,
            search_mode='hybrid'
        )

        assert len(results) > 0
        top_doc = results[0]
        assert top_doc["file_path"] == "databases_doc"
        assert "Write Ahead Logging" in top_doc["full_text"]
        assert "B-Tree indexes" in top_doc["full_text"]
        assert "LSM Trees" in top_doc["full_text"]
        assert top_doc["metadata"]["topic"] == "Storage"
        assert top_doc["hit_chunk_count"] > 0
        assert "matching_chunks" in top_doc

    def test_query_document_content_window_surrounding_chunks(self, document_query_store: SafeStore):
        """Test query_document_content_window expands context before and after hit chunk."""
        windows = document_query_store.query_document_content_window(
            query_text="User Datagram Protocol",
            top_k_hits=1,
            window_before=1,
            window_after=1,
            search_mode='hybrid'
        )

        assert len(windows) == 1
        window = windows[0]
        assert window["file_path"] == "protocols_doc"
        assert "surrounding_chunks" in window
        assert len(window["surrounding_chunks"]) >= 2
        assert "stitched_window_text" in window

        # Target chunk must be flagged
        target_chunk = next(c for c in window["surrounding_chunks"] if c["is_target_hit"])
        assert "User Datagram Protocol" in target_chunk["chunk_text"] or "Datagram" in target_chunk["chunk_text"]

    def test_get_document_content_paginated(self, document_query_store: SafeStore):
        """Test get_document_content_paginated chunk pagination."""
        page1 = document_query_store.get_document_content_paginated("databases_doc", page=1, page_size=2)
        
        assert page1["page"] == 1
        assert page1["page_size"] == 2
        assert page1["total_chunks"] >= 2
        assert len(page1["chunks"]) <= 2
        assert page1["has_previous_page"] is False
        assert "stitched_text" in page1

        if page1["total_pages"] > 1:
            page2 = document_query_store.get_document_content_paginated("databases_doc", page=2, page_size=2)
            assert page2["page"] == 2
            assert page2["has_previous_page"] is True

    def test_reconstruct_document_text_fallback(self, document_query_store: SafeStore):
        """Test direct document reconstruction."""
        full_text = document_query_store.reconstruct_document_text("protocols_doc")
        assert full_text is not None
        assert "Transmission Control Protocol" in full_text
        assert "Border Gateway Protocol" in full_text