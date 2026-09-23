import pytest
from pathlib import Path
from safe_store import (
    SafeStore, LogLevel,
    reconstruct_overlapping_chunks,
    merge_overlapping_texts,
    strip_metadata_header,
    format_metadata_header
)


class TestMergeOverlappingTexts:

    def test_merge_exact_overlap(self):
        text_a = "The quick brown fox jumps over the lazy dog."
        text_b = "jumps over the lazy dog. The cat slept in the corner."
        merged = merge_overlapping_texts(text_a, text_b)
        expected = "The quick brown fox jumps over the lazy dog. The cat slept in the corner."
        assert merged == expected

    def test_merge_whitespace_boundary_overlap(self):
        text_a = "Step 1: Install Python.\nStep 2: Install dependencies.\n"
        text_b = "Step 2: Install dependencies.\nStep 3: Run the tests."
        merged = merge_overlapping_texts(text_a, text_b)
        assert "Step 1: Install Python." in merged
        assert "Step 2: Install dependencies." in merged
        assert "Step 3: Run the tests." in merged
        # Verify Step 2 is not duplicated
        assert merged.count("Step 2: Install dependencies.") == 1

    def test_merge_substring_containment(self):
        text_a = "Complete sentence with full information."
        text_b = "with full information."
        assert merge_overlapping_texts(text_a, text_b) == text_a

    def test_merge_with_breadcrumbs(self):
        text_a = "[Section: Setup]\n\nDownload the package from PyPI."
        text_b = "[Section: Setup]\n\npackage from PyPI. Run pip install."
        merged = merge_overlapping_texts(text_a, text_b)
        assert merged.startswith("[Section: Setup]\n\n")
        assert merged.count("[Section: Setup]") == 1
        assert "Download the package from PyPI. Run pip install." in merged


@pytest.fixture
def store_with_overlapping_docs(tmp_path: Path) -> SafeStore:
    db_path = tmp_path / "test_reconstruction.db"
    store = SafeStore(
        db_path=str(db_path),
        vectorizer_name="st",
        chunk_size=35,
        chunk_overlap=10,
        chunking_strategy="character",
        log_level=LogLevel.DEBUG
    )

    doc_text = (
        "Part 1: In the beginning of the journey, Alice set forth into the dark forest.\n"
        "Part 2: Alice encountered an ancient stone monolith inscribed with glowing runes.\n"
        "Part 3: The glowing runes described a hidden vault deep beneath the mountains.\n"
        "Part 4: Decades later, travelers still spoke of the secret vault discovered by Alice."
    )
    store.add_text("alice_chronicle", doc_text, metadata={"author": "Alice", "genre": "Adventure"})

    doc_unrelated = "Different document about modern quantum computing and entanglement."
    store.add_text("quantum_notes", doc_unrelated, metadata={"topic": "Physics"})

    return store


class TestReconstructOverlappingChunks:

    def test_reconstruct_reorders_and_deduplicates_contiguous_chunks(self, store_with_overlapping_docs: SafeStore):
        """
        Tests that when adjacent chunks are returned in reversed score order,
        reconstruct_overlapping_chunks sorts them chronologically and removes duplicate seams.
        """
        # Simulated raw query results where seq 1 ranked higher than seq 0
        raw_results = [
            {
                "chunk_id": 102,
                "doc_id": 1,
                "chunk_seq": 1,
                "raw_chunk_text": "Alice set forth into the dark forest.\nPart 2: Alice encountered an ancient",
                "chunk_text": "Alice set forth into the dark forest.\nPart 2: Alice encountered an ancient",
                "file_path": "alice_chronicle",
                "document_metadata": {"author": "Alice"},
                "similarity_score": 0.95,
                "similarity_percent": 95.0,
                "relevance_score": 95.0
            },
            {
                "chunk_id": 101,
                "doc_id": 1,
                "chunk_seq": 0,
                "raw_chunk_text": "Part 1: In the beginning of the journey, Alice set forth into the dark forest.",
                "chunk_text": "Part 1: In the beginning of the journey, Alice set forth into the dark forest.",
                "file_path": "alice_chronicle",
                "document_metadata": {"author": "Alice"},
                "similarity_score": 0.85,
                "similarity_percent": 85.0,
                "relevance_score": 85.0
            }
        ]

        reconstructed = store_with_overlapping_docs.reconstruct_overlapping_chunks(raw_results, add_metadata=True)

        assert len(reconstructed) == 1
        doc = reconstructed[0]
        assert doc["file_path"] == "alice_chronicle"
        assert doc["relevance_score"] == 95.0
        assert doc["chunk_seqs"] == [0, 1]
        assert doc["fused_chunk_ids"] == [101, 102]

        text = doc["chunk_text"]
        # Must start with Part 1 (chronological order restored)
        assert "Part 1: In the beginning" in text
        assert "Part 2: Alice encountered" in text
        # Overlapping boundary phrase "Alice set forth into the dark forest." must appear only once!
        assert text.count("Alice set forth into the dark forest.") == 1

        # Single metadata block present
        assert text.count("--- Document Context ---") == 1
        assert "Author: Alice" in text

    def test_reconstruct_non_contiguous_chunks_with_ellipsis_gap(self, store_with_overlapping_docs: SafeStore):
        """
        Tests that when non-contiguous chunks from the same document are retrieved,
        they are separated by a '...' placeholder to signal missing text.
        """
        raw_results = [
            {
                "chunk_id": 101,
                "doc_id": 1,
                "chunk_seq": 0,
                "raw_chunk_text": "Part 1: Journey began in the forest.",
                "chunk_text": "Part 1: Journey began in the forest.",
                "file_path": "alice_chronicle",
                "document_metadata": {"author": "Alice"},
                "relevance_score": 88.0
            },
            {
                "chunk_id": 104,
                "doc_id": 1,
                "chunk_seq": 4,
                "raw_chunk_text": "Part 4: Decades later the secret vault remained.",
                "chunk_text": "Part 4: Decades later the secret vault remained.",
                "file_path": "alice_chronicle",
                "document_metadata": {"author": "Alice"},
                "relevance_score": 92.0
            }
        ]

        reconstructed = store_with_overlapping_docs.reconstruct_overlapping_chunks(raw_results, add_metadata=False)

        assert len(reconstructed) == 1
        text = reconstructed[0]["chunk_text"]

        # Verifies '...' placeholder sits between non-contiguous chunks
        assert "Part 1: Journey began in the forest." in text
        assert "..." in text
        assert "Part 4: Decades later" in text
        assert reconstructed[0]["cluster_count"] == 2
        # Without metadata
        assert "--- Document Context ---" not in text

    def test_query_with_reconstruct_overlapping_chunks_flag(self, store_with_overlapping_docs: SafeStore):
        """
        Tests query() with reconstruct_overlapping_chunks=True natively fuses adjacent chunks.
        """
        with store_with_overlapping_docs:
            results = store_with_overlapping_docs.query(
                "glowing runes ancient stone monolith",
                top_k=4,
                reconstruct_overlapping_chunks=True
            )

            assert len(results) > 0
            first = results[0]
            assert first["file_path"] == "alice_chronicle"
            assert first.get("is_reconstructed") is True
            assert "glowing runes" in first["chunk_text"]
            assert first["relevance_score"] > 0
            # Metadata block appears once
            assert first["chunk_text"].count("--- Document Context ---") <= 1

    def test_hybrid_query_with_reconstruct_overlapping_chunks_flag(self, store_with_overlapping_docs: SafeStore):
        """
        Tests hybrid_query() with reconstruct_overlapping_chunks=True.
        """
        with store_with_overlapping_docs:
            results = store_with_overlapping_docs.hybrid_query(
                "monolith runes Alice",
                top_k=2,
                reconstruct_overlapping_chunks=True
            )

            assert len(results) > 0
            first = results[0]
            assert first.get("is_reconstructed") is True
            assert first["file_path"] == "alice_chronicle"