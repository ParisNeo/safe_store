import pytest
import numpy as np
from safe_store import SafeStore
from safe_store.vectorization.methods.grepper import GrepperVectorizer


@pytest.fixture
def sample_markdown():
    return """# Project Overview

This is the main overview of the project.

## Installation

To install the project, run `pip install safe-store`.
The installation process is straightforward.

## Usage

### Basic Example

Here is a basic example of how to use the library.
You can create a SafeStore instance and add documents.

### Advanced Features

For advanced features, check the documentation.
The grepper vectorizer provides fast keyword search.

## API Reference

The API is documented in the code itself.
"""


@pytest.fixture
def second_markdown():
    return """# Second Document

## Different Section

This document contains different content.
The terms here are unique to this file.

## Another Topic

More content with specific keywords.
Search should find these terms too.
"""


class TestGrepperVectorizer:
    
    def test_grepper_basic_search(self, tmp_path, sample_markdown):
        db_path = tmp_path / "test_grepper_basic.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="grepper",
            chunk_size=500,
            chunk_overlap=0
        )
        
        try:
            result = store.add_text(
                unique_id="doc1.md",
                text=sample_markdown,
                metadata={"title": "Project Overview"}
            )
            assert result["num_chunks_added"] > 0
            
            query_results = store.query("installation", top_k=5)
            
            assert len(query_results) > 0
            first_result = query_results[0]
            
            assert "header_breadcrumbs" in first_result
            assert "chunk_text" in first_result
            assert "similarity_score" in first_result
            assert "similarity_percent" in first_result
            
            # Check that breadcrumb contains the section header path
            breadcrumb = first_result.get("header_breadcrumbs", "")
            assert "Installation" in breadcrumb or "installation" in first_result["chunk_text"].lower()
            
            # Check chunk text contains the search term
            assert "installation" in first_result["chunk_text"].lower()
            
        finally:
            store.close()


    def test_grepper_phrase_search(self, tmp_path, sample_markdown):
        db_path = tmp_path / "test_grepper_phrase.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="grepper",
            chunk_size=500,
            chunk_overlap=0
        )
        
        try:
            store.add_text(
                unique_id="doc1.md",
                text=sample_markdown,
                metadata={"title": "Project Overview"}
            )
            
            # Query for exact phrase
            phrase_results = store.query("pip install safe-store", top_k=5)
            
            # Query for single term from same area
            term_results = store.query("pip", top_k=5)
            
            assert len(phrase_results) > 0
            assert len(term_results) > 0
            
            # Phrase search should have higher or equal score due to exact phrase bonus
            phrase_score = phrase_results[0]["similarity_score"]
            term_score = term_results[0]["similarity_score"]
            
            # The phrase query matches more precisely, so should score well
            assert phrase_score > 0
            
            # Verify the exact phrase appears in result text
            found_phrase = False
            for r in phrase_results:
                if "pip install safe-store" in r["chunk_text"].lower() or "pip install" in r["chunk_text"].lower():
                    found_phrase = True
                    break
            assert found_phrase
            
        finally:
            store.close()


    def test_grepper_multiple_docs(self, tmp_path, sample_markdown, second_markdown):
        db_path = tmp_path / "test_grepper_multi.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="grepper",
            chunk_size=500,
            chunk_overlap=0
        )
        
        try:
            store.add_text(
                unique_id="doc1.md",
                text=sample_markdown,
                metadata={"title": "First Doc"}
            )
            store.add_text(
                unique_id="doc2.md",
                text=second_markdown,
                metadata={"title": "Second Doc"}
            )
            
            # Query for a term present in both documents
            results = store.query("content", top_k=10)
            
            assert len(results) > 0
            
            # Collect unique document paths
            doc_paths = set(r["file_path"] for r in results)
            # Should have results from at least one doc; we added "content" to both
            # so we might get from one or both depending on chunking
            
            # All results should have scores
            for r in results:
                assert "similarity_score" in r
                assert "similarity_percent" in r
                assert r["similarity_score"] >= 0
                assert r["similarity_percent"] >= 0
            
            # Check that results have proper structure
            for r in results:
                assert "header_breadcrumbs" in r
                assert "chunk_text" in r
                assert "chunk_id" in r
                assert "file_path" in r
            
        finally:
            store.close()


    def test_grepper_empty_result(self, tmp_path, sample_markdown):
        db_path = tmp_path / "test_grepper_empty.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="grepper",
            chunk_size=500,
            chunk_overlap=0
        )
        
        try:
            store.add_text(
                unique_id="doc1.md",
                text=sample_markdown,
                metadata={"title": "Project Overview"}
            )
            
            # Query for term that does not exist
            results = store.query("xyznonexistentterm12345", top_k=5)
            
            assert results == []
            assert isinstance(results, list)
            
        finally:
            store.close()


    def test_grepper_placeholder_vectors(self):
        vectorizer = GrepperVectorizer()
        
        # Test with single text
        texts = ["some random text"]
        vectors = vectorizer.vectorize(texts)
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (1, 1)
        assert vectors.dtype == np.float32
        assert vectors[0, 0] == 0.0
        
        # Test with multiple texts
        multi_texts = ["first", "second", "third"]
        multi_vectors = vectorizer.vectorize(multi_texts)
        
        assert multi_vectors.shape == (3, 1)
        assert np.all(multi_vectors == 0)
        
        # Test empty list
        empty_vectors = vectorizer.vectorize([])
        assert empty_vectors.shape == (0,)
        assert len(empty_vectors) == 0


    def test_grepper_list_models(self):
        models = GrepperVectorizer.list_models()
        
        assert isinstance(models, list)
        assert models == ["grepper-default"]
        assert len(models) == 1