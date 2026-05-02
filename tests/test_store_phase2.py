"""
Phase 2 tests for DocumentStore - focusing on search, query, and TF-IDF features
"""

import os
import pytest
from unittest.mock import Mock, patch
import numpy as np

from lollmsvectordb import DocumentStore
from lollmsvectordb.vectorizers.tfidf_vectorizer import TfidfVectorizer


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test data."""
    return str(tmp_path)


@pytest.fixture
def mock_vectorizer():
    """Create a mock vectorizer for testing."""
    vectorizer = Mock()
    vectorizer.name = "MockVectorizer"
    vectorizer.dimension = 384
    
    # Mock vectorize method to return a fixed embedding
    def mock_vectorize(text):
        # Create a deterministic embedding based on text content
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)
    
    vectorizer.vectorize = mock_vectorize
    return vectorizer


class TestStoreQuery:
    """Tests for DocumentStore query functionality."""
    
    def test_query_simple(self, temp_dir, mock_vectorizer):
        """Test basic query functionality."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        # Add some documents
        store.add_document("doc1", "This is a test document about Python programming.")
        store.add_document("doc2", "Another document about Java and coding.")
        
        # Query for Python-related content
        results = store.query("Python", n_results=2)
        
        # Should return results
        assert len(results) > 0
        # First result should be the Python document
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "This is a test document about Python programming."
    
    def test_query_no_results(self, temp_dir, mock_vectorizer):
        """Test query that returns no results."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        store.add_document("doc1", "A document about Python.")
        
        # Query for something completely unrelated
        results = store.query("quantum physics astrophysics", n_results=3)
        
        # Should still return something (the closest match)
        assert isinstance(results, list)
    
    def test_query_limit_results(self, temp_dir, mock_vectorizer):
        """Test limiting the number of query results."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        # Add multiple documents
        for i in range(5):
            store.add_document(f"doc{i}", f"Document number {i} about various topics.")
        
        # Query with limit
        results = store.query("document", n_results=2)
        
        # Should return at most 2 results
        assert len(results) <= 2
    
    def test_query_persistence(self, temp_dir, mock_vectorizer):
        """Test that queries work after closing and reopening store."""
        db_path = os.path.join(temp_dir, "test.db")
        
        # Create and populate store
        store1 = DocumentStore(db_path=db_path, vectorizer=mock_vectorizer)
        store1.add_document("doc1", "Python programming guide")
        store1.close()
        
        # Reopen store
        store2 = DocumentStore(db_path=db_path, vectorizer=mock_vectorizer)
        results = store2.query("Python", n_results=1)
        
        assert len(results) > 0
        assert results[0]["id"] == "doc1"
        store2.close()
    
    def test_query_with_metadata(self, temp_dir, mock_vectorizer):
        """Test querying documents with metadata."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        store.add_document(
            "doc1",
            "Python tutorial",
            metadata={"category": "programming", "level": "beginner"}
        )
        
        results = store.query("Python", n_results=1)
        
        assert len(results) > 0
        assert results[0]["metadata"]["category"] == "programming"
        assert results[0]["metadata"]["level"] == "beginner"


class TestStoreVectorizerCompatibility:
    """Tests for vectorizer compatibility and initialization."""
    
    def test_init_vectorizer_not_found(self, temp_dir):
        """Test initialization when vectorizer class is not found."""
        db_path = os.path.join(temp_dir, "test.db")
        
        # Create a store with a vectorizer first
        store1 = DocumentStore(db_path=db_path)
        # Manually set vectorizer name to something that doesn't exist
        store1._set_vectorizer_name("NonExistentVectorizer")
        store1.close()
        
        # Try to reopen - should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            DocumentStore(db_path=db_path)
    
    def test_add_document_with_tfidf(self, temp_dir):
        """Test adding document with TF-IDF vectorizer."""
        vectorizer = TfidfVectorizer()
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=vectorizer
        )
        
        store.add_document("doc1", "This is a test document")
        # Should not raise any errors
        assert store.get_document("doc1") is not None
    
    def test_add_vectorization_incompatible(self, temp_dir):
        """Test handling incompatible vectorizers."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=None
        )
        
        # Should handle missing vectorizer gracefully
        with pytest.raises(Exception):
            store.add_document("doc1", "Test document")


class TestStoreEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_database_query(self, temp_dir, mock_vectorizer):
        """Test querying an empty database."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        # Query empty database
        results = store.query("anything", n_results=1)
        
        # Should return empty list, not crash
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_very_long_document(self, temp_dir, mock_vectorizer):
        """Test handling very long documents."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        # Create a very long text
        long_text = "word " * 10000
        
        # Should handle long documents without error
        store.add_document("long_doc", long_text)
        retrieved = store.get_document("long_doc")
        assert retrieved["text"] == long_text
    
    def test_special_characters_in_text(self, temp_dir, mock_vectorizer):
        """Test handling special characters in documents."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        special_text = "Special chars: àéèùçñ 中文 🎉 <script>alert('xss')</script>"
        
        store.add_document("special", special_text)
        retrieved = store.get_document("special")
        assert retrieved["text"] == special_text
    
    def test_unicode_normalization(self, temp_dir, mock_vectorizer):
        """Test Unicode handling in documents."""
        store = DocumentStore(
            db_path=os.path.join(temp_dir, "test.db"),
            vectorizer=mock_vectorizer
        )
        
        # Different representations of similar characters
        text1 = "café"  # é as single character
        text2 = "café"  # é as e + combining acute
        
        store.add_document("unicode1", text1)
        store.add_document("unicode2", text2)
        
        # Both should be stored as-is
        assert store.get_document("unicode1")["text"] == text1
        assert store.get_document("unicode2")["text"] == text2