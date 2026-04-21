# tests/test_store_phase2.py
import pytest
import sqlite3
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import re

# Import specific exceptions and modules
from safe_store import store as safe_store_store_module # Target for patching
from safe_store import SafeStore, LogLevel
from safe_store.core import db
from safe_store.core.exceptions import (
    ConfigurationError, VectorizationError, DatabaseError, QueryError, SafeStoreError, FileHandlingError
)
from safe_store.vectorization.methods.tf_idf import TfIdfVectorizer
from safe_store.vectorization.manager import VectorizationManager

# --- REMOVE Availability Checks and Mock Fixtures ---
# try:
#     from sentence_transformers import SentenceTransformer
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False
#     class MockSentenceTransformer: ...
#     @pytest.fixture(autouse=True)
#     def mock_st(monkeypatch): ...

# try:
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.exceptions import NotFittedError
#     SKLEARN_AVAILABLE = True
#     class MockTfidfVectorizer: ...
#     @pytest.fixture(autouse=True)
#     def mock_sklearn(monkeypatch): ...
# except ImportError:
#     SKLEARN_AVAILABLE = False
#     MockTfidfVectorizer = None
#     NotFittedError = None
#     @pytest.fixture(autouse=True)
#     def mock_sklearn(monkeypatch): pass

# --- Helper function (keep) ---
def assert_log_call_containing(mock_logger, expected_substring):
    """Checks if any call to the mock logger contained the substring."""
    found = False
    for call_args in mock_logger.call_args_list:
        args, kwargs = call_args
        if args and isinstance(args[0], str) and expected_substring in args[0]:
            found = True
            break
    if not found:
        for method_call in mock_logger.method_calls:
            call_name, args, kwargs = method_call
            if args and isinstance(args[0], str) and expected_substring in args[0]:
                found = True
                break
    assert found, f"Expected log call containing '{expected_substring}' not found in {mock_logger.call_args_list} or {mock_logger.method_calls}"


# --- Test Fixtures (Keep populated_store, remove skips if mocks are global) ---
# @pytest.fixture
# def populated_store(...): -> Now defined in conftest.py


# --- Query Tests ---
# Remove skipif conditions relying on local variables
@patch('safe_store.search.similarity.ASCIIColors')
@patch(f'{safe_store_store_module.__name__}.ASCIIColors', new_callable=MagicMock)
def test_query_simple(mock_store_colors, mock_sim_colors, populated_store: SafeStore):
    """Test basic query functionality."""
    store = populated_store
    query = "second sentence"
    results = []
    with store:
        results = store.query(query, top_k=2)

    assert len(results) <= 2
    assert len(results) > 0
    assert_log_call_containing(mock_store_colors.info, f"Received query. Searching with '{store.DEFAULT_VECTORIZER}', top_k=2")
    assert_log_call_containing(mock_store_colors.debug, "Vectorizing query text...")
    assert_log_call_containing(mock_sim_colors.debug, "Calculating cosine similarity")


@patch(f'{safe_store_store_module.__name__}.ASCIIColors', new_callable=MagicMock)
def test_init_vectorizer_not_found(mock_store_colors, temp_db_path):
    """Test initializing with a vectorizer that doesn't exist."""
    non_existent_vectorizer = "invalid_vec"
    
    with pytest.raises(ConfigurationError):
        SafeStore(db_path=temp_db_path, vectorizer_name=non_existent_vectorizer)


# --- TF-IDF and Multiple Vectorizer Tests ---
# Remove skipif conditions relying on local variables
@patch('safe_store.vectorization.methods.tf_idf.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_with_tfidf(mock_store_colors, mock_manager_colors, mock_tfidf_colors, safe_store_instance: SafeStore, sample_text_file: Path):
    """Test adding a document using a TF-IDF vectorizer."""
    # Create a fresh store with TF-IDF
    db_path = sample_text_file.parent / "tfidf_store_test.db"
    tfidf_vectorizer_name = "tfidf"
    
    store = SafeStore(db_path=db_path, vectorizer_name=tfidf_vectorizer_name)

    with store:
        store.add_document(
            sample_text_file,
            chunk_size=40,
            chunk_overlap=10
        )

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_text_file.name}")
    assert_log_call_containing(mock_store_colors.info, "Generated 3 chunks")
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 3 chunks using '{tfidf_vectorizer_name}'")
    assert_log_call_containing(mock_tfidf_colors.info, "Fitting TfidfVectorizer on 3 documents")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_text_file.name}'")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM store_metadata WHERE key = 'vectorizer_info'")
    method_result = cursor.fetchone()
    assert method_result is not None
    v_info = json.loads(method_result[0])
    assert v_info['vectorizer_name'] == tfidf_vectorizer_name
    cursor.execute("SELECT COUNT(*) FROM vectors")
    vector_count = cursor.fetchone()[0]
    assert vector_count == 3
    conn.close()


@patch('safe_store.store.ASCIIColors')
def test_add_vectorization_incompatible(mock_store_colors, populated_store: SafeStore, sample_text_file: Path):
    """Test that SafeStore enforces vectorizer consistency for a database."""
    store = populated_store
    new_vectorizer = "tfidf"

    # Re-opening the same DB with a different vectorizer should fail
    with pytest.raises(ConfigurationError, match="is already configured with a different vectorizer"):
        SafeStore(db_path=store.db_path, vectorizer_name=new_vectorizer)

@patch('safe_store.search.similarity.ASCIIColors')
@patch(f'{safe_store_store_module.__name__}.ASCIIColors', new_callable=MagicMock)
def test_query_simple(mock_store_colors, mock_sim_colors, populated_store: SafeStore):
    """Test basic query functionality."""
    store = populated_store
    query = "second sentence"
    results = []
    with store:
        results = store.query(query, top_k=2)

    assert len(results) <= 2
    assert len(results) > 0
    assert_log_call_containing(mock_store_colors.info, f"Received query. Searching with '{store.DEFAULT_VECTORIZER}', top_k=2")
    assert_log_call_containing(mock_store_colors.debug, "Vectorizing query text...")
    assert_log_call_containing(mock_sim_colors.debug, "Calculating cosine similarity")


@patch(f'{safe_store_store_module.__name__}.ASCIIColors', new_callable=MagicMock)
def test_init_vectorizer_not_found(mock_store_colors, temp_db_path):
    """Test initializing with a vectorizer that doesn't exist."""
    non_existent_vectorizer = "invalid_vec"
    
    with pytest.raises(ConfigurationError):
        SafeStore(db_path=temp_db_path, vectorizer_name=non_existent_vectorizer)


# --- TF-IDF and Multiple Vectorizer Tests ---
# Remove skipif conditions relying on local variables
@patch('safe_store.vectorization.methods.tf_idf.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_with_tfidf(mock_store_colors, mock_manager_colors, mock_tfidf_colors, safe_store_instance: SafeStore, sample_text_file: Path):
    """Test adding a document using a TF-IDF vectorizer."""
    # Create a fresh store with TF-IDF
    db_path = sample_text_file.parent / "tfidf_store_test.db"
    tfidf_vectorizer_name = "tfidf"
    
    store = SafeStore(db_path=db_path, vectorizer_name=tfidf_vectorizer_name)

    with store:
        store.add_document(
            sample_text_file,
            chunk_size=40,
            chunk_overlap=10
        )

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_text_file.name}")
    assert_log_call_containing(mock_store_colors.info, "Generated 3 chunks")
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 3 chunks using '{tfidf_vectorizer_name}'")
    assert_log_call_containing(mock_tfidf_colors.info, "Fitting TfidfVectorizer on 3 documents")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_text_file.name}'")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM store_metadata WHERE key = 'vectorizer_info'")
    method_result = cursor.fetchone()
    assert method_result is not None
    v_info = json.loads(method_result[0])
    assert v_info['vectorizer_name'] == tfidf_vectorizer_name
    cursor.execute("SELECT COUNT(*) FROM vectors")
    vector_count = cursor.fetchone()[0]
    assert vector_count == 3
    conn.close()


@patch('safe_store.store.ASCIIColors')
def test_add_vectorization_incompatible(mock_store_colors, populated_store: SafeStore, sample_text_file: Path):
    """Test that SafeStore enforces vectorizer consistency for a database."""
    store = populated_store
    new_vectorizer = "tfidf"

    # Re-opening the same DB with a different vectorizer should fail
    with pytest.raises(ConfigurationError, match="is already configured with a different vectorizer"):
        SafeStore(db_path=store.db_path, vectorizer_name=new_vectorizer)