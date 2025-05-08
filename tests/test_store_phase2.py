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
from safe_store.vectorization.methods.tfidf import TfidfVectorizerWrapper
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
    assert_log_call_containing(mock_store_colors.debug, "Loading all vectors for method_id")
    assert_log_call_containing(mock_sim_colors.debug, "Calculating cosine similarity")
    assert_log_call_containing(mock_store_colors.success, "Query successful.")
    res1 = results[0]
    assert "chunk_id" in res1; assert "similarity" in res1; assert "file_path" in res1
    assert isinstance(res1["similarity"], float); assert -1.01 <= res1["similarity"] <= 1.01
    if len(results) > 1: assert results[0]["similarity"] >= results[1]["similarity"]


@patch(f'{safe_store_store_module.__name__}.ASCIIColors', new_callable=MagicMock)
def test_query_vectorizer_not_found(mock_store_colors, populated_store: SafeStore):
    """Test querying with a vectorizer method that doesn't exist."""
    store = populated_store
    query = "test"
    non_existent_vectorizer = "st:does-not-exist-model"
    error_message = f"Failed to load Sentence Transformer model '{non_existent_vectorizer}'"

    with patch.object(store.vectorizer_manager, 'get_vectorizer', side_effect=VectorizationError(error_message)) as mock_get:
        with pytest.raises(VectorizationError, match=re.escape(error_message)):
             with store:
                 store.query(query, vectorizer_name=non_existent_vectorizer)
        assert mock_get.call_count == 1
        call_args, call_kwargs = mock_get.call_args
        assert call_args[0] == non_existent_vectorizer
        assert isinstance(call_args[1], sqlite3.Connection)
        assert call_args[2] is None

    assert_log_call_containing(mock_store_colors.error, f"Error during query: VectorizationError: {error_message}")


@patch(f'{safe_store_store_module.__name__}.ASCIIColors', new_callable=MagicMock)
def test_query_no_vectors_for_method(mock_store_colors, populated_store: SafeStore):
    """Test querying when the method exists but has no vectors."""
    store = populated_store
    method_name = "empty:method"
    method_id = None

    with store:
        try:
            method_id = db.add_or_get_vectorization_method(store.conn, method_name, "test", 10, "float32", "{}")
            store.conn.commit()
            assert method_id is not None
            cursor_check = store.conn.cursor()
            cursor_check.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
            vector_count = cursor_check.fetchone()[0]
            assert vector_count == 0, f"Test setup failed: Expected 0 vectors for method_id {method_id}, but found {vector_count}"
        except Exception as e:
            pytest.fail(f"Manual method insertion/check failed: {e}")

        class DummyVectorizer:
            dim = 10
            dtype = np.float32
            def vectorize(self, texts): return np.zeros((len(texts), self.dim), dtype=self.dtype)
        dummy_instance = DummyVectorizer()

        with patch.object(store.vectorizer_manager, 'get_vectorizer', return_value=(dummy_instance, method_id)) as mock_get:
            query = "test"
            results = store.query(query, vectorizer_name=method_name)
            mock_get.assert_called_once_with(method_name, store.conn, None)

    assert results == [], "Query should return an empty list when no vectors are found"
    expected_warning = f"No vectors found in the database for method '{method_name}' (ID: {method_id}). Cannot perform query."
    try:
        assert_log_call_containing(mock_store_colors.warning, expected_warning)
    except AssertionError as e:
        print("\n--- DEBUG: Store Mock Calls (method_calls) ---")
        print(mock_store_colors.method_calls)
        print("\n--- DEBUG: Store Warning Calls Args List ---")
        print(mock_store_colors.warning.call_args_list)
        print("-----------------------------\n")
        raise e


# --- TF-IDF and Multiple Vectorizer Tests ---
# Remove skipif conditions relying on local variables
@patch('safe_store.vectorization.methods.tfidf.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_with_tfidf(mock_store_colors, mock_manager_colors, mock_tfidf_colors, safe_store_instance: SafeStore, sample_text_file: Path):
    """Test adding a document using a TF-IDF vectorizer for the first time."""
    store = safe_store_instance
    tfidf_vectorizer_name = "tfidf:test1"

    with store:
        store.add_document(
            sample_text_file,
            vectorizer_name=tfidf_vectorizer_name,
            chunk_size=40,
            chunk_overlap=10
        )

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_text_file.name}")
    assert_log_call_containing(mock_store_colors.info, "Generated 3 chunks for")
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 3 chunks using '{tfidf_vectorizer_name}'")
    expected_warning = f"TF-IDF vectorizer '{tfidf_vectorizer_name}' is not fitted. Fitting ONLY on chunks from '{sample_text_file.name}'"
    assert_log_call_containing(mock_store_colors.warning, expected_warning)
    assert_log_call_containing(mock_tfidf_colors.info, "Fitting TfidfVectorizer on 3 documents")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_text_file.name}' with vectorizer '{tfidf_vectorizer_name}'")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id, vector_dim, params FROM vectorization_methods WHERE method_name = ?", (tfidf_vectorizer_name,))
    method_result = cursor.fetchone()
    assert method_result is not None
    method_id = method_result[0]
    dim = method_result[1]
    params_json = method_result[2]
    assert dim > 0
    assert params_json is not None
    params = json.loads(params_json)
    assert params.get("fitted") is True
    assert "vocabulary" in params and len(params["vocabulary"]) == dim
    assert "idf" in params and len(params["idf"]) == dim
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE v.method_id = ?", (method_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 3
    conn.close()


@patch('safe_store.store.ASCIIColors')
def test_add_vectorization_st(mock_store_colors, populated_store: SafeStore, sample_text_file: Path):
    """Test adding a NEW Sentence Transformer vectorization to existing docs."""
    store = populated_store
    new_st_vectorizer = "st:paraphrase-MiniLM-L3-v2"
    initial_vector_count = 0
    default_method_id = None

    with store:
        conn = store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (store.DEFAULT_VECTORIZER,))
        default_method_id_res = cursor.fetchone(); assert default_method_id_res is not None
        default_method_id = default_method_id_res[0]
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (default_method_id,))
        initial_vector_count = cursor.fetchone()[0]; assert initial_vector_count == 7
        store.add_vectorization(new_st_vectorizer)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting process to add vectorization '{new_st_vectorizer}'")
    assert_log_call_containing(mock_store_colors.info, "Targeting all documents")
    assert_log_call_containing(mock_store_colors.info, f"Found {initial_vector_count} chunks to vectorize.")
    assert_log_call_containing(mock_store_colors.success, f"Successfully added {initial_vector_count} vector embeddings using '{new_st_vectorizer}'.")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (new_st_vectorizer,))
    new_method_id_res = cursor.fetchone(); assert new_method_id_res is not None
    new_method_id = new_method_id_res[0]; assert new_method_id != default_method_id
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (new_method_id,))
    new_vector_count = cursor.fetchone()[0]; assert new_vector_count == initial_vector_count
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (default_method_id,))
    final_default_vector_count = cursor.fetchone()[0]; assert final_default_vector_count == initial_vector_count
    conn.close()


@patch('safe_store.vectorization.methods.tfidf.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_vectorization_tfidf_all_docs(mock_store_colors, mock_manager_colors, mock_tfidf_colors, populated_store: SafeStore):
    """Test adding TF-IDF vectorization to ALL existing docs."""
    store = populated_store
    tfidf_vectorizer_name = "tfidf:global"
    total_chunk_count = 0

    with store:
        conn = store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunk_count = cursor.fetchone()[0]; assert total_chunk_count == 7
        store.add_vectorization(tfidf_vectorizer_name)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting process to add vectorization '{tfidf_vectorizer_name}'")
    assert_log_call_containing(mock_store_colors.info, f"TF-IDF vectorizer '{tfidf_vectorizer_name}' requires fitting.")
    assert_log_call_containing(mock_store_colors.info, "Fetching all chunks from database for fitting TF-IDF...")
    assert_log_call_containing(mock_tfidf_colors.info, f"Fitting TfidfVectorizer on {total_chunk_count} documents")
    assert_log_call_containing(mock_store_colors.info, f"Found {total_chunk_count} chunks to vectorize.")
    assert_log_call_containing(mock_store_colors.success, f"Successfully added {total_chunk_count} vector embeddings using '{tfidf_vectorizer_name}'.")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id, vector_dim, params FROM vectorization_methods WHERE method_name = ?", (tfidf_vectorizer_name,))
    method_result = cursor.fetchone(); assert method_result is not None
    method_id = method_result[0]; params_json = method_result[2]; assert params_json is not None
    params = json.loads(params_json); assert params.get("fitted") is True
    assert method_result[1] > 0
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
    vector_count = cursor.fetchone()[0]; assert vector_count == total_chunk_count
    conn.close()


@patch('safe_store.vectorization.manager.ASCIIColors') # Patch manager's logger
@patch('safe_store.store.ASCIIColors')               # Patch store's logger
def test_remove_vectorization(mock_store_colors, mock_manager_colors, populated_store: SafeStore):
    """Test removing a vectorization method."""
    store = populated_store
    vectorizer_to_remove = store.DEFAULT_VECTORIZER
    method_id = None
    initial_vector_count = 0

    with store:
        conn = store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_to_remove,))
        method_id_res = cursor.fetchone(); assert method_id_res is not None
        method_id = method_id_res[0]
        cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
        initial_vector_count = cursor.fetchone()[0]; assert initial_vector_count > 0
        try:
            store.vectorizer_manager.get_vectorizer(vectorizer_to_remove, store.conn, None)
        except Exception: pass
        assert vectorizer_to_remove in store.vectorizer_manager._cache
        store.remove_vectorization(vectorizer_to_remove)

    # Check logs
    assert_log_call_containing(mock_store_colors.warning, f"Attempting to remove vectorization method '{vectorizer_to_remove}'")
    assert_log_call_containing(mock_store_colors.debug, f"Deleted {initial_vector_count} vector records.")
    assert_log_call_containing(mock_store_colors.debug, "Deleted 1 vectorization method record.")
    # *** FIX: Check the manager's mock for the specific log call ***
    assert_log_call_containing(mock_manager_colors.debug, f"Invalidated cache for method '{vectorizer_to_remove}' (ID: {method_id}) due to removal")
    assert_log_call_containing(mock_store_colors.success, f"Successfully removed vectorization method '{vectorizer_to_remove}'")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_to_remove,))
    method_id_res = cursor.fetchone(); assert method_id_res is None
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
    vector_count = cursor.fetchone()[0]; assert vector_count == 0
    conn.close()

    # Check cache is cleared directly
    assert vectorizer_to_remove not in store.vectorizer_manager._cache


@patch('safe_store.store.ASCIIColors')
def test_remove_vectorization_not_found(mock_store_colors, populated_store: SafeStore):
    """Test attempting to remove a non-existent vectorization method."""
    store = populated_store
    non_existent_vectorizer = "non:existent"

    with store:
        store.remove_vectorization(non_existent_vectorizer)

    # Check logs
    expected_warning = f"Vectorization method '{non_existent_vectorizer}' not found in the database. Nothing to remove."
    assert_log_call_containing(mock_store_colors.warning, expected_warning)
    success_logged = any("Successfully removed" in args[0] for call in mock_store_colors.success.call_args_list for args in call.args if isinstance(args, tuple))
    assert not success_logged