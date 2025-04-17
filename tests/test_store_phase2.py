# tests/test_store_phase2.py
import pytest
import sqlite3
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call
# Import ASCIIColors specifically for patching target
from safestore import store as safestore_module
from safestore import SafeStore, LogLevel
from safestore.core import db
from safestore.vectorization.methods.tfidf import TfidfVectorizerWrapper
from safestore.vectorization.manager import VectorizationManager


# Helper to assert log calls (reuse from phase 1 test)
def assert_log_call_containing(mock_logger, expected_substring):
    found = False
    for call_args in mock_logger.call_args_list:
        args, kwargs = call_args
        if args and isinstance(args[0], str) and expected_substring in args[0]:
            found = True
            break
    assert found, f"Expected log call containing '{expected_substring}' not found in {mock_logger.call_args_list}"

# Mock SentenceTransformer as before
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class MockSentenceTransformer:
        DEFAULT_MODEL = "mock-st-model"
        def __init__(self, model_name):
            self.model_name = model_name
            self._dim = 384 # Standard mock dim
            self._dtype = np.float32
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
             return np.random.rand(len(texts), self._dim).astype(self._dtype)
        def get_sentence_embedding_dimension(self):
            return self._dim
        @property
        def dim(self): return self._dim
        @property
        def dtype(self): return self._dtype

    @pytest.fixture(autouse=True)
    def mock_st(monkeypatch):
        monkeypatch.setattr("safestore.vectorization.methods.sentence_transformer.SentenceTransformer", MockSentenceTransformer, raising=False)
        # The second monkeypatch for the same target is redundant
        # monkeypatch.setattr("safestore.vectorization.methods.sentence_transformer.SentenceTransformer", MockSentenceTransformer, raising=False)

# Mock Scikit-learn TfidfVectorizer
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True

    # Create a semi-realistic mock for TF-IDF testing
    class MockTfidfVectorizer:
        def __init__(self, **kwargs):
            self.params = kwargs
            self._fitted = False
            self.vocabulary_ = {}
            self.idf_ = np.array([])
            self.dtype = np.float64 # Default for mock
            if 'dtype' in kwargs:
                 try:
                    self.dtype = np.dtype(kwargs['dtype'])
                 except: pass # ignore errors
            # Simulate _tfidf attribute expected by load_fitted_state
            class MockTfidfInternal:
                _idf_diag = MagicMock() # Mock the internal attribute
            self._tfidf = MockTfidfInternal()


        def fit(self, texts):
            # Simple mock fit: create vocab based on unique words (lowercase)
            words = set(word for text in texts for word in text.lower().split())
            self.vocabulary_ = {word: i for i, word in enumerate(sorted(list(words)))}
            self.idf_ = np.random.rand(len(self.vocabulary_)).astype(self.dtype) * 5 + 1 # Random IDF > 1
            self._fitted = True
            # print(f"MockTfidf Fit: Vocab Size = {len(self.vocabulary_)}, Dtype={self.dtype}")
            # Set the dtype on the mocked internal attribute after fit
            self._tfidf._idf_diag.dtype = self.dtype
            return self

        def transform(self, texts):
            if not self._fitted:
                # Use the real NotFittedError if available, otherwise generic Exception
                raise (NotFittedError or Exception)("MockTfidfVectorizer not fitted")
            # Simple mock transform: output random sparse-like data matching vocab size
            num_samples = len(texts)
            vocab_size = len(self.vocabulary_)
            # Create dense random array matching shape and type
            dense_array = np.random.rand(num_samples, vocab_size).astype(self.dtype)
            # Simulate sparse output with toarray() method
            class MockSparseMatrix:
                def __init__(self, data):
                    self.data = data
                    self.shape = data.shape
                def toarray(self):
                    return self.data
            # print(f"MockTfidf Transform: Output shape = ({num_samples}, {vocab_size}), Dtype={self.dtype}")
            return MockSparseMatrix(dense_array)

        def get_params(self, deep=True):
             # Return initial params for consistency
             return self.params

    @pytest.fixture(autouse=True)
    def mock_sklearn(monkeypatch):
        # Mock the class where it's imported/used in the wrapper
        monkeypatch.setattr("safestore.vectorization.methods.tfidf.TfidfVectorizer", MockTfidfVectorizer)
        # Mock NotFittedError using the real one if available, or a generic Exception otherwise
        monkeypatch.setattr("safestore.vectorization.methods.tfidf.NotFittedError", NotFittedError or Exception)
        # Mock the check at the top of the module
        monkeypatch.setattr("safestore.vectorization.methods.tfidf.TfidfVectorizer", MockTfidfVectorizer, raising=False)


except ImportError:
    SKLEARN_AVAILABLE = False
    # Define dummy mocks if sklearn is not installed, tests needing it will be skipped
    MockTfidfVectorizer = None
    NotFittedError = None
    @pytest.fixture(autouse=True)
    def mock_sklearn(monkeypatch):
         pass # No mocking needed if sklearn not installed

# Helper to assert log calls (reuse from phase 1 test)
def assert_log_call_containing(mock_logger, expected_substring):
    found = False
    for call_args in mock_logger.call_args_list:
        args, kwargs = call_args
        if args and isinstance(args[0], str) and expected_substring in args[0]:
            found = True
            break
    assert found, f"Expected log call containing '{expected_substring}' not found in {mock_logger.call_args_list}"


# --- Test Fixtures ---
@pytest.fixture
def populated_store(safestore_instance: SafeStore, sample_text_file: Path, tmp_path: Path) -> SafeStore:
    """Provides a SafeStore instance with one document added using the default ST vectorizer."""
    store = safestore_instance
    # Add first doc with default ST
    store.add_document(sample_text_file, chunk_size=30, chunk_overlap=5)

    # Add a second doc
    doc2_content = "Another document.\nWith different content for testing."
    doc2_path = tmp_path / "sample2.txt"
    doc2_path.write_text(doc2_content, encoding='utf-8')
    store.add_document(doc2_path, chunk_size=25, chunk_overlap=5) # Different chunking

    return store

# --- Query Tests ---
@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE and "mock_st" not in globals(), reason="Requires sentence-transformers or mocking")
@patch('safestore.search.similarity.ASCIIColors') # Mock logger in similarity module
@patch('safestore.store.ASCIIColors')             # Mock logger in store module
def test_query_simple(mock_store_colors, mock_sim_colors, populated_store: SafeStore):
    """Test basic query functionality."""
    store = populated_store
    query = "second sentence"
    results = store.query(query, top_k=2)

    assert len(results) <= 2 # Can be less if fewer chunks exist
    assert len(results) > 0 # Should find something relevant

    # Check log messages
    assert_log_call_containing(mock_store_colors.info, f"Received query. Searching with '{store.DEFAULT_VECTORIZER}', top_k=2")
    assert_log_call_containing(mock_store_colors.debug, "Vectorizing query text...")
    assert_log_call_containing(mock_store_colors.debug, "Loading all vectors for method_id") # Check vector loading log
    assert_log_call_containing(mock_sim_colors.debug, "Calculating cosine similarity") # Check similarity log
    assert_log_call_containing(mock_store_colors.success, "Query successful.")

    # Check result structure (example of checking first result)
    res1 = results[0]
    assert "chunk_id" in res1
    assert "chunk_text" in res1
    assert "similarity" in res1
    assert "doc_id" in res1
    assert "file_path" in res1
    assert "start_pos" in res1
    assert "end_pos" in res1
    assert "chunk_seq" in res1
    assert isinstance(res1["similarity"], float)
    # Allow slightly wider range due to potential float inaccuracies or mock data
    assert -1.01 <= res1["similarity"] <= 1.01

    # Check descending order (if multiple results)
    if len(results) > 1:
        assert results[0]["similarity"] >= results[1]["similarity"]

@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE and "mock_st" not in globals(), reason="Requires sentence-transformers or mocking")
@patch('safestore.store.ASCIIColors')
def test_query_vectorizer_not_found(mock_store_colors, populated_store: SafeStore):
    """Test querying with a vectorizer method that doesn't exist."""
    store = populated_store
    query = "test"
    non_existent_vectorizer = "st:does-not-exist-model"

    # Use the actual VectorizerManager instance from the store to mock the get method
    # This avoids issues with potential errors during actual model loading
    with patch.object(store.vectorizer_manager, 'get_vectorizer', side_effect=ValueError("Vectorizer not found")) as mock_get:
        with pytest.raises(ValueError, match="Vectorizer not found"):
            store.query(query, vectorizer_name=non_existent_vectorizer)
        mock_get.assert_called_once_with(non_existent_vectorizer, store.conn)

    # Check appropriate log messages if possible (depends on where error occurs)
    assert_log_call_containing(mock_store_colors.error, "An unexpected error occurred during query:")



# Remove class-level patch for ASCIIColors here
# @patch('safestore.vectorization.manager.ASCIIColors') # Keep if needed for manager interactions, but remove for this test
# @patch('safestore.store.ASCIIColors')
def test_query_no_vectors_for_method(populated_store: SafeStore): # Remove mock args
    """Test querying when the method exists but has no vectors."""
    store = populated_store
    method_name = "empty:method"

    # 1. Manually add method
    try:
        method_id = db.add_or_get_vectorization_method(store.conn, method_name, "test", 10, "float32", "{}")
        assert method_id is not None
    except Exception as e:
        pytest.fail(f"Manual method insertion failed: {e}")

    # 2. Verify DB state directly
    conn_check = sqlite3.connect(store.db_path)
    cursor_check = conn_check.cursor()
    cursor_check.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
    vector_count = cursor_check.fetchone()[0]
    conn_check.close()
    assert vector_count == 0, f"Test setup failed: Expected 0 vectors for method_id {method_id}, but found {vector_count}"

    # 3. Prepare dummy vectorizer
    class DummyVectorizer:
        dim = 10
        dtype = np.float32
        def vectorize(self, texts):
            return np.zeros((len(texts), self.dim), dtype=self.dtype)
    dummy_instance = DummyVectorizer()

    # 4. Patch get_vectorizer AND ASCIIColors within the store module's scope for this call
    # Create a fresh mock for ASCIIColors specifically for this test
    mock_store_colors = MagicMock()
    with patch.object(store.vectorizer_manager, 'get_vectorizer', return_value=(dummy_instance, method_id)) as mock_get, \
         patch.object(safestore_module, 'ASCIIColors', mock_store_colors) as mock_ascii_patch: # Patch within the store module

        query = "test"
        results = store.query(query, vectorizer_name=method_name)
        mock_get.assert_called_once_with(method_name, store.conn)


    # 5. Assert results and the warning log using the new mock
    assert results == [], "Query should return an empty list when no vectors are found"
    try:
        # Check calls on the new mock_store_colors
        assert_log_call_containing(mock_store_colors.info, f"Received query. Searching with '{method_name}', top_k=5") # Check info call
        assert_log_call_containing(mock_store_colors.warning, f"No vectors found in the database for method '{method_name}'") # Check warning call
    except AssertionError as e:
        # Print calls made to the mock for debugging if the assertion fails
        print("\n--- DEBUG: Store Mock Calls (method_calls) ---")
        print(mock_store_colors.method_calls)
        print("\n--- DEBUG: Store Warning Calls Args List ---")
        print(mock_store_colors.warning.call_args_list)
        print("-----------------------------\n")
        raise e


# --- TF-IDF and Multiple Vectorizer Tests ---
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Requires scikit-learn for TF-IDF tests")
@patch('safestore.vectorization.methods.tfidf.ASCIIColors') # Mock loggers
@patch('safestore.vectorization.manager.ASCIIColors')
@patch('safestore.store.ASCIIColors')
def test_add_document_with_tfidf(mock_store_colors, mock_manager_colors, mock_tfidf_colors, safestore_instance: SafeStore, sample_text_file: Path):
    """Test adding a document using a TF-IDF vectorizer for the first time."""
    store = safestore_instance
    tfidf_vectorizer_name = "tfidf:test1"

    # *** FIXED: Provide valid chunk_overlap ***
    store.add_document(
        sample_text_file,
        vectorizer_name=tfidf_vectorizer_name,
        chunk_size=40,
        chunk_overlap=10 # Ensure overlap < chunk_size
    )

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_text_file.name}")
    # With chunk_size=40, overlap=10 on the sample text, it generates 3 chunks
    assert_log_call_containing(mock_store_colors.info, f"Generated 3 chunks for")
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 3 chunks using '{tfidf_vectorizer_name}'")
    assert_log_call_containing(mock_store_colors.warning, f"TF-IDF vectorizer '{tfidf_vectorizer_name}' is not fitted. Fitting on chunks from") # Check fit warning
    assert_log_call_containing(mock_tfidf_colors.info, "Fitting TfidfVectorizer on 3 documents") # Check TFIDF fit log
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
    assert dim > 0 # Dimension should be set after fitting
    assert params_json is not None # Params should be stored
    params = json.loads(params_json)
    assert params.get("fitted") is True # Should be marked as fitted
    assert "vocabulary" in params
    assert "idf" in params
    assert len(params["vocabulary"]) == dim # Vocab size should match dim

    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE v.method_id = ?", (method_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 3 # 3 chunks created by chunk_size=40, overlap=10

    conn.close()

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Requires scikit-learn for TF-IDF tests")
@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE and "mock_st" not in globals(), reason="Requires sentence-transformers or mocking")
@patch('safestore.store.ASCIIColors')
def test_add_vectorization_st(mock_store_colors, populated_store: SafeStore, sample_text_file: Path):
    """Test adding a NEW Sentence Transformer vectorization to existing docs."""
    store = populated_store # Has docs with default ST vectors
    new_st_vectorizer = "st:paraphrase-MiniLM-L3-v2" # Use a different ST model name

    # Get initial vector count for default ST
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (store.DEFAULT_VECTORIZER,))
    default_method_id_res = cursor.fetchone()
    assert default_method_id_res is not None, "Default vectorizer not found in populated store"
    default_method_id = default_method_id_res[0]

    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (default_method_id,))
    initial_vector_count = cursor.fetchone()[0]
    assert initial_vector_count > 0 # Should have vectors from setup
    conn.close()


    # Add the new vectorization to all documents
    store.add_vectorization(new_st_vectorizer)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting process to add vectorization '{new_st_vectorizer}'")
    assert_log_call_containing(mock_store_colors.info, "Targeting all documents")
    # Check the total number of chunks (4 from doc1 + 3 from doc2 = 7)
    assert_log_call_containing(mock_store_colors.info, f"Found {initial_vector_count} chunks to vectorize.") # Should vectorize all existing chunks
    assert_log_call_containing(mock_store_colors.success, f"Successfully added {initial_vector_count} vector embeddings using '{new_st_vectorizer}'.")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (new_st_vectorizer,))
    new_method_id_res = cursor.fetchone()
    assert new_method_id_res is not None
    new_method_id = new_method_id_res[0]
    assert new_method_id != default_method_id

    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (new_method_id,))
    new_vector_count = cursor.fetchone()[0]
    assert new_vector_count == initial_vector_count # Same number of chunks should now have new vectors

    # Verify original vectors still exist
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (default_method_id,))
    final_default_vector_count = cursor.fetchone()[0]
    assert final_default_vector_count == initial_vector_count

    conn.close()


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Requires scikit-learn for TF-IDF tests")
@patch('safestore.vectorization.methods.tfidf.ASCIIColors') # Mock loggers
@patch('safestore.vectorization.manager.ASCIIColors')
@patch('safestore.store.ASCIIColors')
def test_add_vectorization_tfidf_all_docs(mock_store_colors, mock_manager_colors, mock_tfidf_colors, populated_store: SafeStore):
    """Test adding TF-IDF vectorization to ALL existing docs."""
    store = populated_store # Has 2 docs with default ST vectors
    tfidf_vectorizer_name = "tfidf:global"

     # Get total chunk count
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunk_count = cursor.fetchone()[0]
    assert total_chunk_count > 0 # Should be 4 + 3 = 7
    conn.close()

    # Add TF-IDF vectorization to all documents
    store.add_vectorization(tfidf_vectorizer_name)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting process to add vectorization '{tfidf_vectorizer_name}'")
    # *** FIXED ASSERTION BELOW ***
    assert_log_call_containing(mock_store_colors.info, f"TF-IDF vectorizer '{tfidf_vectorizer_name}' requires fitting.")
    assert_log_call_containing(mock_store_colors.info, "Fetching all chunks from database for fitting...")
    assert_log_call_containing(mock_tfidf_colors.info, f"Fitting TfidfVectorizer on {total_chunk_count} documents") # Check fit log
    assert_log_call_containing(mock_store_colors.info, f"Found {total_chunk_count} chunks to vectorize.")
    assert_log_call_containing(mock_store_colors.success, f"Successfully added {total_chunk_count} vector embeddings using '{tfidf_vectorizer_name}'.")

    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id, vector_dim, params FROM vectorization_methods WHERE method_name = ?", (tfidf_vectorizer_name,))
    method_result = cursor.fetchone()
    assert method_result is not None
    method_id = method_result[0]
    params_json = method_result[2]
    assert params_json is not None
    params = json.loads(params_json)
    assert params.get("fitted") is True # Should be fitted now

    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == total_chunk_count

    conn.close()



@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE and "mock_st" not in globals(), reason="Requires sentence-transformers or mocking")
@patch('safestore.vectorization.manager.ASCIIColors') # Need manager logs for DB interactions if remove calls manager directly
@patch('safestore.store.ASCIIColors')                 # Need store logs for the main flow and cache removal log
def test_remove_vectorization(mock_store_colors, mock_manager_colors, populated_store: SafeStore):
    """Test removing a vectorization method."""
    store = populated_store
    vectorizer_to_remove = store.DEFAULT_VECTORIZER

    # Verify method and vectors exist initially
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_to_remove,))
    method_id_res = cursor.fetchone()
    assert method_id_res is not None
    method_id = method_id_res[0]
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,))
    initial_vector_count = cursor.fetchone()[0]
    assert initial_vector_count > 0 # Should be 7 total chunks
    conn.close()


    # Add the vectorizer to the cache manually for testing cache removal
    # This mimics the state where it would have been used before removal
    store.vectorizer_manager.get_vectorizer(vectorizer_to_remove, store.conn)
    assert vectorizer_to_remove in store.vectorizer_manager._cache

    # Remove the vectorization
    store.remove_vectorization(vectorizer_to_remove)

    # Check logs
    assert_log_call_containing(mock_store_colors.warning, f"Attempting to remove vectorization method '{vectorizer_to_remove}'")
    assert_log_call_containing(mock_store_colors.debug, f"Deleted {initial_vector_count} vector records.")
    assert_log_call_containing(mock_store_colors.debug, "Deleted 1 vectorization method record.")
    # *** FIXED: Check store's logger for the cache removal message ***
    assert_log_call_containing(mock_store_colors.debug, f"Removed '{vectorizer_to_remove}' from vectorizer cache.")
    assert_log_call_containing(mock_store_colors.success, f"Successfully removed vectorization method '{vectorizer_to_remove}'")


    # Check DB state
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT method_id FROM vectorization_methods WHERE method_name = ?", (vectorizer_to_remove,))
    method_id_res = cursor.fetchone()
    assert method_id_res is None # Method should be gone

    cursor.execute("SELECT COUNT(*) FROM vectors WHERE method_id = ?", (method_id,)) # Use the old ID to check vectors
    vector_count = cursor.fetchone()[0]
    assert vector_count == 0 # Vectors should be gone

    conn.close()

    # Check cache is cleared directly
    assert vectorizer_to_remove not in store.vectorizer_manager._cache



@patch('safestore.store.ASCIIColors')
def test_remove_vectorization_not_found(mock_store_colors, populated_store: SafeStore):
    """Test attempting to remove a non-existent vectorization method."""
    store = populated_store
    non_existent_vectorizer = "non:existent"

    store.remove_vectorization(non_existent_vectorizer)

    # Check logs
    assert_log_call_containing(mock_store_colors.error, f"Vectorization method '{non_existent_vectorizer}' not found in the database. Cannot remove.")
    # Ensure success message is NOT logged
    success_logged = any("Successfully removed" in args[0] for call in mock_store_colors.success.call_args_list for args in call.args if isinstance(args, tuple))
    assert not success_logged