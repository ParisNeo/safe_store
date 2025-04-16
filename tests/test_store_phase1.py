import pytest
import sqlite3
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock # Import patch
from safestore import SafeStore # Main class
from safestore.core import db # For checking DB directly

# Mock SentenceTransformer if not installed or for speed
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Minimal mock if needed
    class MockSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
             # Return dummy vectors of expected shape/dtype
             # Use a fixed dim like 384 for 'all-MiniLM-L6-v2'
             dim = 384
             return np.random.rand(len(texts), dim).astype(np.float32)
        def get_sentence_embedding_dimension(self):
            return 384

    @pytest.fixture(autouse=True)
    def mock_st(monkeypatch):
        # Replace the class in the module where it's imported and used
        monkeypatch.setattr("safestore.vectorization.methods.sentence_transformer.SentenceTransformer", MockSentenceTransformer)
        # Also need to patch the check at the top level of that module if it runs at import time
        monkeypatch.setattr("safestore.vectorization.methods.sentence_transformer.SentenceTransformer", MockSentenceTransformer, raising=False) # Allow overwrite


# --- Helper function to check if a call with specific text exists ---
def assert_log_call_containing(mock_logger, expected_substring):
    """Checks if any call to the mock logger contained the substring."""
    found = False
    for call in mock_logger.call_args_list:
        # call is like call('Some message', maybe_other_args...)
        args, kwargs = call
        if args and isinstance(args[0], str) and expected_substring in args[0]:
            found = True
            break
    assert found, f"Expected log call containing '{expected_substring}' not found in {mock_logger.call_args_list}"


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE and "mock_st" not in globals(), reason="Requires sentence-transformers or mocking")
# Patch where ASCIIColors is USED
@patch('safestore.vectorization.methods.sentence_transformer.ASCIIColors') # For ST init/vectorize
@patch('safestore.vectorization.manager.ASCIIColors')                     # For manager init/get
@patch('safestore.indexing.chunking.ASCIIColors')                         # For chunking debug
@patch('safestore.indexing.parser.ASCIIColors')                           # For parser debug/error
@patch('safestore.core.db.ASCIIColors')                                   # For DB operations logs *** ADDED ***
@patch('safestore.store.ASCIIColors')                                     # For main store operations
def test_add_document_new(
    mock_store_colors,      # Corresponds to @patch('safestore.store.ASCIIColors')
    mock_db_colors,         # Corresponds to @patch('safestore.core.db.ASCIIColors')
    mock_parser_colors,     # Corresponds to @patch('safestore.indexing.parser.ASCIIColors')
    mock_chunking_colors,   # Corresponds to @patch('safestore.indexing.chunking.ASCIIColors')
    mock_manager_colors,    # Corresponds to @patch('safestore.vectorization.manager.ASCIIColors')
    mock_st_colors,         # Corresponds to @patch('safestore.vectorization.methods.sentence_transformer.ASCIIColors')
    # Original fixtures last
    safestore_instance: SafeStore,
    sample_text_file: Path
):
    """Test adding a completely new document using mocks."""
    store = safestore_instance
    file_path = sample_text_file

    store.add_document(file_path, chunk_size=30, chunk_overlap=5)

    # 1. Check log calls using mocks (access methods via the mock class)
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {file_path.name}")
    assert_log_call_containing(mock_store_colors.info, "Generated 4 chunks") # This call is in store.py
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 4 chunks using '{store.DEFAULT_VECTORIZER}'")
    assert_log_call_containing(mock_store_colors.success, f"Successfully indexed and vectorized '{file_path.name}'")

    # Check debug calls from specific modules if needed
    assert_log_call_containing(mock_parser_colors.debug, f"Successfully parsed TXT file: {file_path}")
    assert_log_call_containing(mock_chunking_colors.debug, "Chunking complete. Generated 4 chunks.")
    assert_log_call_containing(mock_manager_colors.info, f"Initializing vectorizer: {store.DEFAULT_VECTORIZER}")
    assert_log_call_containing(mock_st_colors.info, f"Loading Sentence Transformer model: {store.DEFAULT_VECTORIZER.split(':',1)[1]}")
    assert_log_call_containing(mock_db_colors.debug, "Added document record for") # Example check for DB log

    # Ensure no errors were logged via the store's logger instance
    mock_store_colors.error.assert_not_called()
    mock_db_colors.error.assert_not_called() # Also check DB errors

    # 2. Check database state (remains the same as before)
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    # ... (DB checks are unchanged) ...
    cursor.execute("SELECT doc_id, file_path, full_text, file_hash FROM documents WHERE file_path = ?", (str(file_path.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    assert doc_result[2] == sample_text_file.read_text() # Check full_text stored correctly
    assert doc_result[3] is not None # Check hash was stored

    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 4 # Expected chunk count

    cursor.execute("SELECT method_id, vector_dim, vector_dtype FROM vectorization_methods WHERE method_name = ?", (store.DEFAULT_VECTORIZER,))
    method_result = cursor.fetchone()
    assert method_result is not None
    method_id = method_result[0]
    assert method_result[1] == 384 # Dimension for all-MiniLM-L6-v2 or mock
    assert method_result[2] == 'float32'

    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? AND v.method_id = ?", (doc_id, method_id))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 4 # One vector per chunk for this method
    conn.close()

# Patch only the strictly necessary modules for the second call checks
@patch('safestore.core.db.ASCIIColors')      # Needed for checking existing vectors/doc
@patch('safestore.vectorization.manager.ASCIIColors') # Needed for get_vectorizer cache hit
@patch('safestore.store.ASCIIColors')        # Needed for the main flow logs
def test_add_document_unchanged(
    mock_store_colors,      # Corresponds to @patch('safestore.store.ASCIIColors')
    mock_manager_colors,    # Corresponds to @patch('safestore.vectorization.manager.ASCIIColors')
    mock_db_colors,         # Corresponds to @patch('safestore.core.db.ASCIIColors')
    # Original fixtures last
    safestore_instance: SafeStore,
    sample_text_file: Path
):
    """Test adding the same document again without changes using mocks."""
    store = safestore_instance
    file_path = sample_text_file

    # Add first time
    store.add_document(file_path, chunk_size=30, chunk_overlap=5)

    # Reset only the mocks we are checking for the second call
    mock_store_colors.reset_mock()
    mock_manager_colors.reset_mock()
    mock_db_colors.reset_mock()

    # Add second time
    store.add_document(file_path, chunk_size=30, chunk_overlap=5)

    # Check logs indicate skipping using the specific mock instances
    assert_log_call_containing(mock_store_colors.info, f"Document '{file_path.name}' is unchanged.")

    # *** Check against .success instead of .info ***
    assert_log_call_containing(mock_store_colors.success, f"Vectorization '{store.DEFAULT_VECTORIZER}' already exists for '{file_path.name}'. Skipping.")

    # Check vectorizer manager debug log for cache hit
    assert_log_call_containing(mock_manager_colors.debug, f"Vectorizer '{store.DEFAULT_VECTORIZER}' found in cache.")

    # Check that success message for indexing wasn't called *again*
    # Make sure we don't count the "Skipping" success message here. Check call count.
    # This check might need refinement depending on how many .success calls are expected overall.
    # Let's check that the *final* "Successfully indexed..." success message wasn't called.
    final_success_found = False
    for call in mock_store_colors.success.call_args_list:
        args, _ = call
        if args and "Successfully indexed and vectorized" in args[0]:
            final_success_found = True
            break
    assert not final_success_found, "Final indexing success message should not be logged when skipping."



# Patch all modules involved in the re-indexing path
@patch('safestore.vectorization.methods.sentence_transformer.ASCIIColors') # For ST vectorize
@patch('safestore.vectorization.manager.ASCIIColors')                     # For manager get
@patch('safestore.indexing.chunking.ASCIIColors')                         # For chunking debug
@patch('safestore.indexing.parser.ASCIIColors')                           # For parser debug
@patch('safestore.core.db.ASCIIColors')                                   # For DB delete/update/add
@patch('safestore.store.ASCIIColors')                                     # For main store operations
def test_add_document_changed(
    mock_store_colors,      # Corresponds to @patch('safestore.store.ASCIIColors')
    mock_db_colors,         # Corresponds to @patch('safestore.core.db.ASCIIColors')
    mock_parser_colors,     # Corresponds to @patch('safestore.indexing.parser.ASCIIColors')
    mock_chunking_colors,   # Corresponds to @patch('safestore.indexing.chunking.ASCIIColors')
    mock_manager_colors,    # Corresponds to @patch('safestore.vectorization.manager.ASCIIColors')
    mock_st_colors,         # Corresponds to @patch('safestore.vectorization.methods.sentence_transformer.ASCIIColors')
    # Original fixtures last
    safestore_instance: SafeStore,
    sample_text_file: Path
):
    """Test adding a document that has changed content using mocks."""
    store = safestore_instance
    file_path = sample_text_file

    # Add initial version
    store.add_document(file_path, chunk_size=30, chunk_overlap=5)

    # Reset all mocks before the second call
    mock_store_colors.reset_mock()
    mock_db_colors.reset_mock()
    mock_parser_colors.reset_mock()
    mock_chunking_colors.reset_mock()
    mock_manager_colors.reset_mock()
    mock_st_colors.reset_mock()


    # Modify the file
    new_content = "This is completely new content.\nWith two lines."
    file_path.write_text(new_content, encoding='utf-8')

    # Add again
    store.add_document(file_path, chunk_size=20, chunk_overlap=5) # Use different chunking for change

    # Check logs indicate change and re-indexing
    assert_log_call_containing(mock_store_colors.warning, f"Document '{file_path.name}' has changed (hash mismatch). Re-indexing...")
    assert_log_call_containing(mock_store_colors.debug, "Deleted old chunks/vectors")
    assert_log_call_containing(mock_chunking_colors.debug, "Generated 3 chunks") # Chunking log
    assert_log_call_containing(mock_store_colors.info, "Generated 3 chunks")    # Store log
    assert_log_call_containing(mock_store_colors.info, "Vectorizing 3 chunks")
    assert_log_call_containing(mock_store_colors.success, f"Successfully indexed and vectorized '{file_path.name}'")
    mock_store_colors.error.assert_not_called()


    # Check database state reflects the *new* content (remains the same check)
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(file_path.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    assert doc_result[1] == new_content # Check updated full_text

    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 3 # Check NEW chunk count

    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 3 # Check NEW vector count

    conn.close()

# Add more tests: error handling (file not found), different vectorizers once added etc.