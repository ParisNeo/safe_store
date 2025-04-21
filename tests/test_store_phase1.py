# tests/test_store_phase1.py
import pytest
import sqlite3
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import re

# Import specific exceptions and modules
from safe_store import safe_store, LogLevel
from safe_store.core import db
from safe_store.core.exceptions import FileHandlingError, ConfigurationError, safe_storeError

# --- REMOVE SentenceTransformer Check and Mock Fixture ---
# try:
#     from sentence_transformers import SentenceTransformer
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False
#     class MockSentenceTransformer: ...
#     @pytest.fixture(autouse=True)
#     def mock_st(monkeypatch): ...


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

# --- Tests (keep) ---
# Remove skipif conditions that rely on the local SENTENCE_TRANSFORMERS_AVAILABLE
# @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE and "mock_st" not in globals(), reason="Requires sentence-transformers or mocking")
@patch('safe_store.vectorization.methods.sentence_transformer.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.indexing.chunking.ASCIIColors')
@patch('safe_store.indexing.parser.ASCIIColors')
@patch('safe_store.core.db.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_new(
    mock_store_colors, mock_db_colors, mock_parser_colors,
    mock_chunking_colors, mock_manager_colors, mock_st_colors,
    safe_store_instance: safe_store, sample_text_file: Path
):
    """Test adding a completely new document using mocks."""
    store = safe_store_instance
    file_path = sample_text_file
    vectorizer_name_used = store.DEFAULT_VECTORIZER

    with store:
        store.add_document(file_path, chunk_size=30, chunk_overlap=5)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {file_path.name}")
    assert_log_call_containing(mock_store_colors.info, "Generated 4 chunks")
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 4 chunks using '{vectorizer_name_used}'")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{file_path.name}' with vectorizer '{vectorizer_name_used}'")
    assert_log_call_containing(mock_parser_colors.debug, f"Successfully parsed TXT file: {file_path}")
    assert_log_call_containing(mock_chunking_colors.debug, "Chunking complete. Generated 4 chunks.")
    assert_log_call_containing(mock_manager_colors.info, f"Initializing vectorizer: {vectorizer_name_used}")
    assert_log_call_containing(mock_st_colors.info, f"Loading Sentence Transformer model: {vectorizer_name_used.split(':',1)[1]}")
    assert_log_call_containing(mock_db_colors.debug, "Prepared insertion for document record")
    mock_store_colors.error.assert_not_called()
    mock_db_colors.error.assert_not_called()

    # Check DB
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, file_path, full_text, file_hash FROM documents WHERE file_path = ?", (str(file_path.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    assert doc_result[2] == sample_text_file.read_text(encoding='utf-8')
    assert doc_result[3] is not None
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 4
    cursor.execute("SELECT method_id, vector_dim, vector_dtype FROM vectorization_methods WHERE method_name = ?", (vectorizer_name_used,))
    method_result = cursor.fetchone()
    assert method_result is not None
    method_id = method_result[0]
    assert method_result[1] == 384 # Mock dimension
    assert method_result[2] == 'float32'
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ? AND v.method_id = ?", (doc_id, method_id))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 4
    conn.close()


@patch('safe_store.core.db.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_unchanged(
    mock_store_colors, mock_manager_colors, mock_db_colors,
    safe_store_instance: safe_store, sample_text_file: Path
):
    """Test adding the same document again without changes using mocks."""
    store = safe_store_instance
    file_path = sample_text_file
    vectorizer_name_used = store.DEFAULT_VECTORIZER

    with store:
        store.add_document(file_path, chunk_size=30, chunk_overlap=5)
        mock_store_colors.reset_mock()
        mock_manager_colors.reset_mock()
        mock_db_colors.reset_mock()
        store.add_document(file_path, chunk_size=30, chunk_overlap=5)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Document '{file_path.name}' is unchanged.")
    assert_log_call_containing(mock_store_colors.success, f"Vectorization '{vectorizer_name_used}' already exists for unchanged '{file_path.name}'. Skipping.")
    assert_log_call_containing(mock_manager_colors.debug, f"Vectorizer '{vectorizer_name_used}' found in cache (method_id=")
    process_message_found = any(f"Successfully processed" in args[0] for call in mock_store_colors.success.call_args_list for args in call.args if isinstance(args, tuple))
    assert not process_message_found, "Processing success message should NOT be logged when skipping."


@patch('safe_store.vectorization.methods.sentence_transformer.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.indexing.chunking.ASCIIColors')
@patch('safe_store.indexing.parser.ASCIIColors')
@patch('safe_store.core.db.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_changed(
    mock_store_colors, mock_db_colors, mock_parser_colors,
    mock_chunking_colors, mock_manager_colors, mock_st_colors,
    safe_store_instance: safe_store, sample_text_file: Path
):
    """Test adding a document that has changed content using mocks."""
    store = safe_store_instance
    file_path = sample_text_file
    vectorizer_name_used = store.DEFAULT_VECTORIZER

    with store:
        store.add_document(file_path, chunk_size=30, chunk_overlap=5)
        mock_store_colors.reset_mock()
        mock_db_colors.reset_mock()
        mock_parser_colors.reset_mock()
        mock_chunking_colors.reset_mock()
        mock_manager_colors.reset_mock()
        mock_st_colors.reset_mock()
        new_content = "This is completely new content.\nWith two lines."
        file_path.write_text(new_content, encoding='utf-8')
        store.add_document(file_path, chunk_size=20, chunk_overlap=5)

    # Check logs
    assert_log_call_containing(mock_store_colors.warning, f"Document '{file_path.name}' has changed (hash mismatch). Re-indexing...")
    assert_log_call_containing(mock_store_colors.debug, "Deleted old chunks/vectors")
    assert_log_call_containing(mock_chunking_colors.debug, "Generated 3 chunks")
    assert_log_call_containing(mock_store_colors.info, "Generated 3 chunks")
    assert_log_call_containing(mock_store_colors.info, "Vectorizing 3 chunks")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{file_path.name}' with vectorizer '{vectorizer_name_used}'")
    mock_store_colors.error.assert_not_called()

    # Check DB
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(file_path.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    assert doc_result[1] == new_content
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 3
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 3
    conn.close()


@patch('safe_store.store.ASCIIColors')
def test_add_document_file_not_found(mock_store_colors, safe_store_instance: safe_store, tmp_path: Path):
    """Test adding a document when the source file doesn't exist."""
    store = safe_store_instance
    non_existent_path = tmp_path / "non_existent_file.txt"
    resolved_path_str = str(non_existent_path.resolve())

    with pytest.raises(FileHandlingError, match=re.escape(f"File not found when trying to hash: {resolved_path_str}")):
        with store:
            store.add_document(non_existent_path)

    # Check the error log from where the error is caught in add_document
    assert_log_call_containing(mock_store_colors.error, f"Error during add_document: FileHandlingError: File not found when trying to hash: {resolved_path_str}")
    mock_store_colors.success.assert_not_called()


@patch('safe_store.store.ASCIIColors')
def test_add_document_empty_file(mock_store_colors, safe_store_instance: safe_store, tmp_path: Path):
    """Test adding an empty document."""
    store = safe_store_instance
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()

    with store:
        store.add_document(empty_file)

    # Check logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {empty_file.name}")
    assert_log_call_containing(mock_store_colors.info, "Document 'empty.txt' is new.")
    expected_warning = f"No chunks generated for {empty_file.name}. Document record saved, but skipping vectorization."
    assert_log_call_containing(mock_store_colors.warning, expected_warning)
    process_message_found = any(f"Successfully processed" in args[0] for call in mock_store_colors.success.call_args_list for args in call.args if isinstance(args, tuple))
    assert not process_message_found, "Processing success message should NOT be logged for empty file."

    # Check DB State
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, file_hash FROM documents WHERE file_path = ?", (str(empty_file.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    assert doc_result[1] is not None
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 0
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 0
    conn.close()


@patch('safe_store.store.ASCIIColors')
def test_add_document_hash_failure(mock_store_colors, safe_store_instance: safe_store, sample_text_file: Path):
    """Test behavior when file hashing fails."""
    store = safe_store_instance
    file_path = sample_text_file
    error_message = "Mock hashing failed due to permission error"

    with store:
        conn = store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        initial_doc_count = cursor.fetchone()[0]

    with patch.object(store, '_get_file_hash', side_effect=FileHandlingError(error_message)) as mock_hasher:
        with pytest.raises(FileHandlingError, match=re.escape(error_message)):
            with store:
                 store.add_document(file_path)
        mock_hasher.assert_called_once_with(file_path)

    # Check the log message printed when the error is caught in add_document
    assert_log_call_containing(mock_store_colors.error, f"Error during add_document: FileHandlingError: {error_message}")
    mock_store_colors.success.assert_not_called()

    # Check DB State
    with store:
        conn = store.conn
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        final_doc_count = cursor.fetchone()[0]
    assert final_doc_count == initial_doc_count


@patch('safe_store.indexing.parser.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_unsupported_type(mock_store_colors, mock_parser_colors, safe_store_instance: safe_store, tmp_path: Path):
    """Test adding a file with an unsupported extension."""
    store = safe_store_instance
    unsupported_file = tmp_path / "document.xyz"
    unsupported_file.write_text("Some content", encoding='utf-8')

    expected_error_msg_part = f"Unsupported file type extension: '.xyz' for file: {unsupported_file}. No parser available."
    with pytest.raises(ConfigurationError, match=re.escape(expected_error_msg_part)):
        with store:
            store.add_document(unsupported_file)

    # Check logs
    assert_log_call_containing(mock_parser_colors.warning, expected_error_msg_part)
    assert_log_call_containing(mock_store_colors.error, f"Error during add_document: ConfigurationError: {expected_error_msg_part}")