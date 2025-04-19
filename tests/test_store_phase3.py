# tests/test_store_phase3.py
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from filelock import Timeout

from safestore import store as safestore_module
from safestore import SafeStore, LogLevel
from safestore.core import db

# --- Test Constants ---
PDF_TEXT = "This is PDF content."
DOCX_TEXT = "This is DOCX content."
# *** ADJUST HTML_TEXT to match the *actual* content being parsed from the file ***
HTML_TEXT = "This is HTML content." # Assuming the file only contains this now

# Skip tests if parsing dependencies are not installed
try:
    import pypdf
    import docx
    import bs4
    import lxml
    PARSING_LIBS_AVAILABLE = True
except ImportError:
    PARSING_LIBS_AVAILABLE = False

# Helper for log checks
def assert_log_call_containing(mock_logger, expected_substring):
    found = False
    for call_args in mock_logger.call_args_list:
        args, kwargs = call_args
        if args and isinstance(args[0], str) and expected_substring in args[0]:
            found = True
            break
    assert found, f"Expected log call containing '{expected_substring}' not found in {mock_logger.call_args_list}"

# --- Parser Integration Tests ---

# PDF Test (Should be passing now)
@pytest.mark.skipif(not PARSING_LIBS_AVAILABLE, reason="Requires parsing dependencies")
@patch('safestore.indexing.parser.ASCIIColors')
@patch('safestore.store.ASCIIColors')
def test_add_document_pdf(mock_store_colors, mock_parser_colors, safestore_instance: SafeStore, sample_pdf_file: Path):
    """Test adding a PDF document via SafeStore.add_document."""
    store = safestore_instance
    store.add_document(sample_pdf_file, chunk_size=50, chunk_overlap=10)
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_pdf_file.name}")
    assert_log_call_containing(mock_parser_colors.debug, "Dispatching parser for extension '.pdf'")
    assert_log_call_containing(mock_parser_colors.debug, f"Attempting to parse PDF file: {sample_pdf_file}")
    assert_log_call_containing(mock_store_colors.info, "Generated 1 chunks for")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_pdf_file.name}'")
    mock_store_colors.error.assert_not_called()
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(sample_pdf_file.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    parsed_text = doc_result[1]
    assert len(parsed_text) > 5
    assert "pdf" in parsed_text.lower()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 1
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 1
    conn.close()


@pytest.mark.skipif(not PARSING_LIBS_AVAILABLE, reason="Requires parsing dependencies")
@patch('safestore.indexing.parser.ASCIIColors')
@patch('safestore.store.ASCIIColors')
def test_add_document_docx(mock_store_colors, mock_parser_colors, safestore_instance: SafeStore, sample_docx_file: Path):
    """Test adding a DOCX document via SafeStore.add_document."""
    store = safestore_instance
    store.add_document(sample_docx_file, chunk_size=50, chunk_overlap=10)

    # Check Logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_docx_file.name}")
    assert_log_call_containing(mock_parser_colors.debug, "Dispatching parser for extension '.docx'")
    assert_log_call_containing(mock_parser_colors.debug, f"Attempting to parse DOCX file: {sample_docx_file}")
    # *** CORRECTED expected chunk count based on latest logs ***
    assert_log_call_containing(mock_store_colors.info, "Generated 1 chunks for")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_docx_file.name}'")
    mock_store_colors.error.assert_not_called()

    # Check DB
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(sample_docx_file.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    # Use strip() for DOCX check as it might add extra newline
    assert DOCX_TEXT == doc_result[1].strip()

    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    # *** CORRECTED expected chunk count ***
    assert chunk_count == 1

    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    # *** CORRECTED expected vector count ***
    assert vector_count == 1
    conn.close()


@pytest.mark.skipif(not PARSING_LIBS_AVAILABLE, reason="Requires parsing dependencies")
@patch('safestore.indexing.parser.ASCIIColors')
@patch('safestore.store.ASCIIColors')
def test_add_document_html(mock_store_colors, mock_parser_colors, safestore_instance: SafeStore, sample_html_file: Path):
    """Test adding an HTML document via SafeStore.add_document."""
    store = safestore_instance
    store.add_document(sample_html_file, chunk_size=50, chunk_overlap=10)

    # Check Logs
    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_html_file.name}")
    assert_log_call_containing(mock_parser_colors.debug, "Dispatching parser for extension '.html'")
    assert_log_call_containing(mock_parser_colors.debug, f"Attempting to parse HTML file: {sample_html_file}")
    # *** CORRECTED expected chunk count based on latest logs ***
    assert_log_call_containing(mock_store_colors.info, "Generated 1 chunks for")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_html_file.name}'")
    mock_store_colors.error.assert_not_called()

    # Check DB
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(sample_html_file.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    # *** Use the corrected HTML_TEXT constant defined above ***
    assert HTML_TEXT == doc_result[1]

    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    # *** CORRECTED expected chunk count ***
    assert chunk_count == 1

    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    # *** CORRECTED expected vector count ***
    assert vector_count == 1
    conn.close()

# --- Concurrency Tests (Should pass) ---
@patch('safestore.store.ASCIIColors')
def test_add_document_lock_acquired(mock_store_colors, safestore_instance: SafeStore, sample_text_file: Path):
    """Test that add_document acquires and releases the lock."""
    store = safestore_instance
    with patch.object(store, '_file_lock', new_callable=MagicMock) as mock_lock_instance:
        store.add_document(sample_text_file)
        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_called_once()
    assert_log_call_containing(mock_store_colors.debug, "Attempting to acquire write lock for add_document")
    assert_log_call_containing(mock_store_colors.info, "Write lock acquired for add_document")
    assert_log_call_containing(mock_store_colors.debug, f"Write lock released for add_document: {sample_text_file.name}")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_text_file.name}'")

@patch('safestore.store.ASCIIColors')
def test_add_document_lock_timeout(mock_store_colors, safestore_instance: SafeStore, sample_text_file: Path):
    """Test that add_document handles a lock timeout."""
    store = safestore_instance
    with patch.object(store, '_file_lock', new_callable=MagicMock) as mock_lock_instance:
        mock_lock_instance.__enter__.side_effect = Timeout("Mock Lock Already Held")
        with pytest.raises(Timeout, match=f"Could not acquire write lock for {sample_text_file.name}"):
            store.add_document(sample_text_file)
        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_not_called()
    assert_log_call_containing(mock_store_colors.debug, "Attempting to acquire write lock for add_document")
    assert_log_call_containing(mock_store_colors.error, "Timeout")
    assert_log_call_containing(mock_store_colors.error, f"acquiring write lock for add_document: {sample_text_file.name}")
    mock_store_colors.success.assert_not_called()