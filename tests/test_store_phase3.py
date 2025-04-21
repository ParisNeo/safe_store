# tests/test_store_phase3.py
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import re

# Import exceptions and modules
from safe_store import store as safe_store_store_module
from safe_store import safe_store, LogLevel
from safe_store.core import db
from safe_store.core.exceptions import ConcurrencyError, FileHandlingError, ConfigurationError, safe_storeError

# Import filelock components
from filelock import Timeout, FileLock

# --- FIX: Define availability check locally ---
# Check for parsing libraries availability within this module
try:
    import pypdf
    import docx
    import bs4
    import lxml
    PARSING_LIBS_AVAILABLE = True
except ImportError:
    PARSING_LIBS_AVAILABLE = False

# --- Test Constants ---
PDF_TEXT = "This is PDF content."
DOCX_TEXT = "This is DOCX content."
HTML_TEXT = "This is HTML content." # Adjusted based on sample file

# Helper for log checks
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


# --- Parser Integration Tests ---

# Use the locally defined PARSING_LIBS_AVAILABLE for skipif
@pytest.mark.skipif(not PARSING_LIBS_AVAILABLE, reason="Requires parsing dependencies (pypdf, python-docx, beautifulsoup4, lxml)")
@patch('safe_store.indexing.parser.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_pdf(mock_store_colors, mock_parser_colors, safe_store_instance: safe_store, sample_pdf_file: Path):
    """Test adding a PDF document via safe_store.add_document."""
    store = safe_store_instance
    with store:
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
    doc_result = cursor.fetchone(); assert doc_result is not None
    doc_id = doc_result[0]; parsed_text = doc_result[1]
    assert len(parsed_text) > 5; assert "pdf" in parsed_text.lower()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]; assert chunk_count == 1
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]; assert vector_count == 1
    conn.close()


@pytest.mark.skipif(not PARSING_LIBS_AVAILABLE, reason="Requires parsing dependencies (pypdf, python-docx, beautifulsoup4, lxml)")
@patch('safe_store.indexing.parser.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_docx(mock_store_colors, mock_parser_colors, safe_store_instance: safe_store, sample_docx_file: Path):
    """Test adding a DOCX document via safe_store.add_document."""
    store = safe_store_instance
    with store:
        store.add_document(sample_docx_file, chunk_size=50, chunk_overlap=10)

    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_docx_file.name}")
    assert_log_call_containing(mock_parser_colors.debug, "Dispatching parser for extension '.docx'")
    assert_log_call_containing(mock_parser_colors.debug, f"Attempting to parse DOCX file: {sample_docx_file}")
    assert_log_call_containing(mock_store_colors.info, "Generated 1 chunks for")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_docx_file.name}'")
    mock_store_colors.error.assert_not_called()

    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(sample_docx_file.resolve()),))
    doc_result = cursor.fetchone(); assert doc_result is not None
    doc_id = doc_result[0]; assert DOCX_TEXT == doc_result[1].strip()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]; assert chunk_count == 1
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]; assert vector_count == 1
    conn.close()


@pytest.mark.skipif(not PARSING_LIBS_AVAILABLE, reason="Requires parsing dependencies (pypdf, python-docx, beautifulsoup4, lxml)")
@patch('safe_store.indexing.parser.ASCIIColors')
@patch('safe_store.store.ASCIIColors')
def test_add_document_html(mock_store_colors, mock_parser_colors, safe_store_instance: safe_store, sample_html_file: Path):
    """Test adding an HTML document via safe_store.add_document."""
    store = safe_store_instance
    with store:
        store.add_document(sample_html_file, chunk_size=50, chunk_overlap=10)

    assert_log_call_containing(mock_store_colors.info, f"Starting indexing process for: {sample_html_file.name}")
    assert_log_call_containing(mock_parser_colors.debug, "Dispatching parser for extension '.html'")
    assert_log_call_containing(mock_parser_colors.debug, f"Attempting to parse HTML file: {sample_html_file}")
    assert_log_call_containing(mock_store_colors.info, "Generated 1 chunks for")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_html_file.name}'")
    mock_store_colors.error.assert_not_called()

    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (str(sample_html_file.resolve()),))
    doc_result = cursor.fetchone(); assert doc_result is not None
    doc_id = doc_result[0]; assert HTML_TEXT == doc_result[1].strip()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]; assert chunk_count == 1
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]; assert vector_count == 1
    conn.close()

# --- Concurrency Tests ---
@patch('safe_store.store.ASCIIColors')
def test_add_document_lock_acquired(mock_store_colors, safe_store_instance: safe_store, sample_text_file: Path):
    """Test that add_document acquires and releases the lock."""
    store = safe_store_instance
    with patch.object(store, '_file_lock', autospec=True) as mock_lock_instance:
        mock_lock_instance.__enter__.return_value = None
        mock_lock_instance.__exit__.return_value = None
        with store:
            store.add_document(sample_text_file)
        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_called_once()

    assert_log_call_containing(mock_store_colors.debug, "Attempting to acquire write lock for add_document")
    assert_log_call_containing(mock_store_colors.info, "Write lock acquired for add_document")
    assert_log_call_containing(mock_store_colors.debug, f"Write lock released for add_document: {sample_text_file.name}")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_text_file.name}'")


@patch('safe_store.store.ASCIIColors')
def test_add_document_lock_timeout(mock_store_colors, safe_store_instance: safe_store, sample_text_file: Path):
    """Test that add_document handles a lock timeout."""
    store = safe_store_instance
    mock_lock_instance = MagicMock(spec=FileLock)
    timeout_exception = Timeout(store.lock_path)
    mock_lock_instance.__enter__.side_effect = timeout_exception

    with patch.object(store, '_file_lock', mock_lock_instance):
        expected_error_msg = f"Timeout ({store.lock_timeout}s) acquiring write lock for add_document: {sample_text_file.name}"
        with pytest.raises(ConcurrencyError, match=re.escape(expected_error_msg)):
             store.add_document(sample_text_file)
        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_not_called()

    assert_log_call_containing(mock_store_colors.debug, "Attempting to acquire write lock for add_document")
    assert_log_call_containing(mock_store_colors.error, expected_error_msg)
    mock_store_colors.success.assert_not_called()

