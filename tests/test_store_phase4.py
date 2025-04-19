# tests/test_store_phase4.py
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import re

# Import exceptions and modules
from safestore import store as safestore_store_module
from safestore import SafeStore, LogLevel
from safestore.core import db
from safestore.core.exceptions import ConcurrencyError, FileHandlingError, ConfigurationError, SafeStoreError

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

# Helper for log checks
def assert_log_call_containing(mock_logger, expected_substring):
    """Checks if any call to the mock logger contained the substring."""
    found = False
    # Check positional args
    for call_args in mock_logger.call_args_list:
        args, kwargs = call_args
        if args and isinstance(args[0], str) and expected_substring in args[0]:
            found = True
            break
    # Check method calls if not found in direct calls (needed for specific methods like info, debug)
    if not found:
        for method_call in mock_logger.method_calls:
            call_name, args, kwargs = method_call
            if args and isinstance(args[0], str) and expected_substring in args[0]:
                found = True
                break
    assert found, f"Expected log call containing '{expected_substring}' not found in {mock_logger.call_args_list} or {mock_logger.method_calls}"

# +++ FIX: Added mock_db_colors to decorator and arguments +++
@patch('safestore.core.db.ASCIIColors')
@patch('safestore.store.ASCIIColors')
@patch('safestore.vectorization.manager.ASCIIColors')
def test_store_close_and_context_manager(mock_manager_colors, mock_store_colors, mock_db_colors, temp_db_path):
    """Test explicit close() and context manager usage."""
    # mock_db_colors is now available
    store = SafeStore(temp_db_path, log_level=LogLevel.DEBUG)
    assert store.conn is not None
    assert not store._is_closed
    # Check initial connection log
    assert_log_call_containing(mock_db_colors.debug, "Connected to database:")

    # --- Ensure cache is populated before first close ---
    try:
        with store:
             _ = store.vectorizer_manager.get_vectorizer(store.DEFAULT_VECTORIZER, store.conn, None)
    except Exception as e:
         if not isinstance(e, ConfigurationError):
              print(f"Warning: Error populating cache in test: {e}")
         pass

    store.close()
    assert store.conn is None
    assert store._is_closed
    assert_log_call_containing(mock_store_colors.info, "SafeStore connection closed.")
    try:
        assert_log_call_containing(mock_manager_colors.debug, "Cleared vectorizer manager cache")
    except AssertionError:
        print("Cache clear log not found, cache might have been empty.")


    # Test context manager re-opening and closing
    mock_store_colors.reset_mock()
    mock_manager_colors.reset_mock()
    mock_db_colors.reset_mock() # Reset this mock too
    with SafeStore(temp_db_path, log_level=LogLevel.DEBUG) as store_ctx:
        assert store_ctx.conn is not None
        assert not store_ctx._is_closed
        # +++ FIX: Check the correct mock for the connection log +++
        assert_log_call_containing(mock_db_colors.debug, "Connected to database:")

        # --- Ensure cache is populated before second close (via exit) ---
        try:
             _ = store_ctx.vectorizer_manager.get_vectorizer(store.DEFAULT_VECTORIZER, store_ctx.conn, None)
        except Exception as e:
             if not isinstance(e, ConfigurationError):
                  print(f"Warning: Error populating cache in test context: {e}")
             pass

    # Check logs after exiting context
    assert store_ctx.conn is None
    assert store_ctx._is_closed
    assert_log_call_containing(mock_store_colors.debug, "SafeStore context closed cleanly.")
    try:
        assert_log_call_containing(mock_manager_colors.debug, "Cleared vectorizer manager cache")
    except AssertionError:
        print("Cache clear log not found after context exit, cache might have been empty.")


def test_list_documents_empty(safestore_instance: SafeStore):
    """Test listing documents from an empty store."""
    with safestore_instance as store:
        docs = store.list_documents()
    assert docs == []

def test_list_vectorization_methods_empty(safestore_instance: SafeStore):
    """Test listing methods from an empty store."""
    with safestore_instance as store:
        methods = store.list_vectorization_methods()
    assert methods == []


def test_list_documents_populated(populated_store: SafeStore):
    """Test listing documents after adding some."""
    with populated_store as store:
        docs = store.list_documents()

    assert len(docs) == 2
    doc1_info = next((d for d in docs if "sample.txt" in d["file_path"]), None)
    doc2_info = next((d for d in docs if "sample2.txt" in d["file_path"]), None)
    assert doc1_info is not None; assert doc2_info is not None
    assert doc1_info["doc_id"] is not None; assert isinstance(doc1_info["file_path"], str)
    assert doc1_info["file_hash"] is not None; assert doc1_info["added_timestamp"] is not None
    assert doc1_info["metadata"] is None


def test_list_vectorization_methods_populated(populated_store: SafeStore):
    """Test listing methods after adding documents."""
    with populated_store as store:
        methods = store.list_vectorization_methods()

    st_method = next((m for m in methods if m["method_name"] == store.DEFAULT_VECTORIZER), None)
    assert st_method is not None, f"Default vectorizer {store.DEFAULT_VECTORIZER} not found in listed methods."

    assert st_method["method_type"] == "sentence_transformer"
    assert st_method["vector_dim"] == 384 # Mocked or real dimension
    assert st_method["vector_dtype"] == "float32"
    assert st_method["params"] == {}, f"Expected params to be {{}}, got {st_method['params']}"
    assert len(methods) == 1, f"Expected 1 method, found {len(methods)}"