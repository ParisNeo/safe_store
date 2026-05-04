# tests/test_store_phase2.py
import pytest
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import re

# Import specific exceptions and modules
from safe_store import SafeStore, LogLevel
from safe_store.core import db
from safe_store.core.exceptions import ConfigurationError, SafeStoreError


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


@patch('safe_store.store.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
def test_query_simple(mock_manager_colors, mock_store_colors, populated_store: SafeStore, tmp_path: Path):
    """Test basic query functionality with populated store."""
    store = populated_store

    # Add another document with specific content for querying
    doc3_content = "The quick brown fox jumps over the lazy dog. This is a unique phrase for testing queries."
    doc3_path = tmp_path / "sample3.txt"
    doc3_path.write_text(doc3_content, encoding='utf-8')

    with store:
        store.add_document(doc3_path, chunk_size=50, chunk_overlap=5)
        results = store.query("quick brown fox", top_k=3)

    assert len(results) > 0
    # Check result structure
    first_result = results[0]
    assert 'chunk_text' in first_result
    assert 'similarity_score' in first_result
    assert 'file_path' in first_result
    assert isinstance(first_result['similarity_score'], (int, float))
    assert first_result['similarity_score'] >= 0


def test_query_no_results(safe_store_instance: SafeStore):
    """Test query on store with no matching results."""
    store = safe_store_instance

    with store:
        results = store.query("xyznonexistentquery12345", top_k=5)

    assert results == []


@patch('safe_store.store.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
def test_query_limit_results(mock_manager_colors, mock_store_colors, populated_store: SafeStore):
    """Test that top_k limits the number of results."""
    store = populated_store

    with store:
        results = store.query("the", top_k=1)

    assert len(results) <= 1


def test_query_persistence(safe_store_instance: SafeStore, sample_text_file: Path):
    """Test that queries work after closing and reopening the store."""
    store = safe_store_instance
    db_path = store.db_path

    # Add document and close
    with store:
        store.add_document(sample_text_file, chunk_size=30, chunk_overlap=5, chunking_strategy='character')

    # Reopen store
    store2 = SafeStore(db_path=db_path, log_level=LogLevel.DEBUG)

    with store2:
        results = store2.query("first sentence", top_k=3)

    assert len(results) > 0
    first_result = results[0]
    assert 'chunk_text' in first_result
    assert 'similarity_score' in first_result
    assert 'file_path' in first_result

    store2.close()


def test_init_vectorizer_not_found(temp_db_path: Path):
    """Test creating SafeStore with an invalid vectorizer name."""
    with pytest.raises(ConfigurationError, match="Unsupported vectorizer"):
        SafeStore(db_path=temp_db_path, vectorizer_name='nonexistent_vectorizer_xyz')


@patch('safe_store.store.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
@patch('safe_store.vectorization.methods.tf_idf.ASCIIColors')
def test_add_document_with_tfidf(mock_tfidf_colors, mock_manager_colors, mock_store_colors, temp_db_path: Path, sample_text_file: Path):
    """Test adding a document with TF-IDF vectorizer."""
    store = SafeStore(db_path=temp_db_path, vectorizer_name='tf_idf', log_level=LogLevel.DEBUG)

    with store:
        store.add_document(sample_text_file, chunk_size=30, chunk_overlap=5)

    # Check logs
    assert_log_call_containing(mock_manager_colors.info, "Initializing vectorizer: tf_idf")
    assert_log_call_containing(mock_store_colors.info, f"Vectorizing 2 chunks using 'tf_idf'")
    assert_log_call_containing(mock_store_colors.success, f"Successfully processed '{sample_text_file.name}' with vectorizer 'tf_idf'")

    # Check DB
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id FROM documents WHERE file_path = ?", (str(sample_text_file.resolve()),))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    doc_id = doc_result[0]
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    chunk_count = cursor.fetchone()[0]
    assert chunk_count == 2
    cursor.execute("SELECT COUNT(v.vector_id) FROM vectors v JOIN chunks c ON v.chunk_id = c.chunk_id WHERE c.doc_id = ?", (doc_id,))
    vector_count = cursor.fetchone()[0]
    assert vector_count == 2
    # Check vectorizer metadata
    cursor.execute("SELECT value FROM store_metadata WHERE key = 'vectorizer_info'")
    res = cursor.fetchone()
    assert res is not None
    v_info = json.loads(res[0])
    assert v_info['name'] == 'tf_idf'
    conn.close()


@patch('safe_store.store.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
def test_add_vectorization_incompatible(mock_manager_colors, mock_store_colors, temp_db_path: Path, sample_text_file: Path):
    """Test that reopening store with incompatible vectorizer raises error."""
    # First create store with default vectorizer (st)
    store1 = SafeStore(db_path=temp_db_path, vectorizer_name='st', log_level=LogLevel.DEBUG)

    with store1:
        store1.add_document(sample_text_file, chunk_size=30, chunk_overlap=5)

    store1.close()

    # Now try to open with different vectorizer
    with pytest.raises(ConfigurationError, match="incompatible vectorizer"):
        SafeStore(db_path=temp_db_path, vectorizer_name='tf_idf', log_level=LogLevel.DEBUG)


def test_empty_database_query(safe_store_instance: SafeStore):
    """Test query on completely fresh empty database returns empty list."""
    store = safe_store_instance

    with store:
        results = store.query("anything", top_k=5)

    assert results == []


@patch('safe_store.store.ASCIIColors')
@patch('safe_store.vectorization.manager.ASCIIColors')
def test_from_db_loads_all_configuration(mock_manager_colors, mock_store_colors, tmp_path: Path, sample_text_file: Path):
    """Test that from_db restores all store configuration from the database."""
    db_path = tmp_path / "config_test.db"

    # Create store with explicit non-default configuration
    store1 = SafeStore(
        db_path=db_path,
        vectorizer_name="st",
        vectorizer_config={"model": "all-MiniLM-L6-v2"},
        chunk_size=25,
        chunk_overlap=5,
        chunking_strategy="token",
        expand_before=2,
        expand_after=2,
        text_cleaner="basic",
        log_level=LogLevel.DEBUG
    )

    with store1:
        store1.add_document(sample_text_file, chunk_size=25, chunk_overlap=5)
        # Verify initial config
        assert store1.vectorizer_name == "st"
        assert store1.chunk_size == 25
        assert store1.chunk_overlap == 5
        assert store1.chunking_strategy == "token"
        assert store1.expand_before == 2
        assert store1.expand_after == 2

    store1.close()

    # Reopen using from_db with ONLY the path - no other parameters
    store2 = SafeStore.from_db(db_path)

    with store2:
        # Verify all configuration was restored from database
        assert store2.vectorizer_name == "st"
        assert store2.vectorizer_config == {"model": "all-MiniLM-L6-v2"}
        assert store2.chunk_size == 25
        assert store2.chunk_overlap == 5
        assert store2.chunking_strategy == "token"
        assert store2.expand_before == 2
        assert store2.expand_after == 2
        assert store2.text_cleaner_name == "basic"

        # Verify the store is functional - can query existing data
        results = store2.query("first sentence", top_k=3)
        assert len(results) > 0
        assert "chunk_text" in results[0]
        assert "similarity_score" in results[0]

        # Verify we can add new documents with the restored config
        doc2_path = tmp_path / "sample2.txt"
        doc2_path.write_text("Another document with different content.", encoding="utf-8")
        result = store2.add_document(doc2_path, chunk_size=25, chunk_overlap=5)
        assert result["num_chunks_added"] > 0

    store2.close()


@patch('safe_store.store.ASCIIColors')
def test_special_characters_in_text(mock_store_colors, safe_store_instance: SafeStore):
    """Test add_text and query with unicode and special characters."""
    store = safe_store_instance
    unique_id = "special_chars_doc"
    text = "Héllo wörld! 🎉 Привет мир 你好世界 αβγδ €100 «quoted» — em-dash"

    with store:
        store.add_text(unique_id, text, chunk_size=50, chunk_overlap=5)
        results = store.query("Héllo wörld", top_k=1)

    assert len(results) > 0
    assert 'chunk_text' in results[0]
    assert 'similarity_score' in results[0]
    assert 'file_path' in results[0]

    # Verify document was stored
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, full_text FROM documents WHERE file_path = ?", (unique_id,))
    doc_result = cursor.fetchone()
    assert doc_result is not None
    assert doc_result[1] == text
    conn.close()