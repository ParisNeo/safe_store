import pytest
from safe_store.indexing.chunking import generate_chunks

def test_chunk_simple():
    text = "abcdefghijklmnopqrstuvwxyz"
    # generate_chunks returns List[Tuple[str, str]] (vector_text, storage_text)
    chunks = generate_chunks(text, strategy='character', chunk_size=10, chunk_overlap=3)
    assert len(chunks) == 4
    assert chunks[0][0] == "abcdefghij"
    assert chunks[1][0] == "hijklmnopq"
    assert chunks[2][0] == "opqrstuvwx"
    assert chunks[3][0] == "vwxyz"

def test_chunk_no_overlap():
    text = "abcde fghij klmno"
    chunks = generate_chunks(text, strategy='character', chunk_size=5, chunk_overlap=0)
    assert len(chunks) == 4
    assert chunks[0][0] == "abcde"
    assert chunks[1][0] == " fghi" # Note space included
    assert chunks[2][0] == "j klm"
    assert chunks[3][0] == "no"


def test_chunk_large_overlap_error():
     with pytest.raises(ValueError):
         generate_chunks("abc", strategy='character', chunk_size=5, chunk_overlap=5)

def test_chunk_smaller_than_size():
     text = "short"
     chunks = generate_chunks(text, strategy='character', chunk_size=10, chunk_overlap=2)
     assert len(chunks) == 1
     assert chunks[0][0] == "short"

def test_chunk_exact_size():
     text = "exactsize!" # 10 chars
     chunks = generate_chunks(text, strategy='character', chunk_size=10, chunk_overlap=2)
     assert len(chunks) == 1
     assert chunks[0][0] == "exactsize!"

def test_chunk_edge_case_overlap():
     text = "1234567890"
     chunks = generate_chunks(text, strategy='character', chunk_size=5, chunk_overlap=4)
     assert len(chunks) == 6
     assert chunks[0][0] == "12345"
     assert chunks[-1][0] == "67890"