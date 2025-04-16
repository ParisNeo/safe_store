import pytest
from safestore.indexing.chunking import chunk_text

def test_chunk_simple():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=3)
    # Expected:
    # abcdefghij (0, 10)
    # hijklmnopq (7, 17)
    # opqrstuvwx (14, 24)
    # uvwxyz     (21, 26)
    assert len(chunks) == 4
    assert chunks[0] == ("abcdefghij", 0, 10)
    assert chunks[1] == ("hijklmnopq", 7, 17)
    assert chunks[2] == ("opqrstuvwx", 14, 24)
    assert chunks[3] == ('vwxyz', 21, 26) 

def test_chunk_no_overlap():
    text = "abcde fghij klmno"
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=0)
    assert len(chunks) == 4
    assert chunks[0] == ("abcde", 0, 5)
    assert chunks[1] == (" fghi", 5, 10) # Note space included
    assert chunks[2] == ("j klm", 10, 15)
    assert chunks[3] == ("no", 15, 17)


def test_chunk_large_overlap_error():
     with pytest.raises(ValueError):
         chunk_text("abc", chunk_size=5, chunk_overlap=5)

def test_chunk_smaller_than_size():
     text = "short"
     chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
     assert len(chunks) == 1
     assert chunks[0] == ("short", 0, 5)

def test_chunk_exact_size():
     text = "exactsize!" # 10 chars
     chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
     assert len(chunks) == 1
     assert chunks[0] == ("exactsize!", 0, 10)

def test_chunk_edge_case_overlap():
     # Test where overlap calculation might stall if not handled
     text = "1234567890"
     chunks = chunk_text(text, chunk_size=5, chunk_overlap=4)
     # 12345 (0, 5)
     # 23456 (1, 6)
     # 34567 (2, 7)
     # ...
     # 67890 (5, 10)
     assert len(chunks) == 6
     assert chunks[0] == ("12345", 0, 5)
     assert chunks[-1] == ("67890", 5, 10)