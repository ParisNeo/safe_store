"""Diagnostic tests proving `SafeStore.list_documents()` reports DB-truth chunk counts.

Invariant under test:

    list_documents()[i]["chunk_count"] ==
        SELECT COUNT(*) FROM chunks WHERE doc_id = list_documents()[i]["doc_id"]

Covered scenarios (each corresponds to a plausible distinct regression path):

  1. Live session — same SafeStore instance that ran `add_text`.
  2. Reopened store — fresh SafeStore against an existing DB file, simulating
     a server process restart between indexing and UI listing.
  3. Encrypted store — file_path is blinded to `enc:<hash>`, chunks are
     Fernet-encrypted blobs; the chunk-count must still be correct because
     COUNT(*)-style queries never need decryption.

The failure messages embed `sorted(doc.keys())` so an absent-key bug
(`chunk_count` missing from the return dict — the actual root cause here)
is immediately distinguishable from a wrong-value bug (stale cache,
FK-mismatch join, encryption bypass).
"""
from pathlib import Path

from safe_store import SafeStore


LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 300


def _doc_id(doc: dict) -> int:
    for key in ("doc_id", "document_id", "id"):
        value = doc.get(key)
        if value is not None:
            return int(value)
    raise AssertionError(f"document dict missing id field: {sorted(doc.keys())}")


def _raw_chunk_count(store: SafeStore, doc: dict) -> int:
    row = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (_doc_id(doc),)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _reported_chunk_count(doc: dict) -> int:
    assert "chunk_count" in doc, (
        f"list_documents() returned no 'chunk_count' key: keys={sorted(doc.keys())}"
    )
    value = doc["chunk_count"]
    assert isinstance(value, int) and value >= 0, (
        f"'chunk_count' must be a non-negative int, got "
        f"{type(value).__name__}={value!r}"
    )
    return value


def test_list_documents_reports_chunk_count_live_session(tmp_path: Path):
    db = tmp_path / "live.db"
    with SafeStore(db_path=db) as store:
        store.add_text(unique_id="doc-a", text=LONG_TEXT, metadata={"title": "A"})
        docs = store.list_documents()
        assert len(docs) == 1, f"expected 1 doc, got {len(docs)}"
        doc = docs[0]
        raw = _raw_chunk_count(store, doc)
        reported = _reported_chunk_count(doc)
        assert raw > 0, "chunks must exist in DB after add_text"
        assert reported == raw, (
            f"LIVE-SESSION MISMATCH: reported={reported} raw={raw} "
            f"keys={sorted(doc.keys())}"
        )


def test_list_documents_reports_chunk_count_after_reopen(tmp_path: Path):
    db = tmp_path / "reopened.db"
    with SafeStore(db_path=db) as store:
        store.add_text(unique_id="doc-b", text=LONG_TEXT, metadata={"title": "B"})
    with SafeStore(db_path=db) as store:
        docs = store.list_documents()
        assert len(docs) == 1
        doc = docs[0]
        raw = _raw_chunk_count(store, doc)
        reported = _reported_chunk_count(doc)
        assert raw > 0, "chunks must persist across close/reopen"
        assert reported == raw, (
            f"REOPEN MISMATCH: reported={reported} raw={raw} "
            f"keys={sorted(doc.keys())}"
        )


def test_list_documents_reports_chunk_count_encrypted_store(tmp_path: Path):
    db = tmp_path / "encrypted.db"
    key = "test-encryption-key-42"
    with SafeStore(db_path=db, encryption_key=key) as store:
        store.add_text(unique_id="doc-c", text=LONG_TEXT, metadata={"title": "C"})
        docs = store.list_documents()
        assert len(docs) == 1
        doc = docs[0]
        raw = _raw_chunk_count(store, doc)
        reported = _reported_chunk_count(doc)
        assert raw > 0, "chunks must exist in encrypted DB after add_text"
        assert reported == raw, (
            f"ENCRYPTED MISMATCH: reported={reported} raw={raw} "
            f"keys={sorted(doc.keys())}"
        )