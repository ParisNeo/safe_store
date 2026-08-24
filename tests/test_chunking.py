import pytest
import numpy as np
from safe_store.indexing.chunking import generate_chunks


SAMPLE_MARKDOWN = """# Architecture Overview

SafeStore is an ultra-fast local knowledge engine combining vectors and graphs.

## Storage Subsystem

The storage subsystem manages SQLite connections in WAL mode with robust file locks.

```python
import safe_store
store = safe_store.SafeStore("database.db")
```

### Table Layouts

| Table Name | Purpose |
| :--- | :--- |
| chunks | Stores chunk text and provenance |
| vectors | Stores raw binary embeddings |

## Search Subsystem

Combines BM25 and dense similarity with Reciprocal Rank Fusion.
"""


def test_fixed_size_character_chunking():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = generate_chunks(text, strategy="character", chunk_size=10, chunk_overlap=2)
    assert len(chunks) == 3
    assert chunks[0][0] == "ABCDEFGHIJ"
    assert chunks[1][0] == "IJKLMNOPQR"


def test_recursive_chunking_hierarchy():
    text = "Paragraph one with some text.\n\nParagraph two with different concepts.\n\nParagraph three."
    chunks = generate_chunks(text, strategy="recursive", chunk_size=40, chunk_overlap=5)
    assert len(chunks) >= 2
    for v_text, s_text in chunks:
        assert len(v_text) <= 50


def test_structure_aware_markdown_chunking():
    chunks = generate_chunks(SAMPLE_MARKDOWN, strategy="structure", chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 3

    # Check breadcrumb presence
    has_breadcrumb = any("Architecture Overview" in c[0] for c in chunks)
    has_sub_breadcrumb = any("Storage Subsystem" in c[0] for c in chunks)
    assert has_breadcrumb
    assert has_sub_breadcrumb


def test_semantic_chunking_with_similarity_valleys():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Canines and foxes belong to the Canidae animal family. "
        "Quantum computing relies on superposition and entanglement. "
        "Qubits allow exponential parallel state representations."
    )

    # Mock vectorizer simulating semantic shift
    def mock_embed(sentences):
        vecs = []
        for s in sentences:
            if "fox" in s or "Canidae" in s:
                vecs.append(np.array([1.0, 0.0]))
            else:
                vecs.append(np.array([0.0, 1.0]))
        return np.array(vecs)

    chunks = generate_chunks(
        text,
        strategy="semantic",
        chunk_size=200,
        vectorizer_fn=mock_embed,
        similarity_threshold=0.5
    )
    assert len(chunks) == 2
    assert "fox" in chunks[0][0]
    assert "Quantum" in chunks[1][0]


def test_contextual_retrieval_chunking():
    text = "Revenue grew by 15% in Q3 due to cloud migrations."
    doc = "Acme Corp Financial Report 2026. Detailed quarterly earnings."

    def mock_enricher(full_doc, chunk):
        return f"Document: {full_doc[:25]}..."

    chunks = generate_chunks(
        text,
        strategy="contextual",
        chunk_size=150,
        chunk_overlap=0,
        context_enricher=mock_enricher,
        full_document_text=doc
    )
    assert len(chunks) == 1
    assert "--- Context ---" in chunks[0][0]
    assert "Acme Corp" in chunks[0][0]


def test_late_chunking_boundaries():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = generate_chunks(text, strategy="late", chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2


def test_context_expansion_before_and_after():
    text = "CHUNK_A. CHUNK_B. CHUNK_C."
    chunks = generate_chunks(text, strategy="character", chunk_size=8, chunk_overlap=0, expand_before=5, expand_after=5)
    assert len(chunks) >= 2
    # Storage text should have more context than vector text
    assert len(chunks[1][1]) >= len(chunks[1][0])