import re
from typing import List, Dict, Any, Optional
import numpy as np


def strip_metadata_header(text: str) -> str:
    """Strips any leading '--- Document Context --- ... ------------------------' header."""
    if not isinstance(text, str):
        return ""
    pattern = r'^--- Document Context ---\n.*?------------------------\n\n'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def format_metadata_header(metadata: Optional[Dict[str, Any]]) -> str:
    """Formats a single standardized metadata header block."""
    if not metadata or not isinstance(metadata, dict):
        return ""
    clean_items = {k: v for k, v in metadata.items() if k not in ("error", "raw") and v is not None}
    if not clean_items:
        return ""
    lines = ["--- Document Context ---"]
    for k, v in clean_items.items():
        lines.append(f"{str(k).title()}: {str(v)}")
    lines.append("------------------------\n\n")
    return "\n".join(lines)


def merge_overlapping_texts(text_a: str, text_b: str, min_overlap: int = 4) -> str:
    """
    Stitches text_a and text_b together by identifying and deduplicating
    overlapping text across their boundary.
    """
    if not text_a:
        return text_b or ""
    if not text_b:
        return text_a or ""

    # Check for direct inclusion
    if text_b in text_a:
        return text_a
    if text_a in text_b:
        return text_b

    # Extract optional section breadcrumb headers: [Section: ...]
    header_pattern = r'^(\[[^\]\n]+\]\n\n)'
    m_a = re.match(header_pattern, text_a)
    m_b = re.match(header_pattern, text_b)

    header_a = m_a.group(1) if m_a else ""
    header_b = m_b.group(1) if m_b else ""

    body_a = text_a[len(header_a):]
    body_b = text_b[len(header_b):] if (header_b and header_b == header_a) else text_b

    # 1. Exact suffix-prefix character overlap
    max_k = min(len(body_a), len(body_b))
    best_overlap = 0

    for k in range(max_k, min_overlap - 1, -1):
        if body_a.endswith(body_b[:k]):
            best_overlap = k
            break

    if best_overlap > 0:
        merged_body = body_a + body_b[best_overlap:]
        return header_a + merged_body

    # 2. Whitespace-trimmed suffix-prefix overlap
    stripped_a = body_a.rstrip()
    stripped_b = body_b.lstrip()
    max_k_stripped = min(len(stripped_a), len(stripped_b))

    for k in range(max_k_stripped, min_overlap - 1, -1):
        if stripped_a.endswith(stripped_b[:k]):
            offset = len(body_b) - len(stripped_b) + k
            sep = " " if not stripped_a.endswith(('\n', ' ')) and not body_b[offset:].startswith(('\n', ' ', '.', '!', '?')) else ""
            merged_body = stripped_a + sep + body_b[offset:].lstrip()
            return header_a + merged_body

    # 3. Word-level suffix-prefix overlap
    words_a = body_a.split()
    words_b = body_b.split()
    max_words = min(len(words_a), len(words_b), 100)
    word_overlap = 0

    for m in range(max_words, 1, -1):
        if words_a[-m:] == words_b[:m]:
            word_overlap = m
            break

    if word_overlap > 0:
        remaining_words = words_b[word_overlap:]
        sep = " " if not body_a.endswith((' ', '\n')) else ""
        merged_body = body_a + sep + " ".join(remaining_words)
        return header_a + merged_body

    # 4. Fallback if no overlap detected: join with appropriate spacing
    sep = "\n\n" if ("\n" in body_a or "\n" in body_b) else " "
    return header_a + body_a + sep + body_b


def reconstruct_document_chunks(
    doc_chunks: List[Dict[str, Any]],
    full_document_text: Optional[str] = None,
    add_metadata: bool = True
) -> Dict[str, Any]:
    """
    Reconstructs chunks belonging to a single document into chronologically ordered,
    overlap-deduplicated text with gap indicators ('...') and a single metadata header.
    """
    if not doc_chunks:
        return {}

    # Sort chunks by chunk_seq (chronological order in document), safely handling None
    sorted_chunks = sorted(
        doc_chunks,
        key=lambda c: (
            c.get("chunk_seq") if c.get("chunk_seq") is not None else 0,
            c.get("start_pos") if c.get("start_pos") is not None else 0,
            c.get("chunk_id") if c.get("chunk_id") is not None else 0
        )
    )

    # Deduplicate exact duplicate chunk IDs or sequences
    unique_chunks = []
    seen_ids = set()
    for c in sorted_chunks:
        cid = c.get("chunk_id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(c)

    sorted_chunks = unique_chunks

    # Group into contiguous clusters
    clusters: List[List[Dict[str, Any]]] = []
    current_cluster: List[Dict[str, Any]] = [sorted_chunks[0]]

    for i in range(1, len(sorted_chunks)):
        prev_chunk = sorted_chunks[i - 1]
        curr_chunk = sorted_chunks[i]

        prev_seq = prev_chunk.get("chunk_seq")
        curr_seq = curr_chunk.get("chunk_seq")

        # Contiguous if sequences are consecutive
        is_contiguous = False
        if prev_seq is not None and curr_seq is not None:
            is_contiguous = (curr_seq == prev_seq + 1 or curr_seq == prev_seq)

        if is_contiguous:
            current_cluster.append(curr_chunk)
        else:
            clusters.append(current_cluster)
            current_cluster = [curr_chunk]

    if current_cluster:
        clusters.append(current_cluster)

    # Process each cluster
    merged_cluster_texts: List[str] = []

    for cluster in clusters:
        # Extract clean texts without repeated metadata headers
        cluster_texts = [
            strip_metadata_header(c.get("raw_chunk_text") or c.get("chunk_text", ""))
            for c in cluster
        ]

        # Attempt full_text span reconstruction if available
        span_reconstructed = None
        if full_document_text and len(cluster_texts) > 1:
            first_body = re.sub(r'^(\[[^\]\n]+\]\n\n)', '', cluster_texts[0]).strip()
            last_body = re.sub(r'^(\[[^\]\n]+\]\n\n)', '', cluster_texts[-1]).strip()

            pos_start = full_document_text.find(first_body[:80]) if len(first_body) >= 80 else full_document_text.find(first_body)
            pos_end_match = full_document_text.rfind(last_body[-80:]) if len(last_body) >= 80 else full_document_text.rfind(last_body)

            if pos_start != -1 and pos_end_match != -1 and pos_start <= pos_end_match:
                end_pos = pos_end_match + (80 if len(last_body) >= 80 else len(last_body))
                # Check for breadcrumb header on first chunk
                header_match = re.match(r'^(\[[^\]\n]+\]\n\n)', cluster_texts[0])
                header_prefix = header_match.group(1) if header_match else ""
                span_reconstructed = header_prefix + full_document_text[pos_start:end_pos].strip()

        if span_reconstructed:
            merged_cluster_texts.append(span_reconstructed)
        else:
            # Pairwise overlap merge fallback
            merged = cluster_texts[0]
            for next_text in cluster_texts[1:]:
                merged = merge_overlapping_texts(merged, next_text)
            merged_cluster_texts.append(merged)

    # Fuse non-contiguous clusters with '...' placeholder
    reconstructed_body = "\n\n...\n\n".join(merged_cluster_texts)

    # Extract document-level metadata and top scores
    metadata = sorted_chunks[0].get("document_metadata")
    metadata_header = format_metadata_header(metadata) if add_metadata else ""

    final_chunk_text = f"{metadata_header}{reconstructed_body}".strip()

    # Find highest scoring chunk to anchor relevance metrics
    best_chunk = max(
        sorted_chunks,
        key=lambda c: float(c.get("relevance_score", c.get("similarity_percent", c.get("similarity_score", 0.0))))
    )

    all_chunk_ids = [c["chunk_id"] for c in sorted_chunks if "chunk_id" in c]
    all_seqs = [c["chunk_seq"] for c in sorted_chunks if "chunk_seq" in c]

    result_payload = {
        "doc_id": sorted_chunks[0].get("doc_id"),
        "file_path": sorted_chunks[0].get("file_path", ""),
        "document_title": sorted_chunks[0].get("document_title") or sorted_chunks[0].get("file_path", "").split("/")[-1].split("\\")[-1],
        "chunk_text": final_chunk_text,
        "chunk_id": best_chunk.get("chunk_id"),
        "fused_chunk_ids": all_chunk_ids,
        "chunk_seqs": all_seqs,
        "document_metadata": metadata,
        "similarity_score": best_chunk.get("similarity_score", 0.0),
        "similarity_percent": best_chunk.get("similarity_percent", best_chunk.get("relevance_score", 0.0)),
        "relevance_score": best_chunk.get("relevance_score", best_chunk.get("similarity_percent", 0.0)),
        "is_reconstructed": True,
        "cluster_count": len(clusters),
        "merged_chunk_count": len(sorted_chunks)
    }

    # Pass through optional search scores
    if "raw_rrf_score" in best_chunk:
        result_payload["raw_rrf_score"] = best_chunk["raw_rrf_score"]
    if "fused_score" in best_chunk:
        result_payload["fused_score"] = best_chunk["fused_score"]

    return result_payload


def reconstruct_overlapping_chunks(
    results: List[Dict[str, Any]],
    store: Optional[Any] = None,
    add_metadata: bool = True
) -> List[Dict[str, Any]]:
    """
    Groups search results by document, re-orders chunks chronologically,
    deduplicates contiguous overlaps, bridges non-contiguous gaps with '...',
    and prepends a single metadata header per document.
    """
    if not results:
        return []

    # Group by document key (doc_id or file_path)
    doc_groups: Dict[Any, List[Dict[str, Any]]] = {}
    doc_order: List[Any] = []

    for r in results:
        doc_key = r.get("doc_id")
        if doc_key is None:
            doc_key = r.get("file_path", id(r))

        if doc_key not in doc_groups:
            doc_groups[doc_key] = []
            doc_order.append(doc_key)
        doc_groups[doc_key].append(r)

    reconstructed_results = []

    for doc_key in doc_order:
        chunks = doc_groups[doc_key]

        # Retrieve full document text if store is provided
        full_text = None
        if store is not None:
            try:
                full_text = store.reconstruct_document_text(doc_key)
            except Exception:
                full_text = None

        reconstructed_doc = reconstruct_document_chunks(
            doc_chunks=chunks,
            full_document_text=full_text,
            add_metadata=add_metadata
        )
        if reconstructed_doc:
            reconstructed_results.append(reconstructed_doc)

    # Sort documents descending by top relevance score
    reconstructed_results.sort(
        key=lambda d: float(d.get("relevance_score", d.get("similarity_percent", 0.0))),
        reverse=True
    )

    return reconstructed_results