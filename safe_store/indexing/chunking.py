# safe_store/indexing/chunking.py
import re
from typing import List, Tuple, Callable, Optional, Literal, Any, Union, Dict
import numpy as np
from ascii_colors import ASCIIColors

Chunk = Tuple[str, str] # (text_for_vectorization, text_for_storage)

ChunkingStrategy = Literal[
    'character',
    'token',
    'paragraph',
    'semantic',
    'recursive',
    'structure',
    'markdown',
    'contextual',
    'late'
]

DEFAULT_RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n### ",
    "\n## ",
    "\n# ",
    "\n```",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    " ",
    ""
]


def generate_chunks(
    text: str,
    strategy: ChunkingStrategy = 'token',
    chunk_size: int = 384,
    chunk_overlap: int = 50,
    expand_before: int = 0,
    expand_after: int = 0,
    tokenizer: Optional[Any] = None,
    vectorizer_fn: Optional[Callable[[List[str]], Any]] = None,
    similarity_threshold: float = 0.5,
    initial_semantic_blocks: int = 1,
    strict_size: bool = True,
    separators: Optional[List[str]] = None,
    context_enricher: Optional[Callable[[str, str], str]] = None,
    full_document_text: Optional[str] = None,
    **kwargs
) -> List[Chunk]:
    """
    Generates text chunks using one of 8 specialized RAG chunking strategies:

    1. 'character': Slices text by character count with sliding window overlap.
    2. 'token': Slices by token count, preserving exact line breaks and offsets.
    3. 'paragraph': Groups natural double-newline paragraphs up to chunk_size.
    4. 'recursive': Hierarchically splits text on paragraphs -> headers -> code -> sentences -> words.
    5. 'structure' / 'markdown': Parses Markdown headers (# H1, ## H2), keeping tables/code intact with breadcrumbs.
    6. 'semantic': Embeds sentences and cuts at topical shift points (valleys in cosine similarity).
    7. 'contextual': Enriches each chunk with full-document situating context (Anthropic pattern).
    8. 'late': Segments boundaries for late chunking (embed-first, mean-pool token representations later).
    """
    if not text or not text.strip():
        return []

    strategy_lower = str(strategy).lower()

    if strategy_lower == 'character':
        chunks = _chunk_by_character(text, chunk_size, chunk_overlap, expand_before, expand_after)
    elif strategy_lower == 'token':
        if tokenizer is None:
            raise ValueError("A tokenizer is required for 'token' strategy.")
        chunks = _chunk_by_tokens(text, tokenizer, chunk_size, chunk_overlap, expand_before, expand_after)
    elif strategy_lower == 'paragraph':
        chunks = _chunk_by_paragraph(text, chunk_size, chunk_overlap, tokenizer, strict_size)
    elif strategy_lower == 'recursive':
        chunks = _chunk_recursive(
            text,
            chunk_size,
            chunk_overlap,
            tokenizer,
            separators=separators or DEFAULT_RECURSIVE_SEPARATORS
        )
    elif strategy_lower in ('structure', 'markdown'):
        chunks = _chunk_structure_aware(
            text,
            chunk_size,
            chunk_overlap,
            tokenizer=tokenizer,
            expand_before=expand_before,
            expand_after=expand_after
        )
    elif strategy_lower == 'semantic':
        if vectorizer_fn is None:
            raise ValueError("vectorizer_fn is required for 'semantic' chunking strategy.")
        chunks = _chunk_semantic(
            text,
            chunk_size=chunk_size,
            tokenizer=tokenizer,
            vectorizer_fn=vectorizer_fn,
            similarity_threshold=similarity_threshold,
            initial_blocks=initial_semantic_blocks,
            strict_size=strict_size,
            chunk_overlap=chunk_overlap
        )
    elif strategy_lower == 'contextual':
        chunks = _chunk_contextual(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            context_enricher=context_enricher,
            full_document_text=full_document_text,
            **kwargs
        )
    elif strategy_lower == 'late':
        chunks = _chunk_late_boundaries(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Supported: 'character', 'token', 'paragraph', 'recursive', 'structure', 'markdown', 'semantic', 'contextual', 'late'."
        )

    ASCIIColors.debug("Chunking complete.")
    return chunks


def _get_length(text: str, tokenizer: Optional[Any]) -> int:
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text)


def _split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences while preserving sentence punctuation and spacing."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_into_paragraphs(text: str) -> List[str]:
    """Splits by double newlines while preserving content blocks."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# -----------------------------------------------------------------------------
# 1. Fixed-Size Character Chunking
# -----------------------------------------------------------------------------
def _chunk_by_character(text: str, chunk_size: int, chunk_overlap: int, expand_before: int, expand_after: int) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: List[Chunk] = []
    text_len = len(text)
    start_pos = 0

    while start_pos < text_len:
        end_pos = min(start_pos + chunk_size, text_len)
        vector_text = text[start_pos:end_pos]

        storage_start_pos = max(0, start_pos - expand_before)
        storage_end_pos = min(text_len, end_pos + expand_after)
        storage_text = text[storage_start_pos:storage_end_pos]

        chunks.append((vector_text, storage_text))

        if end_pos == text_len:
            break

        next_start_pos = start_pos + chunk_size - chunk_overlap
        if next_start_pos <= start_pos:
            start_pos += 1
        else:
            start_pos = next_start_pos

    return chunks


# -----------------------------------------------------------------------------
# 2. Fixed-Size Token Chunking (with Character Offset Preservation)
# -----------------------------------------------------------------------------
def _chunk_by_tokens(text: str, tokenizer: Any, chunk_size: int, chunk_overlap: int, expand_before: int, expand_after: int) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("token overlap must be smaller than token chunk_size.")

    offsets = []
    if hasattr(tokenizer, 'encode_with_offsets'):
        all_tokens, offsets = tokenizer.encode_with_offsets(text)
    else:
        all_tokens = tokenizer.encode(text)

    num_tokens = len(all_tokens)
    chunks: List[Chunk] = []
    start_token = 0

    while start_token < num_tokens:
        end_token = min(start_token + chunk_size, num_tokens)

        # When character offsets are available, slice directly from source text to preserve newlines
        if offsets and len(offsets) == num_tokens:
            v_start_char = offsets[start_token][0]
            v_end_char = offsets[end_token - 1][1]
            vector_text = text[v_start_char:v_end_char]

            storage_start_token = max(0, start_token - expand_before)
            storage_end_token = min(num_tokens, end_token + expand_after)
            s_start_char = offsets[storage_start_token][0]
            s_end_char = offsets[storage_end_token - 1][1]
            storage_text = text[s_start_char:s_end_char]
        else:
            vector_tokens = all_tokens[start_token:end_token]
            vector_text = tokenizer.decode(vector_tokens)

            # Snap to newline boundary if close
            if end_token < num_tokens and '\n' in vector_text:
                last_newline = vector_text.rfind('\n')
                if last_newline > len(vector_text) * 0.75:
                    cut_text = vector_text[:last_newline + 1]
                    cut_tokens = tokenizer.encode(cut_text)
                    end_token = start_token + len(cut_tokens)
                    vector_text = cut_text

            storage_start_token = max(0, start_token - expand_before)
            storage_end_token = min(num_tokens, end_token + expand_after)
            storage_tokens = all_tokens[storage_start_token:storage_end_token]
            storage_text = tokenizer.decode(storage_tokens)

        chunks.append((vector_text, storage_text))

        calculated_next = end_token - chunk_overlap
        if calculated_next <= start_token:
            next_start_token = start_token + max(1, (end_token - start_token))
        else:
            next_start_token = calculated_next

        start_token = next_start_token

    return chunks


# -----------------------------------------------------------------------------
# 3. Paragraph Chunking
# -----------------------------------------------------------------------------
def _chunk_by_paragraph(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Optional[Any],
    strict_size: bool
) -> List[Chunk]:
    paragraphs = _split_into_paragraphs(text)
    chunks: List[Chunk] = []

    current_chunk_paras: List[str] = []
    current_chunk_len = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_len = _get_length(para, tokenizer)

        if current_chunk_len == 0 and para_len > chunk_size:
            if strict_size:
                sentences = _split_into_sentences(para)
                current_sent_chunk: List[str] = []
                current_sent_len = 0
                for sent in sentences:
                    sent_len = _get_length(sent, tokenizer)
                    if current_sent_len + sent_len > chunk_size:
                        if current_sent_chunk:
                            txt = " ".join(current_sent_chunk)
                            chunks.append((txt, txt))
                        current_sent_chunk = [sent]
                        current_sent_len = sent_len
                    else:
                        current_sent_chunk.append(sent)
                        current_sent_len += sent_len
                if current_sent_chunk:
                    txt = " ".join(current_sent_chunk)
                    chunks.append((txt, txt))
                i += 1
                continue
            else:
                chunks.append((para, para))
                i += 1
                continue

        if current_chunk_len + para_len > chunk_size:
            if current_chunk_paras:
                txt = "\n\n".join(current_chunk_paras)
                chunks.append((txt, txt))

                overlap_len = 0
                overlap_paras: List[str] = []
                for p in reversed(current_chunk_paras):
                    p_len = _get_length(p, tokenizer)
                    if overlap_len + p_len <= chunk_overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += p_len
                    else:
                        break
                current_chunk_paras = overlap_paras
                current_chunk_len = overlap_len

            if current_chunk_len + para_len > chunk_size:
                current_chunk_paras = []
                current_chunk_len = 0
            else:
                current_chunk_paras.append(para)
                current_chunk_len += para_len
                i += 1
        else:
            current_chunk_paras.append(para)
            current_chunk_len += para_len
            i += 1

    if current_chunk_paras:
        txt = "\n\n".join(current_chunk_paras)
        chunks.append((txt, txt))

    return chunks


# -----------------------------------------------------------------------------
# 4. Recursive Chunking (Hierarchical Multi-Level Splitter with True Overlap)
# -----------------------------------------------------------------------------
def _chunk_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Optional[Any],
    separators: List[str]
) -> List[Chunk]:
    """
    Recursively splits text using a hierarchy of separators (paragraphs -> headers -> code -> sentences -> words)
    and merges small blocks up to chunk_size while maintaining chunk_overlap.
    """
    def _split_text(text_to_split: str, current_separators: List[str]) -> List[str]:
        if not text_to_split:
            return []

        # Find first matching separator
        separator = current_separators[-1]
        next_separators: List[str] = []
        for i, sep in enumerate(current_separators):
            if sep == "":
                separator = ""
                break
            if sep in text_to_split:
                separator = sep
                next_separators = current_separators[i + 1:]
                break

        splits = text_to_split.split(separator) if separator else list(text_to_split)

        good_splits = []
        current_block: List[str] = []
        current_len = 0

        for s in splits:
            s_len = _get_length(s, tokenizer)
            if s_len > chunk_size:
                if current_block:
                    good_splits.append(separator.join(current_block))
                    current_block = []
                    current_len = 0
                if next_separators:
                    good_splits.extend(_split_text(s, next_separators))
                else:
                    good_splits.append(s)
            else:
                join_cost = len(separator) if current_block else 0
                if current_len + s_len + join_cost > chunk_size:
                    if current_block:
                        good_splits.append(separator.join(current_block))
                    current_block = [s]
                    current_len = s_len
                else:
                    current_block.append(s)
                    current_len += s_len + join_cost

        if current_block:
            good_splits.append(separator.join(current_block))

        return [g for g in good_splits if g.strip()]

    raw_blocks = _split_text(text, separators)
    if not raw_blocks:
        return []

    # Merge raw blocks into final chunks respecting chunk_size and chunk_overlap
    chunks: List[Chunk] = []
    current_chunk_blocks: List[str] = []
    current_length = 0

    for block in raw_blocks:
        b_len = _get_length(block, tokenizer)
        if current_length + b_len > chunk_size and current_chunk_blocks:
            chunk_txt = "\n\n".join(current_chunk_blocks)
            chunks.append((chunk_txt, chunk_txt))

            # Maintain overlap by carrying over trailing blocks
            overlap_blocks: List[str] = []
            overlap_length = 0
            for prev_b in reversed(current_chunk_blocks):
                prev_len = _get_length(prev_b, tokenizer)
                if overlap_length + prev_len <= chunk_overlap:
                    overlap_blocks.insert(0, prev_b)
                    overlap_length += prev_len
                else:
                    break

            current_chunk_blocks = overlap_blocks
            current_length = overlap_length

        current_chunk_blocks.append(block)
        current_length += b_len

    if current_chunk_blocks:
        chunk_txt = "\n\n".join(current_chunk_blocks)
        chunks.append((chunk_txt, chunk_txt))

    return chunks


# -----------------------------------------------------------------------------
# 5. Structure-Aware & Markdown Chunking (Header Lineage & Breadcrumbs)
# -----------------------------------------------------------------------------
def _chunk_structure_aware(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Optional[Any] = None,
    expand_before: int = 0,
    expand_after: int = 0
) -> List[Chunk]:
    """
    Structure-aware chunking for Markdown, code, and structured technical manuals:
    - Parses header hierarchy (# H1, ## H2, ### H3)
    - Keeps code blocks (```...```) and tables (|...|) intact
    - Injects section breadcrumb paths [Section: H1 > H2 > H3] into vector & storage texts
    - Recursively splits large sections while preserving the breadcrumb header
    """
    lines = text.split('\n')
    sections = []
    current_lines = []
    current_headers = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    in_code_block = False

    def flush_section():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                # Build breadcrumb path
                active_headers = [
                    current_headers[lvl]
                    for lvl in range(1, 7)
                    if current_headers[lvl] is not None
                ]
                breadcrumb = " > ".join(active_headers)
                sections.append((breadcrumb, content))
            current_lines.clear()

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if not in_code_block and header_match:
            flush_section()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Clear deeper header levels
            for l in range(level, 7):
                current_headers[l] = None
            current_headers[level] = title
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush_section()

    chunks: List[Chunk] = []
    for breadcrumb, content in sections:
        header_prefix = f"[{breadcrumb}]\n\n" if breadcrumb else ""
        total_len = _get_length(header_prefix + content, tokenizer)

        if total_len <= chunk_size:
            chunk_txt = f"{header_prefix}{content}"
            chunks.append((chunk_txt, chunk_txt))
        else:
            # Recursively split oversized section, preserving breadcrumbs on all sub-chunks
            sub_chunks = _chunk_recursive(
                content,
                chunk_size=max(50, chunk_size - _get_length(header_prefix, tokenizer)),
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
                separators=DEFAULT_RECURSIVE_SEPARATORS
            )
            for sub_v, sub_s in sub_chunks:
                chunk_v = f"{header_prefix}{sub_v}"
                chunk_s = f"{header_prefix}{sub_s}"
                chunks.append((chunk_v, chunk_s))

    return chunks


# -----------------------------------------------------------------------------
# 6. Semantic Chunking (Cosine Similarity Valleys)
# -----------------------------------------------------------------------------
def _chunk_semantic(
    text: str,
    chunk_size: int,
    tokenizer: Optional[Any],
    vectorizer_fn: Callable[[List[str]], Any],
    similarity_threshold: float = 0.5,
    initial_blocks: int = 1,
    strict_size: bool = True,
    chunk_overlap: int = 0
) -> List[Chunk]:
    """
    Semantic chunking:
    - Splits text into individual sentences
    - Computes vector embeddings for adjacent sentences
    - Identifies semantic shift points where cosine similarity drops below threshold
    - Groups coherent sentences into topical chunks
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0], sentences[0])]

    try:
        embeddings = vectorizer_fn(sentences)
    except Exception as e:
        ASCIIColors.warning(f"Vectorization failed during semantic chunking: {e}. Falling back to paragraph chunking.")
        return _chunk_by_paragraph(text, chunk_size, chunk_overlap, tokenizer, strict_size)

    # Compute adjacent cosine similarities
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(sentences) - 1)
    ]

    # Group sentences into semantic clusters
    chunks: List[Chunk] = []
    current_sentences = [sentences[0]]
    current_len = _get_length(sentences[0], tokenizer)

    for i in range(len(similarities)):
        next_sent = sentences[i + 1]
        next_len = _get_length(next_sent, tokenizer)
        sim = similarities[i]

        is_topic_shift = sim < similarity_threshold
        is_overflow = strict_size and (current_len + next_len > chunk_size)

        if is_topic_shift or is_overflow:
            chunk_txt = " ".join(current_sentences)
            chunks.append((chunk_txt, chunk_txt))
            current_sentences = [next_sent]
            current_len = next_len
        else:
            current_sentences.append(next_sent)
            current_len += next_len

    if current_sentences:
        chunk_txt = " ".join(current_sentences)
        chunks.append((chunk_txt, chunk_txt))

    return chunks


# -----------------------------------------------------------------------------
# 7. Contextual Retrieval Chunking (Anthropic Situating Context Pattern)
# -----------------------------------------------------------------------------
def _chunk_contextual(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Optional[Any] = None,
    context_enricher: Optional[Callable[[str, str], str]] = None,
    full_document_text: Optional[str] = None,
    **kwargs
) -> List[Chunk]:
    """
    Contextual Retrieval:
    - Splits document using recursive or structure-aware strategy
    - Injects situating full-document context into each chunk's header prefix
    - Eliminates pronoun ambiguity and anchors isolated chunks to the broader narrative
    """
    doc_context = full_document_text or text
    base_chunks = _chunk_recursive(
        text,
        chunk_size=max(50, chunk_size - 40), # Reserve headroom for context prefix
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        separators=DEFAULT_RECURSIVE_SEPARATORS
    )

    enriched_chunks: List[Chunk] = []
    for v_text, s_text in base_chunks:
        if callable(context_enricher):
            try:
                prefix = context_enricher(doc_context, v_text)
                header = f"--- Context ---\n{prefix}\n---------------\n\n"
            except Exception as e:
                ASCIIColors.warning(f"context_enricher callback failed: {e}")
                header = ""
        else:
            # Heuristic context: First line / title of the document
            doc_title = doc_context.strip().split('\n')[0].strip('# ')[:100]
            header = f"[Document Context: {doc_title}]\n\n"

        enriched_v = f"{header}{v_text}"
        enriched_s = f"{header}{s_text}"
        enriched_chunks.append((enriched_v, enriched_s))

    return enriched_chunks


# -----------------------------------------------------------------------------
# 8. Late Chunking Boundaries (Span Preparation for Transformer Pooling)
# -----------------------------------------------------------------------------
def _chunk_late_boundaries(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Optional[Any] = None
) -> List[Chunk]:
    """
    Late Chunking Boundary Preparation:
    Splits text into discrete semantic spans to be mean-pooled from full-document
    transformer token embeddings at vectorization time.
    """
    return _chunk_recursive(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        separators=DEFAULT_RECURSIVE_SEPARATORS
    )