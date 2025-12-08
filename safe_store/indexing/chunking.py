# safe_store/indexing/chunking.py
import re
from typing import List, Tuple, Callable, Optional, Literal, Any, Union
import numpy as np
from ascii_colors import ASCIIColors

Chunk = Tuple[str, str] # (text_for_vectorization, text_for_storage)

def generate_chunks(
    text: str,
    strategy: Literal['character', 'token', 'paragraph', 'semantic'],
    chunk_size: int,
    chunk_overlap: int,
    expand_before: int = 0,
    expand_after: int = 0,
    tokenizer: Optional[Any] = None,
    # New parameters for semantic/paragraph chunking
    vectorizer_fn: Optional[Callable[[List[str]], Any]] = None,
    similarity_threshold: float = 0.5,
    initial_semantic_blocks: int = 3,
    strict_size: bool = True
) -> List[Chunk]:
    if strategy == 'token':
        if tokenizer is None:
            raise ValueError("A tokenizer is required for 'token' strategy.")
        return _chunk_by_tokens(text, tokenizer, chunk_size, chunk_overlap, expand_before, expand_after)
    elif strategy == 'character':
        return _chunk_by_character(text, chunk_size, chunk_overlap, expand_before, expand_after)
    elif strategy == 'paragraph':
        return _chunk_by_paragraph(text, chunk_size, chunk_overlap, tokenizer, strict_size)
    elif strategy == 'semantic':
        if vectorizer_fn is None:
             raise ValueError("vectorizer_fn is required for 'semantic' strategy")
        return _chunk_semantic(text, chunk_size, tokenizer, vectorizer_fn, similarity_threshold, initial_semantic_blocks, strict_size)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'")

def _get_length(text: str, tokenizer: Optional[Any]) -> int:
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text)

def _split_into_sentences(text: str) -> List[str]:
    # Split on punctuation followed by space or newline, keeping punctuation.
    return re.split(r'(?<=[.!?])\s+', text)

def _split_into_paragraphs(text: str) -> List[str]:
    # Split by double newlines, strip whitespace
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]

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
        
        next_start_pos = start_pos + chunk_size - chunk_overlap
        if next_start_pos <= start_pos:
            break
        start_pos = next_start_pos
        
    return chunks

def _chunk_by_tokens(text: str, tokenizer: Any, chunk_size: int, chunk_overlap: int, expand_before: int, expand_after: int) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("token overlap must be smaller than token chunk_size.")

    all_tokens = tokenizer.encode(text)
    num_tokens = len(all_tokens)
    chunks: List[Chunk] = []
    start_token = 0

    while start_token < num_tokens:
        end_token = min(start_token + chunk_size, num_tokens)
        
        vector_tokens = all_tokens[start_token:end_token]
        vector_text = tokenizer.decode(vector_tokens)
        
        storage_start_token = max(0, start_token - expand_before)
        storage_end_token = min(num_tokens, end_token + expand_after)
        storage_tokens = all_tokens[storage_start_token:storage_end_token]
        storage_text = tokenizer.decode(storage_tokens)
        
        chunks.append((vector_text, storage_text))
        
        next_start_token = start_token + chunk_size - chunk_overlap
        if next_start_token <= start_token:
            break
        start_token = next_start_token
        
    return chunks

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
        
        # 1. Handle single paragraph larger than chunk_size
        if current_chunk_len == 0 and para_len > chunk_size:
            if strict_size:
                # Split huge paragraph into sentences
                sentences = _split_into_sentences(para)
                current_sent_chunk = []
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
                # Ignore size limit, allow overflow
                chunks.append((para, para))
                i += 1
                continue

        # 2. Accumulate paragraphs
        if current_chunk_len + para_len > chunk_size:
            # Emit current chunk
            if current_chunk_paras:
                txt = "\n\n".join(current_chunk_paras)
                chunks.append((txt, txt))
                
                # Handle Overlap: keep last paragraphs that fit in chunk_overlap
                overlap_len = 0
                overlap_paras = []
                for p in reversed(current_chunk_paras):
                    p_len = _get_length(p, tokenizer)
                    if overlap_len + p_len <= chunk_overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += p_len
                    else:
                        break
                current_chunk_paras = overlap_paras
                current_chunk_len = overlap_len
            
            # If buffer is empty (overlap was 0 or cleared), loop handles para as new chunk
            # or potentially as a large para in next iteration
            if current_chunk_len + para_len > chunk_size and current_chunk_len == 0:
                continue # Back to top to handle 'single large para' logic
            elif current_chunk_len + para_len > chunk_size:
                 # Even with overlap, it's too big. Clear overlap to start fresh.
                 current_chunk_paras = []
                 current_chunk_len = 0
                 continue
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

def _cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def _chunk_semantic(
    text: str,
    chunk_size: int,
    tokenizer: Optional[Any],
    vectorizer_fn: Callable[[List[str]], Any],
    similarity_threshold: float,
    initial_blocks: int,
    strict_size: bool
) -> List[Chunk]:
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    
    i = 0
    while i < len(paragraphs):
        # Initialize chunk with minimum blocks
        current_paras = paragraphs[i : i + initial_blocks]
        i += len(current_paras)
        
        # Check semantic similarity for subsequent blocks
        while i < len(paragraphs):
            current_text = "\n\n".join(current_paras)
            next_para = paragraphs[i]
            
            # Vectorize to check similarity
            vecs = vectorizer_fn([current_text, next_para])
            sim = _cosine_similarity(vecs[0], vecs[1])
            
            if sim >= similarity_threshold:
                # Check size constraint
                next_len = _get_length(next_para, tokenizer)
                current_len = _get_length(current_text, tokenizer)
                
                if strict_size and (current_len + next_len > chunk_size):
                    # Stop if it would exceed size
                    break
                else:
                    current_paras.append(next_para)
                    i += 1
            else:
                # Semantic drift, stop chunk
                break
        
        txt = "\n\n".join(current_paras)
        chunks.append((txt, txt))
        # Note: No overlap logic implemented for semantic chunking to ensure clean semantic partitions
        
    return chunks
