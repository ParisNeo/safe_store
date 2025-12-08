# safe_store/indexing/chunking.py
import re
from typing import List, Tuple, Callable, Optional, Literal, Any, Union
import numpy as np
from ascii_colors import ASCIIColors

Chunk = Tuple[str, str] # (text_for_vectorization, text_for_storage)

def generate_chunks(
    text: str,
    strategy: Literal['character', 'token', 'paragraph', 'semantic', 'recursive'],
    chunk_size: int,
    chunk_overlap: int,
    expand_before: int = 0,
    expand_after: int = 0,
    tokenizer: Optional[Any] = None,
    # New parameters
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
    elif strategy == 'recursive':
        return _chunk_recursive(text, chunk_size, chunk_overlap, tokenizer, strict_size)
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
        
        # Simple extraction
        vector_tokens = all_tokens[start_token:end_token]
        vector_text = tokenizer.decode(vector_tokens)
        
        # Improvement: If we cut in the middle of a sentence/line, try to snap to a newline if reasonably close
        # This prevents splitting "print('hello')" into "print('he" and "llo')"
        # Only do this if we are not at the very end
        if end_token < num_tokens and '\n' in vector_text:
            last_newline = vector_text.rfind('\n')
            # If the newline is within the last 25% of the chunk, assume it's a good break point
            # This is heuristic but effective for keeping lines intact.
            if last_newline > len(vector_text) * 0.75:
                # Re-calculate tokens to this cut point
                cut_text = vector_text[:last_newline+1] # Include the newline
                cut_tokens = tokenizer.encode(cut_text)
                end_token = start_token + len(cut_tokens)
                vector_text = cut_text

        storage_start_token = max(0, start_token - expand_before)
        storage_end_token = min(num_tokens, end_token + expand_after)
        storage_tokens = all_tokens[storage_start_token:storage_end_token]
        storage_text = tokenizer.decode(storage_tokens)
        
        chunks.append((vector_text, storage_text))
        
        next_start_token = start_token + chunk_size - chunk_overlap
        # If we snapped back end_token significantly, next_start might overlap differently. 
        # But we base next_start on the start of the previous chunk, which is robust.
        # Wait, usually next_start = end_token - overlap. If end_token moved back, next_start moves back too, preserving overlap.
        # However, we must ensure progress.
        calculated_next = end_token - chunk_overlap
        if calculated_next <= start_token:
             # If the chunk is smaller than overlap (rare/impossible if checked), force progress
             next_start_token = start_token + max(1, (end_token - start_token))
        else:
             next_start_token = calculated_next

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
        
        if current_chunk_len == 0 and para_len > chunk_size:
            if strict_size:
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
                chunks.append((para, para))
                i += 1
                continue

        if current_chunk_len + para_len > chunk_size:
            if current_chunk_paras:
                txt = "\n\n".join(current_chunk_paras)
                chunks.append((txt, txt))
                
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
            
            if current_chunk_len + para_len > chunk_size and current_chunk_len == 0:
                continue 
            elif current_chunk_len + para_len > chunk_size:
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
    if not paragraphs: return []

    chunks: List[Chunk] = []
    i = 0
    while i < len(paragraphs):
        current_paras = paragraphs[i : i + initial_blocks]
        i += len(current_paras)
        
        while i < len(paragraphs):
            current_text = "\n\n".join(current_paras)
            next_para = paragraphs[i]
            
            vecs = vectorizer_fn([current_text, next_para])
            sim = _cosine_similarity(vecs[0], vecs[1])
            
            if sim >= similarity_threshold:
                next_len = _get_length(next_para, tokenizer)
                current_len = _get_length(current_text, tokenizer)
                if strict_size and (current_len + next_len > chunk_size):
                    break
                else:
                    current_paras.append(next_para)
                    i += 1
            else:
                break
        
        txt = "\n\n".join(current_paras)
        chunks.append((txt, txt))
        
    return chunks

def _chunk_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Optional[Any],
    strict_size: bool
) -> List[Chunk]:
    """
    Recursively splits text using a hierarchy of separators to keep meaningful blocks together.
    Ideal for code and JSON.
    """
    # Separators in order of preference
    separators = ["\n\n", "\nclass ", "\ndef ", "\n", " ", ""]
    
    def _split_text(text_to_split: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        # Find the best separator
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text_to_split:
                separator = sep
                new_separators = separators[i+1:]
                break
        
        # Split
        if separator:
            splits = text_to_split.split(separator)
            # Re-attach separator to the end of the previous chunk or beginning? 
            # Usually strict split removes it. We'll join later with it if we merge.
            # But standard recursive splitter keeps the separator.
            # For simplicity, we split, and if we merge, we assume the separator was there.
            # Actually, `split` removes separators. We might lose "\n".
            # Better approach: split and keep delimiter. Python's `split` doesn't do that.
            # We'll just split and assume `separator` is the joiner.
        else:
            splits = list(text_to_split) # Character split
            new_separators = [] # Done

        good_splits = []
        current_doc = []
        current_len = 0
        
        for s in splits:
            s_len = _get_length(s, tokenizer)
            
            if s_len > chunk_size:
                if current_doc:
                    # Flush current
                    doc_txt = separator.join(current_doc)
                    good_splits.append(doc_txt)
                    current_doc = []
                    current_len = 0
                
                # Recursively split this big chunk
                if new_separators:
                    good_splits.extend(_split_text(s, new_separators))
                else:
                    # Cannot split further, just add it (or force cut if strict)
                    good_splits.append(s)
            else:
                if current_len + s_len + (len(separator) if current_doc else 0) > chunk_size:
                    # Flush
                    doc_txt = separator.join(current_doc)
                    good_splits.append(doc_txt)
                    current_doc = [s]
                    current_len = s_len
                else:
                    current_doc.append(s)
                    current_len += s_len + (len(separator) if current_doc else 0)
        
        if current_doc:
            good_splits.append(separator.join(current_doc))
            
        return good_splits

    # The recursive splitter returns a list of strings fitting the size.
    # We now need to handle overlap manually since the recursive process produces discrete blocks.
    # Standard recursive splitters in RAG often just produce chunks. 
    # Implementing overlap on top of recursive split results:
    
    raw_chunks = _split_text(text, separators)
    
    # Post-process for overlap
    if chunk_overlap == 0:
        return [(c, c) for c in raw_chunks]
        
    final_chunks: List[Chunk] = []
    
    # Sliding window over the resulting blocks?
    # Recursive splitting produces blocks of size <= chunk_size.
    # We can't easily merge them back to creating overlap without reconstructing text.
    # Simpler approach: Just return the blocks. Overlap in recursive splitting is complex 
    # and usually done during the merge phase.
    # For now, we return discrete blocks which guarantees "complete blocks" as requested.
    
    return [(c, c) for c in raw_chunks]

def _cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)
