# safe_store/indexing/chunking.py
from typing import List, Tuple, Callable, Optional, Literal, Any
from ascii_colors import ASCIIColors

Chunk = Tuple[str, str] # (text_for_vectorization, text_for_storage)

def generate_chunks(
    text: str,
    strategy: Literal['character', 'token'],
    chunk_size: int,
    chunk_overlap: int,
    expand_before: int = 0,
    expand_after: int = 0,
    tokenizer: Optional[Any] = None # Expects a TokenizerWrapper now
) -> List[Chunk]:
    if strategy == 'token':
        if tokenizer is None:
            raise ValueError("A tokenizer is required for 'token' strategy.")
        return _chunk_by_tokens(text, tokenizer, chunk_size, chunk_overlap, expand_before, expand_after)
    elif strategy == 'character':
        return _chunk_by_character(text, chunk_size, chunk_overlap, expand_before, expand_after)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'")

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
        # --- FIX: Call decode without extra arguments ---
        vector_text = tokenizer.decode(vector_tokens)
        
        storage_start_token = max(0, start_token - expand_before)
        storage_end_token = min(num_tokens, end_token + expand_after)
        storage_tokens = all_tokens[storage_start_token:storage_end_token]
        # --- FIX: Call decode without extra arguments ---
        storage_text = tokenizer.decode(storage_tokens)
        
        chunks.append((vector_text, storage_text))
        
        next_start_token = start_token + chunk_size - chunk_overlap
        if next_start_token <= start_token:
            break
        start_token = next_start_token
        
    return chunks