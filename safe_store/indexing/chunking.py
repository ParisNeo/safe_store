# safe_store/indexing/chunking.py
import collections.abc
from typing import List, Tuple, Callable, TypeVar
from ascii_colors import ASCIIColors

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Tuple[str, int, int]]:
    """
    Splits a given text into overlapping chunks based on character count.

    Each chunk includes its start and end character position relative to the
    original text.

    Args:
        text: The input text string to be chunked.
        chunk_size: The target maximum size of each chunk in characters.
                    Defaults to 1000.
        chunk_overlap: The number of characters to overlap between consecutive
                       chunks. Defaults to 150. Must be less than `chunk_size`.

    Returns:
        A list of tuples, where each tuple contains:
        (chunk_text: str, start_char_offset: int, end_char_offset: int).
        Returns an empty list if the input text is empty.

    Raises:
        TypeError: If the input `text` is not a string.
        ValueError: If `chunk_overlap` is greater than or equal to `chunk_size`.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: List[Tuple[str, int, int]] = []
    start_index: int = 0
    text_len: int = len(text)

    ASCIIColors.debug(f"Starting chunking: text_len={text_len}, chunk_size={chunk_size}, overlap={chunk_overlap}")

    if text_len == 0:
        ASCIIColors.debug("Input text is empty, returning no chunks.")
        return []

    while start_index < text_len:
        # Determine the end index for the current chunk
        end_index = min(start_index + chunk_size, text_len)
        chunk = text[start_index:end_index]
        chunks.append((chunk, start_index, end_index))

        # If this chunk reached the end of the text, we are done
        if end_index == text_len:
            break

        # Calculate the start of the next chunk, ensuring progress
        next_start_index = start_index + chunk_size - chunk_overlap

        # Safety check: Ensure next_start_index advances.
        # This should ideally not happen if overlap < chunk_size, but prevents infinite loops.
        if next_start_index <= start_index:
             # Force advancement by at least one character if stuck
             ASCIIColors.warning(f"Chunking calculation resulted in non-progressing start index ({start_index} -> {next_start_index}). Forcing advance by 1.")
             next_start_index = start_index + 1

        start_index = next_start_index
        # The loop condition `start_index < text_len` handles termination naturally

    num_chunks = len(chunks)
    ASCIIColors.debug(f"Chunking complete. Generated {num_chunks} chunks.")
    if num_chunks > 0:
        ASCIIColors.debug(f"First chunk preview (len {len(chunks[0][0])}, pos {chunks[0][1]}-{chunks[0][2]}): '{chunks[0][0][:50]}...'")
        if num_chunks > 1:
             ASCIIColors.debug(f"Last chunk preview (len {len(chunks[-1][0])}, pos {chunks[-1][1]}-{chunks[-1][2]}): '...{chunks[-1][0][-50:]}'")
        else: # Only one chunk
             ASCIIColors.debug(f"Last chunk preview (len {len(chunks[-1][0])}, pos {chunks[-1][1]}-{chunks[-1][2]}): '{chunks[-1][0][-50:]}'")


    return chunks

TokenT = TypeVar('TokenT')  # Represents the type of a single token (e.g., int for ID, str for string token)

def chunk_text_by_tokens(
    text: str,
    tokenize: Callable[[str], List[TokenT]],
    detokenize: Callable[[List[TokenT]], str],
    chunk_size_tokens: int = 300,       # Common default for LLM context windows
    chunk_overlap_tokens: int = 50      # Common default for overlap
) -> List[Tuple[str, int, int]]:
    """
    Splits a given text into overlapping chunks based on token count.

    Each chunk's text is reconstructed using `detokenize` from its tokens.
    The start and end character positions are relative to the text that would
    be formed by detokenizing all tokens from the beginning of the document
    up to the start/end of the current chunk's tokens. If `detokenize(tokenize(text))`
    is identical to `text`, these offsets are effectively relative to the original text.

    Args:
        text: The input text string to be chunked.
        tokenize: A callable that takes a string and returns a list of tokens
                  (e.g., List[int] for token IDs, or List[str] for token strings).
        detokenize: A callable that takes a list of tokens (of the same type
                    returned by `tokenize`) and returns a string.
        chunk_size_tokens: The target maximum size of each chunk in tokens.
                           Defaults to 300.
        chunk_overlap_tokens: The number of tokens to overlap between consecutive
                              chunks. Defaults to 50. Must be less than
                              `chunk_size_tokens`.

    Returns:
        A list of tuples, where each tuple contains:
        (chunk_text: str, start_char_offset: int, end_char_offset: int).
        Returns an empty list if the input text tokenizes to an empty list of tokens.

    Raises:
        TypeError: If `text` is not a string, or `tokenize`/`detokenize`
                   are not callable.
        ValueError: If `chunk_overlap_tokens` is negative, `chunk_size_tokens`
                    is not positive, or `chunk_overlap_tokens` is greater
                    than or equal to `chunk_size_tokens`.
    """
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not isinstance(tokenize, collections.abc.Callable):
        raise TypeError("Input 'tokenize' must be a callable function.")
    if not isinstance(detokenize, collections.abc.Callable):
        raise TypeError("Input 'detokenize' must be a callable function.")

    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative.")
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")

    ASCIIColors.debug(
        f"Starting token chunking: text_len_chars={len(text)}, "
        f"chunk_size_tokens={chunk_size_tokens}, overlap_tokens={chunk_overlap_tokens}"
    )

    all_tokens: List[TokenT] = tokenize(text)
    num_all_tokens: int = len(all_tokens)

    if num_all_tokens == 0:
        ASCIIColors.debug("Input text tokenized to an empty list, returning no chunks.")
        return []
    
    ASCIIColors.debug(f"Total tokens generated from text: {num_all_tokens}")

    # Store chunks with their original token counts for more accurate debug messages
    # List[Tuple[str, int, int, int]] where the last int is num_tokens_in_chunk
    processed_chunks_with_meta: List[Tuple[str, int, int, int]] = []
    start_token_idx: int = 0

    while start_token_idx < num_all_tokens:
        # Determine the end token index for the current chunk
        end_token_idx = min(start_token_idx + chunk_size_tokens, num_all_tokens)
        
        current_token_chunk: List[TokenT] = all_tokens[start_token_idx:end_token_idx]
        num_tokens_in_current_chunk = len(current_token_chunk)
        chunk_sub_text: str = detokenize(current_token_chunk)

        # Calculate character offsets. These are relative to the text reconstructed
        # by detokenizing tokens from the beginning.
        
        # Text formed by tokens *before* the current chunk
        tokens_before_chunk: List[TokenT] = all_tokens[:start_token_idx]
        start_char_offset: int = len(detokenize(tokens_before_chunk))

        # Text formed by tokens *up to and including* the current chunk
        tokens_up_to_chunk_end: List[TokenT] = all_tokens[:end_token_idx]
        end_char_offset: int = len(detokenize(tokens_up_to_chunk_end))
        
        processed_chunks_with_meta.append(
            (chunk_sub_text, start_char_offset, end_char_offset, num_tokens_in_current_chunk)
        )

        # If this chunk's tokens reached the end of all tokens, we are done
        if end_token_idx == num_all_tokens:
            break

        # Calculate the start of the next chunk's tokens
        next_start_token_idx = start_token_idx + chunk_size_tokens - chunk_overlap_tokens
        
        # Safety check: Ensure next_start_token_idx advances.
        # This should not happen with valid inputs (chunk_overlap_tokens < chunk_size_tokens
        # and chunk_size_tokens > 0 implies (chunk_size_tokens - chunk_overlap_tokens) >= 1).
        if next_start_token_idx <= start_token_idx:
             ASCIIColors.warning(
                 f"Token chunking calculation resulted in non-progressing start token index "
                 f"({start_token_idx} -> {next_start_token_idx}). Forcing advance by 1 token."
             )
             next_start_token_idx = start_token_idx + 1 # Force progress
        
        start_token_idx = next_start_token_idx
        # The loop condition `start_token_idx < num_all_tokens` handles termination.

    # Prepare final result by stripping the temporary token count used for debugging
    final_chunks: List[Tuple[str, int, int]] = [
        (data[0], data[1], data[2]) for data in processed_chunks_with_meta
    ]

    num_gen_chunks = len(final_chunks)
    ASCIIColors.debug(f"Token chunking complete. Generated {num_gen_chunks} chunks.")
    if num_gen_chunks > 0:
        first_chunk_text, first_s, first_e, first_n_tok = processed_chunks_with_meta[0]
        ASCIIColors.debug(
            f"First chunk preview (actual_tokens {first_n_tok}, len {len(first_chunk_text)} chars, "
            f"char_pos {first_s}-{first_e}): '{first_chunk_text[:50]}...'"
        )
        if num_gen_chunks > 1:
            last_chunk_text, last_s, last_e, last_n_tok = processed_chunks_with_meta[-1]
            ASCIIColors.debug(
                f"Last chunk preview (actual_tokens {last_n_tok}, len {len(last_chunk_text)} chars, "
                f"char_pos {last_s}-{last_e}): '...{last_chunk_text[-50:]}'"
            )
        else: # Only one chunk
            # Use first_chunk_text variables as it's the only one
            ASCIIColors.debug(
                f"Single chunk preview (actual_tokens {first_n_tok}, len {len(first_chunk_text)} chars, "
                f"char_pos {first_s}-{first_e}): '{first_chunk_text[-50:]}'" 
            )
    return final_chunks