# safe_store/indexing/chunking.py
from typing import List, Tuple
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