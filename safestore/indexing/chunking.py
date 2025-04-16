from typing import List, Tuple
from ascii_colors import ASCIIColors

def chunk_text(
    text: str,
    chunk_size: int = 1000, # Characters
    chunk_overlap: int = 150 # Characters
) -> List[Tuple[str, int, int]]:
    """
    Splits text into overlapping chunks with start/end positions.

    Args:
        text: The input text.
        chunk_size: The target size of each chunk in characters.
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of tuples, where each tuple contains:
        (chunk_text, start_character_offset, end_character_offset).
    """
    if not isinstance(text, str):
         # Adding type check for safety
         raise TypeError("Input 'text' must be a string.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start_index = 0
    text_len = len(text)

    # ASCIIColors.debug(f"Starting chunking: text_len={text_len}, chunk_size={chunk_size}, overlap={chunk_overlap}") # Keep logging if desired

    if text_len == 0:
        return [] # Handle empty string input

    while start_index < text_len:
        end_index = min(start_index + chunk_size, text_len)
        chunk = text[start_index:end_index]
        chunks.append((chunk, start_index, end_index))

        # ****** ADDED CHECK ******
        # If this chunk reached the absolute end of the text, we are done.
        if end_index == text_len:
            break
        # *************************

        # Calculate the start of the next chunk
        next_start_index = start_index + chunk_size - chunk_overlap

        # Prevent infinite loops if overlap is too large or chunk_size is too small
        # relative to overlap, causing next_start_index to not advance.
        # Should ideally not happen if overlap < chunk_size, but good safety check.
        if next_start_index <= start_index:
             # Force advancement if stuck, ensuring loop terminates eventually
             # This case should only be hit if overlap is very close to chunk_size
             # or if the remaining text is smaller than the overlap step.
             next_start_index = start_index + 1


        start_index = next_start_index
        # The main loop condition (start_index < text_len) will eventually catch
        # the case where next_start_index >= text_len after the update.


    num_chunks = len(chunks)
    ASCIIColors.debug(f"Chunking complete. Generated {num_chunks} chunks.") # Keep logging if desired
    # Optional: Add back preview logging if helpful
    if num_chunks > 0:
        ASCIIColors.debug(f"First chunk preview (len {len(chunks[0][0])}): '{chunks[0][0][:50]}...'")
        ASCIIColors.debug(f"Last chunk preview (len {len(chunks[-1][0])}): '...{chunks[-1][0][-50:]}'")

    return chunks