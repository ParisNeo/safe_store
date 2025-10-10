# safe_store/processing/text_cleaning.py
import re
from typing import Callable, Union

def basic_text_cleaner(text: str) -> str:
    """
    A basic text cleaner that performs several common cleanup tasks.

    - Replaces multiple spaces, newlines, and tabs with a single space.
    - Removes standalone newlines and tabs that are surrounded by whitespace.
    - Consolidates multiple newlines into a double newline (paragraph break).

    Args:
        text: The input string to clean.

    Returns:
        The cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Consolidate multiple newlines into a standard paragraph break
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove single newlines that are likely just line breaks within a paragraph
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Consolidate multiple spaces into a single space
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def get_cleaner(cleaner: Union[str, Callable[[str], str], None]) -> Callable[[str], str]:
    """
    Returns a callable cleaner function.

    Args:
        cleaner: Can be the name of a predefined cleaner ('basic') or a custom
                 callable function. If None, returns an identity function that
                 does nothing.

    Returns:
        A callable function that takes a string and returns a string.
    """
    if cleaner is None:
        return lambda x: x # Identity function
    if callable(cleaner):
        return cleaner
    if isinstance(cleaner, str):
        if cleaner == 'basic':
            return basic_text_cleaner
        else:
            raise ValueError(f"Unknown predefined cleaner: '{cleaner}'")
    raise TypeError("cleaner must be a string, a callable, or None")