# safe_store/processing/text_cleaning.py
import re
from typing import Callable, Union

def basic_text_cleaner(text: str) -> str:
    """
    An enhanced text cleaner that performs several common cleanup tasks, designed
    to be safe for code and structured text while improving quality for LLMs.

    - Normalizes all line endings to a single newline character (`\n`).
    - Removes non-printable ASCII control characters (except tab and newline) that
      can break LLM tokenizers.
    - Preserves leading whitespace (indentation) on each line, which is crucial for code.
    - Replaces repetitive dot sequences (e.g., '....') with a standard ellipsis ('...').
    - Collapses multiple spaces *within* a line into a single space, but leaves indentation untouched.
    - Reduces three or more consecutive newlines down to just two, preserving paragraph
      breaks without creating excessive empty space. Single newlines are kept.

    Args:
        text: The input string to clean.

    Returns:
        The cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalize line endings to \n.
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Remove non-printable control characters except for tab, newline.
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Replace long sequences of dots with a standard ellipsis.
    text = re.sub(r'\.{4,}', '...', text)
    
    # 4. Process line by line to preserve indentation while cleaning inline spaces.
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Separate leading whitespace (indentation) from the rest of the content
        match = re.match(r'^(\s*)', line)
        leading_whitespace = match.group(1) if match else ""
        content = line[len(leading_whitespace):]
        
        # Collapse multiple spaces in the content part only
        cleaned_content = re.sub(r' {2,}', ' ', content)
        
        cleaned_lines.append(leading_whitespace + cleaned_content)
    
    text = '\n'.join(cleaned_lines)

    # 5. Reduce 3 or more newlines to a maximum of two.
    text = re.sub(r'\n{3,}', '\n\n', text)
    
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