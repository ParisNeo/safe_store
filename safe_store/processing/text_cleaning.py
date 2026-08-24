# safe_store/processing/text_cleaning.py
import re
from typing import Callable, Union

def basic_text_cleaner(text: str, remove_line_returns: bool = False) -> str:
    """
    An enhanced text cleaner designed to be safe for code, markdown, and structured text.

    - Normalizes all line endings to `\n`.
    - Preserves all line returns by default (deactivated removal).
    - Removes non-printable ASCII control characters (except tab and newline).
    - Preserves leading whitespace and indentation on each line.
    - Replaces repetitive dot sequences with a standard ellipsis (`...`).
    - Collapses multiple spaces within a line without altering line breaks.
    - Optionally flattens line returns into spaces when `remove_line_returns=True`.

    Args:
        text: The input string to clean.
        remove_line_returns: If True, replaces newlines with single spaces. Defaults to False.

    Returns:
        The cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalize line endings to \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2. Remove non-printable control characters except for tab and newline
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Replace long sequences of dots with standard ellipsis
    text = re.sub(r'\.{4,}', '...', text)

    # 4. Optional line return removal (deactivated by default)
    if remove_line_returns:
        text = re.sub(r'\s*\n\s*', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    # 5. Process line-by-line to preserve indentation and line returns
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        match = re.match(r'^(\s*)', line)
        leading_whitespace = match.group(1) if match else ""
        content = line[len(leading_whitespace):]
        cleaned_content = re.sub(r' {2,}', ' ', content)
        cleaned_lines.append(leading_whitespace + cleaned_content)

    text = '\n'.join(cleaned_lines)

    # 6. Reduce excessive blank lines (3+ down to maximum of 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def get_cleaner(
    cleaner: Union[str, Callable[[str], str], None],
    remove_line_returns: bool = False
) -> Callable[[str], str]:
    """
    Returns a callable cleaner function configured with line return preferences.

    Args:
        cleaner: Name of cleaner ('basic'), custom callable, or None.
        remove_line_returns: Whether line returns should be flattened. Defaults to False.

    Returns:
        A callable function (str -> str).
    """
    if cleaner is None:
        if remove_line_returns:
            return lambda x: re.sub(r'\s*\n\s*', ' ', str(x)).strip() if isinstance(x, str) else ""
        return lambda x: x
    if callable(cleaner):
        return cleaner
    if isinstance(cleaner, str):
        if cleaner == 'basic':
            return lambda x: basic_text_cleaner(x, remove_line_returns=remove_line_returns)
        else:
            raise ValueError(f"Unknown predefined cleaner: '{cleaner}'")
    raise TypeError("cleaner must be a string, a callable, or None")