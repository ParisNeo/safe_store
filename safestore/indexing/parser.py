from pathlib import Path
from ascii_colors import ASCIIColors

def parse_txt(file_path: str | Path) -> str:
    """Parses a plain text file."""
    file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse TXT file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ASCIIColors.debug(f"Successfully parsed TXT file: {file_path}")
        return content
    except FileNotFoundError:
        ASCIIColors.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        ASCIIColors.error(f"Error parsing TXT file {file_path}: {e}", exc_info=True)
        raise

# In the future, add functions like parse_pdf, parse_docx, etc.
# and a dispatcher function based on file extension.
def parse_document(file_path: str | Path) -> str:
    """Parses a document based on its extension."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == '.txt':
        return parse_txt(file_path)
    # Add other parsers later
    # elif extension == '.pdf':
    #     return parse_pdf(file_path)
    else:
        ASCIIColors.warning(f"Unsupported file type '{extension}' for file: {file_path}. Skipping parsing.")
        raise ValueError(f"Unsupported file type: {extension}")