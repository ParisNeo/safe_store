
### `safestore\indexing\parser.py`
from pathlib import Path
from ascii_colors import ASCIIColors
from typing import Protocol, runtime_checkable # For optional typing

# --- Define protocols for optional dependencies ---
# This helps with static analysis even if libs aren't installed

@runtime_checkable
class PdfReaderProtocol(Protocol):
    pages: list # Simplified representation

    def __init__(self, stream: bytes | Path | str, strict: bool = True): ...

@runtime_checkable
class PageObjectProtocol(Protocol):
    def extract_text(self) -> str: ...

@runtime_checkable
class DocumentProtocol(Protocol):
    paragraphs: list

    def __init__(self, file_path: str | Path | None = None): ...

@runtime_checkable
class ParagraphProtocol(Protocol):
    text: str

@runtime_checkable
class BeautifulSoupProtocol(Protocol):
    def __init__(self, markup: str | bytes, features: str, **kwargs): ...
    def get_text(self, separator: str = "", strip: bool = False) -> str: ...


# --- Parsing Functions ---

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


def parse_pdf(file_path: str | Path) -> str:
    """Parses a PDF file to extract text."""
    file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse PDF file: {file_path}")
    try:
        # Lazy import pypdf
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
        # Check protocol conformity if needed (optional)
        # assert isinstance(PdfReader, PdfReaderProtocol)
    except ImportError:
        ASCIIColors.error("Parsing PDF files requires 'pypdf'. Install with: pip install safestore[parsing]")
        raise ImportError("pypdf not installed.")

    full_text = ""
    try:
        reader = PdfReader(file_path) # type: ignore # Ignore type check due to Protocol mismatch sometimes
        ASCIIColors.debug(f"PDF '{file_path.name}' loaded with {len(reader.pages)} pages.")
        for i, page in enumerate(reader.pages):
            # assert isinstance(page, PageObjectProtocol) # Optional protocol check
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n" # Add newline between pages
                else:
                     ASCIIColors.debug(f"No text extracted from page {i+1} of '{file_path.name}'.")
            except Exception as page_err:
                 ASCIIColors.warning(f"Could not extract text from page {i+1} of '{file_path.name}': {page_err}")
        ASCIIColors.debug(f"Successfully parsed PDF file: {file_path}")
        return full_text.strip()
    except FileNotFoundError:
        ASCIIColors.error(f"File not found: {file_path}")
        raise
    except PdfReadError as e:
        ASCIIColors.error(f"Error reading PDF file {file_path}: {e}")
        raise ValueError(f"Invalid or corrupted PDF file: {file_path}") from e
    except Exception as e:
        ASCIIColors.error(f"Error parsing PDF file {file_path}: {e}", exc_info=True)
        raise


def parse_docx(file_path: str | Path) -> str:
    """Parses a DOCX file to extract text."""
    file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse DOCX file: {file_path}")
    try:
        # Lazy import python-docx
        from docx import Document
        # assert isinstance(Document, DocumentProtocol) # Optional protocol check
    except ImportError:
        ASCIIColors.error("Parsing DOCX files requires 'python-docx'. Install with: pip install safestore[parsing]")
        raise ImportError("python-docx not installed.")

    full_text = ""
    try:
        document = Document(file_path) # type: ignore # Ignore type check due to Protocol mismatch sometimes
        ASCIIColors.debug(f"DOCX '{file_path.name}' loaded.")
        for para in document.paragraphs:
            # assert isinstance(para, ParagraphProtocol) # Optional protocol check
            full_text += para.text + "\n"
        ASCIIColors.debug(f"Successfully parsed DOCX file: {file_path}")
        return full_text.strip()
    except FileNotFoundError:
        ASCIIColors.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        # Catch potential errors from python-docx (e.g., corrupted file)
        ASCIIColors.error(f"Error parsing DOCX file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Could not parse DOCX file: {file_path}") from e


def parse_html(file_path: str | Path) -> str:
    """Parses an HTML file to extract text content."""
    file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse HTML file: {file_path}")
    try:
        # Lazy import BeautifulSoup and check for lxml
        from bs4 import BeautifulSoup
        # assert isinstance(BeautifulSoup, BeautifulSoupProtocol) # Optional protocol check
        try:
            import lxml
            HTML_PARSER = 'lxml'
        except ImportError:
            ASCIIColors.debug("lxml not found, using Python's built-in 'html.parser' for HTML.")
            HTML_PARSER = 'html.parser'
    except ImportError:
        ASCIIColors.error("Parsing HTML files requires 'BeautifulSoup4'. Install with: pip install safestore[parsing]")
        raise ImportError("BeautifulSoup4 not installed.")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        soup = BeautifulSoup(content, HTML_PARSER)
        # Extract text, separating paragraphs with newlines might be better
        # Use .get_text() with a separator
        text = soup.get_text(separator='\n', strip=True)
        ASCIIColors.debug(f"Successfully parsed HTML file: {file_path}")
        return text
    except FileNotFoundError:
        ASCIIColors.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        ASCIIColors.error(f"Error parsing HTML file {file_path}: {e}", exc_info=True)
        raise


# --- Dispatcher Function ---

def parse_document(file_path: str | Path) -> str:
    """Parses a document based on its extension, dispatching to specific parsers."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    ASCIIColors.debug(f"Dispatching parser for extension '{extension}' on file: {file_path.name}")

    parser_map = {
        '.txt': parse_txt,
        '.pdf': parse_pdf,
        '.docx': parse_docx,
        '.html': parse_html,
        '.htm': parse_html, # Add .htm alias
    }

    if extension in parser_map:
        try:
            parser_func = parser_map[extension]
            return parser_func(file_path)
        except ImportError as e:
            # Catch import errors from lazy imports if dependencies are missing
            ASCIIColors.error(f"Missing dependency for parsing '{extension}' files: {e}")
            raise ValueError(f"Missing dependency to parse {extension} files. Install with 'pip install safestore[parsing]'") from e
        except Exception as e:
             # Catch errors from the specific parser (e.g., file corruption, parse error)
             ASCIIColors.error(f"Failed to parse {file_path.name} using {parser_map[extension].__name__}: {e}")
             # Re-raise specific errors if needed, or a generic one
             raise # Re-raise the original error (e.g., ValueError, FileNotFoundError)
    else:
        ASCIIColors.warning(f"Unsupported file type '{extension}' for file: {file_path}. No parser available.")
        raise ValueError(f"Unsupported file type: {extension}")