# safe_store/indexing/parser.py
from pathlib import Path
from typing import Callable, Dict, Union # Added Union
from ascii_colors import ASCIIColors
# Import specific custom exceptions
from ..core.exceptions import ParsingError, FileHandlingError, ConfigurationError

# Protocols remain unchanged, add type hints
from typing import Protocol, runtime_checkable, BinaryIO, TextIO
@runtime_checkable
class PdfReaderProtocol(Protocol): # noqa
    pages: list
    def __init__(self, stream: Union[bytes, Path, str, BinaryIO], strict: bool = True): ...
@runtime_checkable
class PageObjectProtocol(Protocol): # noqa
    def extract_text(self) -> str: ...
@runtime_checkable
class DocumentProtocol(Protocol): # noqa
    paragraphs: list
    def __init__(self, file_path: Union[str, Path, None] = None): ...
@runtime_checkable
class ParagraphProtocol(Protocol): # noqa
    text: str
@runtime_checkable
class BeautifulSoupProtocol(Protocol): # noqa
    def __init__(self, markup: Union[str, bytes], features: str, **kwargs): ...
    def get_text(self, separator: str = "", strip: bool = False) -> str: ...


# --- Parsing Functions ---

def parse_txt(file_path: Union[str, Path]) -> str:
    """
    Parses a plain text file (UTF-8 encoding).

    Args:
        file_path: Path to the text file.

    Returns:
        The content of the file as a string.

    Raises:
        FileHandlingError: If the file is not found or cannot be read.
        ParsingError: If decoding fails or other unexpected errors occur.
    """
    _file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse TXT file: {_file_path}")
    try:
        with open(_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ASCIIColors.debug(f"Successfully parsed TXT file: {_file_path}")
        return content
    except FileNotFoundError as e:
        msg = f"File not found: {_file_path}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except UnicodeDecodeError as e:
        msg = f"Encoding error parsing TXT file {_file_path} as UTF-8: {e}"
        ASCIIColors.error(msg)
        raise ParsingError(msg) from e
    except OSError as e:
        msg = f"OS error reading TXT file {_file_path}: {e}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except Exception as e:
        msg = f"Unexpected error parsing TXT file {_file_path}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise ParsingError(msg) from e


def parse_pdf(file_path: Union[str, Path]) -> str:
    """
    Parses a PDF file to extract text content using pypdf.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The extracted text content, concatenated from all pages.

    Raises:
        ConfigurationError: If 'pypdf' is not installed.
        FileHandlingError: If the file is not found or is empty.
        ParsingError: If the PDF structure is invalid or text extraction fails.
    """
    _file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse PDF file: {_file_path}")
    try:
        # Import dynamically to check for dependency
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError, EmptyFileError
    except ImportError as e:
        msg = "Parsing PDF files requires 'pypdf'. Install with: pip install safe_store[parsing]"
        ASCIIColors.error(msg)
        raise ConfigurationError(msg) from e

    full_text = ""
    try:
        # Use strict=False to be more tolerant of minor PDF spec violations
        reader: PdfReaderProtocol = PdfReader(_file_path, strict=False)
        num_pages = len(reader.pages)
        ASCIIColors.debug(f"PDF '{_file_path.name}' loaded with {num_pages} pages.")

        if num_pages == 0:
             ASCIIColors.warning(f"PDF file '{_file_path.name}' contains zero pages.")
             return "" # Return empty string for zero-page PDFs

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n" # Add newline between pages
                else:
                    ASCIIColors.debug(f"No text extracted from page {page_num} of '{_file_path.name}'.")
            except Exception as page_err: # Catch errors during individual page extraction
                 ASCIIColors.warning(f"Could not extract text from page {page_num} of '{_file_path.name}': {page_err}")
                 # Decide whether to continue or fail? Continue for now.
                 # Consider adding a flag to control strictness on page errors.

        ASCIIColors.debug(f"Successfully parsed PDF file: {_file_path}")
        return full_text.strip() # Remove leading/trailing whitespace

    except FileNotFoundError as e:
        msg = f"File not found: {_file_path}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except EmptyFileError as e:
         msg = f"Cannot parse empty PDF file: {_file_path}"
         ASCIIColors.error(msg)
         # Consider empty file a FileHandlingError subtype? Or keep as ParsingError?
         # Let's use FileHandlingError as it relates to the file state.
         raise FileHandlingError(msg) from e
    except PdfReadError as e: # Catch specific pypdf structural errors
        msg = f"Error reading PDF structure in {_file_path}: {e}"
        ASCIIColors.error(msg)
        raise ParsingError(msg) from e
    except OSError as e: # Catch potential OS errors during file reading by pypdf
        msg = f"OS error reading PDF file {_file_path}: {e}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except Exception as e: # Catch other unexpected errors from pypdf
        msg = f"Unexpected error parsing PDF file {_file_path}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise ParsingError(msg) from e


def parse_docx(file_path: Union[str, Path]) -> str:
    """
    Parses a DOCX file to extract text content using python-docx.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        The extracted text content, concatenated from all paragraphs.

    Raises:
        ConfigurationError: If 'python-docx' is not installed.
        FileHandlingError: If the file is not found.
        ParsingError: If the file is not a valid DOCX format or text extraction fails.
    """
    _file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse DOCX file: {_file_path}")
    try:
        # Import dynamically
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as e:
        msg = "Parsing DOCX files requires 'python-docx'. Install with: pip install safe_store[parsing]"
        ASCIIColors.error(msg)
        raise ConfigurationError(msg) from e

    full_text = ""
    try:
        document: DocumentProtocol = Document(_file_path)
        ASCIIColors.debug(f"DOCX '{_file_path.name}' loaded.")
        for para in document.paragraphs:
            full_text += para.text + "\n" # Add newline between paragraphs
        ASCIIColors.debug(f"Successfully parsed DOCX file: {_file_path}")
        return full_text.strip() # Remove leading/trailing whitespace

    except FileNotFoundError as e:
        msg = f"File not found: {_file_path}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except PackageNotFoundError as e: # Specific error for invalid zip/docx format
         msg = f"File is not a valid DOCX (Zip) file: {_file_path}. Error: {e}"
         ASCIIColors.error(msg)
         raise ParsingError(msg) from e
    except OSError as e: # Catch potential OS errors during file reading by python-docx
        msg = f"OS error reading DOCX file {_file_path}: {e}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except Exception as e: # Catch other unexpected errors from python-docx
        msg = f"Unexpected error parsing DOCX file {_file_path}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise ParsingError(msg) from e


def parse_html(file_path: Union[str, Path]) -> str:
    """
    Parses an HTML file to extract text content using BeautifulSoup.

    Uses 'lxml' if available for performance, otherwise falls back to
    Python's built-in 'html.parser'.

    Args:
        file_path: Path to the HTML file.

    Returns:
        The extracted text content, with tags stripped.

    Raises:
        ConfigurationError: If 'beautifulsoup4' (or 'lxml' if used) is not installed.
        FileHandlingError: If the file is not found or cannot be read.
        ParsingError: If the file cannot be decoded or parsed as HTML.
    """
    _file_path = Path(file_path)
    ASCIIColors.debug(f"Attempting to parse HTML file: {_file_path}")
    try:
        # Import dynamically
        from bs4 import BeautifulSoup
        try:
            import lxml # noqa F401 ensure it's installed for best performance
            HTML_PARSER = 'lxml'
            ASCIIColors.debug("Using 'lxml' for HTML parsing.")
        except ImportError:
            ASCIIColors.debug("lxml not found, using Python's built-in 'html.parser' for HTML.")
            HTML_PARSER = 'html.parser'
    except ImportError as e:
        # This catches missing BeautifulSoup4
        msg = "Parsing HTML files requires 'BeautifulSoup4'. Install with: pip install safe_store[parsing]"
        ASCIIColors.error(msg)
        raise ConfigurationError(msg) from e

    try:
        with open(_file_path, 'r', encoding='utf-8') as f:
            # Read content first, then parse
             content = f.read()

        # Parse the HTML content
        soup: BeautifulSoupProtocol = BeautifulSoup(content, HTML_PARSER)
        # Get text, stripping extra whitespace and using newline as separator
        text = soup.get_text(separator='\n', strip=True)
        ASCIIColors.debug(f"Successfully parsed HTML file: {_file_path}")
        return text

    except FileNotFoundError as e:
        msg = f"File not found: {_file_path}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except UnicodeDecodeError as e:
        msg = f"Encoding error parsing HTML file {_file_path} as UTF-8: {e}"
        ASCIIColors.error(msg)
        raise ParsingError(msg) from e
    except OSError as e:
        msg = f"OS error reading HTML file {_file_path}: {e}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) from e
    except Exception as e: # Catch potential BeautifulSoup errors or other issues
        msg = f"Error parsing HTML file {_file_path}: {e}"
        ASCIIColors.error(msg, exc_info=True)
        raise ParsingError(msg) from e


# --- Dispatcher Function ---

# Define a type hint for the parser functions
ParserFunc = Callable[[Union[str, Path]], str]

def parse_document(file_path: Union[str, Path]) -> str:
    """
    Parses a document based on its file extension.

    Dispatches to the appropriate parser (.txt, .pdf, .docx, .html/.htm).

    Args:
        file_path: Path to the document file.

    Returns:
        The extracted text content of the document.

    Raises:
        FileHandlingError: If the input path is not a file.
        ConfigurationError: If the file extension is unsupported.
        ParsingError: If the dispatched parser encounters an error.
    """
    _file_path = Path(file_path)
    if not _file_path.is_file():
        msg = f"Input path is not a file: {_file_path}"
        ASCIIColors.error(msg)
        raise FileHandlingError(msg) # Use specific error

    extension = _file_path.suffix.lower()
    ASCIIColors.debug(f"Dispatching parser for extension '{extension}' on file: {_file_path.name}")

    # Map extensions to parser functions
    parser_map: Dict[str, ParserFunc] = {
        '.txt': parse_txt,
        '.pdf': parse_pdf,
        '.docx': parse_docx,
        '.html': parse_html,
        '.htm': parse_html, # Treat .htm the same as .html
    }

    parser_func = parser_map.get(extension)

    if parser_func:
        try:
            # Call the appropriate parser function
            return parser_func(_file_path)
        except (ConfigurationError, FileHandlingError, ParsingError) as e:
             # Re-raise specific known errors directly
             raise e
        except Exception as e:
             # Wrap unexpected errors from the specific parser
             msg = f"Unexpected error during parsing dispatch for {_file_path.name} (extension '{extension}'): {e}"
             ASCIIColors.error(msg, exc_info=True)
             # Wrap in ParsingError as it occurred during the parsing stage
             raise ParsingError(msg) from e
    else:
        # Handle unsupported file type
        msg = f"Unsupported file type extension: '{extension}' for file: {_file_path}. No parser available."
        ASCIIColors.warning(msg)
        # Use ConfigurationError for unsupported type, as it's a setup/config issue
        raise ConfigurationError(msg)