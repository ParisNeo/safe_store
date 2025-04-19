# tests/conftest.py
import pytest
from pathlib import Path
import sqlite3
from safestore import SafeStore, LogLevel # Adjust import if needed
import shutil # Import shutil for copying fixtures

# --- Fixture Directory ---
FIXTURES_DIR = Path(__file__).parent / "fixtures"
print(f"DEBUG [conftest.py]: Conftest __file__ is {__file__}")
print(f"DEBUG [conftest.py]: Calculated FIXTURES_DIR is {FIXTURES_DIR}")
# Check if the directory itself exists
if not FIXTURES_DIR.is_dir():
    print(f"ERROR [conftest.py]: FIXTURES_DIR {FIXTURES_DIR} does not exist or is not a directory!")


@pytest.fixture(scope="function")
def temp_db_path(tmp_path: Path) -> Path:
    """Provides a path to a temporary database file."""
    return tmp_path / "test_safestore.db"

# Shorten lock timeout for testing by default
@pytest.fixture(scope="function")
def safestore_instance(temp_db_path: Path) -> SafeStore:
    """Provides a SafeStore instance with a clean temporary database."""
    if temp_db_path.exists():
        temp_db_path.unlink()
    lock_path = temp_db_path.with_suffix(".db.lock")
    if lock_path.exists():
        lock_path.unlink()
    store = SafeStore(db_path=temp_db_path, log_level=LogLevel.DEBUG, lock_timeout=0.1) # Use shorter timeout
    yield store
    store.close()

@pytest.fixture(scope="session")
def sample_text_content() -> str:
    return "This is the first sentence.\nThis is the second sentence, it is a bit longer.\nAnd a third one."

@pytest.fixture
def sample_text_file(tmp_path: Path, sample_text_content: str) -> Path:
    """Creates a temporary text file."""
    p = tmp_path / "sample.txt"
    p.write_text(sample_text_content, encoding='utf-8')
    return p

# --- New Fixtures for Phase 3 ---

@pytest.fixture
def sample_pdf_file(tmp_path: Path) -> Path:
    """Copies the sample PDF to the temp directory."""
    source = FIXTURES_DIR / "sample.pdf"
    print(f"DEBUG [sample_pdf_file fixture]: Source path: {source}") # DEBUG PRINT
    if not source.exists():
        print(f"ERROR [sample_pdf_file fixture]: Source file {source} does not exist!") # DEBUG PRINT
        pytest.skip("sample.pdf fixture file not found")
    dest = tmp_path / "sample.pdf"
    print(f"DEBUG [sample_pdf_file fixture]: Dest path: {dest}") # DEBUG PRINT
    try:
        shutil.copy(source, dest)
        print(f"DEBUG [sample_pdf_file fixture]: Copied {source} to {dest}") # DEBUG PRINT
    except Exception as e:
        print(f"ERROR [sample_pdf_file fixture]: Failed to copy {source} to {dest}: {e}") # DEBUG PRINT
        pytest.fail(f"Failed to copy fixture file {source}: {e}")
    return dest

@pytest.fixture
def sample_docx_file(tmp_path: Path) -> Path:
    """Copies the sample DOCX to the temp directory."""
    source = FIXTURES_DIR / "sample.docx"
    print(f"DEBUG [sample_docx_file fixture]: Source path: {source}") # DEBUG PRINT
    if not source.exists():
        print(f"ERROR [sample_docx_file fixture]: Source file {source} does not exist!") # DEBUG PRINT
        pytest.skip("sample.docx fixture file not found")
    dest = tmp_path / "sample.docx"
    print(f"DEBUG [sample_docx_file fixture]: Dest path: {dest}") # DEBUG PRINT
    try:
        shutil.copy(source, dest)
        print(f"DEBUG [sample_docx_file fixture]: Copied {source} to {dest}") # DEBUG PRINT
    except Exception as e:
        print(f"ERROR [sample_docx_file fixture]: Failed to copy {source} to {dest}: {e}") # DEBUG PRINT
        pytest.fail(f"Failed to copy fixture file {source}: {e}")
    return dest

@pytest.fixture
def sample_html_file(tmp_path: Path) -> Path:
    """Copies the sample HTML to the temp directory."""
    source = FIXTURES_DIR / "sample.html"
    print(f"DEBUG [sample_html_file fixture]: Source path: {source}") # DEBUG PRINT
    if not source.exists():
        print(f"ERROR [sample_html_file fixture]: Source file {source} does not exist!") # DEBUG PRINT
        pytest.skip("sample.html fixture file not found")
    dest = tmp_path / "sample.html"
    print(f"DEBUG [sample_html_file fixture]: Dest path: {dest}") # DEBUG PRINT
    try:
        shutil.copy(source, dest)
        print(f"DEBUG [sample_html_file fixture]: Copied {source} to {dest}") # DEBUG PRINT
    except Exception as e:
        print(f"ERROR [sample_html_file fixture]: Failed to copy {source} to {dest}: {e}") # DEBUG PRINT
        pytest.fail(f"Failed to copy fixture file {source}: {e}")
    return dest