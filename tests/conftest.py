import pytest
from pathlib import Path
import sqlite3
from safestore import SafeStore, LogLevel # Adjust import if needed

@pytest.fixture(scope="function") # Use function scope for clean DB each test
def temp_db_path(tmp_path: Path) -> Path:
    """Provides a path to a temporary database file."""
    return tmp_path / "test_safestore.db"

@pytest.fixture(scope="function")
def safestore_instance(temp_db_path: Path) -> SafeStore:
    """Provides a SafeStore instance with a clean temporary database."""
    # Ensure the db file doesn't exist from a previous failed run if scope changes
    if temp_db_path.exists():
        temp_db_path.unlink()
    store = SafeStore(db_path=temp_db_path, log_level=LogLevel.DEBUG)
    yield store # Provide the instance to the test
    store.close() # Ensure connection is closed after test
    # Optional: cleanup the db file, but tmp_path fixture handles directory cleanup
    # if temp_db_path.exists():
    #     temp_db_path.unlink()


@pytest.fixture(scope="session")
def sample_text_content() -> str:
    return "This is the first sentence.\nThis is the second sentence, it is a bit longer.\nAnd a third one."

@pytest.fixture
def sample_text_file(tmp_path: Path, sample_text_content: str) -> Path:
    """Creates a temporary text file."""
    p = tmp_path / "sample.txt"
    p.write_text(sample_text_content, encoding='utf-8')
    return p