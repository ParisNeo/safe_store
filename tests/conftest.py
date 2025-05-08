# tests/conftest.py
import pytest
from pathlib import Path
import sqlite3
import shutil
import numpy as np
from unittest.mock import MagicMock

# Import the class for type hinting
from safe_store import SafeStore, LogLevel

# --- Fixture Directory ---
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# --- Dependency Availability Check ---
# Check for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None # Define as None if not available

# Check for scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    NotFittedError = None

# --- Global Mocking Fixtures ---

# Mock SentenceTransformer if not available
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    class MockSentenceTransformer:
        DEFAULT_MODEL = "mock-st-model"
        def __init__(self, model_name):
            self.model_name = model_name
            self._dim = 384
            self._dtype = np.float32
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
             if not texts: return np.empty((0, self._dim), dtype=self._dtype)
             return np.random.rand(len(texts), self._dim).astype(self._dtype)
        def get_sentence_embedding_dimension(self): return self._dim
        @property
        def dim(self): return self._dim
        @property
        def dtype(self): return self._dtype

    @pytest.fixture(scope="session", autouse=True)
    def mock_st_globally(session_mocker):
        # Use session_mocker if available, otherwise regular monkeypatch might work in module scope
        # Using monkeypatch fixture is generally preferred within test functions/fixtures
        # For autouse session scope, directly patching might be necessary if mocker isn't standard
        # Let's use monkeypatch fixture within other fixtures instead for safety.
        pass # We will apply this mock conditionally in test files or fixtures needing it

# Mock Scikit-learn if not available
if not SKLEARN_AVAILABLE:
    class MockTfidfVectorizer:
        def __init__(self, **kwargs):
            self.params = kwargs; self._fitted = False; self.vocabulary_ = {}; self.idf_ = np.array([])
            self.dtype = np.float64
            if 'dtype' in kwargs:
                 try: self.dtype = np.dtype(kwargs['dtype'])
                 except: pass
            class MockTfidfInternal: _idf_diag = MagicMock()
            self._tfidf = MockTfidfInternal(); self._dim = None
        def fit(self, texts):
            if not texts: self.vocabulary_ = {}; self.idf_ = np.array([], dtype=self.dtype); self._dim = 0
            else:
                words = set(w for t in texts for w in t.lower().split())
                self.vocabulary_ = {w: i for i, w in enumerate(sorted(list(words)))}
                if not self.vocabulary_: self._dim = 0; self.idf_ = np.array([], dtype=self.dtype)
                else: self._dim = len(self.vocabulary_); self.idf_ = np.random.rand(self._dim).astype(self.dtype)*5+1
            self._fitted = True
            if hasattr(self, '_tfidf') and hasattr(self._tfidf, '_idf_diag'): self._tfidf._idf_diag.dtype = self.dtype
            return self
        def transform(self, texts):
            if not self._fitted: raise (NotFittedError or Exception)("MockTfidfVectorizer not fitted")
            if not texts: return MagicMock(**{'toarray.return_value': np.empty((0, self._dim or 0), dtype=self.dtype)})
            num_samples=len(texts); vocab_size=self._dim if self._dim is not None else 0
            if vocab_size is None: vocab_size = 0
            dense_array = np.random.rand(num_samples, vocab_size).astype(self.dtype)
            return MagicMock(**{'toarray.return_value': dense_array, 'shape': dense_array.shape})
        def get_params(self, deep=True): return self.params
        @property
        def dim(self): return self._dim

    @pytest.fixture(scope="session", autouse=True)
    def mock_sklearn_globally(session_mocker):
        # Similar caveat as mock_st_globally - apply conditionally where needed
        pass


# --- Helper to conditionally apply mocks ---
@pytest.fixture(autouse=True)
def apply_mocks_conditionally(monkeypatch):
    """Applies mocks only if the libraries are unavailable."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        monkeypatch.setattr("safe_store.vectorization.methods.sentence_transformer.SentenceTransformer", MockSentenceTransformer, raising=False)
        monkeypatch.setattr("safe_store.vectorization.methods.sentence_transformer._SENTENCE_TRANSFORMERS_AVAILABLE", True, raising=False) # Make wrapper think it's ok
    if not SKLEARN_AVAILABLE:
        monkeypatch.setattr("safe_store.vectorization.methods.tfidf.TfidfVectorizer", MockTfidfVectorizer, raising=False)
        monkeypatch.setattr("safe_store.vectorization.methods.tfidf.NotFittedError", NotFittedError or Exception, raising=False)
        monkeypatch.setattr("safe_store.vectorization.methods.tfidf._SKLEARN_AVAILABLE", True, raising=False) # Make wrapper think it's ok


# --- Standard Fixtures ---
@pytest.fixture(scope="function")
def temp_db_path(tmp_path: Path) -> Path:
    """Provides a path to a temporary database file."""
    return tmp_path / "test_safe_store.db"

@pytest.fixture(scope="function")
def safe_store_instance(temp_db_path: Path) -> SafeStore:
    """Provides a safe_store instance with a clean temporary database."""
    # Ensure clean slate
    if temp_db_path.exists(): temp_db_path.unlink()
    lock_path = temp_db_path.with_suffix(".db.lock")
    if lock_path.exists(): lock_path.unlink()
    wal_path = temp_db_path.with_suffix(".db-wal")
    if wal_path.exists(): wal_path.unlink(missing_ok=True)
    shm_path = temp_db_path.with_suffix(".db-shm")
    if shm_path.exists(): shm_path.unlink(missing_ok=True)

    # Use DEBUG level for more verbose test output
    store = SafeStore(db_path=temp_db_path, log_level=LogLevel.DEBUG, lock_timeout=0.1)
    yield store
    store.close() # Ensure closure after test function finishes

@pytest.fixture(scope="session")
def sample_text_content() -> str:
    return "This is the first sentence.\nThis is the second sentence, it is a bit longer.\nAnd a third one."

@pytest.fixture
def sample_text_file(tmp_path: Path, sample_text_content: str) -> Path:
    """Creates a temporary text file."""
    p = tmp_path / "sample.txt"
    p.write_text(sample_text_content, encoding='utf-8')
    return p

# --- Phase 3 Fixtures ---
@pytest.fixture
def sample_pdf_file(tmp_path: Path) -> Path:
    """Copies the sample PDF to the temp directory."""
    source = FIXTURES_DIR / "sample.pdf"
    if not source.exists(): pytest.skip("sample.pdf fixture file not found")
    dest = tmp_path / "sample.pdf"
    try: shutil.copy(source, dest)
    except Exception as e: pytest.fail(f"Failed to copy fixture file {source}: {e}")
    return dest

@pytest.fixture
def sample_docx_file(tmp_path: Path) -> Path:
    """Copies the sample DOCX to the temp directory."""
    source = FIXTURES_DIR / "sample.docx"
    if not source.exists(): pytest.skip("sample.docx fixture file not found")
    dest = tmp_path / "sample.docx"
    try: shutil.copy(source, dest)
    except Exception as e: pytest.fail(f"Failed to copy fixture file {source}: {e}")
    return dest

@pytest.fixture
def sample_html_file(tmp_path: Path) -> Path:
    """Copies the sample HTML to the temp directory."""
    source = FIXTURES_DIR / "sample.html"
    if not source.exists(): pytest.skip("sample.html fixture file not found")
    dest = tmp_path / "sample.html"
    try: shutil.copy(source, dest)
    except Exception as e: pytest.fail(f"Failed to copy fixture file {source}: {e}")
    return dest


# --- Phase 2 Fixture ---
@pytest.fixture
def populated_store(safe_store_instance: SafeStore, sample_text_file: Path, tmp_path: Path) -> SafeStore:
    """Provides a safe_store instance with two documents added using the default ST vectorizer."""
    store = safe_store_instance
    doc2_content = "Another document.\nWith different content for testing."
    doc2_path = tmp_path / "sample2.txt"
    doc2_path.write_text(doc2_content, encoding='utf-8')

    # No need for availability check here due to global autouse fixture apply_mocks_conditionally
    try:
        with store:
            store.add_document(sample_text_file, chunk_size=30, chunk_overlap=5)
            store.add_document(doc2_path, chunk_size=25, chunk_overlap=5)
    except Exception as e:
         pytest.fail(f"Populated store fixture setup failed: {e}")

    return store