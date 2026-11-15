# safe_store/vectorization/methods/onnx/__init__.py
import importlib
import site
import sys
import threading
import numpy as np
from typing import List, Optional, Dict, Any, Tuple

from safe_store.vectorization.base import BaseVectorizer
from safe_store.core.exceptions import ConfigurationError, VectorizationError
from safe_store.processing.tokenizers import HuggingFaceTokenizerWrapper
from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

class_name = "ONNXVectorizer"

# In-process registry: one ONNX session per (model, cache, providers, threads).
_IN_PROCESS_MODEL_REGISTRY: Dict[str, Tuple[Any, int]] = {}
_REGISTRY_LOCK = threading.Lock()

_TextEmbedding = None
_FASTEMBED_DEPS_READY = False


def _refresh_sys_path_after_install() -> None:
    """Make a wheel installed by pipmaster importable in THIS process.

    `pipmaster.ensure_packages(...)` may run a real `pip install` in a
    subprocess. The running interpreter's `importlib` caches directory
    listings for every entry on `sys.path` at startup; new packages that
    land in `site-packages` (or in a `--user` site directory) after that
    point are **not** discoverable until the caches are invalidated.

    This helper:
      1. Calls `importlib.invalidate_caches()` so existing path entries
         are re-scanned.
      2. Adds the user-site `site-packages` if pip installed `--user`
         (common when the venv is not writable / on locked-down Windows
         environments) and it isn't already on `sys.path`.

    Idempotent and safe to call even when no install actually happened.
    """
    importlib.invalidate_caches()
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site or user_site in sys.path:
        return
    try:
        import os
        if os.path.isdir(user_site):
            site.addsitedir(user_site)
    except Exception:
        pass


def _try_import_fastembed():
    """Return the imported `fastembed` module, or None if not importable yet.

    Probes the import in isolation so callers can distinguish "not installed
    yet" from "installed but temporarily invisible due to importlib caches".
    Never raises.
    """
    try:
        import fastembed  # noqa: F401 - probe only
        return fastembed
    except (ImportError, ModuleNotFoundError):
        return None


def _ensure_fastembed_deps_installed() -> None:
    """Install `fastembed` via pipmaster on first use and verify importability.

    Raises `ConfigurationError` if the package cannot be imported even after
    the pip step. The flag `_FASTEMBED_DEPS_READY` is only set TRUE once the
    import has been *verified* — otherwise a subsequent caller would short-
    circuit the (possibly deferred) install attempt and re-raise ModuleNotFoundError
    at the enclosing `_ensure_fastembed_loaded` call site.

    pipmaster may run the install synchronously or in a backgrounded/"scheduled"
    mode (log line: "Scheduled for installation."). When scheduled, the wheel
    is not on disk yet at the next line, so we retry the import for a bounded
    window after invalidating importlib caches.
    """
    global _FASTEMBED_DEPS_READY
    if _FASTEMBED_DEPS_READY:
        return

    if _try_import_fastembed() is not None:
        _FASTEMBED_DEPS_READY = True
        return

    ASCIIColors.info(
        "ONNXVectorizer: installing 'fastembed' (first use only, ~30 s)..."
    )
    pm.ensure_packages(["fastembed"])
    _refresh_sys_path_after_install()

    # Bounded wait: pipmaster may have *scheduled* the install rather than
    # completed it. Retry the import for up to ~60 s (poll every 1 s) so a
    # deferred install becomes visible before we give up and raise a clear
    # ConfigurationError with a manual-install hint.
    import time as _time
    deadline = _time.monotonic() + 60.0
    while _time.monotonic() < deadline:
        if _try_import_fastembed() is not None:
            _FASTEMBED_DEPS_READY = True
            return
        _refresh_sys_path_after_install()
        _time.sleep(1.0)

    raise ConfigurationError(
        "ONNXVectorizer: 'fastembed' was requested for installation but is "
        "not importable after 60 s. pipmaster may have scheduled a deferred "
        "install that did not complete. Please run `pip install fastembed` "
        "manually, then retry."
    )


def _ensure_fastembed_loaded():
    """Import fastembed.TextEmbedding on first use. No torch involved."""
    global _TextEmbedding
    if _TextEmbedding is not None:
        return _TextEmbedding

    # Ensure installable + importable first; `_ensure_fastembed_deps_installed`
    # only returns normally once a real import has succeeded.
    _ensure_fastembed_deps_installed()

    try:
        from fastembed import TextEmbedding as _TE
    except Exception as e:
        trace_exception(e)
        raise ConfigurationError(
            "ONNXVectorizer requires 'fastembed' (auto-installed via pipmaster "
            "on first use). If this persists, run `pip install fastembed`."
        ) from e

    _TextEmbedding = _TE
    return _TextEmbedding


_FALLBACK_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-small",
]


def list_available_models(**kwargs) -> List[str]:
    try:
        te = _ensure_fastembed_loaded()
        names = [m.get("model") for m in te.list_supported_models() if isinstance(m, dict)]
        names = [n for n in names if n]
        if names:
            return names
    except Exception:
        pass
    return list(_FALLBACK_MODELS)


class ONNXVectorizer(BaseVectorizer):
    """Lightweight text vectorizer: ONNX Runtime via fastembed. No PyTorch."""

    DEFAULT_MODEL: str = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_config: Dict[str, Any], cache_folder: Optional[str] = None, **kwargs):
        super().__init__(vectorizer_name="onnx")

        self.model_name: str = model_config.get("model_name") or model_config.get("model") or self.DEFAULT_MODEL
        self.cache_folder = cache_folder
        self.threads: Optional[int] = model_config.get("threads")
        providers = model_config.get("providers")
        self.providers: Optional[List[str]] = list(providers) if providers else None
        self.batch_size: int = int(model_config.get("batch_size", 64))
        self.reuse_model_in_process: bool = bool(model_config.get("reuse_model_in_process", True))

        self._dim: Optional[int] = None
        self._dtype: np.dtype = np.dtype(np.float32)
        self.model: Optional[Any] = None
        self._registry_key: Optional[str] = None

        self._init_local_model()

    def _init_local_model(self) -> None:
        te_cls = _ensure_fastembed_loaded()
        try:
            if self.reuse_model_in_process:
                self._registry_key = (
                    f"{self.model_name}:{self.cache_folder}:{self.providers}:{self.threads}"
                )
                with _REGISTRY_LOCK:
                    if self._registry_key in _IN_PROCESS_MODEL_REGISTRY:
                        shared, count = _IN_PROCESS_MODEL_REGISTRY[self._registry_key]
                        _IN_PROCESS_MODEL_REGISTRY[self._registry_key] = (shared, count + 1)
                        self.model = shared
                        ASCIIColors.info(
                            f"Reusing in-memory ONNX model '{self.model_name}' (Active instances: {count + 1})"
                        )
                        return

            ASCIIColors.info(f"Loading ONNX embedding model: {self.model_name}")
            kw: Dict[str, Any] = {"model_name": self.model_name}
            if self.cache_folder is not None:
                kw["cache_dir"] = self.cache_folder
            if self.threads is not None:
                kw["threads"] = int(self.threads)
            if self.providers:
                kw["providers"] = self.providers
            self.model = te_cls(**kw)

            if self.reuse_model_in_process and self._registry_key:
                with _REGISTRY_LOCK:
                    _IN_PROCESS_MODEL_REGISTRY[self._registry_key] = (self.model, 1)

            ASCIIColors.info(f"ONNX model '{self.model_name}' loaded.")
        except Exception as e:
            raise VectorizationError(f"Failed to load ONNX model '{self.model_name}': {e}") from e

    def _lookup_dim_from_catalog(self) -> Optional[int]:
        try:
            for m in _ensure_fastembed_loaded().list_supported_models():
                if isinstance(m, dict) and m.get("model") == self.model_name:
                    d = m.get("dim")
                    if isinstance(d, (int, np.integer)) and int(d) > 0:
                        return int(d)
        except Exception:
            pass
        return None

    @property
    def dim(self) -> Optional[int]:
        if self._dim is not None:
            return self._dim
        self._dim = self._lookup_dim_from_catalog()
        if self._dim is not None:
            return self._dim
        try:
            probe = self.vectorize(["dimension_probe"])
            self._dim = int(probe.shape[1])
        except Exception:
            return None
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def get_tokenizer(self) -> Optional[HuggingFaceTokenizerWrapper]:
        return None

    def supports_late_chunking(self) -> bool:
        # fastembed returns pooled vectors, not token-level embeddings.
        return False

    def late_chunk_embed(self, text: str, chunk_spans: List[Tuple[int, int]]) -> np.ndarray:
        return self.vectorize([text[s:e] for s, e in chunk_spans])

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim or 1), dtype=self._dtype)
        if self.model is None:
            raise VectorizationError("ONNX model is not loaded.")
        try:
            embeddings = np.asarray(
                list(self.model.embed(list(texts), batch_size=self.batch_size)),
                dtype=self._dtype,
            )
            if embeddings.ndim != 2:
                raise VectorizationError(
                    f"Unexpected embedding shape {embeddings.shape} from ONNX model."
                )
            if self._dim is None:
                self._dim = int(embeddings.shape[1])
            return embeddings
        except VectorizationError:
            raise
        except Exception as e:
            raise VectorizationError(f"Error during ONNX encoding: {e}") from e

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return list_available_models()

    def unload(self) -> None:
        if self.model is None:
            return
        if self._registry_key and self.reuse_model_in_process:
            with _REGISTRY_LOCK:
                entry = _IN_PROCESS_MODEL_REGISTRY.get(self._registry_key)
                if entry is not None:
                    shared, count = entry
                    if count > 1:
                        _IN_PROCESS_MODEL_REGISTRY[self._registry_key] = (shared, count - 1)
                    else:
                        del _IN_PROCESS_MODEL_REGISTRY[self._registry_key]
        self.model = None
        import gc
        gc.collect()

    def close(self) -> None:
        self.unload()

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass