# safe_store/vectorization/methods/sentense_transformer/__init__.py
import importlib
import os
import site
import time
import socket
import subprocess
import sys
import secrets
import tempfile
import threading
import numpy as np
from typing import List, Optional, Dict, Any, Tuple

from safe_store.vectorization.base import BaseVectorizer
from safe_store.core.exceptions import ConfigurationError, VectorizationError
from safe_store.processing.tokenizers import HuggingFaceTokenizerWrapper
from ascii_colors import ASCIIColors, trace_exception
import pipmaster as pm

try:
    from filelock import FileLock, Timeout
except ImportError:
    FileLock = None
    Timeout = None

class_name = "STVectorizer"

# In-process thread-safe model registry to prevent duplicate model allocations in the same process
_IN_PROCESS_MODEL_REGISTRY: Dict[str, Tuple[Any, int]] = {}
_REGISTRY_LOCK = threading.Lock()


def disable_broken_torchvision() -> None:
    """
    Guards against PyTorch/torchvision binary mismatches where an incompatible
    torchvision build causes: RuntimeError: operator torchvision::nms does not exist.
    """
    for mod in list(sys.modules.keys()):
        if mod == "torchvision" or mod.startswith("torchvision."):
            try:
                del sys.modules[mod]
            except KeyError:
                pass
    sys.modules["torchvision"] = None
    sys.modules["torchvision.ops"] = None


disable_broken_torchvision()


def _log_st_timing(label: str, started: float) -> None:
    """Emit a precise elapsed-ms line for a STVectorizer construction step.

    Diagnostic instrumentation: attributes startup time to a specific
    sub-phase of `_init_shared_server_impl` (log-streamer start, spawn-lock
    acquire, spawn, server-ready wait, connect retries, attach message).
    """
    elapsed_ms = (time.monotonic() - started) * 1000.0
    ASCIIColors.info(f"[STVectorizer timing] {label}: {elapsed_ms:.1f} ms")


# sentence_transformers / torch are imported LAZILY, only when a LOCAL model is
# needed. In shared-server mode (`use_shared_server=True`), the main process must
# NEVER import torch — the daemon (a separate process) owns the model. On Windows,
# `import torch` costs ~20-25 s cold, which previously surfaced as a mysterious
# stall inside `SafeStore.__init__` even when the daemon was perfectly lazy.
# The module-global name is kept for backward compatibility and populated on
# first use via `_ensure_st_class_loaded()`.
SentenceTransformer = None

# Proactive one-shot install guard: even if `sentence_transformers` is already
# importable, we still call ensure_packages once per process to validate that
# `torch` is importable too (some users install sentence_transformers without
# torch via --no-deps). Idempotent; the flag short-circuits after success.
_ST_DEPS_READY = False


def _refresh_sys_path_after_install() -> None:
    """Make wheels installed by pipmaster importable in THIS process.

    `pm.ensure_packages(...)` runs a real `pip install` in a subprocess. The
    running interpreter caches directory listings for `sys.path` at startup,
    so new packages dropped into `site-packages` (or user-site) after that
    point are invisible until the caches are invalidated. This is a no-op if
    the install was a no-op.
    """
    importlib.invalidate_caches()
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site or user_site in sys.path:
        return
    try:
        if os.path.isdir(user_site):
            site.addsitedir(user_site)
    except Exception:
        pass


def _ensure_st_deps_installed() -> None:
    """Install `torch` + `sentence-transformers` via pipmaster on first use.

    Called proactively by `_ensure_st_class_loaded()` (main process, local mode)
    and by `ModelServer._load_model_locked()` (shared-daemon process). Skips the
    pip check once the deps are known-present, so the second + Nth callers pay
    only a boolean check.
    """
    global _ST_DEPS_READY
    if _ST_DEPS_READY:
        return
    try:
        import torch  # noqa: F401 - probe only
        import sentence_transformers  # noqa: F401 - probe only
        _ST_DEPS_READY = True
        return
    except (ImportError, ModuleNotFoundError):
        pass
    ASCIIColors.info(
        "STVectorizer: installing 'torch' + 'sentence-transformers' "
        "(first use only — this can take 30–90 s on a cold environment)..."
    )
    pm.ensure_packages(["torch", "sentence-transformers"])
    _refresh_sys_path_after_install()
    _ST_DEPS_READY = True


def _ensure_st_class_loaded():
    """Import `sentence_transformers.SentenceTransformer` on first use.

    Idempotent. Installs `torch` + `sentence-transformers` via pipmaster if
    missing, then imports the class. Raises `ConfigurationError` if the
    dependency cannot be installed/imported. Never called on the shared-server
    path in the main process — the daemon installs on its own.
    """
    global SentenceTransformer
    if SentenceTransformer is not None:
        return SentenceTransformer

    try:
        _ensure_st_deps_installed()
        disable_broken_torchvision()
        from sentence_transformers import SentenceTransformer as _ST
    except Exception as e:
        trace_exception(e)
        raise ConfigurationError(
            "STVectorizer requires 'sentence-transformers' and 'torch' "
            "(auto-installed via pipmaster on first use). If this persists, "
            "run `pip install torch sentence-transformers` manually."
        ) from e

    SentenceTransformer = _ST
    return SentenceTransformer


def list_available_models(**kwargs) -> List[str]:
    return [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "paraphrase-albert-small-v2",
        "LaBSE"
    ]


class STVectorizer(BaseVectorizer):
    """Vectorizes text using models from the sentence-transformers library, supporting shared background server mode."""

    DEFAULT_MODEL: str = "all-MiniLM-L6-v2"
    DEFAULT_HOST: str = "127.0.0.1"
    SERVER_STARTUP_TIMEOUT: float = 180.0
    SERVER_PING_INTERVAL: float = 0.1

    def __init__(self, model_config: Dict[str, Any], cache_folder: Optional[str] = None, **kwargs):
        super().__init__(vectorizer_name="st")

        disable_broken_torchvision()

        self.model_name: str = model_config.get("model_name") or model_config.get("model", self.DEFAULT_MODEL)
        if not self.model_name:
            raise ConfigurationError("STVectorizer config must include a 'model_name' or 'model' key.")

        self.use_shared_server: bool = bool(
            model_config.get("use_shared_server", False) or
            model_config.get("shared_server", False) or
            model_config.get("shared_mode", False) or
            model_config.get("shared", False)
        )
        self.reuse_model_in_process: bool = bool(
            model_config.get("reuse_model_in_process", True) and
            not self.use_shared_server
        )

        # Only local mode needs the model class in THIS process. Shared mode
        # never imports torch — the daemon owns the model.
        if not self.use_shared_server:
            _ensure_st_class_loaded()
        self.port: int = int(model_config.get("port", 8765))
        self.host: str = str(model_config.get("host", self.DEFAULT_HOST))
        self.idle_timeout: float = float(model_config.get("idle_timeout", 0.0))
        self.batch_window: float = float(model_config.get("batch_window", 0.02))
        self.max_batch_size: int = int(model_config.get("max_batch_size", 64))
        self.stream_server_logs: bool = bool(model_config.get("stream_server_logs", True))
        self.cache_folder = cache_folder

        self._server_process: Optional[subprocess.Popen] = None
        self._client = None
        self._registry_key: Optional[str] = None
        self._dim: Optional[int] = None
        self._dtype: np.dtype = np.dtype(np.float32)
        self.model: Optional[Any] = None
        self._token_file: Optional[str] = None
        self._lock_file: Optional[str] = None
        self._server_log_file: Optional[str] = None
        self._log_streamer_thread: Optional[threading.Thread] = None
        self._log_streamer_stop: Optional[threading.Event] = None

        if self.use_shared_server:
            try:
                self._init_shared_server()
            except (VectorizationError, ConfigurationError) as e:
                if model_config.get("fail_on_shared_server_error", False):
                    raise
                ASCIIColors.warning(
                    f"Shared model server unavailable: {e}. "
                    "Falling back to in-process SentenceTransformer loading."
                )
                self._cleanup_shared_server_state()
                self.use_shared_server = False
                self.reuse_model_in_process = bool(
                    model_config.get("reuse_model_in_process", True)
                )
                self._init_local_model()
        else:
            self._init_local_model()

    def _resolve_model_dimension(self) -> int:
        """
        Resolves the embedding dimension of the loaded model across
        sentence-transformers versions and test fakes/mocks.

        Resolution order:
        1. `get_embedding_dimension()` — the modern API
           (sentence-transformers >= 3.x). Accepted only when it returns a
           genuine integer, guarding against MagicMock auto-stubs and
           attribute-name drift in test doubles.
        2. `get_sentence_embedding_dimension()` — the legacy alias, still
           present on real ST models (deprecated) and used by existing
           test fixtures.

        Raises VectorizationError if neither yields a usable integer.
        """
        # Attempt 1: modern API
        modern_getter = getattr(self.model, "get_embedding_dimension", None)
        if callable(modern_getter):
            try:
                dim = modern_getter()
                if isinstance(dim, (int, np.integer)) and not isinstance(dim, bool):
                    return int(dim)
            except Exception:
                pass

        # Attempt 2: legacy API
        legacy_getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(legacy_getter):
            try:
                dim = legacy_getter()
                if isinstance(dim, (int, np.integer)) and not isinstance(dim, bool):
                    return int(dim)
            except Exception:
                pass

        raise VectorizationError(
            f"Could not determine embedding dimension for model '{self.model_name}': "
            "neither get_embedding_dimension() nor get_sentence_embedding_dimension() "
            "returned a valid integer."
        )

    def _is_spawnable_python(self) -> bool:
        """Returns True if ``sys.executable`` is a standard Python launcher
        suitable for spawning the shared model server as a script subprocess.

        Rejects:
        - empty/None interpreter paths,
        - ``pythonw.exe`` / ``pythonw`` (GUI launcher, no console),
        - frozen interpreters (PyInstaller / Nuitka bundles),
        - non-Python ``.exe`` launchers (e.g. ``AITIANA_studio.exe``).
        """
        exe = sys.executable
        if not exe:
            return False
        if getattr(sys, "frozen", False):
            return False
        base = os.path.basename(exe).lower()
        if base in {"pythonw.exe", "pythonw"}:
            return False
        if base.endswith(".exe") and "python" not in base:
            return False
        return True

    @staticmethod
    def _resolve_run_server_path() -> str:
        """Absolute path to ``run_server.py`` shipped next to this vectorizer.

        Resolved from this file's location rather than ``-m`` module name so the
        daemon starts regardless of the spawner's ``cwd`` or whether ``safe_store``
        is pip-installed in the child interpreter.
        """
        return os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..",
                "model_server",
                "run_server.py",
            )
        )

    def _read_last_log_line(self) -> Optional[str]:
        """Returns the last non-empty stripped line of the server log, or None."""
        if not self._server_log_file or not os.path.exists(self._server_log_file):
            return None
        try:
            with open(self._server_log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return lines[-1] if lines else None
        except (IOError, OSError):
            return None

    def _cleanup_shared_server_state(self) -> None:
        """Stops the log streamer, closes the client, and terminates any
        spawned server process. Idempotent and safe to call multiple times."""
        if self._log_streamer_stop is not None:
            try:
                self._log_streamer_stop.set()
            except Exception:
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._server_process is not None:
            try:
                if self._server_process.poll() is None:
                    self._server_process.terminate()
            except Exception:
                pass
            self._server_process = None

    def _init_local_model(self) -> None:
        _ensure_st_class_loaded()
        try:
            if self.reuse_model_in_process:
                reg_key = f"{self.model_name}:{self.cache_folder}"
                self._registry_key = reg_key
                with _REGISTRY_LOCK:
                    if reg_key in _IN_PROCESS_MODEL_REGISTRY:
                        shared_model, count = _IN_PROCESS_MODEL_REGISTRY[reg_key]
                        _IN_PROCESS_MODEL_REGISTRY[reg_key] = (shared_model, count + 1)
                        self.model = shared_model
                        self._dim = self._resolve_model_dimension()
                        ASCIIColors.info(f"Reusing existing in-memory Sentence Transformer model '{self.model_name}' (Active instances: {count + 1})")
                        return

            ASCIIColors.info(f"Loading Sentence Transformer model locally: {self.model_name}")
            st_kwargs = {}
            if self.cache_folder is not None:
                st_kwargs["cache_folder"] = self.cache_folder
            self.model = SentenceTransformer(self.model_name, **st_kwargs)
            self._dim = self._resolve_model_dimension()

            if self.reuse_model_in_process and self._registry_key:
                with _REGISTRY_LOCK:
                    _IN_PROCESS_MODEL_REGISTRY[self._registry_key] = (self.model, 1)

            ASCIIColors.info(f"Model '{self.model_name}' loaded. Dimension: {self._dim}")
        except Exception as e:
            raise VectorizationError(f"Failed to load Sentence Transformer model '{self.model_name}': {e}") from e

    def _init_shared_server(self) -> None:
        try:
            self._init_shared_server_impl()
        except Exception:
            self._cleanup_shared_server_state()
            raise

    def _init_shared_server_impl(self) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient

        if FileLock is None:
            raise ConfigurationError("filelock package is required for shared server mode.")

        temp_dir = tempfile.gettempdir()
        self._token_file = os.path.join(temp_dir, f"safe_store_server_{self.port}.token")
        self._lock_file = os.path.join(temp_dir, f"safe_store_server_{self.port}.lock")
        self._server_log_file = os.path.join(temp_dir, f"safe_store_server_{self.port}.log")

        # Start live real-time console log streamer for server internal events
        if self.stream_server_logs:
            _t = time.monotonic()
            self._start_log_streamer()
            _log_st_timing("_start_log_streamer", _t)

        auth_token: Optional[str] = None
        spawn_lock = FileLock(self._lock_file, timeout=self.SERVER_STARTUP_TIMEOUT)
        # Track a working client across the branches so we do NOT double-handshake
        # after a successful ping (the redundant re-connect is a known multi-second
        # stall on Windows loopback).
        probe_client: Optional[ModelServerClient] = None

        try:
            _t = time.monotonic()
            spawn_lock.acquire(timeout=self.SERVER_STARTUP_TIMEOUT)
            _log_st_timing("spawn_lock.acquire", _t)
            try:
                _t = time.monotonic()
                auth_token = self._read_token_file()
                if auth_token:
                    if not self._is_port_in_use():
                        # Stale token file from a crashed previous daemon: the
                        # port isn't bound, so a ping would just burn the socket
                        # timeout (up to 1.5 s on Windows). Skip it and go
                        # straight to spawn.
                        try:
                            os.remove(self._token_file)
                        except OSError:
                            pass
                        auth_token = None
                    else:
                        # Check if running server is already answering pings
                        client_probe = ModelServerClient(self.host, self.port, auth_token)
                        if client_probe.ping(timeout=1.5):
                            ASCIIColors.info(f"Connected to active shared model server on port {self.port}.")
                            probe_client = client_probe
                        else:
                            auth_token = None
                            try:
                                os.remove(self._token_file)
                            except OSError:
                                pass
                _log_st_timing("token read + existing-daemon probe", _t)

                if not auth_token:
                    _t = time.monotonic()
                    auth_token = self._spawn_server()
                    _log_st_timing("_spawn_server", _t)

                    _t = time.monotonic()
                    probe_client = self._wait_for_server_ready(auth_token)
                    _log_st_timing("_wait_for_server_ready", _t)
            finally:
                spawn_lock.release()
        except Timeout:
            auth_token = self._read_token_file()
            if not auth_token:
                raise VectorizationError("Timeout waiting for shared server spawn lock and no token file found.")

        if not auth_token:
            auth_token = self._read_token_file()
            if not auth_token:
                raise VectorizationError("Failed to obtain auth token for shared model server.")

        _t = time.monotonic()
        if probe_client is not None:
            # The probe/_wait already produced a live client. Reuse it; do not
            # re-probe or re-connect (avoid the Windows loopback double handshake).
            self._client = probe_client
        else:
            # Fallback: no live client was produced above (e.g. token-file based
            # path in the Timeout branch). Use the retrying connector.
            self._client = ModelServerClient(self.host, self.port, auth_token)
            self._connect_with_retries(auth_token)
        _log_st_timing("attach to shared model server", _t)
        ASCIIColors.success(f"Attached to shared model server (model={self.model_name}, port={self.port}).")
        # NOTE: Do NOT probe the dimension here. Previous implementations sent a real
        # `vectorize(["dimension_probe"])` request, which forced the daemon to fully
        # load the SentenceTransformer at client-construction time. `dim` is now a
        # lazy property: it will answer from the daemon's cheap config hint, or from
        # an already-loaded peer, and only fall back to a single probe if neither is
        # available. This keeps STVectorizer() construction O(ms) so the UI panel
        # does not wait on a multi-second model load.

    def _start_log_streamer(self) -> None:
        """Starts a background daemon thread that prints server logs live to the main console."""
        if self._log_streamer_thread is None and self._server_log_file:
            self._log_streamer_stop = threading.Event()
            self._log_streamer_thread = threading.Thread(
                target=self._stream_server_logs_worker,
                args=(self._server_log_file, self._log_streamer_stop),
                daemon=True
            )
            self._log_streamer_thread.start()

    @staticmethod
    def _stream_server_logs_worker(log_path: str, stop_event: threading.Event) -> None:
        # Start reading from the END of the existing log so we only stream NEW
        # lines emitted during this session. Without this, every new
        # STVectorizer instance re-streams the entire historical startup log
        # (model-load weights, ping probes, prior inferences) — which looks
        # exactly like the model being loaded multiple times.
        try:
            last_pos = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        except OSError:
            last_pos = 0

        while not stop_event.is_set():
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        current_size = os.fstat(f.fileno()).st_size
                        if current_size < last_pos:
                            # File was truncated by a fresh spawn — reset.
                            last_pos = 0
                        f.seek(last_pos)
                        lines = f.readlines()
                        last_pos = f.tell()
                        if lines:
                            # Use raw stdout writes (no ANSI mangling) and one
                            # flush per poll instead of per line: print(...,
                            # flush=True) per line is a known multi-second stall
                            # source on Windows consoles and can starve the
                            # constructor thread right after "Attached".
                            for line in lines:
                                text = line.rstrip()
                                if text:
                                    sys.stdout.write("\033[36m" + text + "\033[0m\n")
                            sys.stdout.flush()
                except Exception:
                    pass
            time.sleep(0.1)

    def _connect_with_retries(self, auth_token: str) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient

        start_time = time.time()
        last_error: Optional[Exception] = None
        iteration = 0

        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            iteration += 1
            iter_t = time.monotonic()
            try:
                temp_client = ModelServerClient(self.host, self.port, auth_token)
                if temp_client.ping(timeout=2.0):
                    self._client = temp_client
                    _log_st_timing(f"_connect_with_retries [iter {iteration}] OK", iter_t)
                    return
            except (ConnectionError, OSError, RuntimeError) as e:
                last_error = e
            _log_st_timing(f"_connect_with_retries [iter {iteration}] failed", iter_t)
            time.sleep(self.SERVER_PING_INTERVAL)

        self._dump_server_log()
        raise VectorizationError(f"Failed to connect and ping shared model server: {last_error}")

    def _read_token_file(self) -> Optional[str]:
        if not self._token_file or not os.path.exists(self._token_file):
            return None
        try:
            with open(self._token_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except (IOError, OSError):
            return None

    def _spawn_server(self) -> str:
        if not self._is_spawnable_python():
            raise ConfigurationError(
                f"Interpreter {sys.executable!r} is not a suitable Python launcher "
                "for subprocess mode; cannot spawn shared model server."
            )

        run_server_path = self._resolve_run_server_path()
        if not os.path.isfile(run_server_path):
            raise ConfigurationError(
                f"Shared model server script not found at '{run_server_path}'. "
                "Your safe_store installation is incomplete."
            )

        auth_token = secrets.token_hex(16)

        command = [
            sys.executable, run_server_path,
            "--host", self.host,
            "--port", str(self.port),
            "--model", self.model_name,
            "--token", auth_token,
            "--token-file", self._token_file,
            "--idle-timeout", str(self.idle_timeout),
            "--batch-window", str(self.batch_window),
            "--max-batch-size", str(self.max_batch_size)
        ]

        # Seed the child's PYTHONPATH with the directory containing the
        # `safe_store` package so the daemon's `from safe_store.…` imports
        # resolve even when safe_store is not pip-installed in that interpreter.
        pkg_parent = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(run_server_path)
                )
            )
        )
        child_env = dict(os.environ)
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            pkg_parent + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        )

        print(f"[STVectorizer] Spawning shared model server daemon on port {self.port}...", flush=True)

        log_handle = open(self._server_log_file, 'w', encoding='utf-8')
        try:
            if os.name == "nt":
                self._server_process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(run_server_path),
                    env=child_env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
            else:
                self._server_process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(run_server_path),
                    env=child_env,
                    start_new_session=True
                )
        except Exception as e:
            log_handle.close()
            raise VectorizationError(f"Failed to spawn shared model server: {e}") from e
        finally:
            log_handle.close()

        return auth_token

    def _dump_server_log(self) -> None:
        if self._server_log_file and os.path.exists(self._server_log_file):
            try:
                with open(self._server_log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                if log_content.strip():
                    print(f"\n--- Shared Model Server Log ({self._server_log_file}) ---\n{log_content}\n--- End Log ---\n", flush=True)
            except (IOError, OSError) as e:
                print(f"[!] Could not read server log: {e}", flush=True)

    def _is_port_in_use(self) -> bool:
        # 0.2 s: a bound port answers in microseconds; a not-yet-bound port on
        # Windows can stall on connect. Shorter timeout makes the "not ready
        # yet" case fail faster without penalizing a healthy daemon.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex((self.host, self.port)) == 0

    def _wait_for_server_ready(self, auth_token: str):
        """Poll until the daemon answers a ping, then return the ready client.

        Returning the proven-live client avoids the caller re-running a full
        `_connect_with_retries` handshake (a redundant re-connect that is a
        known multi-second stall on Windows loopback sockets).
        """
        from safe_store.vectorization.model_server.client import ModelServerClient

        start_time = time.time()
        print(f"[STVectorizer] Waiting for shared model server to initialize on port {self.port}...", flush=True)
        iteration = 0
        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            iteration += 1
            iter_t = time.monotonic()
            if self._server_process and self._server_process.poll() is not None:
                self._dump_server_log()
                tail = self._read_last_log_line()
                raise VectorizationError(
                    f"Shared model server process exited prematurely with return code {self._server_process.returncode}. "
                    f"Last server log line: {tail or '<no output>'}"
                )
            if self._is_port_in_use():
                _ping_t = time.monotonic()
                try:
                    temp_client = ModelServerClient(self.host, self.port, auth_token)
                    if temp_client.ping(timeout=2.0):
                        print(f"[STVectorizer] Shared model server is online and ready!", flush=True)
                        _log_st_timing(
                            f"_wait_for_server_ready [iter {iteration}] ping OK",
                            _ping_t,
                        )
                        return temp_client
                except Exception:
                    pass
                _log_st_timing(
                    f"_wait_for_server_ready [iter {iteration}] ping failed",
                    _ping_t,
                )
            time.sleep(self.SERVER_PING_INTERVAL)

        self._dump_server_log()
        raise VectorizationError(f"Timeout waiting for shared model server to bind on port {self.port}.")

    def _fetch_dimension_from_server(self) -> int:
        dummy_vector = self.vectorize(["dimension_probe"])
        if dummy_vector.size == 0:
            raise VectorizationError("Server returned empty vector during dimension probe.")
        return int(dummy_vector.shape[1])

    def get_tokenizer(self) -> Optional[HuggingFaceTokenizerWrapper]:
        if self.model is not None and hasattr(self.model, 'tokenizer'):
            return HuggingFaceTokenizerWrapper(self.model.tokenizer)
        return None

    def supports_late_chunking(self) -> bool:
        return self.model is not None

    def late_chunk_embed(self, text: str, chunk_spans: List[Tuple[int, int]]) -> np.ndarray:
        if self.model is None or not chunk_spans:
            chunk_texts = [text[s:e] for s, e in chunk_spans]
            return self.vectorize(chunk_texts)

        try:
            import torch
            tokenizer = getattr(self.model, 'tokenizer', None)
            if tokenizer is None:
                chunk_texts = [text[s:e] for s, e in chunk_spans]
                return self.vectorize(chunk_texts)

            encoded = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=8192
            )

            offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
            device = self.model.device
            inputs = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                transformer_module = self.model[0]
                outputs = transformer_module.auto_model(**inputs)
                token_embeddings = outputs.last_hidden_state[0]

            vectors = []
            for start_char, end_char in chunk_spans:
                token_indices = []
                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == 0 and tok_end == 0:
                        continue
                    if tok_start < end_char and tok_end > start_char:
                        token_indices.append(tok_idx)

                if token_indices:
                    span_tokens = token_embeddings[token_indices]
                    chunk_vec = span_tokens.mean(dim=0).cpu().numpy()
                    norm = np.linalg.norm(chunk_vec)
                    if norm > 0:
                        chunk_vec = chunk_vec / norm
                else:
                    chunk_vec = self.vectorize([text[start_char:end_char]])[0]

                vectors.append(chunk_vec)

            return np.array(vectors, dtype=self.dtype)

        except Exception as e:
            ASCIIColors.warning(f"Late chunking fallback: {e}")
            chunk_texts = [text[s:e] for s, e in chunk_spans]
            return self.vectorize(chunk_texts)

    def vectorize(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim if self._dim else 1), dtype=self.dtype)

        if self.use_shared_server and self._client is not None:
            try:
                embeddings = self._client.vectorize(texts)
                if not isinstance(embeddings, np.ndarray):
                    raise VectorizationError("Shared server client did not return a NumPy array.")
                if embeddings.dtype != self._dtype:
                    embeddings = embeddings.astype(self._dtype)
                return embeddings
            except Exception as e:
                raise VectorizationError(f"Error during shared server vectorization: {e}") from e

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if not isinstance(embeddings, np.ndarray):
                raise VectorizationError("SentenceTransformer model did not return a NumPy array.")
            if embeddings.dtype != self._dtype:
                embeddings = embeddings.astype(self._dtype)
            return embeddings
        except Exception as e:
            raise VectorizationError(f"Error during sentence-transformer encoding: {e}") from e

    @property
    def dim(self) -> Optional[int]:
        """
        Embedding dimension, resolved lazily.

        Resolution order (cheapest first):
          1. Cached `self._dim` (set by local init, or by an earlier resolution).
          2. The shared daemon's `health()["dim"]` — answers instantly from config
             when the daemon itself hasn't loaded the model, or from the live model
             when a peer already loaded it. Never blocks on model load.
          3. A one-shot `_fetch_dimension_from_server()` probe (which DOES trigger a
             lazy load) — only reached if the daemon cannot report a dimension at all.

        Returns None on transient transport failure rather than raising, so callers
        that merely introspect (e.g. DB diagnostics) are never blocked by a model load.
        """
        if self._dim is not None:
            return self._dim

        client = getattr(self, "_client", None)
        if client is not None:
            try:
                info = client.health()
                if isinstance(info, dict):
                    remote_dim = info.get("dim")
                    if isinstance(remote_dim, int) and remote_dim > 0:
                        self._dim = remote_dim
                        return self._dim
            except Exception:
                pass

        try:
            self._dim = self._fetch_dimension_from_server()
        except Exception:
            return None
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased-v1",
            "all-roberta-large-v1"
        ]

    def shutdown_shared_server(self) -> bool:
        """Sends the explicit shutdown command to stop the shared server."""
        if self._client is not None:
            return self._client.shutdown()
        return False

    @classmethod
    def stop_shared_server(cls, port: int = 8765, host: str = "127.0.0.1", token: Optional[str] = None) -> bool:
        """Class method to shut down a running shared model server on the given port."""
        from safe_store.vectorization.model_server.client import ModelServerClient

        temp_dir = tempfile.gettempdir()
        token_path = os.path.join(temp_dir, f"safe_store_server_{port}.token")

        if token is None and os.path.exists(token_path):
            try:
                with open(token_path, 'r', encoding='utf-8') as f:
                    token = f.read().strip()
            except OSError:
                pass

        if not token:
            print(f"[STVectorizer] No auth token found for shared server on port {port}.", flush=True)
            return False

        client = ModelServerClient(host, port, token)
        res = client.shutdown(timeout=3.0)
        if res:
            print(f"[STVectorizer] Shared model server on {host}:{port} has been shut down.", flush=True)
            if os.path.exists(token_path):
                try: os.remove(token_path)
                except OSError: pass
        return res

    def unload(self) -> None:
        """
        Unloads model weights from RAM/VRAM, cleans up shared server clients,
        and releases PyTorch CUDA/MPS memory back to the operating system.
        """
        if self._log_streamer_stop is not None:
            self._log_streamer_stop.set()

        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

        if self.model is not None:
            should_purge = True
            if self._registry_key and self.reuse_model_in_process:
                with _REGISTRY_LOCK:
                    if self._registry_key in _IN_PROCESS_MODEL_REGISTRY:
                        shared_model, count = _IN_PROCESS_MODEL_REGISTRY[self._registry_key]
                        if count > 1:
                            _IN_PROCESS_MODEL_REGISTRY[self._registry_key] = (shared_model, count - 1)
                            should_purge = False
                        else:
                            del _IN_PROCESS_MODEL_REGISTRY[self._registry_key]

            if should_purge:
                try:
                    if hasattr(self.model, "to"):
                        try:
                            self.model.to("cpu")
                        except Exception:
                            pass
                except Exception:
                    pass

                del self.model
                self.model = None

                # Release PyTorch GPU caches
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except Exception:
                    pass
            else:
                self.model = None

    def close(self) -> None:
        self.unload()

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass