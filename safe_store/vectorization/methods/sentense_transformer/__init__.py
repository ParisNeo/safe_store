# safe_store/vectorization/methods/sentense_transformer/__init__.py
import os
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

# Fast-path import: do not run pip if sentence_transformers is already installed!
try:
    disable_broken_torchvision()
    from sentence_transformers import SentenceTransformer
except (ImportError, ModuleNotFoundError):
    try:
        pm.ensure_packages(["torch", "sentence-transformers"])
        disable_broken_torchvision()
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        trace_exception(e)
        SentenceTransformer = None
except Exception as e:
    trace_exception(e)
    SentenceTransformer = None


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
    SERVER_PING_INTERVAL: float = 0.5

    def __init__(self, model_config: Dict[str, Any], cache_folder: Optional[str] = None, **kwargs):
        super().__init__(vectorizer_name="st")

        disable_broken_torchvision()

        if SentenceTransformer is None:
            raise ConfigurationError(
                "STVectorizer requires 'sentence-transformers'. "
                "Install with: pip install safe_store[sentence-transformers]"
            )

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
            self._init_shared_server()
        else:
            self._init_local_model()

    def _init_local_model(self) -> None:
        try:
            if self.reuse_model_in_process:
                reg_key = f"{self.model_name}:{self.cache_folder}"
                self._registry_key = reg_key
                with _REGISTRY_LOCK:
                    if reg_key in _IN_PROCESS_MODEL_REGISTRY:
                        shared_model, count = _IN_PROCESS_MODEL_REGISTRY[reg_key]
                        _IN_PROCESS_MODEL_REGISTRY[reg_key] = (shared_model, count + 1)
                        self.model = shared_model
                        self._dim = self.model.get_sentence_embedding_dimension()
                        ASCIIColors.info(f"Reusing existing in-memory Sentence Transformer model '{self.model_name}' (Active instances: {count + 1})")
                        return

            ASCIIColors.info(f"Loading Sentence Transformer model locally: {self.model_name}")
            st_kwargs = {}
            if self.cache_folder is not None:
                st_kwargs["cache_folder"] = self.cache_folder
            self.model = SentenceTransformer(self.model_name, **st_kwargs)
            self._dim = self.model.get_sentence_embedding_dimension()

            if self.reuse_model_in_process and self._registry_key:
                with _REGISTRY_LOCK:
                    _IN_PROCESS_MODEL_REGISTRY[self._registry_key] = (self.model, 1)

            ASCIIColors.info(f"Model '{self.model_name}' loaded. Dimension: {self._dim}")
        except Exception as e:
            raise VectorizationError(f"Failed to load Sentence Transformer model '{self.model_name}': {e}") from e

    def _init_shared_server(self) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient

        if FileLock is None:
            raise ConfigurationError("filelock package is required for shared server mode.")

        temp_dir = tempfile.gettempdir()
        self._token_file = os.path.join(temp_dir, f"safe_store_server_{self.port}.token")
        self._lock_file = os.path.join(temp_dir, f"safe_store_server_{self.port}.lock")
        self._server_log_file = os.path.join(temp_dir, f"safe_store_server_{self.port}.log")

        # Start live real-time console log streamer for server internal events
        if self.stream_server_logs:
            self._start_log_streamer()

        auth_token: Optional[str] = None
        spawn_lock = FileLock(self._lock_file, timeout=self.SERVER_STARTUP_TIMEOUT)

        try:
            spawn_lock.acquire(timeout=self.SERVER_STARTUP_TIMEOUT)
            try:
                auth_token = self._read_token_file()
                if auth_token:
                    # Check if running server is already answering pings
                    client_probe = ModelServerClient(self.host, self.port, auth_token)
                    if client_probe.ping(timeout=1.5):
                        ASCIIColors.info(f"Connected to active shared model server on port {self.port}.")
                    else:
                        auth_token = None
                        try:
                            os.remove(self._token_file)
                        except OSError:
                            pass
                
                if not auth_token:
                    auth_token = self._spawn_server()
                    self._wait_for_server_ready(auth_token)
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

        self._client = ModelServerClient(self.host, self.port, auth_token)
        self._connect_with_retries(auth_token)

        ASCIIColors.success(f"Attached to shared model server (model={self.model_name}, port={self.port}).")
        self._dim = self._fetch_dimension_from_server()

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
        last_pos = 0
        while not stop_event.is_set():
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_pos)
                        lines = f.readlines()
                        last_pos = f.tell()
                        for line in lines:
                            text = line.rstrip()
                            if text:
                                # Stream server events in cyan for high visibility
                                print(f"\033[36m{text}\033[0m", flush=True)
                except Exception:
                    pass
            time.sleep(0.05)

    def _connect_with_retries(self, auth_token: str) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient
        
        start_time = time.time()
        last_error: Optional[Exception] = None
        
        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            try:
                temp_client = ModelServerClient(self.host, self.port, auth_token)
                if temp_client.ping(timeout=2.0):
                    self._client = temp_client
                    return
            except (ConnectionError, OSError, RuntimeError) as e:
                last_error = e
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
        auth_token = secrets.token_hex(16)

        server_module = "safe_store.vectorization.model_server.run_server"
        command = [
            sys.executable, "-m", server_module,
            "--host", self.host,
            "--port", str(self.port),
            "--model", self.model_name,
            "--token", auth_token,
            "--token-file", self._token_file,
            "--idle-timeout", str(self.idle_timeout),
            "--batch-window", str(self.batch_window),
            "--max-batch-size", str(self.max_batch_size)
        ]

        print(f"[STVectorizer] Spawning shared model server daemon on port {self.port}...", flush=True)

        log_handle = open(self._server_log_file, 'w', encoding='utf-8')
        try:
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW
                self._server_process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags
                )
            else:
                self._server_process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def _wait_for_server_ready(self, auth_token: str) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient

        start_time = time.time()
        print(f"[STVectorizer] Waiting for shared model server to initialize on port {self.port}...", flush=True)
        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            if self._server_process and self._server_process.poll() is not None:
                self._dump_server_log()
                raise VectorizationError(f"Shared model server process exited prematurely with return code {self._server_process.returncode}.")
            if self._is_port_in_use():
                try:
                    temp_client = ModelServerClient(self.host, self.port, auth_token)
                    if temp_client.ping(timeout=2.0):
                        print(f"[STVectorizer] Shared model server is online and ready!", flush=True)
                        return
                except Exception:
                    pass
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