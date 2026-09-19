# safe_store/vectorization/methods/sentense_transformer/__init__.py
import os
import time
import socket
import subprocess
import sys
import secrets
import tempfile
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

try:
    pm.ensure_packages(["torch", "sentence-transformers"])
    try:
        import torchvision
    except Exception:
        sys.modules["torchvision"] = None

    from sentence_transformers import SentenceTransformer
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
    """Vectorizes text using models from the sentence-transformers library."""

    DEFAULT_MODEL: str = "all-MiniLM-L6-v2"
    DEFAULT_HOST: str = "127.0.0.1"
    SERVER_STARTUP_TIMEOUT: float = 180.0
    SERVER_PING_INTERVAL: float = 0.5

    def __init__(self, model_config: Dict[str, Any], cache_folder: Optional[str] = None, **kwargs):
        super().__init__(vectorizer_name="st")

        if SentenceTransformer is None:
            raise ConfigurationError(
                "STVectorizer requires 'sentence-transformers'. "
                "Install with: pip install safe_store[sentence-transformers]"
            )

        self.model_name: str = model_config.get("model_name") or model_config.get("model", self.DEFAULT_MODEL)
        if not self.model_name:
            raise ConfigurationError("STVectorizer config must include a 'model_name' or 'model' key.")

        self.use_shared_server: bool = bool(model_config.get("use_shared_server", False))
        self.port: int = int(model_config.get("port", 8765))
        self.host: str = str(model_config.get("host", self.DEFAULT_HOST))
        self.cache_folder = cache_folder

        self._server_process: Optional[subprocess.Popen] = None
        self._client = None
        self._dim: Optional[int] = None
        self._dtype: np.dtype = np.dtype(np.float32)
        self.model: Optional[Any] = None
        self._token_file: Optional[str] = None
        self._lock_file: Optional[str] = None
        self._server_log_file: Optional[str] = None

        if self.use_shared_server:
            self._init_shared_server()
        else:
            self._init_local_model()

    def _init_local_model(self) -> None:
        try:
            ASCIIColors.info(f"Loading Sentence Transformer model locally: {self.model_name}")
            st_kwargs = {}
            if self.cache_folder is not None:
                st_kwargs["cache_folder"] = self.cache_folder
            self.model = SentenceTransformer(self.model_name, **st_kwargs)
            self._dim = self.model.get_sentence_embedding_dimension()
            ASCIIColors.info(f"Model '{self.model_name}' loaded. Dimension: {self._dim}")
        except Exception as e:
            raise VectorizationError(f"Failed to load Sentence Transformer model '{self.model_name}': {e}") from e

    def _init_shared_server(self) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient

        if FileLock is None:
            raise ConfigurationError("filelock package is required for shared server mode.")

        self._token_file = os.path.join(tempfile.gettempdir(), f"safe_store_server_{self.port}.token")
        self._lock_file = os.path.join(tempfile.gettempdir(), f"safe_store_server_{self.port}.lock")
        self._server_log_file = os.path.join(tempfile.gettempdir(), f"safe_store_server_{self.port}.log")

        auth_token: Optional[str] = None
        spawn_lock = FileLock(self._lock_file, timeout=self.SERVER_STARTUP_TIMEOUT)

        try:
            spawn_lock.acquire(timeout=self.SERVER_STARTUP_TIMEOUT)
            try:
                auth_token = self._read_token_file()
                if auth_token:
                    if not self._is_port_in_use():
                        auth_token = None
                        try:
                            os.remove(self._token_file)
                        except OSError:
                            pass
                
                if not auth_token:
                    auth_token = self._spawn_server()
                    self._wait_for_server_ready()
                    self._verify_server_ping(auth_token)
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

        ASCIIColors.success(f"Connected to shared model server (model={self.model_name}).")
        self._dim = self._fetch_dimension_from_server()

    def _connect_with_retries(self, auth_token: str) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient
        
        start_time = time.time()
        last_error: Optional[Exception] = None
        
        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            try:
                temp_client = ModelServerClient(self.host, self.port, auth_token)
                temp_client.connect()
                if temp_client.ping():
                    self._client = temp_client
                    return
                temp_client.close()
            except (ConnectionError, OSError, RuntimeError) as e:
                last_error = e
            time.sleep(self.SERVER_PING_INTERVAL)
            
        self._dump_server_log()
        raise VectorizationError(f"Failed to connect and ping shared model server: {last_error}")

    def _verify_server_ping(self, auth_token: str) -> None:
        from safe_store.vectorization.model_server.client import ModelServerClient
        
        start_time = time.time()
        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            if self._server_process and self._server_process.poll() is not None:
                self._dump_server_log()
                raise VectorizationError("Shared model server process exited prematurely during ping verification.")
            
            try:
                temp_client = ModelServerClient(self.host, self.port, auth_token)
                temp_client.connect()
                ping_ok = temp_client.ping()
                temp_client.close()
                if ping_ok:
                    return
            except (ConnectionError, OSError, RuntimeError):
                pass
            time.sleep(self.SERVER_PING_INTERVAL)
            
        self._dump_server_log()
        raise VectorizationError("Timeout waiting for shared model server to respond to ping.")

    def _read_token_file(self) -> Optional[str]:
        if not os.path.exists(self._token_file):
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
            "--token-file", self._token_file
        ]

        log_handle = open(self._server_log_file, 'w', encoding='utf-8')
        try:
            self._server_process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
        except Exception as e:
            log_handle.close()
            raise VectorizationError(f"Failed to spawn shared model server: {e}") from e

        return auth_token

    def _dump_server_log(self) -> None:
        if self._server_log_file and os.path.exists(self._server_log_file):
            try:
                with open(self._server_log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                if log_content.strip():
                    ASCIIColors.error(f"--- Shared Model Server Log ---\n{log_content}\n--- End Log ---")
            except (IOError, OSError):
                pass

    def _is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def _wait_for_server_ready(self) -> None:
        start_time = time.time()
        while time.time() - start_time < self.SERVER_STARTUP_TIMEOUT:
            if self._server_process and self._server_process.poll() is not None:
                self._dump_server_log()
                raise VectorizationError("Shared model server process exited prematurely during startup.")
            if self._is_port_in_use():
                time.sleep(1.0)
                return
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
            return np.empty((0, self.dim if self._dim else 1), dtype=self.dtype)

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

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None