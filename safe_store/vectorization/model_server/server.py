import sys
import os
import gc
import json
import time
import hmac
import socket
import threading
import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .protocol import send_message, send_array, recv_message

DEFAULT_IDLE_TIMEOUT_SECONDS = 0.0  # 0.0 = stay alive indefinitely until explicit shutdown
DEFAULT_MODEL_IDLE_TIMEOUT_SECONDS = 300.0  # 0.0 = never unload weights on idle
DEFAULT_BATCH_WINDOW_SECONDS = 0.02  # 20ms micro-batching window
DEFAULT_MAX_BATCH_SIZE = 64
MAX_TEXTS_PER_REQUEST = 10000


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


def _torch():
    """Import torch lazily so this module is cheap to import in control mode."""
    import torch  # noqa: PLC0415 - intentional lazy import
    return torch


def _read_hidden_size(config_path: str) -> Optional[int]:
    """Read `hidden_size` (or close variants) from an HF-style config.json."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("hidden_size", "n_embd", "d_model", "embedding_dimension", "dim"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            try:
                return int(value)
            except ValueError:
                pass
    return None


def _hf_cache_roots(cache_folder: Optional[str]) -> List[str]:
    """Candidate directories that may hold an HF / ST model checkpoint on disk."""
    roots: List[str] = []
    if cache_folder:
        roots.append(os.path.expanduser(cache_folder))
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        env_value = os.environ.get(env_var)
        if env_value:
            expanded = os.path.expanduser(env_value)
            roots.append(expanded)
            if env_var == "HF_HOME":
                roots.append(os.path.join(expanded, "hub"))
    home = os.path.expanduser("~")
    roots.extend([
        os.path.join(home, ".cache", "huggingface", "hub"),
        os.path.join(home, ".cache", "huggingface", "transformers"),
        os.path.join(home, ".cache", "torch", "sentence_transformers"),
    ])
    # On Windows `expanduser("~")` is `%USERPROFILE%`, so the explicit `USERPROFILE`
    # entries below only add value for unusual HOME overrides. Keep them for safety.
    if os.name == "nt" and os.environ.get("USERPROFILE"):
        up = os.environ["USERPROFILE"]
        roots.extend([
            os.path.join(up, ".cache", "huggingface", "hub"),
            os.path.join(up, ".cache", "huggingface", "transformers"),
            os.path.join(up, ".cache", "torch", "sentence_transformers"),
        ])
    seen = set()
    unique_roots = []
    for root in roots:
        if root and root not in seen:
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def _model_folder_aliases(model_name: str) -> List[str]:
    """
    All on-disk folder names a given model id may use.

    Covers:
    - the raw id (`all-MiniLM-L6-v2`)
    - the sentence-transformers cache prefix (`sentence-transformers_all-MiniLM-L6-v2`)
    - the older ST underscore alias (`_all-MiniLM-L6-v2`)
    - the HF hub cache convention (`models--sentence-transformers--all-MiniLM-L6-v2`,
      `models--<org>--<name>`)
    """
    aliases: List[str] = [model_name, f"_{model_name}"]
    if "/" in model_name:
        org, _, name = model_name.partition("/")
        aliases.append(f"sentence-transformers_{name}")
        aliases.append(f"_sentence-transformers_{name}")
        aliases.append(f"models--{org}--{name}")
        aliases.append(f"models--sentence-transformers--{name}")
    else:
        aliases.append(f"sentence-transformers_{model_name}")
        aliases.append(f"_sentence-transformers_{model_name}")
        aliases.append(f"models--sentence-transformers--{model_name}")
    seen = set()
    unique = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            unique.append(alias)
    return unique


def _find_dim_in_model_dir(model_dir: str) -> Optional[int]:
    """
    Inspect a model directory and return its embedding dimension, if discoverable.

    Handles:
    - flat layout: `<dir>/config.json`, `<dir>/config_sentence_transformers.json`
    - sentence-transformers saved layout: `<dir>/0_Transformer/config.json`
      (this is where `hidden_size` lives for ST-saved models)
    - any first-level subfolder holding a `config.json`
    """
    if not os.path.isdir(model_dir):
        return None

    # Top-level config.json (HF hub snapshot or plain folder)
    dim = _read_hidden_size(os.path.join(model_dir, "config.json"))
    if dim:
        return dim

    # Top-level sentence-transformers config
    dim = _read_hidden_size(os.path.join(model_dir, "config_sentence_transformers.json"))
    if dim:
        return dim

    # sentence-transformers module subfolders (0_Transformer, 1_Pooling, ...)
    for module_name in ("0_Transformer", "1_Pooling", "2_Dense"):
        dim = _read_hidden_size(os.path.join(model_dir, module_name, "config.json"))
        if dim:
            return dim

    # Bounded scan of immediate subfolders as a last resort (some ST layouts
    # nest the transformer under a differently-numbered module).
    try:
        entries = os.listdir(model_dir)
    except OSError:
        entries = []
    for entry in entries:
        sub = os.path.join(model_dir, entry)
        if os.path.isdir(sub):
            dim = _read_hidden_size(os.path.join(sub, "config.json"))
            if dim:
                return dim
    return None


def _iter_snapshot_dirs(model_dir: str) -> Iterator[str]:
    """Yield snapshot hash directories of an HF hub repo folder."""
    snapshots = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return
    try:
        for entry in os.listdir(snapshots):
            candidate = os.path.join(snapshots, entry)
            if os.path.isdir(candidate):
                yield candidate
    except OSError:
        return


def _resolve_dim_from_config(model_name: str, cache_folder: Optional[str] = None) -> Optional[int]:
    """
    Best-effort embedding-dimension lookup WITHOUT loading any weights.

    Never raises; returns None if nothing usable is found.
    """
    if not model_name:
        return None

    aliases = _model_folder_aliases(model_name)

    for root in _hf_cache_roots(cache_folder):
        if not os.path.isdir(root):
            continue
        for alias in aliases:
            model_dir = os.path.join(root, alias)
            if not os.path.isdir(model_dir):
                continue

            # Direct folder (flat config or sentence-transformers module layout)
            dim = _find_dim_in_model_dir(model_dir)
            if dim:
                return dim

            # HF hub snapshot subtree (models--org--name/snapshots/<hash>/config.json)
            for snapshot_dir in _iter_snapshot_dirs(model_dir):
                dim = _find_dim_in_model_dir(snapshot_dir)
                if dim:
                    return dim

    return None


def _bytes_on_device(model) -> Tuple[int, int]:
    """
    Best-effort (vram_bytes, ram_bytes) accounting for the loaded model.

    ram_bytes is exact for parameters/buffers (numel * element_size). vram_bytes
    is the memory currently allocated on the model's device as reported by torch;
    it is 0 when the model is CPU-resident or torch/CUDA is unavailable.
    """
    if model is None:
        return 0, 0
    try:
        torch = _torch()
    except Exception:
        return 0, 0

    ram_bytes = 0
    device = None
    try:
        for param in model.parameters():
            ram_bytes += param.numel() * param.element_size()
            if device is None:
                device = param.device
    except Exception:
        pass
    try:
        for buf in model.buffers():
            ram_bytes += buf.numel() * buf.element_size()
            if device is None:
                device = buf.device
    except Exception:
        pass

    vram_bytes = 0
    try:
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            vram_bytes = int(torch.cuda.memory_allocated(device))
    except Exception:
        vram_bytes = 0
    return vram_bytes, int(ram_bytes)


class ModelServer:
    """Loads a SentenceTransformer model on demand and serves batched vectorization requests across processes."""

    def __init__(
        self,
        host: str,
        port: int,
        auth_token: str,
        model_name: str,
        model_config: Optional[Dict[str, Any]] = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        batch_window: float = DEFAULT_BATCH_WINDOW_SECONDS,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        token_file: Optional[str] = None,
        lazy: bool = True,
        model_idle_timeout: float = DEFAULT_MODEL_IDLE_TIMEOUT_SECONDS,
    ):
        self.host = host
        port = int(port)
        if not (1 <= port <= 65535):
            raise ValueError(f"Invalid port number: {port}")
        self.port = port
        self.auth_token = auth_token
        self.model_name = model_name
        self.model_config = model_config or {}
        self.idle_timeout = float(idle_timeout)
        self.model_idle_timeout = float(model_idle_timeout)
        self.batch_window = float(batch_window)
        self.max_batch_size = int(max_batch_size)
        self.token_file = token_file
        self.lazy = bool(lazy)

        self._model = None
        self._dim: Optional[int] = None
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None
        # Cheap dimension hint read from on-disk config (no weights required). Lets
        # ping/health/status answer `dim` on a fresh daemon without triggering load.
        self._dim_hint: Optional[int] = _resolve_dim_from_config(
            model_name, self.model_config.get("cache_folder")
        )
        self._load_lock = threading.RLock()

        self._shutdown = threading.Event()
        self._pending: List[Tuple[socket.socket, Dict[str, Any]]] = []
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._last_activity = time.time()
        self._last_used_ts: Optional[float] = None
        self._started_at = time.time()
        self._listener: Optional[socket.socket] = None

    def _effective_dim(self) -> Optional[int]:
        """Exact dim if the model is loaded, otherwise the config-derived hint."""
        return self._dim if self._dim is not None else self._dim_hint

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        # 1. Bind socket and write token file FIRST so port is occupied immediately.
        self._listener = self._create_listener()
        if self.token_file:
            from .run_server import write_token_file
            write_token_file(self.token_file, self.auth_token)

        if self._dim_hint is not None:
            hint_fragment = f", dim_hint={self._dim_hint}"
        else:
            hint_fragment = ", dim_hint=<unresolved: will probe on demand>"
        print(
            f"[model-server] Bound port {self.port} on {self.host}. "
            f"Lazy loading is {'enabled' if self.lazy else 'disabled'}{hint_fragment}.",
            flush=True,
        )

        # 2. Optionally load model eagerly (legacy opt-in).
        if not self.lazy:
            t0 = time.perf_counter()
            self._load_model_locked()
            load_time = time.perf_counter() - t0
            print(
                f"[model-server] Model '{self.model_name}' loaded eagerly in {load_time:.2f}s "
                f"(dim={self._dim}, device={self._device}).",
                flush=True,
            )

        print(
            f"[model-server] Ready to accept requests on {self.host}:{self.port} (PID={os.getpid()})",
            flush=True,
        )

        # 3. Start acceptor thread
        acceptor = threading.Thread(target=self._accept_loop, args=(self._listener,), daemon=True)
        acceptor.start()

        # 4. Run dynamic micro-batching loop (blocks).
        self._batch_loop()

    def _create_listener(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.settimeout(1.0)
        listener.listen(128)
        return listener

    # ------------------------------------------------------------- model I/O

    def _load_model_locked(self) -> None:
        """Instantiate the SentenceTransformer. Caller must hold the load lock.

        Ensures the runtime deps (`torch`, `sentence-transformers`) are installed
        via pipmaster before importing — so a freshly spawned daemon on a cold
        host self-heals instead of raising `ModuleNotFoundError`. Reuses the
        same install helper as the local (in-process) vectorizer so the install
        is a no-op if either side already triggered it.
        """
        if self._model is not None:
            return
        # Import the install helper (cheap — it just reads a module-global flag).
        try:
            from safe_store.vectorization.methods.sentense_transformer import (
                _ensure_st_deps_installed,
            )
        except ImportError:
            # Fallback path when this module is executed outside the package
            # context (e.g. direct `python run_server.py` in the source tree).
            import pipmaster as _pm
            try:
                import torch  # noqa: F401
                import sentence_transformers  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                print(
                    "[model-server] Installing 'torch' + 'sentence-transformers' "
                    "(first use only, this can take 30-90 s)...",
                    flush=True,
                )
                _pm.ensure_packages(["torch", "sentence-transformers"])
        else:
            _ensure_st_deps_installed()

        disable_broken_torchvision()
        from sentence_transformers import SentenceTransformer

        st_kwargs = {}
        if "device" in self.model_config:
            st_kwargs["device"] = self.model_config["device"]
        if "cache_folder" in self.model_config:
            st_kwargs["cache_folder"] = self.model_config["cache_folder"]

        t0 = time.perf_counter()
        self._model = SentenceTransformer(self.model_name, **st_kwargs)
        self._dim = int(self._model.get_sentence_embedding_dimension())

        # Record device/dtype for health reporting.
        try:
            representation = self._model._first_module().auto_model  # type: ignore[attr-defined]
            first_param = next(representation.parameters())
            self._device = str(first_param.device)
            self._dtype = str(first_param.dtype).replace("torch.", "")
        except Exception:
            self._device = str(self.model_config.get("device", "cpu"))
            self._dtype = None

        self._last_used_ts = time.time()
        print(
            f"[model-server] Loaded model '{self.model_name}' in {time.perf_counter() - t0:.2f}s "
            f"(dim={self._dim}, device={self._device}).",
            flush=True,
        )

    def ensure_loaded(self) -> Dict[str, Any]:
        """Idempotent: materialize the model if not present. Returns load metrics."""
        with self._load_lock:
            already_loaded = self._model is not None
            t0 = time.perf_counter()
            self._load_model_locked()
            elapsed = time.perf_counter() - t0
        return {
            "loaded": True,
            "was_loaded": already_loaded,
            "load_time_ms": int(elapsed * 1000),
            "model_name": self.model_name,
            "device": self._device,
            "dim": self._effective_dim(),
        }

    def unload_model(self) -> Dict[str, Any]:
        """Idempotent: release the model and free VRAM/RAM. Returns freed bytes."""
        with self._load_lock:
            if self._model is None:
                return {"loaded": False, "was_loaded": False, "freed_vram_bytes": 0, "freed_ram_bytes": 0}
            vram_before, ram_before = _bytes_on_device(self._model)
            device = self._device
            self._model = None
            self._dim = None
            self._device = None
            self._dtype = None
        gc.collect()
        vram_freed = 0
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram_freed = vram_before
        except Exception:
            vram_freed = vram_before
        print(
            f"[model-server] Model unloaded (device_was={device}, freed_vram={vram_freed}B, freed_ram={ram_before}B).",
            flush=True,
        )
        return {
            "loaded": False,
            "was_loaded": True,
            "freed_vram_bytes": int(vram_freed),
            "freed_ram_bytes": int(ram_before),
        }

    def health(self) -> Dict[str, Any]:
        """Non-blocking snapshot of the server + model state."""
        with self._load_lock:
            loaded = self._model is not None
            if loaded:
                vram_bytes, ram_bytes = _bytes_on_device(self._model)
            else:
                vram_bytes, ram_bytes = 0, 0
            device = self._device if loaded else None
            dtype = self._dtype if loaded else None
            last_used = self._last_used_ts

        now = time.time()
        idle_s = (now - last_used) if last_used else (now - self._started_at)
        with self._pending_lock:
            pending = len(self._pending)

        return {
            "status": "ok",
            "model_name": self.model_name,
            "model_loaded": loaded,
            "dim": self._effective_dim(),
            "device": device,
            "dtype": dtype,
            "vram_bytes": int(vram_bytes),
            "ram_bytes": int(ram_bytes),
            "pending_requests": pending,
            "uptime_s": round(now - self._started_at, 3),
            "idle_s": round(idle_s, 3),
            "last_used_ts": last_used,
            "lazy": self.lazy,
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
        }

    def status(self) -> Dict[str, Any]:
        """Lightweight liveness snapshot (no model accounting)."""
        with self._pending_lock:
            pending = len(self._pending)
        return {
            "status": "ok",
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "uptime_s": round(time.time() - self._started_at, 3),
            "model_loaded": self._model is not None,
            "dim": self._effective_dim(),
            "pending_requests": pending,
        }

    # ------------------------------------------------------------ networking

    def _accept_loop(self, listener: socket.socket) -> None:
        while not self._shutdown.is_set():
            try:
                conn, addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_connection(conn, addr)

    def _handle_connection(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        try:
            payload = recv_message(conn)
        except (ConnectionError, OSError, ValueError):
            conn.close()
            return

        if not self._authenticate(payload):
            print(f"[model-server] Rejected unauthorized connection from {addr[0]}:{addr[1]}", flush=True)
            self._send_error(conn, "unauthorized")
            conn.close()
            return

        self._last_activity = time.time()
        kind = payload.get("kind")

        if kind == "ping":
            try:
                send_message(conn, {
                    "status": "ok",
                    "dim": self._effective_dim(),
                    "model_loaded": self._model is not None,
                })
            except OSError:
                pass
            conn.close()
            return

        if kind == "status":
            self._reply_dict(conn, self.status())
            return

        if kind == "health":
            self._reply_dict(conn, self.health())
            return

        if kind == "load":
            try:
                result = self.ensure_loaded()
                self._reply_dict(conn, {"status": "ok", **result})
            except Exception as exc:  # noqa: BLE001 - surface to the caller
                self._send_error(conn, f"load failed: {exc}")
            return

        if kind == "unload":
            try:
                result = self.unload_model()
                self._reply_dict(conn, {"status": "ok", **result})
            except Exception as exc:  # noqa: BLE001
                self._send_error(conn, f"unload failed: {exc}")
            return

        if kind == "shutdown":
            print(f"[model-server] Shutdown command received from {addr[0]}:{addr[1]}. Tearing down...", flush=True)
            try:
                send_message(conn, {"status": "ok", "message": "Server shutting down."})
            except OSError:
                pass
            conn.close()
            self.shutdown()
            return

        if kind == "vectorize":
            text_count = len(payload.get("texts", []))
            print(f"[model-server] Enqueued vectorization request from {addr[0]}:{addr[1]} ({text_count} text chunks)", flush=True)
            with self._pending_lock:
                self._pending.append((conn, payload))
                self._pending_event.set()
            return

        self._send_error(conn, f"unknown kind: {kind}")
        conn.close()

    def _reply_dict(self, conn: socket.socket, payload: Dict[str, Any]) -> None:
        try:
            send_message(conn, payload)
        except OSError:
            pass
        finally:
            conn.close()

    def _send_error(self, conn: socket.socket, error: str) -> None:
        try:
            send_message(conn, {"status": "error", "error": error})
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _authenticate(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        token = payload.get("token")
        if not isinstance(token, str):
            return False
        return hmac.compare_digest(token, self.auth_token)

    # ------------------------------------------------------------- batching

    def _batch_loop(self) -> None:
        while not self._shutdown.is_set():
            signaled = self._pending_event.wait(timeout=1.0)
            if not signaled:
                now = time.time()
                # Process-level idle exit takes precedence.
                if self.idle_timeout > 0 and (now - self._last_activity > self.idle_timeout):
                    print(f"[model-server] Process idle timeout of {self.idle_timeout}s reached. Shutting down.", flush=True)
                    break
                # Weight-only idle unload (process stays alive, model reloads on demand).
                if (
                    self.model_idle_timeout > 0
                    and self._model is not None
                    and self._last_used_ts is not None
                    and (now - self._last_used_ts > self.model_idle_timeout)
                ):
                    try:
                        self.unload_model()
                    except Exception as exc:  # noqa: BLE001 - never kill the loop on unload failure
                        print(f"[model-server] Idle unload failed: {exc}", flush=True)
                continue

            batch = self._collect_batch()
            if batch:
                self._last_activity = time.time()
                self._process_batch(batch)

        self._cleanup()

    def _collect_batch(self) -> List[Tuple[socket.socket, Dict[str, Any]]]:
        if self.batch_window > 0:
            time.sleep(self.batch_window)

        batch: List[Tuple[socket.socket, Dict[str, Any]]] = []
        with self._pending_lock:
            while self._pending and len(batch) < self.max_batch_size:
                batch.append(self._pending.pop(0))

            if not self._pending:
                self._pending_event.clear()

        return batch

    def _process_batch(self, batch: List[Tuple[socket.socket, Dict[str, Any]]]) -> None:
        texts: List[str] = []
        for _conn, payload in batch:
            texts.extend(payload.get("texts", []))

        if len(texts) > MAX_TEXTS_PER_REQUEST:
            for conn, _payload in batch:
                self._send_error(conn, "too many texts")
            return

        # Lazy load on first real embedding work.
        try:
            self.ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            print(f"[model-server] Failed to load model for request: {exc}", flush=True)
            for conn, _payload in batch:
                self._send_error(conn, f"model load failed: {exc}")
            return

        client_count = len(batch)
        print(f"[model-server] Dynamic batch formed: {client_count} client(s) with {len(texts)} total text chunk(s). Running inference...", flush=True)

        t0 = time.perf_counter()
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            encode_time = time.perf_counter() - t0
            self._last_used_ts = time.time()
            print(f"[model-server] Inferred {len(texts)} embeddings in {encode_time:.4f}s. Dispatching to {client_count} client socket(s)...", flush=True)
        except Exception as e:
            print(f"[model-server] Error during model encoding: {e}", flush=True)
            for conn, _payload in batch:
                self._send_error(conn, f"model encode failed: {e}")
            return

        offset = 0
        for conn, payload in batch:
            count = len(payload.get("texts", []))
            chunk = embeddings[offset:offset + count]
            offset += count
            try:
                send_message(conn, {"status": "ok"})
                send_array(conn, chunk)
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    # ----------------------------------------------------------- shutdown

    def shutdown(self) -> None:
        self._shutdown.set()
        self._pending_event.set()
        if self._listener:
            try:
                self._listener.close()
            except OSError:
                pass

    def _cleanup(self) -> None:
        try:
            self.unload_model()
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            print(f"[model-server] Cleanup unload failed: {exc}", flush=True)
        if self.token_file and os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
            except OSError:
                pass
        print(f"[model-server] Server daemon on {self.host}:{self.port} has terminated cleanly.", flush=True)


if __name__ == "__main__":
    print("Run via run_server.py", file=sys.stderr)
    sys.exit(1)