import sys
import os
import time
import hmac
import socket
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .protocol import send_message, send_array, recv_message

DEFAULT_IDLE_TIMEOUT_SECONDS = 0.0  # 0.0 = stay alive indefinitely until explicit shutdown
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


class ModelServer:
    """Loads a SentenceTransformer model once and serves batched vectorization requests across processes."""

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
        token_file: Optional[str] = None
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
        self.batch_window = float(batch_window)
        self.max_batch_size = int(max_batch_size)
        self.token_file = token_file

        self._model = None
        self._dim: Optional[int] = None
        self._shutdown = threading.Event()
        self._pending: List[Tuple[socket.socket, Dict[str, Any]]] = []
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()
        self._last_activity = time.time()
        self._started_at = time.time()
        self._listener: Optional[socket.socket] = None

    def start(self) -> None:
        # 1. Bind socket and write token file FIRST so port is occupied immediately
        self._listener = self._create_listener()
        if self.token_file:
            from .run_server import write_token_file
            write_token_file(self.token_file, self.auth_token)

        print(f"[model-server] Bound port {self.port} on {self.host}. Starting model loading...", flush=True)

        # 2. Load model with timing
        t0 = time.perf_counter()
        self._load_model()
        load_time = time.perf_counter() - t0
        print(f"[model-server] Model '{self.model_name}' loaded in {load_time:.2f}s! Vector dimension: {self._dim}", flush=True)
        print(f"[model-server] Ready to accept vectorization requests on {self.host}:{self.port} (PID={os.getpid()})", flush=True)

        # 3. Start acceptor thread
        acceptor = threading.Thread(target=self._accept_loop, args=(self._listener,), daemon=True)
        acceptor.start()

        # 4. Run dynamic micro-batching loop
        self._batch_loop()

    def _create_listener(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.settimeout(1.0)
        listener.listen(128)
        return listener

    def _load_model(self):
        disable_broken_torchvision()
        from sentence_transformers import SentenceTransformer

        st_kwargs = {}
        if "device" in self.model_config:
            st_kwargs["device"] = self.model_config["device"]
        if "cache_folder" in self.model_config:
            st_kwargs["cache_folder"] = self.model_config["cache_folder"]

        self._model = SentenceTransformer(self.model_name, **st_kwargs)
        self._dim = int(self._model.get_sentence_embedding_dimension())

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
            print(f"[model-server] Received ping probe from {addr[0]}:{addr[1]} -> pong (dim={self._dim})", flush=True)
            try:
                send_message(conn, {"status": "ok", "dim": self._dim})
            except OSError:
                pass
            conn.close()
            return

        if kind == "shutdown":
            print(f"[model-server] Special shutdown command received from {addr[0]}:{addr[1]}. Initiating daemon teardown...", flush=True)
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

    def _send_error(self, conn: socket.socket, error: str) -> None:
        try:
            send_message(conn, {"status": "error", "error": error})
        except OSError:
            pass

    def _authenticate(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        token = payload.get("token")
        if not isinstance(token, str):
            return False
        return hmac.compare_digest(token, self.auth_token)

    def _batch_loop(self) -> None:
        while not self._shutdown.is_set():
            signaled = self._pending_event.wait(timeout=1.0)
            if not signaled:
                if self.idle_timeout > 0 and (time.time() - self._last_activity > self.idle_timeout):
                    print(f"[model-server] Idle timeout of {self.idle_timeout}s reached with no traffic. Shutting down.", flush=True)
                    break
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
                conn.close()
            return

        client_count = len(batch)
        print(f"[model-server] Dynamic batch formed: {client_count} client(s) with {len(texts)} total text chunk(s). Running inference...", flush=True)

        t0 = time.perf_counter()
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            encode_time = time.perf_counter() - t0
            print(f"[model-server] Inferred {len(texts)} embeddings in {encode_time:.4f}s. Dispatching results to {client_count} client sockets...", flush=True)
        except Exception as e:
            print(f"[model-server] Error during model encoding: {e}", flush=True)
            for conn, _payload in batch:
                self._send_error(conn, f"model encode failed: {e}")
                conn.close()
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
                conn.close()

    def shutdown(self) -> None:
        self._shutdown.set()
        self._pending_event.set()
        if self._listener:
            try:
                self._listener.close()
            except OSError:
                pass

    def _cleanup(self) -> None:
        if self.token_file and os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
            except OSError:
                pass
        print(f"[model-server] Server daemon on {self.host}:{self.port} has terminated cleanly.", flush=True)


if __name__ == "__main__":
    print("Run via run_server.py", file=sys.stderr)
    sys.exit(1)