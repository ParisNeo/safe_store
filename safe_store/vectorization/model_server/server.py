import sys
import time
import hmac
import socket
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .protocol import send_message, send_array, recv_message

IDLE_TIMEOUT_SECONDS = 3600.0
BATCH_WINDOW_SECONDS = 0.05
MAX_BATCH_SIZE = 64
MAX_TEXTS_PER_REQUEST = 10000


class ModelServer:
    """Loads a model once and serves vectorization requests with batching."""

    def __init__(
        self,
        host: str,
        port: int,
        auth_token: str,
        model_name: str,
        model_config: Optional[Dict[str, Any]] = None,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
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
        self._model = None
        self._dim: Optional[int] = None
        self._shutdown = threading.Event()
        self._pending: List[Tuple[socket.socket, Dict[str, Any]]] = []
        self._pending_lock = threading.Lock()
        self._last_activity = time.time()
        self._started_at = time.time()

    def start(self) -> None:
        self._load_model()
        listener = self._create_listener()
        print(f"[model-server] listening on {self.host}:{self.port} model={self.model_name}", flush=True)
        acceptor = threading.Thread(target=self._accept_loop, args=(listener,), daemon=True)
        acceptor.start()
        self._batch_loop()

    def _create_listener(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.settimeout(1.0)
        return listener

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def _accept_loop(self, listener: socket.socket) -> None:
        while not self._shutdown.is_set():
            try:
                conn, _addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle_connection(conn)

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            payload = recv_message(conn)
        except (ConnectionError, OSError, ValueError):
            conn.close()
            return
        if not self._authenticate(payload):
            self._send_error(conn, "unauthorized")
            conn.close()
            return
        kind = payload.get("kind")
        if kind == "ping":
            try:
                send_message(conn, {"status": "ok", "dim": self._dim})
            except OSError:
                pass
            conn.close()
            return
        if kind == "vectorize":
            with self._pending_lock:
                self._pending.append((conn, payload))
            return
        self._send_error(conn, f"unknown kind: {kind}")
        conn.close()

    def _send_error(self, conn: socket.socket, error: str) -> None:
        try:
            send_message(conn, {"status": "error", "error": error})
        except OSError:
            pass

    def _authenticate(self, payload: Dict[str, Any]) -> bool:
        token = payload
        if not isinstance(token, str):
            return False
        return hmac.compare_digest(token, self.auth_token)

    def _batch_loop(self) -> None:
        while not self._shutdown.is_set():
            batch = self._collect_batch()
            if not batch:
                if time.time() - self._last_activity > self.idle_timeout:
                    break
                continue
            self._process_batch(batch)

    def _collect_batch(self) -> List[Tuple[socket.socket, Dict[str, Any]]]:
        deadline = time.time() + BATCH_WINDOW_SECONDS
        batch: List[Tuple[socket.socket, Dict[str, Any]]] = []
        while time.time() < deadline and len(batch) < MAX_BATCH_SIZE:
            with self._pending_lock:
                if self._pending:
                    batch.append(self._pending.pop(0))
                else:
                    break
            if not batch:
                time.sleep(0.001)
        return batch

    def _process_batch(self, batch) -> None:
        texts: List[str] = []
        for _conn, payload in batch:
            texts.extend(payload.get("texts", []))
        if len(texts) > MAX_TEXTS_PER_REQUEST:
            for conn, _payload in batch:
                self._send_error(conn, "too many texts")
                conn.close()
            return
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            embeddings = np.asarray(embeddings, dtype=np.float32)
        except Exception:
            for conn, _payload in batch:
                self._send_error(conn, "model encode failed")
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


if __name__ == "__main__":
    print("Run via run_server.py", file=sys.stderr)
    sys.exit(1)