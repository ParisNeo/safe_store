import json
import socket
import threading
import numpy as np
from typing import Any, Dict, List, Optional

from .protocol import send_message, recv_message, recv_array


__doc__ = """\
Thread-safe client for the SafeStore local shared model server.

Supported control verbs (all sent as JSON with an auth token; the server
closes the socket after one request, so the client reconnects per call):

    vectorize(texts)      -> np.ndarray           ; raises RuntimeError on server error
    ping(timeout=3)       -> bool                 ; liveness, never raises
    status(timeout=3)     -> Optional[dict]       ; lightweight snapshot
    health(timeout=5)     -> Optional[dict]       ; full state (model_loaded/device/vram/ram)
    ensure_loaded()       -> Optional[dict]       ; force lazy load; up to 120s
    unload_model()        -> Optional[dict]       ; release weights; returns freed bytes
    shutdown(timeout=5)   -> bool                 ; remote graceful stop

`health`, `status`, `ensure_loaded`, `unload_model` return ``None`` on
transport/protocol failure. ``vectorize`` may be slow on the first call because
the daemon lazily loads the embedding model; the client allows up to 120 s.

See docs:
    docs/shared_model_server_architecture.md
"""


class ModelServerClient:
    """Thread-safe client for interacting with the local shared model server."""

    def __init__(self, host: str, port: int, auth_token: str):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self, timeout: float = 10.0) -> None:
        with self._lock:
            self._connect_locked(timeout=timeout)

    def _connect_locked(self, timeout: float = 10.0) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((self.host, self.port))
        self._sock = sock

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _request_locked(self, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        self._close_locked()
        self._connect_locked(timeout=timeout)
        assert self._sock is not None
        send_message(self._sock, payload)
        return recv_message(self._sock)

    def vectorize(self, texts: List[str]) -> np.ndarray:
        with self._lock:
            try:
                # First call may trigger a lazy load on the server; give it headroom.
                response = self._request_locked({
                    "token": self.auth_token,
                    "kind": "vectorize",
                    "texts": texts,
                }, timeout=120.0)
                if response.get("status") != "ok":
                    raise RuntimeError(f"Server error: {response.get('error', 'Unknown')}")
                arr = recv_array(self._sock)
                return arr
            finally:
                self._close_locked()

    def ping(self, timeout: float = 3.0) -> bool:
        with self._lock:
            try:
                response = self._request_locked({
                    "token": self.auth_token,
                    "kind": "ping"
                }, timeout=timeout)
                return response.get("status") == "ok"
            except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
                return False
            finally:
                self._close_locked()

    def status(self, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Lightweight liveness snapshot. Returns None if unavailable."""
        return self._simple_request("status", timeout=timeout)

    def health(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Full server + model health snapshot. Returns None on failure."""
        return self._simple_request("health", timeout=timeout)

    def dim(self, timeout: float = 3.0) -> Optional[int]:
        """Return the server-reported embedding dimension (cheap: no model load required)."""
        info = self.health(timeout=timeout)
        if not info:
            return None
        value = info.get("dim")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def ensure_loaded(self, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        """Ask the server to load the vectorizer now. Returns load metrics."""
        return self._simple_request("load", timeout=timeout)

    def unload_model(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Ask the server to release the vectorizer. Returns freed-byte metrics."""
        return self._simple_request("unload", timeout=timeout)

    def _simple_request(self, kind: str, timeout: float) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                return self._request_locked({
                    "token": self.auth_token,
                    "kind": kind,
                }, timeout=timeout)
            except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
                return None
            finally:
                self._close_locked()

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Sends the explicit shutdown command to stop the remote model server."""
        with self._lock:
            try:
                response = self._request_locked({
                    "token": self.auth_token,
                    "kind": "shutdown"
                }, timeout=timeout)
                return response.get("status") == "ok"
            except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
                return False
            finally:
                self._close_locked()