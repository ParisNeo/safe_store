import json
import socket
import threading
import numpy as np
from typing import Any, Dict, List, Optional

from .protocol import send_message, recv_message, recv_array


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
                response = self._request_locked({
                    "token": self.auth_token,
                    "kind": "vectorize",
                    "texts": texts,
                }, timeout=30.0)
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