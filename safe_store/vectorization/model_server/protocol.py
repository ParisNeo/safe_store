import io
import json
import struct
import socket
import numpy as np
from typing import Any, Dict

HEADER_SIZE = 8
MAX_MESSAGE_SIZE = 512 * 1024 * 1024


def encode_message(payload: Dict[str, Any]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return struct.pack(">Q", len(body)) + body


def send_message(sock: socket.socket, payload: Dict[str, Any]) -> None:
    sock.sendall(encode_message(payload))


def recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    received = 0
    while received < count:
        chunk = sock.recv(min(65536, count - received))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Dict[str, Any]:
    header = recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(">Q", header)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message exceeds maximum size: {length} bytes")
    return json.loads(recv_exact(sock, length).decode("utf-8"))


def encode_array(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    payload = buffer.getvalue()
    return struct.pack(">Q", len(payload)) + payload


def send_array(sock: socket.socket, arr: np.ndarray) -> None:
    sock.sendall(encode_array(arr))


def recv_array(sock: socket.socket) -> np.ndarray:
    header = recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(">Q", header)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Array exceeds maximum size: {length} bytes")
    data = recv_exact(sock, length)
    return np.load(io.BytesIO(data), allow_pickle=False)