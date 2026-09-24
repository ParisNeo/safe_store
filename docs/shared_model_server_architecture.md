# Shared Model Server Architecture & Hardening Blueprint
**Author:** ParisNeo & Lollms  
**Scope:** Multi-Process AI Model Serving, IPC Architecture, Dynamic Micro-Batching & Security

---

## 1. Executive Summary & Problem Statement

In modern AI service architectures (such as FastAPI/Uvicorn multi-worker setups, Celery task workers, or desktop GUIs with worker processes), hosting deep learning models (SentenceTransformers, LLMs via llama.cpp/transformers, or vision encoders) directly within each application process introduces severe bottlenecks:

1. **Out-of-Memory (OOM) Explosions**: If each worker process loads its own model weights into RAM or GPU VRAM, memory usage scales linearly ($N \times \text{Model Size}$). For four 1.5 GB workers, 6 GB of VRAM is wasted solely on redundant weight copies.
2. **Zero Request Batching**: Concurrent requests arriving from different users or workers are processed sequentially in isolation, failing to utilize GPU tensor cores.
3. **Slow Process Spawning**: Every time an application worker restarts or scales, it incurs a multi-second delay to re-load model weights from disk.

### The Solution: The Self-Spawning Shared Daemon
A lightweight, zero-dependency, local Inter-Process Communication (IPC) daemon running on loopback (`127.0.0.1`):
- **First Instance Wins**: The first process that needs the model spawns the daemon in the background; all subsequent processes attach to it.
- **Single Model Footprint**: Exactly **one** instance of the model resides in memory across the entire machine.
- **Dynamic Micro-Batching**: Requests from independent processes are gathered across an adaptive time window (e.g., 20ms) and executed in a single vector/tensor pass.
- **Persistent Daemon with Explicit Shutdown**: The daemon outlives individual worker processes and stays alive until commanded to terminate via an authenticated shutdown RPC.

---

## 2. High-Level Architecture & Lifecycle

```
[ Application Process 1 ] ──┐
  (FastAPI Worker / User A) │
                            ├─► (Port 8765, TCP Loopback) ──► [ ModelServer Daemon ]
[ Application Process 2 ] ──┤   [HMAC Token Authenticated]     │  - Single Model in VRAM
  (FastAPI Worker / User B) │                                  │  - Dynamic Micro-Batching Loop
                            │                                  │  - Model Forward Pass
[ Application Process N ] ──┘                                  ▼
                                                    [ Special Shutdown Command ]
                                                      (SafeStore.shutdown_shared_vectorizer)
```

### The Lifecycle Flow
1. **Lock Acquisition (`FileLock`)**: When an application process needs the model, it acquires an OS-level file lock (`safe_store_server_<port>.lock`).
2. **Probe Existing Daemon**: The process reads the local token file (`safe_store_server_<port>.token`) and sends a fast `ping` command (timeout $\le 1.5$s).
3. **Self-Spawning**: If no daemon responds, the process spawns `run_server.py` as an independent background daemon and writes a cryptographically secure token (`secrets.token_hex(16)`).
4. **Immediate Port Binding**: The daemon binds the TCP socket **first**, ensuring subsequent processes immediately detect the port as active.
5. **Model Initialization**: The daemon loads model weights into memory while client connections wait in the OS TCP backlog (up to 128 queue length).
6. **Release Lock & Ready**: Once `ping` returns `{"status": "ok"}`, the spawning process releases the lock. All workers can now issue concurrent requests.
7. **Explicit RPC Termination**: At deployment teardown or server maintenance, any authorized process sends `{"kind": "shutdown", "token": "..."}`, allowing the server to finish in-flight batches, close sockets, and exit cleanly.

---

## 3. Dynamic Micro-Batching Algorithm

Traditional sequential serving processes one request at a time. The shared daemon implements **event-driven dynamic micro-batching**:

```python
# Server-side micro-batching loop (Zero CPU busy-wait)
def _batch_loop(self) -> None:
    while not self._shutdown.is_set():
        # 1. Sleep on an OS event until at least one request arrives (0% CPU idle)
        signaled = self._pending_event.wait(timeout=1.0)
        if not signaled:
            if self.idle_timeout > 0 and (time.time() - self._last_activity > self.idle_timeout):
                break # Optional auto-exit if configured
            continue

        # 2. Open a short micro-batch window (e.g., 20ms) to allow concurrent requests to join
        if self.batch_window > 0:
            time.sleep(self.batch_window)

        # 3. Drain pending requests up to max_batch_size
        batch = []
        with self._pending_lock:
            while self._pending and len(batch) < self.max_batch_size:
                batch.append(self._pending.pop(0))
            if not self._pending:
                self._pending_event.clear()

        # 4. Perform a single batched inference pass
        if batch:
            self._last_activity = time.time()
            self._process_batch(batch)
```

### Why this beats individual processing:
- **Zero Busy-Waiting**: The thread sleeps on `threading.Event.wait()`, consuming 0% CPU while idle.
- **Latency vs. Throughput Balance**: A 20ms collection window adds negligible latency to interactive requests while allowing 5–20 concurrent user queries to merge into a single tensor core operation.
- **Fair Slicing**: The server groups all input strings, encodes them in one pass, slices the resulting matrix per client, and delivers the exact response slice back to each client's socket.

---

## 4. Critical Quirks & Pitfalls to Avoid Next Time

During the design and implementation of this architecture, several subtle cross-platform and deep learning quirks were diagnosed and resolved. Keep this checklist handy for any future LOLLMS bindings or daemon services:

### 4.1. Binary Length Framing vs. Magic Headers (The `\x93NUMPY` Bug)
* **The Quirk**: `np.save()` writes a raw binary stream starting with the magic header `b"\x93NUMPY\x01\x00"`. If you send this over a socket without an explicit 8-byte big-endian length prefix, the receiver’s `recv_array()` reads `\x93NUMPY\x01\x00` and interprets it as a 64-bit integer (`struct.unpack(">Q", header)`). In decimal, that is:
  ```text
  10,614,515,162,307,690,752 bytes (> 10 Exabytes!)
  ```
  This immediately triggers a `ValueError: Array exceeds maximum size`.
* **The Rule**: **Always maintain strict protocol symmetry.** Every message—whether a JSON dictionary or a NumPy binary array—must be preceded by an exact 8-byte length prefix:
  ```python
  def encode_array(arr: np.ndarray) -> bytes:
      buffer = io.BytesIO()
      np.save(buffer, arr, allow_pickle=False)
      payload = buffer.getvalue()
      return struct.pack(">Q", len(payload)) + payload # MUST prepend 8-byte header!
  ```

### 4.2. Windows `subprocess.Popen` Creation Flags Conflict
* **The Quirk**: On Windows, combining `subprocess.CREATE_NO_WINDOW` and `subprocess.DETACHED_PROCESS` in `creationflags` violates the Windows Process Creation API. Furthermore, passing `close_fds=True` alongside redirected `stdout=log_handle` strips file handle inheritance, causing the child Python process to crash with `[WinError 6] The handle is invalid` on its first `print()`.
* **The Rule**:
  - On Windows: Use `creationflags = subprocess.CREATE_NO_WINDOW` without `DETACHED_PROCESS` or `close_fds=True`.
  - On POSIX/Linux/macOS: Use `start_new_session=True` with `close_fds=True`.
  - In the parent process: Always call `log_handle.close()` immediately after `Popen()` to avoid Windows file-sharing locks (`WinError 32: The process cannot access the file because it is being used by another process`).

### 4.3. Socket Lifecycle & Stale Connection Recycling
* **The Quirk**: If the server closes the connection after sending the response (`finally: conn.close()`), a client that caches its socket without resetting it will attempt to send subsequent requests over a dead socket, producing `BrokenPipeError` or `ConnectionResetError`.
* **The Rule**: The client must explicitly reset its socket handle after completing every request cycle:
  ```python
  def vectorize(self, texts: List[str]) -> np.ndarray:
      with self._lock:
          try:
              # connect, send request, receive response
              ...
              return arr
          finally:
              self._close_locked() # Cleanly close and reset self._sock to None
  ```

### 4.4. Aggressive Probe Timeouts on Loopback
* **The Quirk**: Defaulting client socket timeouts to 30s or 60s for `ping()` or `shutdown()` causes the application to freeze for an entire minute if an old or un-killable process is holding the port.
* **The Rule**: Local loopback (`127.0.0.1`) control frames (`ping` and `shutdown`) should timeout in **1.5 to 3.0 seconds**. If a local daemon does not respond to a ping in 2 seconds, it is either uninitialized or dead.

### 4.5. Race-Free Socket Binding
* **The Quirk**: If `run_server.py` loads model weights *before* creating its listener socket, the port remains closed for 5–15 seconds during startup. Other worker processes checking `is_port_in_use()` will see the port closed, assume the server failed, delete the token file, and attempt to spawn duplicate servers.
* **The Rule**: **Bind the socket and write the token file first.** Only then proceed to load model weights. Any concurrent process checking the port will see that it is occupied and wait for the ping probe to succeed.

### 4.6. Subprocess Package Name Mismatches (`pm.ensure_packages`)
* **The Quirk**: Distribution names on PyPI often contain hyphens (e.g., `sentence-transformers`), but Python module import names use underscores (`sentence_transformers`). If an automatic package installer checks `find_spec("sentence-transformers")` at the module level, it fails to find it and silently blocks execution running `pip install` in the background on every import!
* **The Rule**: Always use a fast-path try/import before invoking any package manager:
  ```python
  try:
      from sentence_transformers import SentenceTransformer
  except ImportError:
      pm.ensure_packages(["sentence-transformers"])
      from sentence_transformers import SentenceTransformer
  ```

### 4.7. PyTorch / Torchvision C++ Kernel Incompatibility
* **The Quirk**: When `torchvision` is installed with a mismatched PyTorch build, PyTorch's C++ operator registry fails with:
  ```text
  RuntimeError: operator torchvision::nms does not exist
  ```
  Because text embedding models do not use vision kernels, having `torchvision` probe these C++ symbols during `transformers` initialization causes an unhandled fatal exception.
* **The Rule**: Neutralize `torchvision` by removing it from `sys.modules` and assigning `sys.modules["torchvision"] = None`. Python's import system will cleanly inform `transformers` that vision dependencies are absent without triggering C++ binary crashes.

---

## 5. Security Architecture & Hardening Doctrine

When deploying a shared daemon across processes or user accounts on the same machine, follow these security invariants:

### 5.1. Constant-Time Authentication (`hmac.compare_digest`)
- Never use standard string equality (`==`) to verify authentication tokens, as it is vulnerable to timing attacks.
- Generate high-entropy tokens:
  ```python
  auth_token = secrets.token_hex(32) # 256 bits of entropy
  ```
- Authenticate requests in constant time:
  ```python
  if not hmac.compare_digest(request_token, self.auth_token):
      self._send_error(conn, "unauthorized")
      conn.close()
      return
  ```

### 5.2. Strict Loopback Binding (`127.0.0.1`)
- The server must **never** bind to `0.0.0.0` or public network interfaces.
- Binding to `127.0.0.1` ensures that only local processes on the same host can establish TCP connections.

### 5.3. File Permissions on Tokens
- Token files must be written with strict user-only read/write permissions (`0o600`):
  ```python
  fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  with os.fdopen(fd, 'w', encoding='utf-8') as f:
      f.write(token)
  ```
- This prevents other unprivileged users on shared Linux/macOS multi-user servers from reading the authentication token.

### 5.4. Denial-of-Service & Message Cap Protection
- Enforce strict size limits before allocating buffers in memory:
  ```python
  HEADER_SIZE = 8
  MAX_MESSAGE_SIZE = 512 * 1024 * 1024  # 512 MB hard ceiling
  MAX_TEXTS_PER_REQUEST = 10000         # Maximum chunks per batch
  ```
- If a client sends a header indicating a length greater than `MAX_MESSAGE_SIZE`, the socket is closed immediately before attempting to allocate memory.

### 5.5. Safe Serialization (Zero Pickle Vulnerability)
- Never use Python's `pickle` module for inter-process communication, as untrusted data can trigger arbitrary Remote Code Execution (RCE).
- Use **JSON** for control messages and **`allow_pickle=False`** for NumPy arrays:
  ```python
  np.save(buffer, arr, allow_pickle=False)
  np.load(io.BytesIO(data), allow_pickle=False)
  ```

---

## 6. Generalizing to Other LOLLMS Bindings

This architecture can be reused across any LOLLMS binding (e.g., `llama.cpp`, `exllama`, `transformers`, `diffusers`, `whisper`).

### Recommended Protocol Schema for LLM Generation
For text generation models, expand the message protocol:

| Kind | Payload | Response | Notes |
| :--- | :--- | :--- | :--- |
| `"ping"` | `{"token": "..."}` | `{"status": "ok", "model": "..."}` | Health and readiness check. |
| `"vectorize"` | `{"token": "...", "texts": [...]}` | `{"status": "ok"}` + Binary Array | Batched embedding matrix. |
| `"generate"` | `{"token": "...", "prompt": "...", "params": {...}}` | `{"status": "ok", "text": "..."}` | Batched or queued text generation. |
| `"generate_stream"`| `{"token": "...", "prompt": "..."}` | Chunked stream over socket | Real-time token streaming. |
| `"shutdown"` | `{"token": "..."}` | `{"status": "ok"}` | Clean daemon teardown. |

### Boilerplate for Future Bindings
To adapt this for another backend:
1. Create a `ModelServer` subclass implementing `_load_model()` and `_process_batch()`.
2. Keep `client.py` and `protocol.py` identical.
3. In your binding's `__init__.py`, use the same `_init_shared_server()` and `_spawn_server()` helper methods.

---

## 7. Operational Cheat Sheet

### Programmatic Usage (Python)
```python
import safe_store

# Connect to or automatically spawn the shared daemon
store = safe_store.SafeStore(
    "database.db",
    vectorizer_name="st",
    vectorizer_config={
        "model_name": "all-MiniLM-L6-v2",
        "use_shared_server": True, # Enables shared daemon
        "port": 8765,
        "idle_timeout": 0.0        # Stay alive until shutdown command
    }
)

# Shutdown the shared daemon when taking down the application
safe_store.shutdown_shared_vectorizer(port=8765)
```

### Command-Line Usage (CLI)
```bash
# Start a standalone model server on port 8765
python -m safe_store.vectorization.model_server.run_server --port 8765 --model all-MiniLM-L6-v2 --token "my_secret_token"

# Send the special shutdown command to terminate a running server
python -m safe_store.vectorization.model_server.run_server --stop --port 8765
```