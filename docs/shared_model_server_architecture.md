# Shared Local Model Server (Vectorizer)

The shared model server provides a single-process, TCP-based embedding service
that many `SafeStore` clients can reuse. It avoids loading one `SentenceTransformer`
per process and gives the host application (e.g. lollms) a way to release VRAM
on demand so the LLM/TTI models can take priority.

The implementation lives in `safe_store/vectorization/model_server/`:

- `run_server.py` — CLI entry point and control verbs (start / status / health / load / unload / stop).
- `server.py` — `ModelServer`, the socket daemon with lazy loading and micro-batching.
- `client.py` — `ModelServerClient`, a thread-safe client wrapper.
- `protocol.py` — framing helpers (`send_message`, `recv_message`, `send_array`, `recv_array`).

## 1. Design goals

- **Fast startup.** Starting the daemon must not load a model, import `torch`,
  or allocate GPU memory. Binding the port and writing the token file is all
  the daemon does before going live.
- **On-demand load.** The embedding model is materialized on the first
  `vectorize` request, or explicitly via the `load` control command.
- **On-demand unload.** The host can release the model at any time, and an
  idle timer can release it automatically without killing the daemon.
- **Cross-process sharing.** All local clients benefit from the same resident
  model via a single TCP endpoint.
- **Observability.** Callers can query whether the model is loaded, which
  device it lives on, and roughly how much VRAM/RAM it occupies.

## 2. Architecture

```
┌────────────────────┐     JSON + ndarray over TCP     ┌────────────────────┐
│  ModelServerClient │ ──────────────────────────────▶ │    ModelServer     │
│  (SafeStore / app) │ ◀────────────────────────────── │  (daemon process)  │
└────────────────────┘                                 └────────────────────┘
        │                                                       │
        │  one socket per request, guarded by a Lock            │
        │                                                       ├─► auth token check
        │                                                       ├─► command dispatch
        │                                                       ├─► micro-batch queue
        │                                                       └─► SentenceTransformer
```

- **Transport:** TCP (default `127.0.0.1:5555`).
- **Framing:** JSON control messages via `send_message`/`recv_message`,
  optionally followed by a raw NumPy array via `send_array`/`recv_array` for
  `vectorize` responses.
- **Auth:** every message carries a `token` field compared with
  `hmac.compare_digest`. Requests without a valid token get
  `{"status": "error", "error": "unauthorized"}` and the connection is closed.
- **Concurrency:** the acceptor thread handles one connection at a time; the
  batch loop drains the pending queue and runs a single inference pass across
  all enqueued texts. The model-load path is serialized by an `RLock`.
- **Persistence:** the daemon writes the auth token to a token file so control
  verbs (`--status`, `--health`, `--load`, `--unload`, `--stop`) can reconnect
  without passing `--token` explicitly. The file is removed on clean shutdown.

## 3. Protocol commands

Every request is a JSON object with `token` and `kind`. Supported `kind` values:

| `kind`      | Purpose                                   | Response                                        |
| ----------- | ----------------------------------------- | ----------------------------------------------- |
| `ping`      | Liveness without side effects             | `{"status": "ok", "dim": ..., "model_loaded": ...}` |
| `status`    | Lightweight daemon snapshot               | `{"status": "ok", "pid", "uptime_s", "model_loaded", "pending_requests"}` |
| `health`    | Full state incl. resource usage           | see §4                                           |
| `load`      | Force the model to load unless loaded     | `{"status": "ok", "loaded", "was_loaded", "load_time_ms", ...}` |
| `unload`    | Release model weights, free VRAM/RAM      | `{"status": "ok", "loaded": false, "freed_vram_bytes", "freed_ram_bytes"}` |
| `shutdown`  | Graceful daemon stop                      | `{"status": "ok", "message": "Server shutting down."}` |
| `vectorize` | Enqueue texts for batched embedding       | `{"status": "ok"}` followed by a `float32` ndarray (`send_array`) |

`vectorize` responses are two-frame: the JSON header confirms success, then
raw NumPy bytes carry the embeddings. On server-side error the JSON body has
`status: "error"` and an `error` string, and no array is sent.

## 4. `health` payload

```json
{
  "status": "ok",
  "model_name": "all-MiniLM-L6-v2",
  "model_loaded": true,
  "device": "cuda:0",
  "dtype": "float32",
  "vram_bytes": 431509504,
  "ram_bytes": 138412032,
  "pending_requests": 0,
  "uptime_s": 812.4,
  "idle_s": 12.7,
  "last_used_ts": 1730000000.0,
  "lazy": true,
  "pid": 12345,
  "host": "127.0.0.1",
  "port": 5555
}
```

- `vram_bytes` is `torch.cuda.memory_allocated(device)` for the model's device,
  or `0` on CPU-only builds or when the model is unloaded.
- `ram_bytes` is `Σ numel × element_size` over the model's parameters and
  buffers.
- `idle_s` is time since the last successful encode, or time since startup
  when the model has never been used.
- `model_loaded: false` is a normal, non-error state.

## 5. Server lifecycle

1. **Start.** `run_server.py` parses arguments and constructs `ModelServer`.
   `ModelServer.start()` binds the socket and writes the token file, then
   enters the accept/batch loops. **No model is loaded at this point** unless
   `--eager` was passed.
2. **First request.** A `vectorize` (or explicit `load`) triggers
   `ensure_loaded()`, which instantiates the `SentenceTransformer` under the
   load lock. Concurrent requests wait on that lock and then share the batch.
3. **Idle unload.** After `model_idle_timeout` seconds with no encode, the
   batch loop calls `unload_model()`. The daemon stays alive; the next request
   reloads the model transparently.
4. **Process idle exit (optional).** If `idle_timeout > 0`, the daemon exits
   completely after that many idle seconds, using the same `_cleanup` path
   (unload + token file removal).
5. **Explicit shutdown.** `shutdown` (or `--stop`) sets the shutdown event,
   unloads the model, removes the token file, and prints a clean-termination
   line.

## 6. Client contract

`ModelServerClient` is thread-safe via an internal `threading.Lock`. Each call
opens a fresh connection, sends one request, reads the response, and closes
the socket — no long-lived state on the wire.

| Method                                 | Returns                | Notes                                                      |
| -------------------------------------- | ---------------------- | ---------------------------------------------------------- |
| `vectorize(texts)`                     | `np.ndarray`           | Raises `RuntimeError` on server-side error; 120 s timeout. |
| `ping(timeout=3)`                      | `bool`                 | Never raises.                                              |
| `status(timeout=3)`                    | `Optional[dict]`       | `None` on transport/protocol failure.                      |
| `health(timeout=5)`                    | `Optional[dict]`       | `None` on transport/protocol failure.                      |
| `ensure_loaded(timeout=120)`           | `Optional[dict]`       | `None` on transport/protocol failure.                      |
| `unload_model(timeout=30)`             | `Optional[dict]`       | `None` on transport/protocol failure.                      |
| `shutdown(timeout=5)`                  | `bool`                 | Never raises.                                              |

Because `vectorize` may trigger a lazy load, its timeout is 120 s while
control verbs default to shorter timeouts.

## 7. CLI reference (run_server.py)

| Flag                        | Default              | Meaning                                                   |
| --------------------------- | -------------------- | --------------------------------------------------------- |
| `--host`                    | `127.0.0.1`          | Bind host.                                                |
| `--port`                    | *(required)*         | Port to listen on or connect to.                          |
| `--model`                   | `all-MiniLM-L6-v2`   | SentenceTransformer model name.                           |
| `--token`                   | *(required when starting)* | Auth token.                                        |
| `--token-file`              | temp file            | Path to persist/read the token.                           |
| `--idle-timeout SEC`        | `0` (never)          | Exit the process entirely after N idle seconds.           |
| `--model-idle-timeout SEC`  | `300` (`0` = never)  | Unload weights after N idle seconds, keep daemon alive.   |
| `--eager`                   | off                  | Load the model at startup (legacy behavior).              |
| `--batch-window SEC`        | `0.02`               | Micro-batching window.                                    |
| `--max-batch-size N`        | `64`                 | Cap on queued requests per batch.                         |
| `--status` / `--health` / `--load` / `--unload` / `--stop` | | Mutually exclusive control verbs that print JSON. |

Control verbs run in a lightweight process: they never import `torch` or
allocate GPU memory, and they exit with `0` on success, `2` when the daemon is
unreachable, `3` when no token is available, `4` when the daemon is too old to
support the requested endpoint, and `5` on server-side failure.

## 7.1 Runtime configuration via `VectorizationManager`

`SafeStore` callers pass shared-server behaviour through `vectorizer_config`.
`VectorizationManager.RUNTIME_TRANSPORT_KEYS` (see `safe_store/vectorization/manager.py`)
strips these from the vectorizer identity so they do **not** create a distinct
cached vectorizer:

`use_shared_server`, `shared_server`, `shared_mode`, `shared`, `port`, `host`,
`idle_timeout`, `batch_window`, `max_batch_size`, `stream_server_logs`,
`reuse_model_in_process`, `cache_folder`, `api_key`, `service_key`,
`verify_ssl_certificate`.

> ⚠️ `idle_timeout` here is the **process-level** idle exit (maps to the
> daemon's `--idle-timeout`). The weights-only idle timer (`--model-idle-timeout`,
> default 300 s) is currently **CLI-only** and is not wired through
> `vectorizer_config` — set it on the daemon launch command.

The vectorizer/model identity (`_filter_semantic_config`) therefore only keeps
`model` / `model_name` (normalized to `model`) plus any model-specific knobs
(e.g. `device`), so switching ports won't spawn duplicate entries in the
`VectorizationManager` cache.

## 8. Integrating with a host (e.g. lollms)

- **VRAM arbitration.** Before loading an LLM/TTI model, call
  `client.unload_model()`. The daemon drops the vectorizer and keeps serving
  control commands; call `client.ensure_loaded()` before the next embedding
  request if you want to pre-warm.
- **Observability.** Poll `client.health()` to decide whether it is safe to
  allocate GPU memory elsewhere, and to log `vram_bytes` / `ram_bytes`.
- **Automatic relaxation.** Set `--model-idle-timeout` low (e.g. 60 s) on
  shared hosts so the vectorizer releases VRAM while other workloads run.

## 9. Troubleshooting

- **`{"error": "not_running"}`** — nothing is listening on the port. Start the
  daemon first, or confirm the port/host.
- **`{"error": "unauthorized"}`** — the token does not match the daemon's
  `--token`. Use the token file written at startup or pass `--token` explicitly.
- **`{"status": "unsupported"}`** from a control verb — the running daemon is
  older than the CLI and does not implement that endpoint. Restart it with the
  updated package.
- **First `vectorize` is slow** — that is the lazy model load. Use
  `client.ensure_loaded()` at process start to pre-warm if desired.
- **`vram_bytes` stays `0` while `device` is `cuda`** — the process may not see
  other allocations via `memory_allocated`; the value tracks *this* daemon's
  own usage, which is what matters for unload decisions.