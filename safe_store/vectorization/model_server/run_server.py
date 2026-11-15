# safe_store/vectorization/model_server/run_server.py
"""
CLI entry point for the SafeStore local shared model server.

This module is intentionally cheap to import and cheap to run in "control"
mode: it never imports torch, sentence-transformers, or allocates any GPU
memory unless the caller explicitly starts the daemon (or issues `--load`
against a running daemon, in which case the *daemon* is what pays the cost).

Control verbs (fast, no model allocation in this process):
    --status          Lightweight liveness probe (running, pid, uptime).
    --health          Full JSON health incl. model_loaded / device / vram / ram.
    --load            Ask the running daemon to materialize the vectorizer now.
    --unload          Ask the running daemon to release the vectorizer (VRAM/RAM).
    --stop/--shutdown Ask the running daemon to exit.

Daemon mode (no verb, or --serve):
    Starts the server, binds the socket, writes the token file and returns to
    the accept loop. The embedder is loaded lazily on the first embed/vectorize
    request unless --eager is passed. Runtime deps ('torch', 'sentence-transformers')
    are auto-installed via pipmaster on first model load — no manual pre-install
    is required. Two independent idle timers are honored:
      --model-idle-timeout  Unload weights only; process stays alive/serving.
      --idle-timeout        Exit the process entirely (legacy behaviour).

Backward compatibility:
    All pre-existing flags keep their meaning. The only observable change for
    existing clients is that the very first embed request pays a one-time
    model-load latency.

Docs:
    docs/shared_model_server_architecture.md
"""
import argparse
import json
import os
import sys
import time
import socket


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

# Actively ensure the project root is importable before resolving siblings.
# Without this, subprocess spawners running with a non-project cwd (e.g. %TEMP%)
# can fail with "No module named safe_store.vectorization.model_server.run_server".
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Resolve imports whether this module is executed as:
#   python -m safe_store.vectorization.model_server.run_server   (relative imports)
#   python <abs_path>/run_server.py                              (direct file execution)
try:
    from .server import ModelServer
    from .client import ModelServerClient
except ImportError:
    from safe_store.vectorization.model_server.server import ModelServer
    from safe_store.vectorization.model_server.client import ModelServerClient


CONTROL_VERBS = ("status", "health", "load", "unload", "stop")


def default_token_path(port: int) -> str:
    tmp = os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp"))
    return os.path.join(tmp, f"safe_store_server_{port}.token")


def write_token_file(token_file: str, token: str) -> None:
    token_dir = os.path.dirname(token_file)
    if token_dir:
        os.makedirs(token_dir, exist_ok=True)
    for _ in range(10):
        try:
            fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(token)
            return
        except (IOError, OSError):
            time.sleep(0.1)
    raise OSError(f"Failed to write token file: {token_file}")


def read_token_file(token_file: str) -> str:
    with open(token_file, 'r', encoding='utf-8') as f:
        return f.read().strip()


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _emit_json(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    sys.exit(exit_code)


def _resolve_token(args: argparse.Namespace, token_path: str) -> str | None:
    """Resolve the auth token from --token, then the token file, else None."""
    if args.token:
        return args.token
    if os.path.exists(token_path):
        try:
            return read_token_file(token_path)
        except OSError as exc:
            print(f"[!] Could not read token file '{token_path}': {exc}", file=sys.stderr, flush=True)
    return None


def _require_client(args: argparse.Namespace, token_path: str) -> ModelServerClient:
    """Build a connected ModelServerClient or exit with a structured error."""
    if not is_port_in_use(args.host, args.port):
        _emit_json(
            {"error": "not_running", "host": args.host, "port": args.port},
            exit_code=2,
        )
    token = _resolve_token(args, token_path)
    if not token:
        _emit_json(
            {"error": "missing_token", "hint": "Provide --token or run on a host with a token file."},
            exit_code=3,
        )
    return ModelServerClient(args.host, args.port, token)


def _call_optional(client: ModelServerClient, method_name: str) -> bool:
    """True if the client exposes the given control method."""
    return callable(getattr(client, method_name, None))


def handle_status(args: argparse.Namespace, token_path: str) -> None:
    """Liveness probe. Works even if the daemon lacks a rich status endpoint."""
    running = is_port_in_use(args.host, args.port)
    payload = {"running": running, "host": args.host, "port": args.port}
    if not running:
        _emit_json(payload, exit_code=2)
    client = _require_client(args, token_path)
    if _call_optional(client, "status"):
        try:
            extra = client.status() or {}
            if isinstance(extra, dict):
                payload.update(extra)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            payload["status_error"] = str(exc)
    _emit_json(payload, exit_code=0)


def handle_health(args: argparse.Namespace, token_path: str) -> None:
    """Full health. Always exits 0 with a JSON payload when socket is up."""
    if not is_port_in_use(args.host, args.port):
        _emit_json(
            {
                "status": "unreachable",
                "model_name": None,
                "model_loaded": False,
                "device": None,
                "dtype": None,
                "vram_bytes": 0,
                "ram_bytes": 0,
                "pending_requests": 0,
                "uptime_s": 0.0,
                "idle_s": None,
                "last_used_ts": None,
                "error": "not_running",
            },
            exit_code=2,
        )
    client = _require_client(args, token_path)
    if not _call_optional(client, "health"):
        _emit_json(
            {
                "status": "unsupported",
                "model_loaded": None,
                "error": "server does not implement health(); upgrade the daemon.",
            },
            exit_code=4,
        )
    try:
        payload = client.health() or {}
    except Exception as exc:  # noqa: BLE001
        _emit_json({"status": "error", "error": str(exc)}, exit_code=5)
    if not isinstance(payload, dict):
        _emit_json({"status": "error", "error": "malformed health payload"}, exit_code=5)
    payload.setdefault("status", "ok")
    _emit_json(payload, exit_code=0)


def handle_load(args: argparse.Namespace, token_path: str) -> None:
    """Force-load the vectorizer in the running daemon."""
    client = _require_client(args, token_path)
    if not _call_optional(client, "ensure_loaded"):
        _emit_json(
            {"error": "unsupported", "hint": "server does not implement ensure_loaded()"},
            exit_code=4,
        )
    started = time.monotonic()
    try:
        result = client.ensure_loaded()
    except Exception as exc:  # noqa: BLE001
        _emit_json({"loaded": False, "error": str(exc)}, exit_code=5)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    payload = result if isinstance(result, dict) else {"loaded": True}
    payload.setdefault("loaded", True)
    payload.setdefault("load_time_ms", elapsed_ms)
    _emit_json(payload, exit_code=0)


def handle_unload(args: argparse.Namespace, token_path: str) -> None:
    """Release the vectorizer's weights from VRAM/RAM in the running daemon."""
    client = _require_client(args, token_path)
    if not _call_optional(client, "unload_model"):
        _emit_json(
            {"error": "unsupported", "hint": "server does not implement unload_model()"},
            exit_code=4,
        )
    try:
        result = client.unload_model()
    except Exception as exc:  # noqa: BLE001
        _emit_json({"loaded": None, "error": str(exc)}, exit_code=5)
    payload = result if isinstance(result, dict) else {}
    payload.setdefault("loaded", False)
    payload.setdefault("freed_vram_bytes", 0)
    payload.setdefault("freed_ram_bytes", 0)
    _emit_json(payload, exit_code=0)


def handle_stop(args: argparse.Namespace, token_path: str) -> None:
    """Legacy shutdown path (unchanged semantics, exit 0 when already stopped)."""
    if not is_port_in_use(args.host, args.port):
        print(f"[model-server] Server on {args.host}:{args.port} is not running.", flush=True)
        if os.path.exists(token_path):
            try:
                os.remove(token_path)
            except OSError:
                pass
        sys.exit(0)
    token = _resolve_token(args, token_path)
    if not token:
        print("[!] Error: Authentication token is required to shut down the server.", file=sys.stderr, flush=True)
        sys.exit(1)
    client = ModelServerClient(args.host, args.port, token)
    if client.shutdown():
        print(f"[model-server] Successfully sent shutdown command to server on {args.host}:{args.port}.", flush=True)
        if os.path.exists(token_path):
            try:
                os.remove(token_path)
            except OSError:
                pass
        sys.exit(0)
    print(f"[!] Failed to shut down server on {args.host}:{args.port}.", file=sys.stderr, flush=True)
    sys.exit(1)


def _chosen_verb(args: argparse.Namespace) -> str | None:
    selected = [name for name in CONTROL_VERBS if getattr(args, name.replace("-", "_"), False)]
    if len(selected) > 1:
        print(
            f"[!] Only one control verb may be specified at a time (got: {', '.join('--' + s for s in selected)}).",
            file=sys.stderr, flush=True,
        )
        sys.exit(64)
    return selected[0] if selected else None


def run_daemon(args: argparse.Namespace) -> None:
    """Start the server process. Model loading is deferred to the daemon."""
    if not args.token:
        print("[!] Error: --token is required when starting the server.", file=sys.stderr, flush=True)
        sys.exit(1)

    # Persist the token so control verbs can find it without an explicit flag.
    if args.token_file:
        try:
            write_token_file(args.token_file, args.token)
        except OSError as exc:
            print(f"[!] Warning: could not persist token file '{args.token_file}': {exc}", file=sys.stderr, flush=True)

    print(
        f"[model-server-process] PID={os.getpid()} starting server on {args.host}:{args.port} "
        f"(lazy_load={not args.eager}, model_idle_timeout={args.model_idle_timeout}s)...",
        flush=True,
    )

    server = ModelServer(
        host=args.host,
        port=args.port,
        auth_token=args.token,
        model_name=args.model,
        idle_timeout=args.idle_timeout,
        batch_window=args.batch_window,
        max_batch_size=args.max_batch_size,
        token_file=args.token_file,
        # New lazy-loading control plane. If the installed ModelServer predates
        # these kwargs, the Coder must update server.py accordingly.
        lazy=not args.eager,
        model_idle_timeout=args.model_idle_timeout,
    )
    server.start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or manage the SafeStore local shared model server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on or connect to.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--token", default=None, help="Authentication token.")
    parser.add_argument("--token-file", default=None, help="File path to write or read the auth token.")
    parser.add_argument(
        "--idle-timeout", type=float, default=0.0,
        help="Seconds of process-level idle before auto-exit (0 = stay alive until shutdown).",
    )
    parser.add_argument(
        "--model-idle-timeout", "--unload-on-idle", dest="model_idle_timeout",
        type=float, default=300.0,
        help="Seconds of idle before unloading the vectorizer only (0 = never unload).",
    )
    parser.add_argument(
        "--eager", action="store_true",
        help="Load the vectorizer at startup (legacy behaviour). Default is lazy.",
    )
    parser.add_argument("--batch-window", type=float, default=0.02, help="Micro-batching window (s).")
    parser.add_argument("--max-batch-size", type=int, default=64, help="Maximum batch size for inference.")

    verbs = parser.add_mutually_exclusive_group()
    verbs.add_argument("--status", action="store_true", help="Print lightweight server liveness as JSON.")
    verbs.add_argument("--health", action="store_true", help="Print full server health as JSON.")
    verbs.add_argument("--load", dest="load", action="store_true", help="Force-load the vectorizer in the running daemon.")
    verbs.add_argument("--unload", dest="unload", action="store_true", help="Unload the vectorizer in the running daemon.")
    verbs.add_argument(
        "--stop", "--shutdown", dest="stop", action="store_true",
        help="Send the shutdown command to the running server.",
    )
    return parser


def main() -> None:
    disable_broken_torchvision()
    args = build_parser().parse_args()

    token_path = args.token_file or default_token_path(args.port)
    verb = _chosen_verb(args)

    if verb == "stop":
        handle_stop(args, token_path)
    elif verb == "status":
        handle_status(args, token_path)
    elif verb == "health":
        handle_health(args, token_path)
    elif verb == "load":
        handle_load(args, token_path)
    elif verb == "unload":
        handle_unload(args, token_path)
    else:
        run_daemon(args)


if __name__ == "__main__":
    main()