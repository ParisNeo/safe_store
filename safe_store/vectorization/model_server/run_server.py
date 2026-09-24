# safe_store/vectorization/model_server/run_server.py
import argparse
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

from .server import ModelServer
from .client import ModelServerClient


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


def main():
    disable_broken_torchvision()

    parser = argparse.ArgumentParser(description="Run or manage the SafeStore local shared model server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on or connect to")
    parser.add_argument("--model", required=False, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--token", required=False, default=None, help="Authentication token")
    parser.add_argument("--token-file", required=False, default=None, help="File path to write or read the auth token")
    parser.add_argument("--idle-timeout", type=float, default=0.0, help="Seconds of idle before auto-exit (0 = stay alive until shutdown command)")
    parser.add_argument("--batch-window", type=float, default=0.02, help="Micro-batching collection window in seconds (default: 0.02)")
    parser.add_argument("--max-batch-size", type=int, default=64, help="Maximum batch size for inference")
    parser.add_argument("--stop", "--shutdown", dest="stop", action="store_true", help="Send shutdown command to running server")
    args = parser.parse_args()

    # Handle explicit shutdown command
    if args.stop:
        token = args.token
        token_path = args.token_file or os.path.join(os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp")), f"safe_store_server_{args.port}.token")

        if not token and os.path.exists(token_path):
            try:
                token = read_token_file(token_path)
            except Exception as e:
                print(f"[!] Could not read token file '{token_path}': {e}", file=sys.stderr, flush=True)

        if not is_port_in_use(args.host, args.port):
            print(f"[model-server] Server on {args.host}:{args.port} is not running.", flush=True)
            if os.path.exists(token_path):
                try: os.remove(token_path)
                except OSError: pass
            sys.exit(0)

        if not token:
            print(f"[!] Error: Authentication token is required to shut down the server.", file=sys.stderr, flush=True)
            sys.exit(1)

        client = ModelServerClient(args.host, args.port, token)
        if client.shutdown():
            print(f"[model-server] Successfully sent shutdown command to server on {args.host}:{args.port}.", flush=True)
            if os.path.exists(token_path):
                try: os.remove(token_path)
                except OSError: pass
            sys.exit(0)
        else:
            print(f"[!] Failed to shut down server on {args.host}:{args.port}.", file=sys.stderr, flush=True)
            sys.exit(1)

    # Server execution
    if not args.token:
        print("[!] Error: --token is required when starting the server.", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[model-server-process] PID={os.getpid()} starting server on {args.host}:{args.port}...", flush=True)

    server = ModelServer(
        host=args.host,
        port=args.port,
        auth_token=args.token,
        model_name=args.model,
        idle_timeout=args.idle_timeout,
        batch_window=args.batch_window,
        max_batch_size=args.max_batch_size,
        token_file=args.token_file
    )
    server.start()


if __name__ == "__main__":
    main()