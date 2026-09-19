# safe_store/vectorization/model_server/run_server.py
import argparse
import os
import time
from .server import ModelServer


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


def main():
    parser = argparse.ArgumentParser(description="Run the SafeStore local model server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--model", required=True, help="SentenceTransformer model name")
    parser.add_argument("--token", required=True, help="Authentication token")
    parser.add_argument("--token-file", required=False, default=None, help="File path to write the auth token for cross-process discovery")
    args = parser.parse_args()

    if args.token_file:
        write_token_file(args.token_file, args.token)

    server = ModelServer(
        host=args.host,
        port=args.port,
        auth_token=args.token,
        model_name=args.model
    )
    server.start()


if __name__ == "__main__":
    main()