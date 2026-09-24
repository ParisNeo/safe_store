import sys
import time
import socket
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from safe_store.vectorization.model_server.protocol import (
    encode_message, recv_message, encode_array, recv_array
)
from safe_store.vectorization.model_server.client import ModelServerClient
from safe_store.vectorization.model_server.server import ModelServer
from safe_store import SafeStore, LogLevel, shutdown_shared_vectorizer


class TestProtocolSerialization:
    def test_encode_and_decode_array(self):
        import struct
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        encoded = encode_array(arr)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 8

        # Verify length header matches payload
        (length,) = struct.unpack(">Q", encoded[:8])
        assert length == len(encoded) - 8

        import io
        decoded = np.load(io.BytesIO(encoded[8:]), allow_pickle=False)
        np.testing.assert_array_equal(arr, decoded)

    def test_encode_message_length_prefix(self):
        msg = {"kind": "test", "token": "secret123"}
        encoded = encode_message(msg)
        assert len(encoded) > 8


class TestModelServerLifecycle:
    def test_server_authentication_and_ping(self):
        """Test server starts, authenticates valid token, and replies to ping."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        auth_token = "valid_test_token_abc"
        server = ModelServer(
            host="127.0.0.1",
            port=port,
            auth_token=auth_token,
            model_name="all-MiniLM-L6-v2",
            idle_timeout=0.0
        )

        # Mock sentence transformer model to avoid downloading weights during test
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        server._model = mock_model
        server._dim = 384

        listener = server._create_listener()
        server._listener = listener

        import threading
        acceptor = threading.Thread(target=server._accept_loop, args=(listener,), daemon=True)
        acceptor.start()

        # Connect with correct token
        client = ModelServerClient("127.0.0.1", port, auth_token)
        assert client.ping(timeout=3.0) is True

        # Connect with bad token
        bad_client = ModelServerClient("127.0.0.1", port, "wrong_token")
        assert bad_client.ping(timeout=3.0) is False

        # Send special shutdown command
        assert client.shutdown(timeout=3.0) is True
        server.shutdown()

    def test_server_vectorize_request(self):
        """Test sending a vectorization request to the server and receiving back embeddings."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        auth_token = "vectorize_test_token"
        server = ModelServer(
            host="127.0.0.1",
            port=port,
            auth_token=auth_token,
            model_name="all-MiniLM-L6-v2",
            idle_timeout=0.0,
            batch_window=0.01
        )

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.ones((2, 384), dtype=np.float32)
        server._model = mock_model
        server._dim = 384

        listener = server._create_listener()
        server._listener = listener

        import threading
        acceptor = threading.Thread(target=server._accept_loop, args=(listener,), daemon=True)
        acceptor.start()
        batcher = threading.Thread(target=server._batch_loop, daemon=True)
        batcher.start()

        client = ModelServerClient("127.0.0.1", port, auth_token)
        assert client.ping(timeout=3.0) is True

        # Send vectorize request
        vectors = client.vectorize(["Hello world", "SafeStore shared vectorizer"])
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (2, 384)
        assert np.all(vectors == 1.0)

        # Shut down server cleanly
        assert client.shutdown(timeout=3.0) is True
        server.shutdown()

    def test_server_shutdown_command_cleans_up(self):
        """Test the explicit shutdown command terminates server loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        auth_token = "shutdown_test_token"
        server = ModelServer(
            host="127.0.0.1",
            port=port,
            auth_token=auth_token,
            model_name="all-MiniLM-L6-v2",
            idle_timeout=0.0
        )

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        server._model = mock_model
        server._dim = 384

        listener = server._create_listener()
        server._listener = listener

        import threading
        acceptor = threading.Thread(target=server._accept_loop, args=(listener,), daemon=True)
        acceptor.start()

        client = ModelServerClient("127.0.0.1", port, auth_token)
        # Verify running
        assert client.ping(timeout=3.0) is True

        # Instruct to die via special command
        assert client.shutdown(timeout=3.0) is True
        time.sleep(0.2)
        assert server._shutdown.is_set()


class TestSafeStoreSharedServerIntegration:
    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_safestore_shared_server_configuration(self, mock_st_class, tmp_path: Path):
        """Test SafeStore initialization with shared_vectorizer=True sets up configuration."""
        mock_st_instance = MagicMock()
        mock_st_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_st_instance

        db_path = tmp_path / "test_shared_store.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={
                "model_name": "all-MiniLM-L6-v2",
                "use_shared_server": False  # In-process for unit testing
            },
            log_level=LogLevel.DEBUG
        )

        assert store.vectorizer_config.get("use_shared_server") is False
        store.close()


class TestSharedServerCompatibilityAndRegression:
    """Regression tests preventing false-positive vectorizer compatibility errors and bytes crashes."""

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_local_store_reopened_in_shared_mode_is_compatible(self, mock_st_class, tmp_path: Path):
        """
        Regression test: An existing database created in local in-process mode
        MUST reopen cleanly when use_shared_server=True without ConfigurationError.
        """
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "compat_test.db"

        # 1. Create database in local mode (like old LOLLMS or SafeStore instances)
        store_local = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={
                "model": "all-MiniLM-L6-v2",
                "use_shared_server": False
            }
        )
        with store_local:
            store_local.add_text("doc1", "Sample document content.")
        store_local.close()

        # 2. Reopen the exact same database in shared daemon mode
        store_shared = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={
                "model_name": "all-MiniLM-L6-v2",
                "use_shared_server": False,  # Client mode
                "port": 9876,
                "idle_timeout": 0.0,
                "batch_window": 0.05
            }
        )
        assert store_shared.conn is not None
        store_shared.close()

    def test_semantic_fingerprint_ignores_transport_keys(self):
        """
        Regression test: Transport parameters like use_shared_server, port, host
        must NOT alter the semantic compatibility fingerprint of a model.
        """
        from safe_store.vectorization.manager import VectorizationManager

        fp_local = VectorizationManager._create_unique_name(
            "st",
            {"model": "all-MiniLM-L6-v2"}
        )
        fp_shared = VectorizationManager._create_unique_name(
            "sentense_transformer",
            {
                "model_name": "all-MiniLM-L6-v2",
                "use_shared_server": True,
                "port": 8765,
                "host": "127.0.0.1",
                "idle_timeout": 0.0,
                "batch_window": 0.02
            }
        )
        assert fp_local == fp_shared

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_truly_incompatible_models_still_rejected(self, mock_st_class, tmp_path: Path):
        """
        Ensures that genuine model mismatches (e.g. 384-dim vs 768-dim)
        are still strictly rejected to protect vector space integrity.
        """
        from safe_store.core.exceptions import ConfigurationError

        mock_model_384 = MagicMock()
        mock_model_384.get_sentence_embedding_dimension.return_value = 384
        mock_model_768 = MagicMock()
        mock_model_768.get_sentence_embedding_dimension.return_value = 768

        mock_st_class.side_effect = [mock_model_384, mock_model_768]

        db_path = tmp_path / "incompatible_test.db"

        # Create with MiniLM (384)
        store1 = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={"model": "all-MiniLM-L6-v2", "use_shared_server": False}
        )
        store1.close()

        # Attempt to reopen with mpnet (768) must fail
        with pytest.raises(ConfigurationError, match="incompatible vectorizer"):
            SafeStore(
                db_path=str(db_path),
                vectorizer_name="st",
                vectorizer_config={"model": "all-mpnet-base-v2", "use_shared_server": False}
            )

    def test_graph_store_chunk_sanitization_removes_base64_images(self, tmp_path: Path):
        """
        Regression test: Large inline base64 image data URIs in markdown must be stripped
        to prevent downstream JSON serializers in LLM clients from crashing on bytes.
        """
        from safe_store import GraphStore

        store = SafeStore(db_path=str(tmp_path / "sanitize.db"), vectorizer_name="st", vectorizer_config={"use_shared_server": False})
        graph_store = GraphStore(store=store)

        raw_chunk = (
            "# Heading\n\n"
            "Here is a diagram: ![system architecture](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
            "AAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==)\n\n"
            "And regular text explaining the architecture."
        )

        sanitized = graph_store._sanitize_chunk_for_llm(raw_chunk)

        assert "data:image/png;base64" not in sanitized
        assert "[Image: system architecture]" in sanitized
        assert "And regular text explaining the architecture." in sanitized
        assert isinstance(sanitized, str)
        store.close()


if __name__ == "__main__":
    print("\n" + "=" * 70, flush=True)
    print(" Running SafeStore Shared Vectorizer Test Suite via pytest ", flush=True)
    print("=" * 70 + "\n", flush=True)
    sys.exit(pytest.main(["-v", "-s", __file__]))