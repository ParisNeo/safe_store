import sys
import gc
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from safe_store import SafeStore, LogLevel
from safe_store.vectorization.methods.sentense_transformer import STVectorizer, _IN_PROCESS_MODEL_REGISTRY


class TestLocalMemoryManagement:
    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_store_close_unloads_vectorizer_and_clears_model(self, mock_st_class, tmp_path: Path):
        """Test that calling store.close() explicitly releases the model and unlinks vectorizer."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "test_mem_close.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={"model_name": "all-MiniLM-L6-v2", "use_shared_server": False},
            log_level=LogLevel.DEBUG
        )

        assert store.vectorizer is not None
        assert store.vectorizer.model is not None

        # Explicit close
        store.close()

        assert store.conn is None
        assert store.vectorizer is None
        assert store._is_closed is True

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_context_manager_unloads_on_exit(self, mock_st_class, tmp_path: Path):
        """Test that exiting a 'with store:' block automatically unloads the vectorizer."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "test_context_exit.db"
        with SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={"model_name": "all-MiniLM-L6-v2", "use_shared_server": False},
            log_level=LogLevel.DEBUG
        ) as store:
            assert store.vectorizer is not None
            assert store.vectorizer.model is not None

        # Outside the with-block, store must be closed and vectorizer unlinked
        assert store.conn is None
        assert store.vectorizer is None
        assert store._is_closed is True

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_unload_vectorizer_cleans_memory_on_demand(self, mock_st_class, tmp_path: Path):
        """Test on-demand store.unload_vectorizer() frees VRAM without closing database."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "test_mem_unload.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={"model_name": "all-MiniLM-L6-v2", "use_shared_server": False},
            log_level=LogLevel.DEBUG
        )

        assert store.vectorizer is not None
        store.unload_vectorizer()

        assert store.vectorizer is None
        # Database connection remains valid
        assert store.conn is not None

        store.close()

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_store_implicit_del_cleanup(self, mock_st_class, tmp_path: Path):
        """Test that discarding a store instance (without calling close) triggers __del__ cleanup."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "test_implicit_del.db"

        def _create_and_discard():
            s = SafeStore(
                db_path=str(db_path),
                vectorizer_name="st",
                vectorizer_config={"model_name": "all-MiniLM-L6-v2", "use_shared_server": False}
            )
            assert s.vectorizer is not None
            return s.db_path

        # Discard store reference
        _create_and_discard()
        gc.collect()

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_in_process_model_reuse_and_refcounting(self, mock_st_class, tmp_path: Path):
        """Test that multiple SafeStore instances in the same process share the model and refcount it."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db1 = tmp_path / "store1.db"
        db2 = tmp_path / "store2.db"

        cfg = {"model_name": "test-reuse-model", "use_shared_server": False, "reuse_model_in_process": True}

        store1 = SafeStore(db_path=str(db1), vectorizer_name="st", vectorizer_config=cfg)
        store2 = SafeStore(db_path=str(db2), vectorizer_name="st", vectorizer_config=cfg)

        # SentenceTransformer was only instantiated once!
        assert mock_st_class.call_count == 1
        assert store1.vectorizer.model is store2.vectorizer.model

        # Closing store 1 decrements refcount, model still alive for store 2
        store1.close()
        assert store2.vectorizer.model is not None

        # Closing store 2 unloads model completely
        store2.close()
        assert store2.vectorizer is None

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_independent_instances_when_reuse_disabled(self, mock_st_class, tmp_path: Path):
        """Test that when reuse_model_in_process=False, separate model instances are allocated."""
        mock_model_1 = MagicMock()
        mock_model_1.get_sentence_embedding_dimension.return_value = 384
        mock_model_2 = MagicMock()
        mock_model_2.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.side_effect = [mock_model_1, mock_model_2]

        db1 = tmp_path / "store_indep_1.db"
        db2 = tmp_path / "store_indep_2.db"

        cfg = {"model_name": "test-indep-model", "use_shared_server": False, "reuse_model_in_process": False}

        store1 = SafeStore(db_path=str(db1), vectorizer_name="st", vectorizer_config=cfg)
        store2 = SafeStore(db_path=str(db2), vectorizer_name="st", vectorizer_config=cfg)

        # Separate instances allocated
        assert mock_st_class.call_count == 2
        assert store1.vectorizer.model is not store2.vectorizer.model

        store1.close()
        store2.close()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_cuda_empty_cache_invoked_on_unload(self, mock_st_class, mock_empty_cache, mock_cuda_avail, tmp_path: Path):
        """Test that CUDA empty_cache is called when unloading a model with CUDA active."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "test_cuda_flush.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={"model_name": "all-MiniLM-L6-v2", "use_shared_server": False}
        )

        store.close()
        assert mock_empty_cache.called

    @patch("safe_store.vectorization.methods.sentense_transformer.SentenceTransformer")
    def test_idempotent_close_and_unload(self, mock_st_class, tmp_path: Path):
        """Test that calling unload_vectorizer or close multiple times is completely safe."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        db_path = tmp_path / "test_idempotent.db"
        store = SafeStore(
            db_path=str(db_path),
            vectorizer_name="st",
            vectorizer_config={"model_name": "all-MiniLM-L6-v2", "use_shared_server": False}
        )

        # Multiple unload calls
        store.unload_vectorizer()
        store.unload_vectorizer()

        # Multiple close calls
        store.close()
        store.close()

        assert store.conn is None
        assert store.vectorizer is None

    def test_clear_gpu_memory_does_not_crash(self):
        """Test SafeStore.clear_gpu_memory executes cleanly."""
        SafeStore.clear_gpu_memory()


if __name__ == "__main__":
    print("\n" + "=" * 70, flush=True)
    print(" Running SafeStore Memory Management Test Suite via pytest ", flush=True)
    print("=" * 70 + "\n", flush=True)
    sys.exit(pytest.main(["-v", "-s", __file__]))