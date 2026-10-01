"""
Example demonstrating how to use SafeStore with raw Hugging Face Transformers.

Shows how you can run entirely local, sovereign models using Hugging Face pipelines
or raw PyTorch models without any external API dependencies.
"""

from pathlib import Path
import json
from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "transformers_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore + Hugging Face Transformers Integration Demo ")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Define Transformers Pipeline Generator Adapter
    # -------------------------------------------------------------------------
    # In a real environment with transformers and torch installed:
    # from transformers import pipeline
    # pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cuda:0")

    # For this standalone demonstration, we demonstrate the adapter pattern:
    class MockTransformersPipeline:
        def __call__(self, full_prompt, max_new_tokens=512, **kwargs):
            return [{"generated_text": '{"nodes": [{"label": "Model", "properties": {"identifying_value": "Transformers", "name": "Transformers"}}], "relationships": []}'}]

    hf_pipeline = MockTransformersPipeline()

    def transformers_generator(prompt: str, system_prompt: str = None, json_mode: bool = False, **kwargs) -> str:
        """Adapts any Hugging Face transformers pipeline to SafeStore's LLM generator signature."""
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" if system_prompt else prompt
        results = hf_pipeline(full_prompt, max_new_tokens=kwargs.get("max_tokens", 512))
        return results[0]["generated_text"]

    # -------------------------------------------------------------------------
    # 2. Initialize SafeStore with Transformers Generator
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing SafeStore with local Transformers generator...")
    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        llm_generator=transformers_generator,
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.INFO
    )

    with store:
        store.add_text("hf_doc", "Hugging Face Transformers provides thousands of pretrained models for NLP tasks.")

        graph = store.graph
        print("\n[Step 2] Executing extraction using Transformers...")
        stats = graph.build_graph_for_all_documents()
        print(f"✓ Extracted {stats['nodes_created']} nodes and {stats['relationships_created']} relationships via Transformers!")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" Transformers integration complete! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()