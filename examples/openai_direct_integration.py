"""
Example demonstrating how to use SafeStore directly with the official OpenAI SDK.

SafeStore is not tied to OpenAI SDK or any vendor; any callable wrapping
client.chat.completions.create works out of the box.
"""

from pathlib import Path
import json
from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "openai_direct_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore + Direct OpenAI SDK Integration Demo ")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Define OpenAI SDK Generator Callback
    # -------------------------------------------------------------------------
    # In a real environment:
    # from openai import OpenAI
    # client = OpenAI(api_key="sk-...")

    class MockOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, response_format=None, **kwargs):
                    prompt = messages[-1]["content"]
                    if "nodes" in prompt.lower() or "extract" in prompt.lower():
                        content = json.dumps({
                            "nodes": [{"label": "Service", "properties": {"identifying_value": "OpenAI", "name": "OpenAI"}}],
                            "relationships": []
                        })
                    else:
                        content = "Processed prompt successfully via OpenAI API."
                    return type("Response", (), {
                        "choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]
                    })()

    client = MockOpenAIClient()

    def openai_generator(prompt: str, system_prompt: str = None, json_mode: bool = False, **kwargs) -> str:
        """Adapts OpenAI chat completions to SafeStore's standard generator signature."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=kwargs.get("temperature", 0.1)
        )
        return response.choices[0].message.content

    # -------------------------------------------------------------------------
    # 2. Initialize SafeStore with OpenAI Generator
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing SafeStore with direct OpenAI generator...")
    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        llm_generator=openai_generator,
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.INFO
    )

    with store:
        store.add_text("openai_doc", "OpenAI provides advanced multimodal language models through REST APIs.")

        graph = store.graph
        print("\n[Step 2] Executing extraction using OpenAI...")
        stats = graph.build_graph_for_all_documents()
        print(f"✓ Extracted {stats['nodes_created']} nodes and {stats['relationships_created']} relationships via OpenAI!")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" OpenAI direct integration complete! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()