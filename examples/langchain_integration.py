"""
Example demonstrating how to use SafeStore with LangChain.

SafeStore remains 100% independent and framework-agnostic. You can plug in ANY
LangChain ChatModel, LLM, or LCEL chain into SafeStore via the clean callback system.
"""

from pathlib import Path
import json
from safe_store import SafeStore, LogLevel


def cleanup_db(db_file: str):
    for ext in ["", ".lock", "-wal", "-shm"]:
        Path(f"{db_file}{ext}").unlink(missing_ok=True)


def main():
    db_file = "langchain_demo.db"
    cleanup_db(db_file)

    print("=" * 80)
    print(" SafeStore + LangChain Integration Demo ")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Define LangChain Generator Adapter
    # -------------------------------------------------------------------------
    # In a real app with langchain installed:
    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import SystemMessage, HumanMessage
    # chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # For this standalone demonstration, we mock the LangChain runnable interface:
    class MockLangChainChatModel:
        def invoke(self, messages):
            # Inspect prompt and return mock response
            content = messages[-1]["content"] if isinstance(messages[-1], dict) else str(messages[-1])
            if "nodes" in content.lower() or "extract" in content.lower():
                return type("AIMessage", (), {
                    "content": json.dumps({
                        "nodes": [{"label": "Framework", "properties": {"identifying_value": "LangChain", "name": "LangChain"}}],
                        "relationships": []
                    })
                })()
            return type("AIMessage", (), {"content": "SafeStore integrates cleanly with LangChain via callbacks."})()

    lc_model = MockLangChainChatModel()

    def langchain_generator(prompt: str, system_prompt: str = None, json_mode: bool = False, **kwargs) -> str:
        """Adapts any LangChain ChatModel or Chain to SafeStore's LLM generator signature."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call LangChain runnable
        ai_message = lc_model.invoke(messages)
        return ai_message.content

    # -------------------------------------------------------------------------
    # 2. Initialize SafeStore with LangChain Generator
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing SafeStore with LangChain generator callback...")
    store = SafeStore(
        db_path=db_file,
        vectorizer_name="st",
        llm_generator=langchain_generator,
        chunk_size=50,
        chunk_overlap=5,
        log_level=LogLevel.INFO
    )

    with store:
        store.add_text("overview_doc", "LangChain is a popular orchestrator for LLM pipelines and autonomous agents.")

        graph = store.graph
        print("\n[Step 2] Executing extraction using LangChain...")
        stats = graph.build_graph_for_all_documents()
        print(f"✓ Extracted {stats['nodes_created']} nodes and {stats['relationships_created']} relationships via LangChain!")

    store.close()
    cleanup_db(db_file)
    print("\n" + "=" * 80)
    print(" LangChain integration complete! ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()