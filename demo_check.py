# Pre-flight for the web demo. Run this well before the session -- NOT live.
#
#     python demo_check.py
#
# Verifies every dependency the demo touches and pays the one-time costs up
# front (embedding model load/download, Ollama warm-up) so the first live
# turn is fast. Exits non-zero if anything the demo needs is missing.

import sys
import time


def check_imports():
    import sentence_transformers  # noqa: F401
    import numpy  # noqa: F401
    import dateparser  # noqa: F401
    import fastapi  # noqa: F401
    import ollama  # noqa: F401


def check_embedding_model():
    # Loads the model and builds the exemplar index. If the weights aren't
    # cached this pulls from HuggingFace -- exactly what must not happen live.
    from embeddings import retrieve
    start = time.perf_counter()
    retrieve("warm up the embedding index")
    return time.perf_counter() - start


def check_ollama():
    import ollama
    from llm import MODEL_NAME

    client = ollama.Client()
    resp = client.list()
    models = getattr(resp, "models", None) or (resp.get("models", []) if isinstance(resp, dict) else [])
    names = []
    for m in models:
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    if not any(MODEL_NAME in n for n in names):
        raise RuntimeError(f"{MODEL_NAME} not pulled. Run:  ollama pull {MODEL_NAME}")

    # Warm it so the first live escalation isn't a cold start.
    start = time.perf_counter()
    client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with the word ok."}],
        options={"temperature": 0},
    )
    return time.perf_counter() - start


def check_pipeline_end_to_end():
    # Intent -> slot -> MCP call, the spine of the demo.
    from state_store import ConversationStateStore

    store = ConversationStateStore()
    store.process_utterance("Where is my order")
    trace = store.process_utterance("ORD-48213")
    calls = trace.get("tool_calls") or []
    if not calls or calls[0].get("tool") != "get_order_status":
        raise RuntimeError(f"expected a get_order_status tool call, got {calls!r}")
    if calls[0]["arguments"].get("order_id") != "ORD-48213":
        raise RuntimeError(f"order_id not extracted into the tool call: {calls[0]!r}")


CHECKS = [
    ("python deps import", check_imports),
    ("embedding model (loads / downloads if uncached)", check_embedding_model),
    ("ollama + escalation model, warmed", check_ollama),
    ("pipeline end-to-end: intent -> slot -> MCP call", check_pipeline_end_to_end),
]


def main():
    failed = False
    for name, fn in CHECKS:
        try:
            result = fn()
            extra = f"  ({result:.1f}s)" if isinstance(result, float) else ""
            print(f"  [ok]   {name}{extra}")
        except Exception as exc:
            failed = True
            print(f" [FAIL]  {name}\n         {type(exc).__name__}: {exc}")

    if failed:
        print("\nOne or more checks failed -- fix before the demo.")
        sys.exit(1)

    print("\nAll checks passed.")
    print("Start:  uvicorn server:app")
    print("Open:   http://127.0.0.1:8000/demo")


if __name__ == "__main__":
    main()
