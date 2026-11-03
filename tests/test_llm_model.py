# One Ollama model for escalations and answers: llm.py defaults to the answer
# model and calls it with the answer calls' num_ctx / keep_alive, so Ollama
# never reloads or unloads it between the two. Run: pytest tests

import json

from ciap import config
from ciap.pipeline import llm


class _Recorder:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": json.dumps({"resolved_intent": "track_order"})}}


def _escalate(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(llm, "_client", rec)
    llm.escalate("mid_confidence_ambiguous_intent", "where is it", context=[("track_order", 0.6), ("cancel_order", 0.55)],
                 fallback_intent="track_order")
    return rec.calls[0]


def test_escalations_default_to_the_answer_model():
    assert config.LLM_MODEL_NAME == config.RESPONDER_MODEL_NAME
    assert llm.MODEL_NAME == config.RESPONDER_MODEL_NAME


def test_shared_model_gets_the_answer_calls_settings(monkeypatch):
    monkeypatch.setattr(config, "RESPONDER_NUM_CTX", 8192)
    call = _escalate(monkeypatch)
    assert call["model"] == config.RESPONDER_MODEL_NAME
    assert call["think"] is False
    assert call["keep_alive"] == config.RESPONDER_KEEP_ALIVE_SECONDS
    assert call["options"] == {"temperature": 0, "num_ctx": 8192}


def test_no_num_ctx_when_the_answer_calls_send_none(monkeypatch):
    monkeypatch.setattr(config, "RESPONDER_NUM_CTX", None)
    assert _escalate(monkeypatch)["options"] == {"temperature": 0}


def test_a_separate_escalation_model_keeps_ollamas_defaults(monkeypatch):
    monkeypatch.setattr(llm, "MODEL_NAME", "gemma3:1b")
    monkeypatch.setattr(config, "RESPONDER_NUM_CTX", 8192)
    call = _escalate(monkeypatch)
    assert call["model"] == "gemma3:1b"
    assert "keep_alive" not in call
    assert call["options"] == {"temperature": 0}
    assert call["think"] is False
