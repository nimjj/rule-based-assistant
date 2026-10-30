# Regression tests for the master code-review fixes: order_id extraction,
# LLM escalation fallbacks, intent bindings needing allowed_tools, turns
# racing a delete/reset, and rotated OAuth refresh tokens. Run: pytest tests

import json
import threading

import pytest

from ciap.pipeline import llm
from ciap.pipeline.slots import extract_order_id

from test_api import TENANT, NEW_INTENT, client, live_call, turn, write_connections  # noqa: F401


# -- order_id -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("my order is ORD-48213", "ORD-48213"),
    ("the order number is 48213", "48213"),
    ("account 1234567890 please", "1234567890"),
    ("Can you move it to March 15 2025", None),
    ("deliver it on 15 Dec, 2026 instead", None),
    ("call me on 555-555-5555", None),
    ("my number is (555) 555 5555", None),
    ("move it to 2025-03-15", None),
    ("ship it by 03/15/2025", None),
    ("order 48213, delivered March 3rd 2025", "48213"),
    ("the market rate is 2025", "2025"),  # "market" isn't a month
])
def test_extract_order_id(text, expected):
    assert extract_order_id(text) == expected


# -- LLM escalation fallbacks -------------------------------------------------

class _Reply:
    def __init__(self, content):
        self.content = content

    def chat(self, **_kwargs):
        if isinstance(self.content, Exception):
            raise self.content
        return {"message": {"content": self.content}}


@pytest.mark.parametrize("content", ['["track_order"]', '"track_order"', "42", "not json", ConnectionError("down")])
def test_ask_returns_error_dict_for_anything_but_an_object(monkeypatch, content):
    monkeypatch.setattr(llm, "_client", _Reply(content))
    result = llm._ask("prompt")
    assert isinstance(result, dict) and "_error" in result


def test_failed_correction_verdict_does_not_revert(monkeypatch):
    monkeypatch.setattr(llm, "_client", _Reply(ConnectionError("down")))
    result = llm.escalate("correction_or_reference", "I was asking about the delivery",
                          context={"last_frame": 1, "intent": "track_order"})
    assert result["is_correction_or_reference"] is False


def test_duplicated_multi_intent_is_one_intent(monkeypatch):
    monkeypatch.setattr(llm, "_client", _Reply(json.dumps({"resolved_intents": ["track_order", "track_order"]})))
    result = llm.escalate("multiple_intents_in_utterance", "track it", context=("track_order", "cancel_order"))
    assert result["resolved_intents"] == ["track_order"]


def test_escalation_client_has_a_timeout():
    assert llm._client._client.timeout.read is not None


# -- intent bindings ----------------------------------------------------------

def test_binding_needs_connection_allowed_tools(client):
    shipped = json.loads((__import__("pathlib").Path(__file__).parent.parent / "connections.json").read_text())["zoho"]
    write_connections({"zoho": {k: v for k, v in shipped.items() if k != "allowed_tools"}})
    body = {**NEW_INTENT, "remote": {"connection": "zoho", "tool": "ZohoCRM_searchRecords",
                                     "arguments": {"path_variables": {"module": "Leads"}}}}
    r = client.post("/v1/intents", json=body)
    assert r.status_code == 422 and "allowed_tools" in r.json()["error"]["message"]


# -- turns racing delete/reset ------------------------------------------------

@pytest.mark.parametrize("method,path", [("delete", ""), ("post", "/reset")])
def test_turn_waiting_on_a_deleted_or_reset_call_is_refused(client, method, path):
    from ciap.api import server

    cid = live_call(client)
    session = server._calls[(TENANT, cid)]
    session.lock.acquire()  # a turn in flight
    done = {}
    t = threading.Thread(target=lambda: done.setdefault("r", getattr(client, method)(f"/v1/tenants/{TENANT}/calls/{cid}{path}")))
    t.start()
    t.join(0.2)
    assert t.is_alive()  # delete/reset waits for the turn
    session.lock.release()
    t.join(5)
    assert done["r"].status_code in (200, 204)
    with pytest.raises(server.ApiError) as exc:
        server._run_turn((TENANT, cid), session, "track my order", "customer")
    assert exc.value.code == "call_replaced"
