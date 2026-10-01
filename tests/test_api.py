# API tests. The embedding model and the LLM are faked so these run offline
# and deterministically; slot extraction, the state machine, views and the
# HTTP layer are real.  Run: pytest tests

import itertools
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import call_log
import config
import intents
import mcp_tools
import state_store
import zoho_client


def fake_retrieve(text, top_k=3):
    t = text.lower()
    if "both" in t:
        return [("track_order", 0.70), ("cancel_order", 0.65), ("payment_info", 0.1)]
    if "maybe" in t:
        return [("track_order", 0.40), ("cancel_order", 0.30), ("payment_info", 0.1)]
    if "track" in t:
        return [("track_order", 0.80), ("cancel_order", 0.2), ("payment_info", 0.1)]
    if "cancel" in t:
        return [("cancel_order", 0.80), ("track_order", 0.2), ("payment_info", 0.1)]
    if "look me up" in t:
        return [("lookup_crm_record", 0.80), ("track_order", 0.1), ("payment_info", 0.1)]
    if "reschedule" in t:
        return [("change_delivery_date", 0.80), ("track_order", 0.2), ("payment_info", 0.1)]
    return [("payment_info", 0.10), ("track_order", 0.05), ("cancel_order", 0.04)]


def fake_escalate(reason, utterance, context=None, fallback_intent=None):
    base = {"escalated": True, "reason": reason, "model": "fake", "note": "fake verdict"}
    if reason == "multiple_intents_in_utterance":
        return {**base, "resolved_intents": list(context)}
    if reason == "mid_confidence_ambiguous_intent":
        return {**base, "resolved_intent": fallback_intent}
    return {**base, "resolved_intent": fallback_intent, "is_correction_or_reference": True}


@pytest.fixture
def client(monkeypatch, tmp_path, request):
    import server

    monkeypatch.setattr(state_store, "retrieve", fake_retrieve)
    monkeypatch.setattr(state_store.llm, "escalate", fake_escalate)
    monkeypatch.setattr(call_log, "TURN_LOG", tmp_path / "turns.jsonl")
    monkeypatch.setattr(zoho_client, "search_lead", lambda email: {"data": [{"Email": email}]})
    # Catalog edits mutate these shared dicts in place: restore them after.
    for mod, attr in [(intents, "INTENTS"), (mcp_tools, "TOOLS"), (mcp_tools, "INTENT_TO_TOOL")]:
        original = getattr(mod, attr)
        snapshot = dict(original)
        request.addfinalizer(lambda o=original, s=snapshot: (o.clear(), o.update(s)))
    monkeypatch.setattr(config, "CATALOG_PATH", str(tmp_path / "catalog.json"))
    server._calls.clear()
    server._expired_ids.clear()
    monkeypatch.setattr(server, "_now", lambda: clock.now)
    clock.now = 1000.0
    monkeypatch.setattr(config, "CALL_IDLE_TIMEOUT_SECONDS", 600)
    monkeypatch.setattr(config, "SLOT_OVERRIDE_ENABLED", False)
    return TestClient(server.app, raise_server_exceptions=False)


@pytest.fixture
def slot_override(monkeypatch):
    """Turn on the archived slot override endpoint for one test."""
    monkeypatch.setattr(config, "SLOT_OVERRIDE_ENABLED", True)


class clock:
    """Fake monotonic clock for idle-expiry tests; advance with clock.now += s."""
    now = 1000.0


_pbx_ids = itertools.count(1)


def pbx_id():
    """A PBXware/Asterisk-style call ID, unique per test run."""
    return f"1727690000.{next(_pbx_ids)}"


def new_call(client):
    cid = pbx_id()
    r = client.post("/v1/calls", json={"call_id": cid})
    assert r.status_code == 201, r.text
    assert r.json() == {"call_id": cid}
    return cid


def turn(client, call_id, utterance, speaker="customer", **params):
    return client.post(f"/v1/calls/{call_id}/turns", json={"utterance": utterance, "speaker": speaker}, params=params)


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()["error"]
    assert body["code"] == code and body["message"]


# -- lifecycle ----------------------------------------------------------------

def test_lifecycle_get_reset_delete(client):
    cid = new_call(client)
    turn(client, cid, "track my order")
    state = client.get(f"/v1/calls/{cid}").json()
    assert state["turn_count"] == 1 and state["frames"][0]["intent"] == "track_order"

    assert client.post(f"/v1/calls/{cid}/reset").json() == {"call_id": cid}
    state = client.get(f"/v1/calls/{cid}").json()
    assert state["turn_count"] == 0 and state["frames"] == []

    assert client.delete(f"/v1/calls/{cid}").status_code == 204
    assert_error(client.get(f"/v1/calls/{cid}"), 404, "call_not_found")


# -- idle expiry --------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("get", ""), ("delete", ""), ("post", "/reset"),
])
def test_idle_call_expires_on_access(client, method, path):
    import server

    cid = new_call(client)
    clock.now += 601
    assert_error(getattr(client, method)(f"/v1/calls/{cid}{path}"), 404, "call_expired")
    # Gone from memory, and stays call_expired on later requests.
    assert cid not in server._calls
    assert_error(client.get(f"/v1/calls/{cid}"), 404, "call_expired")


def test_active_call_outlives_timeout(client):
    cid = new_call(client)
    for _ in range(3):
        clock.now += 500
        assert turn(client, cid, "track my order").status_code == 200
    clock.now += 500
    assert client.post(f"/v1/calls/{cid}/reset").status_code == 200
    clock.now += 500
    assert client.get(f"/v1/calls/{cid}").status_code == 200


def test_get_does_not_keep_call_alive(client):
    cid = new_call(client)
    clock.now += 400
    assert client.get(f"/v1/calls/{cid}").status_code == 200
    clock.now += 201
    assert_error(client.get(f"/v1/calls/{cid}"), 404, "call_expired")


def test_sweep_evicts_only_idle_calls(client):
    import server

    old, fresh = new_call(client), new_call(client)
    clock.now += 500
    turn(client, fresh, "track my order")
    clock.now += 101
    assert server.evict_idle_calls() == 1
    assert list(server._calls) == [fresh]
    assert_error(client.get(f"/v1/calls/{old}"), 404, "call_expired")


def test_sweep_skips_call_with_turn_in_flight(client):
    import server

    cid = new_call(client)
    clock.now += 601
    with server._calls[cid].lock:
        assert server.evict_idle_calls() == 0
    assert server.evict_idle_calls() == 1


def test_zero_timeout_disables_expiry(client, monkeypatch):
    import server

    monkeypatch.setattr(config, "CALL_IDLE_TIMEOUT_SECONDS", 0)
    cid = new_call(client)
    clock.now += 10**6
    assert server.evict_idle_calls() == 0
    assert client.get(f"/v1/calls/{cid}").status_code == 200


@pytest.mark.parametrize("method,path", [
    ("get", "/v1/calls/nope"), ("delete", "/v1/calls/nope"), ("post", "/v1/calls/nope/reset"),
])
def test_unknown_call_404(client, method, path):
    assert_error(getattr(client, method)(path), 404, "call_not_found")


def test_unknown_call_on_override(client, slot_override):
    r = client.patch("/v1/calls/nope/frames/1/slots/order_id", json={"value": "ORD-1234"})
    assert_error(r, 404, "call_not_found")


# -- PBXware call IDs -----------------------------------------------------------

def test_create_is_idempotent_and_keeps_state(client):
    cid = new_call(client)
    turn(client, cid, "track my order")
    r = client.post("/v1/calls", json={"call_id": cid})
    assert r.status_code == 200 and r.json() == {"call_id": cid}
    assert client.get(f"/v1/calls/{cid}").json()["turn_count"] == 1


@pytest.mark.parametrize("body", [{}, {"call_id": ""}, {"call_id": "has space"}, {"call_id": "a/b"},
                                  {"call_id": "x" * 129}, {"call_id": 12345}])
def test_create_requires_valid_call_id(client, body):
    import server

    assert_error(client.post("/v1/calls", json=body), 422, "validation_error")
    assert server._calls == {}


@pytest.mark.parametrize("cid", ["1727690000.12345", "7c9e6679-7425-40de-944b-e07fc1f90ae7", "pbx:A_1"])
def test_accepted_call_id_formats(client, cid):
    assert client.post("/v1/calls", json={"call_id": cid}).status_code == 201
    assert turn(client, cid, "track my order").status_code == 200


def test_first_turn_starts_the_call(client):
    cid = pbx_id()
    v = turn(client, cid, "track my order")
    assert v.status_code == 200 and v.json()["turn_index"] == 1
    assert client.get(f"/v1/calls/{cid}").json()["frames"][0]["intent"] == "track_order"
    # The later create (e.g. a late call-start event) doesn't wipe it.
    assert client.post("/v1/calls", json={"call_id": cid}).status_code == 200
    assert client.get(f"/v1/calls/{cid}").json()["turn_count"] == 1


def test_invalid_call_id_on_turn_creates_nothing(client):
    import server

    assert_error(turn(client, "x" * 129, "hi"), 422, "validation_error")
    assert server._calls == {}


def test_transcript_starts_the_call_only_if_valid(client):
    import server

    cid = pbx_id()
    bad = client.post(f"/v1/calls/{cid}/transcript", json={"transcript": "no tagged lines"})
    assert_error(bad, 422, "validation_error")
    assert cid not in server._calls
    r = client.post(f"/v1/calls/{cid}/transcript", json={"transcript": "Customer: track my order\nCustomer: ORD-12345"})
    assert r.status_code == 200 and r.json()["frames"][0]["status"] == "FIRED"


def test_expired_call_starts_over_on_turn_or_create(client):
    cid = new_call(client)
    turn(client, cid, "track my order")
    clock.now += 601
    v = turn(client, cid, "cancel my order").json()
    assert v["turn_index"] == 1 and [f["intent"] for f in v["slots"]["frames"]] == ["cancel_order"]

    clock.now += 601
    assert client.post("/v1/calls", json={"call_id": cid}).status_code == 201
    assert client.get(f"/v1/calls/{cid}").json() == {"call_id": cid, "turn_count": 0, "frames": []}


def test_unknown_route_uses_error_shape(client):
    assert_error(client.get("/v1/nothing"), 404, "not_found")


# -- turns --------------------------------------------------------------------

def test_turn_stable_view_and_slot_fill_fires_tool(client):
    cid = new_call(client)
    v1 = turn(client, cid, "please track my order").json()
    assert v1["turn_index"] == 1
    assert v1["intent"]["turn_type"] == "new_intent" and v1["intent"]["decision"] == "accepted"
    assert v1["intent"]["selected_intent"] == "track_order"
    frame = v1["slots"]["frames"][0]
    assert frame["status"] == "AWAITING_INFO" and frame["missing"] == ["order_id"]
    assert v1["mcp"]["fired_this_turn"] == [] and v1["debug"] is None

    v2 = turn(client, cid, "ORD-99213").json()
    assert v2["intent"]["turn_type"] == "slot_fill"
    assert v2["slots"]["frames"][0]["filled_this_turn"] == ["order_id"]
    assert v2["mcp"]["fired_this_turn"] == [
        {"tool": "get_order_status", "tool_label": "Get Order Status", "arguments": {"order_id": "ORD-99213"}}
    ]
    assert v2["slots"]["frames"][0]["status"] == "FIRED"


def test_speaker_recorded_as_initiated_by(client):
    cid = new_call(client)
    v = turn(client, cid, "let me track that", speaker="agent").json()
    assert v["speaker"] == "agent" and v["slots"]["frames"][0]["initiated_by"] == "agent"


def test_debug_flag_includes_json_safe_trace(client):
    cid = new_call(client)
    v = turn(client, cid, "track my order", debug="true").json()
    assert v["debug"]["turn_type"] == "new_intent"


def test_escalated_and_multi_intent_and_unknown(client):
    cid = new_call(client)
    esc = turn(client, cid, "maybe track it").json()
    assert esc["intent"]["turn_type"] == "new_intent_escalated"
    assert esc["intent"]["escalation"]["reason"] == "mid_confidence_ambiguous_intent"

    cid = new_call(client)
    multi = turn(client, cid, "both please").json()
    assert multi["intent"]["turn_type"] == "multi_intent_blocked"
    assert multi["intent"]["message"] and multi["slots"]["frames"] == []

    cid = new_call(client)
    assert turn(client, cid, "hello there").json()["intent"]["turn_type"] == "unknown_intent"


def test_new_date_slot_is_flattened_to_resolved_string(client):
    cid = new_call(client)
    turn(client, cid, "reschedule ORD-12345 to 2030-01-15")
    frame = client.get(f"/v1/calls/{cid}").json()["frames"][0]
    assert frame["status"] == "FIRED"
    assert frame["slots"]["new_date"] == "2030-01-15T00:00:00"  # resolved ISO-8601, not the {"text","date"} dict


def test_zoho_tool_result_exposed(client):
    cid = new_call(client)
    v = turn(client, cid, "look me up at jane@example.com").json()
    frame = v["slots"]["frames"][0]
    assert frame["tool_call"]["tool"] == "search_zoho_lead"
    assert frame["tool_result"] == {"data": [{"Email": "jane@example.com"}]}
    assert client.get(f"/v1/calls/{cid}").json()["frames"][0]["tool_result"] == frame["tool_result"]


def test_zoho_failure_is_a_tool_error_not_a_500(client, monkeypatch):
    def boom(email):
        raise RuntimeError("zoho down")
    monkeypatch.setattr(zoho_client, "search_lead", boom)
    cid = new_call(client)
    r = turn(client, cid, "look me up at jane@example.com")
    assert r.status_code == 200
    assert "zoho down" in r.json()["slots"]["frames"][0]["tool_result"]["error"]


# -- validation & errors -------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"utterance": ""}, {"utterance": "   "}, {"utterance": "x" * 2001},
    {"utterance": "hi", "speaker": "robot"}, {},
])
def test_turn_validation(client, payload):
    cid = new_call(client)
    r = client.post(f"/v1/calls/{cid}/turns", json=payload)
    assert_error(r, 422, "validation_error")
    assert r.json()["error"]["details"]


def test_pipeline_exception_is_500_with_error_shape(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(state_store, "retrieve", boom)
    cid = new_call(client)
    r = turn(client, cid, "track my order")
    assert_error(r, 500, "pipeline_error")
    assert "exploded" not in r.text  # internals aren't leaked
    # A failed turn doesn't consume a turn index.
    assert client.get(f"/v1/calls/{cid}").json()["turn_count"] == 0


# -- slot override (archived) ----------------------------------------------------

def test_override_disabled_by_default(client):
    cid = new_call(client)
    turn(client, cid, "track my order")
    r = client.patch(f"/v1/calls/{cid}/frames/1/slots/order_id", json={"value": "ORD-22222"})
    assert_error(r, 404, "endpoint_disabled")
    assert client.get(f"/v1/calls/{cid}").json()["turn_count"] == 1
    assert client.get("/demo/config").json() == {"slot_override_enabled": False}


def test_override_listed_as_archived(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/v1/calls/{call_id}/frames/{frame_id}/slots/{slot_name}"]["patch"]
    assert op["tags"] == ["archived"] and op["deprecated"] is True
    assert spec["tags"][-1]["name"] == "archived"


def test_override_refires_tool_with_corrected_value(client, slot_override):
    cid = new_call(client)
    turn(client, cid, "track my order")
    turn(client, cid, "ORD-11111")
    r = client.patch(f"/v1/calls/{cid}/frames/1/slots/order_id", json={"value": "ORD-22222"})
    assert r.status_code == 200
    v = r.json()
    assert v["intent"]["turn_type"] == "slot_override" and v["speaker"] == "agent"
    assert v["turn_index"] == 3
    assert v["mcp"]["fired_this_turn"] == [
        {"tool": "get_order_status", "tool_label": "Get Order Status", "arguments": {"order_id": "ORD-22222"}}
    ]
    assert v["slots"]["frames"][0]["slots"]["order_id"] == "ORD-22222"


def test_override_errors(client, slot_override):
    cid = new_call(client)
    turn(client, cid, "track my order")
    url = f"/v1/calls/{cid}/frames/1/slots"
    assert_error(client.patch(f"/v1/calls/{cid}/frames/99/slots/order_id", json={"value": "ORD-1234"}), 404, "frame_not_found")
    assert_error(client.patch(f"{url}/email", json={"value": "a@b.co"}), 422, "invalid_slot")
    assert_error(client.patch(f"{url}/order_id", json={"value": "no id here"}), 422, "invalid_slot_value")
    assert_error(client.patch(f"{url}/order_id", json={"value": ""}), 422, "validation_error")
    # Nothing above changed anything or consumed a turn index.
    assert client.get(f"/v1/calls/{cid}").json()["turn_count"] == 1


def test_override_on_reverted_frame_is_409(client, slot_override):
    cid = new_call(client)
    turn(client, cid, "cancel it")  # opens cancel_order, awaiting order_id
    assert turn(client, cid, "actually never mind").json()["intent"]["turn_type"] == "ambiguous_continuation"
    # Complete the frame first, then revert it with a correction.
    turn(client, cid, "ORD-12345")
    assert turn(client, cid, "actually never mind").json()["intent"]["turn_type"] == "correction"
    r = client.patch(f"/v1/calls/{cid}/frames/1/slots/order_id", json={"value": "ORD-99999"})
    assert_error(r, 409, "frame_reverted")


# -- transcript ----------------------------------------------------------------

def test_transcript_text(client):
    cid = new_call(client)
    text = "Customer: please track my order\nAgent: sure, what is the order number?\nCustomer: ORD-99213\nnoise line"
    r = client.post(f"/v1/calls/{cid}/transcript", json={"transcript": text})
    assert r.status_code == 200
    body = r.json()
    assert [t["speaker"] for t in body["turns"]] == ["customer", "agent", "customer"]
    assert [t["turn_index"] for t in body["turns"]] == [1, 2, 3]
    assert body["frames"][0]["status"] == "FIRED"


def test_transcript_turns_list(client):
    cid = new_call(client)
    r = client.post(f"/v1/calls/{cid}/transcript", json={"turns": [
        {"utterance": "track my order"}, {"utterance": "ORD-99213", "speaker": "customer"},
    ]})
    assert r.status_code == 200 and r.json()["frames"][0]["status"] == "FIRED"


@pytest.mark.parametrize("payload", [
    {}, {"transcript": "a", "turns": [{"utterance": "b"}]}, {"transcript": "no tags at all"},
    {"turns": [{"utterance": "ok"}, {"utterance": ""}]},
    {"transcript": "Customer: " + "x" * 2001},
])
def test_transcript_validation_processes_nothing(client, payload):
    cid = new_call(client)
    assert_error(client.post(f"/v1/calls/{cid}/transcript", json=payload), 422, "validation_error")
    assert client.get(f"/v1/calls/{cid}").json()["turn_count"] == 0


# -- catalog, log, health, demo -------------------------------------------------

def test_intent_and_tool_catalogs(client):
    intents = {i["name"]: i for i in client.get("/v1/intents").json()}
    assert intents["track_order"]["required_slots"] == ["order_id"]
    assert intents["track_order"]["tool"] == "get_order_status"
    tools = {t["name"]: t for t in client.get("/v1/tools").json()}
    assert tools["search_zoho_lead"]["executes_live"] is True
    assert tools["get_order_status"]["executes_live"] is False
    assert tools["get_order_status"]["intents"] == ["track_order"]
    assert tools["get_order_status"]["input_schema"]["required"] == ["order_id"]
    assert all(i["tool"] in tools for i in intents.values() if i["tool"])


# -- catalog edits ------------------------------------------------------------

NEW_INTENT = {"name": "gift_wrap", "exemplars": ["Can you gift wrap it"], "required_slots": ["order_id"],
              "tool": "get_order_status"}
NEW_TOOL = {"name": "add_gift_wrap", "description": "Add gift wrap to an order.",
            "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                             "required": ["order_id"]}}


def test_intent_crud_persists_and_reloads(client):
    import catalog

    r = client.post("/v1/intents", json=NEW_INTENT)
    assert r.status_code == 201, r.text
    assert r.json()["tool"] == "get_order_status" and r.json()["exemplar_count"] == 1
    assert client.get("/v1/intents/gift_wrap").json()["exemplars"] == ["Can you gift wrap it"]
    assert "gift_wrap" in client.get("/v1/tools/get_order_status").json()["intents"]

    r = client.put("/v1/intents/gift_wrap", json={"exemplars": ["Wrap it please", "Gift wrap"], "tool": None})
    assert r.json()["tool"] is None and r.json()["required_slots"] == []

    saved = json.loads(Path(config.CATALOG_PATH).read_text(encoding="utf-8"))
    assert saved["intents"]["gift_wrap"] == {"exemplars": ["Wrap it please", "Gift wrap"],
                                             "required_slots": [], "tool": None}

    assert client.delete("/v1/intents/gift_wrap").status_code == 204
    assert_error(client.get("/v1/intents/gift_wrap"), 404, "intent_not_found")

    # A restart loads the saved file over the defaults.
    intents.INTENTS["stale"] = {"exemplars": ["x"], "required_slots": []}
    assert catalog.load()
    assert "stale" not in intents.INTENTS and "gift_wrap" not in intents.INTENTS
    assert mcp_tools.INTENT_TO_TOOL["track_order"] == "get_order_status"


# -- labels -------------------------------------------------------------------

def test_every_catalog_entry_ships_with_a_label(client):
    for path in ("/v1/intents", "/v1/tools"):
        for item in client.get(path).json():
            assert item["label"] and item["label"] != item["name"], item["name"]
    assert client.get("/v1/intents/track_order").json()["label"] == "Track Order"
    assert client.get("/v1/tools/search_zoho_lead").json()["label"] == "Search Zoho CRM Lead"


def test_labels_on_turn_frame_and_call_views(client):
    cid = new_call(client)
    v = turn(client, cid, "track my order ORD-12345").json()
    assert v["intent"]["selected_intent_label"] == "Track Order"
    assert v["intent"]["candidates"][0] == {"intent": "track_order", "label": "Track Order", "score": 0.8}
    frame = v["slots"]["frames"][0]
    assert frame["intent_label"] == "Track Order"
    assert frame["tool_call"]["tool_label"] == "Get Order Status"
    assert v["mcp"]["all_calls"][0]["intent_label"] == "Track Order"
    assert v["mcp"]["all_calls"][0]["tool_label"] == "Get Order Status"

    state = client.get(f"/v1/calls/{cid}").json()
    assert state["frames"][0]["intent_label"] == "Track Order"

    blocked = turn(client, cid, "track and cancel both").json()
    assert blocked["intent"]["selected_intent_label"] == "Track Order + Cancel Order"

    unknown = turn(client, cid, "hello there").json()
    assert unknown["intent"]["selected_intent"] is None and unknown["intent"]["selected_intent_label"] is None


def test_label_create_update_persist_and_fallback(client):
    import catalog

    r = client.post("/v1/intents", json={**NEW_INTENT, "label": "  Gift Wrapping  "})
    assert r.status_code == 201 and r.json()["label"] == "Gift Wrapping"
    r = client.post("/v1/tools", json={**NEW_TOOL, "label": "Add Gift Wrap Now"})
    assert r.json()["label"] == "Add Gift Wrap Now"

    saved = json.loads(Path(config.CATALOG_PATH).read_text(encoding="utf-8"))
    assert saved["intents"]["gift_wrap"]["label"] == "Gift Wrapping"
    assert saved["tools"]["add_gift_wrap"]["label"] == "Add Gift Wrap Now"
    assert catalog.load()
    assert client.get("/v1/intents/gift_wrap").json()["label"] == "Gift Wrapping"

    # PUT replaces the whole entry: no label -> humanized name.
    body = {k: v for k, v in NEW_INTENT.items() if k != "name"}
    assert client.put("/v1/intents/gift_wrap", json=body).json()["label"] == "Gift Wrap"
    assert "label" not in json.loads(Path(config.CATALOG_PATH).read_text(encoding="utf-8"))["intents"]["gift_wrap"]


@pytest.mark.parametrize("label", ["", "   ", "x" * 81])
def test_label_validation(client, label):
    assert_error(client.post("/v1/intents", json={**NEW_INTENT, "label": label}), 422, "validation_error")
    assert_error(client.post("/v1/tools", json={**NEW_TOOL, "label": label}), 422, "validation_error")


def test_label_edit_shows_on_live_call(client):
    cid = new_call(client)
    turn(client, cid, "track my order ORD-12345")
    spec = client.get("/v1/intents/track_order").json()
    body = {"label": "Order Tracking", "exemplars": spec["exemplars"],
            "required_slots": spec["required_slots"], "tool": spec["tool"]}
    assert client.put("/v1/intents/track_order", json=body).status_code == 200
    assert client.get(f"/v1/calls/{cid}").json()["frames"][0]["intent_label"] == "Order Tracking"


# -- renames ------------------------------------------------------------------

def test_intent_rename_moves_spec_mapping_and_live_frames(client, monkeypatch):
    import catalog
    import embeddings

    original = client.get("/v1/intents/track_order").json()
    cid = new_call(client)
    turn(client, cid, "track my order")  # frame waiting on order_id

    monkeypatch.setattr(embeddings, "_index", (["track_order"], None))
    r = client.post("/v1/intents/track_order/rename", json={"new_name": "order_tracking"})
    assert r.status_code == 200, r.text
    assert r.json() == {**original, "name": "order_tracking"}
    assert embeddings._index is None
    assert_error(client.get("/v1/intents/track_order"), 404, "intent_not_found")
    assert client.get("/v1/tools/get_order_status").json()["intents"] == ["order_tracking"]

    # The waiting frame carries on under the new name and still fires.
    frame = client.get(f"/v1/calls/{cid}").json()["frames"][0]
    assert frame["intent"] == "order_tracking" and frame["status"] == "AWAITING_INFO"
    monkeypatch.setattr(state_store, "retrieve", lambda text, top_k=3: [("order_tracking", 0.1)])
    v = turn(client, cid, "ORD-12345").json()
    assert v["intent"]["turn_type"] == "slot_fill" and v["intent"]["selected_intent"] == "order_tracking"
    assert v["mcp"]["all_calls"][0]["intent"] == "order_tracking"

    saved = json.loads(Path(config.CATALOG_PATH).read_text(encoding="utf-8"))
    assert "track_order" not in saved["intents"] and saved["intents"]["order_tracking"]["tool"] == "get_order_status"
    assert catalog.load() and "order_tracking" in intents.INTENTS


def test_tool_rename_repoints_intents_and_live_calls(client):
    cid = new_call(client)
    turn(client, cid, "track my order ORD-12345")  # fires get_order_status

    r = client.post("/v1/tools/get_order_status/rename", json={"new_name": "get_status_order"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "get_status_order" and r.json()["intents"] == ["track_order"]
    assert r.json()["label"] == "Get Order Status"
    assert client.get("/v1/intents/track_order").json()["tool"] == "get_status_order"
    assert_error(client.get("/v1/tools/get_order_status"), 404, "tool_not_found")

    frame = client.get(f"/v1/calls/{cid}").json()["frames"][0]
    assert frame["tool_call"]["tool"] == "get_status_order"

    cid2 = new_call(client)
    v = turn(client, cid2, "track my order ORD-777777").json()
    assert v["mcp"]["fired_this_turn"][0]["tool"] == "get_status_order"


def test_renamed_live_tool_still_executes(client):
    r = client.post("/v1/tools/search_zoho_lead/rename", json={"new_name": "crm_lead_search"})
    assert r.json()["executes_live"] is True
    # A PUT (which can't set the backend) keeps it live too.
    spec = r.json()
    r = client.put("/v1/tools/crm_lead_search", json={"description": "Find a CRM lead.", "input_schema": spec["input_schema"]})
    assert r.json()["executes_live"] is True

    cid = new_call(client)
    v = turn(client, cid, "look me up, jane@example.com").json()
    frame = v["slots"]["frames"][0]
    assert frame["tool_call"]["tool"] == "crm_lead_search"
    assert frame["tool_result"] == {"data": [{"Email": "jane@example.com"}]}


def test_old_saved_catalog_keeps_zoho_live(client):
    import catalog

    tools = {n: {k: v for k, v in spec.items() if k != "backend"} for n, spec in mcp_tools.TOOLS.items()}
    Path(config.CATALOG_PATH).write_text(json.dumps({"intents": {}, "tools": tools}), encoding="utf-8")
    assert catalog.load()
    assert client.get("/v1/tools/search_zoho_lead").json()["executes_live"] is True


@pytest.mark.parametrize("path,body,status,code", [
    ("/v1/intents/nope/rename", {"new_name": "x_new"}, 404, "intent_not_found"),
    ("/v1/intents/track_order/rename", {"new_name": "cancel_order"}, 409, "intent_exists"),
    ("/v1/intents/track_order/rename", {"new_name": "Bad Name"}, 422, "validation_error"),
    ("/v1/tools/nope/rename", {"new_name": "x_new"}, 404, "tool_not_found"),
    ("/v1/tools/get_order_status/rename", {"new_name": "cancel_order"}, 409, "tool_exists"),
    ("/v1/tools/get_order_status/rename", {}, 422, "validation_error"),
])
def test_rename_errors(client, path, body, status, code):
    assert_error(client.post(path, json=body), status, code)
    assert "track_order" in intents.INTENTS and "get_order_status" in mcp_tools.TOOLS


def test_intent_edit_rebuilds_embedding_index(client, monkeypatch):
    import embeddings

    monkeypatch.setattr(embeddings, "_index", (["track_order"], None))
    client.post("/v1/intents", json=NEW_INTENT)
    assert embeddings._index is None


@pytest.mark.parametrize("body,code", [
    ({**NEW_INTENT, "name": "track_order"}, "intent_exists"),
    ({**NEW_INTENT, "name": "Bad Name"}, "validation_error"),
    ({**NEW_INTENT, "exemplars": []}, "validation_error"),
    ({**NEW_INTENT, "required_slots": ["shoe_size"]}, "unknown_slot"),
    ({**NEW_INTENT, "tool": "nope"}, "tool_not_found"),
    ({**NEW_INTENT, "required_slots": []}, "tool_slot_mismatch"),
])
def test_intent_create_validation(client, body, code):
    r = client.post("/v1/intents", json=body)
    assert_error(r, 409 if code == "intent_exists" else 422, code)
    assert "gift_wrap" not in intents.INTENTS
    assert not Path(config.CATALOG_PATH).exists()


def test_intent_update_and_delete_unknown_404(client):
    assert_error(client.put("/v1/intents/nope", json={"exemplars": ["x"]}), 404, "intent_not_found")
    assert_error(client.delete("/v1/intents/nope"), 404, "intent_not_found")


def test_intent_delete_refused_while_frame_waiting(client):
    cid = new_call(client)
    turn(client, cid, "track my order")
    assert_error(client.delete("/v1/intents/track_order"), 409, "intent_in_use")
    turn(client, cid, "ORD-12345")  # frame fires -> no longer waiting
    assert client.delete("/v1/intents/track_order").status_code == 204
    assert client.get(f"/v1/calls/{cid}").json()["frames"][0]["required"] == []


def test_tool_crud_and_mapping(client, monkeypatch):
    r = client.post("/v1/tools", json=NEW_TOOL)
    assert r.status_code == 201 and r.json()["executes_live"] is False and r.json()["intents"] == []
    client.post("/v1/intents", json={**NEW_INTENT, "tool": "add_gift_wrap"})

    cid = new_call(client)
    monkeypatch.setattr(state_store, "retrieve", lambda text, top_k=3: [("gift_wrap", 0.9)])
    v = turn(client, cid, "gift wrap ORD-5555").json()
    assert v["mcp"]["fired_this_turn"] == [
        {"tool": "add_gift_wrap", "tool_label": "Add Gift Wrap", "arguments": {"order_id": "ORD-5555"}}
    ]

    r = client.put("/v1/tools/add_gift_wrap", json={**NEW_TOOL, "description": "Wrap it."})
    assert r.json()["description"] == "Wrap it." and r.json()["intents"] == ["gift_wrap"]
    assert_error(client.delete("/v1/tools/add_gift_wrap"), 409, "tool_in_use")

    client.put("/v1/intents/gift_wrap", json={"exemplars": ["Wrap it"], "tool": None})
    assert client.delete("/v1/tools/add_gift_wrap").status_code == 204
    assert_error(client.get("/v1/tools/add_gift_wrap"), 404, "tool_not_found")


def test_tool_validation(client):
    assert_error(client.post("/v1/tools", json={**NEW_TOOL, "name": "cancel_order"}), 409, "tool_exists")
    bad = {**NEW_TOOL, "input_schema": {"type": "object", "properties": {}, "required": ["order_id"]}}
    assert_error(client.post("/v1/tools", json=bad), 422, "invalid_input_schema")
    assert_error(client.post("/v1/tools", json={**NEW_TOOL, "input_schema": {"type": "string"}}),
                 422, "invalid_input_schema")
    # track_order doesn't collect `email`, so get_order_status can't start requiring it.
    schema = {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}
    r = client.put("/v1/tools/get_order_status", json={"description": "x", "input_schema": schema})
    assert_error(r, 422, "tool_slot_mismatch")
    assert_error(client.put("/v1/tools/nope", json={"description": "x", "input_schema": schema}),
                 404, "tool_not_found")


def test_slots_catalog(client):
    assert client.get("/v1/slots").json() == ["discount_code", "email", "new_date", "order_id", "product_name"]

def test_log_records_and_filters(client):
    a, b = new_call(client), new_call(client)
    turn(client, a, "track my order")
    turn(client, b, "cancel it")
    all_recs = client.get("/v1/log").json()
    assert all_recs["total"] == 4  # 2 call_created + 2 turn
    only_a = client.get("/v1/log", params={"call_id": a}).json()
    assert {r["call_id"] for r in only_a["records"]} == {a}
    assert [r["event"] for r in only_a["records"]] == ["call_created", "turn"]
    assert client.get("/v1/log", params={"limit": 1}).json()["records"][-1]["event"] == "turn"
    assert_error(client.get("/v1/log", params={"limit": 0}), 422, "validation_error")


def test_health_ok_and_degraded(client, monkeypatch):
    import llm

    monkeypatch.setattr(llm._client, "list", lambda: [])
    assert client.get("/health").json() == {"status": "ok", "ollama": True, "model": llm.MODEL_NAME}

    def down():
        raise ConnectionError
    monkeypatch.setattr(llm._client, "list", down)
    assert client.get("/health").json()["status"] == "degraded"


def test_demo_page_and_assets_and_no_legacy_routes(client):
    assert client.get("/demo").status_code == 200
    assert client.get("/demo/static/app.js").status_code == 200
    assert client.get("/demo/static/../server.py").status_code == 404
    assert client.post("/calls").status_code == 404
    assert client.post("/demo/calls").status_code == 404


def test_openapi_documents_all_v1_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/v1/calls", "/v1/calls/{call_id}", "/v1/calls/{call_id}/turns", "/v1/calls/{call_id}/transcript",
              "/v1/calls/{call_id}/reset", "/v1/calls/{call_id}/frames/{frame_id}/slots/{slot_name}",
              "/v1/intents", "/v1/intents/{name}", "/v1/tools", "/v1/tools/{name}", "/v1/slots",
              "/v1/log", "/health"]:
        assert p in paths, p
