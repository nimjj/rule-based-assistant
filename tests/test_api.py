# API tests. The embedding model and the LLM are faked so these run offline
# and deterministically, and the Zoho MCP server is an in-process fake; slot
# extraction, the state machine, views, the MCP client and the HTTP layer are
# real.  Run: pytest tests

import itertools
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ciap import call_log
from ciap import config
from ciap.pipeline import intents
from ciap.pipeline import responder
from ciap.integrations import mcp_client
from ciap.integrations import mcp_tools
from ciap.pipeline import state_store

# Stands in for Zoho's hosted MCP server behind the "zoho" connection,
# including its {"status", "data"} result envelope.
fake_zoho = MCPServer("fake-zoho")
fake_zoho_state = {"down": False, "unauthorised": False, "searches": []}


@fake_zoho.tool()
def ZohoCRM_searchRecords(path_variables: dict, query_params: dict) -> dict:
    """Search a CRM module."""
    fake_zoho_state["searches"].append(path_variables["module"])
    if fake_zoho_state["down"]:
        raise ToolError("zoho down")
    if fake_zoho_state["unauthorised"]:
        return {"status": "failure", "data": {"message": "You cannot perform this operation. Connection not authorised"}}
    assert path_variables["module"] in ("Leads", "Contacts")
    email = query_params["email"]
    # contact@...: no lead, but an existing customer in Contacts.
    if email.startswith("contact") and path_variables["module"] == "Contacts":
        return {"status": "success", "data": {"data": [{"Email": email, "Account_Name": {"name": "King"}}]}}
    if email.startswith(("nobody", "contact")):
        return {"status": "success", "data": {"message": "Empty response received from API. Service - CRM, "
                                                         "Tool - ZohoCRM_searchRecords, With Status - 204."}}
    return {"status": "success", "data": {"data": [{"Email": email}]}}


@fake_zoho.tool()
def empty_tool(email: str) -> None:
    """Returns nothing."""
    return None


@fake_zoho.tool(annotations={"readOnlyHint": False})
def slow_tool(email: str) -> dict:
    """Takes too long."""
    time.sleep(1)
    return {"email": email}


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


fake_answer = {"text": None, "prompts": []}


@pytest.fixture
def client(monkeypatch, tmp_path, request):
    from ciap.api import server

    monkeypatch.setattr(state_store, "retrieve", fake_retrieve)
    monkeypatch.setattr(state_store.llm, "escalate", fake_escalate)
    # The responder model replies with fake_answer["text"] (None: no answer).
    fake_answer.update(text=None, prompts=[])
    monkeypatch.setattr(
        responder, "_ask_model",
        lambda prompt, mode="focused": fake_answer["prompts"].append(prompt) or fake_answer["text"],
    )
    monkeypatch.setattr(call_log, "TURN_LOG", tmp_path / "turns.jsonl")
    connections = tmp_path / "connections.json"
    shipped_zoho = json.loads((ROOT / "connections.json").read_text())["zoho"]
    connections.write_text(json.dumps({"zoho": {**shipped_zoho, "url": "http://zoho.invalid/mcp"}}), encoding="utf-8")
    monkeypatch.setattr(config, "MCP_CONNECTIONS_PATH", str(connections))
    # Only "zoho" is configured; it connects to the in-process fake.
    monkeypatch.setattr(mcp_client, "_server_for", lambda name, spec: fake_zoho)
    fake_zoho_state.update(down=False, unauthorised=False, searches=[])
    request.addfinalizer(mcp_client.shutdown)
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
    monkeypatch.setattr(config, "DEMO_UI_ENABLED", False)
    return TestClient(server.app, raise_server_exceptions=False)


@pytest.fixture
def slot_override(monkeypatch):
    """Turn on the archived slot override endpoint for one test."""
    monkeypatch.setattr(config, "SLOT_OVERRIDE_ENABLED", True)


@pytest.fixture
def demo_ui(monkeypatch):
    """Serve the /demo browser page for one test."""
    monkeypatch.setattr(config, "DEMO_UI_ENABLED", True)


class clock:
    """Fake monotonic clock for idle-expiry tests; advance with clock.now += s."""
    now = 1000.0


ROOT = Path(__file__).resolve().parent.parent
TENANT = "acme"
_pbx_ids = itertools.count(1)


def pbx_id():
    """A PBXware/Asterisk-style call ID, unique per test run."""
    return f"1727690000.{next(_pbx_ids)}"


def new_call(client):
    """A fresh call ID. Nothing exists server-side until its first turn."""
    return pbx_id()


def live_call(client):
    """A call started by its first turn (track_order, awaiting order_id)."""
    cid = pbx_id()
    assert turn(client, cid, "track my order").status_code == 200
    return cid


def turn(client, call_id, utterance, speaker="customer", **params):
    return client.post(f"/v1/tenants/{TENANT}/calls/{call_id}/turns", json={"utterance": utterance, "speaker": speaker}, params=params)


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()["error"]
    assert body["code"] == code and body["message"]


# -- lifecycle ----------------------------------------------------------------

def test_lifecycle_get_reset_delete(client):
    cid = new_call(client)
    turn(client, cid, "track my order")
    state = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()
    assert state["turn_count"] == 1 and state["frames"][0]["intent"] == "track_order"

    assert client.post(f"/v1/tenants/{TENANT}/calls/{cid}/reset").json() == {"tenant_id": TENANT, "call_id": cid}
    state = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()
    assert state["turn_count"] == 0 and state["frames"] == []

    assert client.delete(f"/v1/tenants/{TENANT}/calls/{cid}").status_code == 204
    assert_error(client.get(f"/v1/tenants/{TENANT}/calls/{cid}"), 404, "call_not_found")


# -- idle expiry --------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("get", ""), ("delete", ""), ("post", "/reset"),
])
def test_idle_call_expires_on_access(client, method, path):
    from ciap.api import server

    cid = live_call(client)
    clock.now += 601
    assert_error(getattr(client, method)(f"/v1/tenants/{TENANT}/calls/{cid}{path}"), 404, "call_expired")
    # Gone from memory, and stays call_expired on later requests.
    assert (TENANT, cid) not in server._calls
    assert_error(client.get(f"/v1/tenants/{TENANT}/calls/{cid}"), 404, "call_expired")


def test_active_call_outlives_timeout(client):
    cid = new_call(client)
    for _ in range(3):
        clock.now += 500
        assert turn(client, cid, "track my order").status_code == 200
    clock.now += 500
    assert client.post(f"/v1/tenants/{TENANT}/calls/{cid}/reset").status_code == 200
    clock.now += 500
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").status_code == 200


def test_get_does_not_keep_call_alive(client):
    cid = live_call(client)
    clock.now += 400
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").status_code == 200
    clock.now += 201
    assert_error(client.get(f"/v1/tenants/{TENANT}/calls/{cid}"), 404, "call_expired")


def test_sweep_evicts_only_idle_calls(client):
    from ciap.api import server

    old, fresh = live_call(client), live_call(client)
    clock.now += 500
    turn(client, fresh, "track my order")
    clock.now += 101
    assert server.evict_idle_calls() == 1
    assert list(server._calls) == [(TENANT, fresh)]
    assert_error(client.get(f"/v1/tenants/{TENANT}/calls/{old}"), 404, "call_expired")


def test_sweep_skips_call_with_turn_in_flight(client):
    from ciap.api import server

    cid = live_call(client)
    clock.now += 601
    with server._calls[(TENANT, cid)].lock:
        assert server.evict_idle_calls() == 0
    assert server.evict_idle_calls() == 1


def test_zero_timeout_disables_expiry(client, monkeypatch):
    from ciap.api import server

    monkeypatch.setattr(config, "CALL_IDLE_TIMEOUT_SECONDS", 0)
    cid = live_call(client)
    clock.now += 10**6
    assert server.evict_idle_calls() == 0
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").status_code == 200


@pytest.mark.parametrize("method,path", [
    ("get", f"/v1/tenants/{TENANT}/calls/nope"), ("delete", f"/v1/tenants/{TENANT}/calls/nope"),
    ("post", f"/v1/tenants/{TENANT}/calls/nope/reset"),
])
def test_unknown_call_404(client, method, path):
    assert_error(getattr(client, method)(path), 404, "call_not_found")


def test_unknown_call_on_override(client, slot_override):
    r = client.patch(f"/v1/tenants/{TENANT}/calls/nope/frames/1/slots/order_id", json={"value": "ORD-1234"})
    assert_error(r, 404, "call_not_found")


# -- PBXware call IDs -----------------------------------------------------------

def test_create_endpoint_is_gone(client):
    assert_error(client.post("/v1/calls", json={"call_id": pbx_id()}), 404, "not_found")


@pytest.mark.parametrize("cid", ["1727690000.12345", "7c9e6679-7425-40de-944b-e07fc1f90ae7", "pbx:A_1"])
def test_accepted_call_id_formats(client, cid):
    assert turn(client, cid, "track my order").status_code == 200
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["call_id"] == cid


def test_first_turn_starts_the_call(client):
    cid = pbx_id()
    v = turn(client, cid, "track my order")
    assert v.status_code == 200 and v.json()["turn_index"] == 1
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]["intent"] == "track_order"
    # Later turns continue the same call.
    assert turn(client, cid, "ORD-12345").json()["turn_index"] == 2


def test_same_call_id_under_two_tenants_is_two_calls(client):
    cid = pbx_id()
    turn(client, cid, "track my order")
    other = client.post(f"/v1/tenants/globex/calls/{cid}/turns", json={"utterance": "cancel my order"})
    assert other.json()["turn_index"] == 1
    a = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()
    b = client.get(f"/v1/tenants/globex/calls/{cid}").json()
    assert (a["tenant_id"], a["frames"][0]["intent"]) == (TENANT, "track_order")
    assert (b["tenant_id"], b["frames"][0]["intent"]) == ("globex", "cancel_order")
    assert client.delete(f"/v1/tenants/globex/calls/{cid}").status_code == 204
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").status_code == 200


@pytest.mark.parametrize("tenant", ["x" * 129, "has space"])
def test_invalid_tenant_id_creates_nothing(client, tenant):
    from ciap.api import server

    r = client.post(f"/v1/tenants/{tenant}/calls/{pbx_id()}/turns", json={"utterance": "hi"})
    assert_error(r, 422, "validation_error")
    assert server._calls == {}


def test_invalid_call_id_on_turn_creates_nothing(client):
    from ciap.api import server

    assert_error(turn(client, "x" * 129, "hi"), 422, "validation_error")
    assert server._calls == {}


def test_transcript_starts_the_call_only_if_valid(client):
    from ciap.api import server

    cid = pbx_id()
    bad = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/transcript", json={"transcript": "no tagged lines"})
    assert_error(bad, 422, "validation_error")
    assert (TENANT, cid) not in server._calls
    r = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/transcript", json={"transcript": "Customer: track my order\nCustomer: ORD-12345"})
    assert r.status_code == 200 and r.json()["frames"][0]["status"] == "FIRED"


def test_expired_call_starts_over_on_turn(client):
    cid = live_call(client)
    clock.now += 601
    v = turn(client, cid, "cancel my order").json()
    assert v["turn_index"] == 1 and [f["intent"] for f in v["slots"]["frames"]] == ["cancel_order"]


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
    frame = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]
    assert frame["status"] == "FIRED"
    assert frame["slots"]["new_date"] == "2030-01-15T00:00:00"  # resolved ISO-8601, not the {"text","date"} dict


def test_zoho_tool_result_exposed(client):
    cid = new_call(client)
    v = turn(client, cid, "look me up at jane@example.com").json()
    frame = v["slots"]["frames"][0]
    assert frame["tool_call"]["tool"] == "search_zoho_lead"
    assert frame["tool_result"] == {"status": "success", "data": {"data": [{"Email": "jane@example.com"}]}, "message": None}
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]["tool_result"] == frame["tool_result"]


def test_zoho_answer_written_by_model(client):
    fake_answer["text"] = "Jane (jane@example.com) is in Zoho as a lead."
    cid = new_call(client)
    frame = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]
    assert frame["answer"] == {
        "text": fake_answer["text"], "source": "llm", "model": responder.MODEL_NAME, "note": None, "mode": "focused",
    }
    prompt = fake_answer["prompts"][0]
    assert 'Customer: "look me up at jane@example.com"' in prompt and "Email: jane@example.com" in prompt
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]["answer"] == frame["answer"]


def test_zoho_answer_falls_back_on_invented_value(client):
    fake_answer["text"] = "Lead 98765 belongs to jane@example.com."
    cid = new_call(client)
    answer = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]["answer"]
    assert answer["source"] == "fallback" and "98765" in answer["note"]
    assert answer["text"] == "Zoho CRM Customer Lookup found 1 result: email jane@example.com."


def test_zoho_answer_falls_back_when_model_fails(client, monkeypatch):
    def down(prompt, mode="focused"):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(responder, "_ask_model", down)
    cid = new_call(client)
    answer = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]["answer"]
    assert answer["source"] == "fallback" and "ollama down" in answer["note"] and answer["model"] is None


def test_answer_question_spans_slot_fill_turns(client):
    fake_answer["text"] = "Found Jane."
    cid = new_call(client)
    turn(client, cid, "can you look me up")
    turn(client, cid, "what's your email?", speaker="agent")
    turn(client, cid, "it's jane@example.com")
    prompt = fake_answer["prompts"][0]
    assert 'Customer: "can you look me up"\nCustomer: "it\'s jane@example.com"' in prompt


def test_answer_rules_for_no_results_and_error(client):
    cid = new_call(client)
    answer = turn(client, cid, "look me up at nobody@example.com").json()["slots"]["frames"][0]["answer"]
    assert answer == {"text": "Zoho CRM Customer Lookup: nothing found for email nobody@example.com.",
                      "source": "rule", "model": None, "note": None, "mode": "focused"}
    fake_zoho_state["down"] = True
    cid = new_call(client)
    answer = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]["answer"]
    assert answer["source"] == "rule" and "zoho down" in answer["text"]
    assert fake_answer["prompts"] == []


def test_full_answer_mode(client, monkeypatch):
    modes = []
    monkeypatch.setattr(
        responder, "_ask_model",
        lambda prompt, mode="focused": modes.append(mode) or fake_answer["prompts"].append(prompt) or "Found Jane.",
    )
    cid = new_call(client)
    body = {"utterance": "look me up at jane@example.com", "answer_mode": "full"}
    answer = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/turns", json=body).json()["slots"]["frames"][0]["answer"]
    assert answer["source"] == "llm" and answer["mode"] == "full" and modes == ["full"]
    assert responder.INSTRUCTIONS["full"] in fake_answer["prompts"][0]
    assert responder.INSTRUCTIONS["focused"] not in fake_answer["prompts"][0]


def test_full_answer_mode_on_transcript(client):
    fake_answer["text"] = "Found Jane."
    cid = new_call(client)
    body = {"transcript": "Customer: look me up at jane@example.com", "answer_mode": "full"}
    r = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/transcript", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["frames"][0]["answer"]["mode"] == "full"


def test_answer_mode_is_per_request(client):
    fake_answer["text"] = "Found Jane."
    cid = new_call(client)
    turn(client, cid, "can you look me up")
    body = {"utterance": "it's jane@example.com", "answer_mode": "full"}
    frame = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/turns", json=body).json()["slots"]["frames"][0]
    # The turn that fires the tool decides, not the one that opened the frame.
    assert frame["answer"]["mode"] == "full"


def test_invalid_answer_mode_rejected(client):
    body = {"utterance": "look me up at jane@example.com", "answer_mode": "verbose"}
    assert_error(client.post(f"/v1/tenants/{TENANT}/calls/{new_call(client)}/turns", json=body), 422, "validation_error")


def test_mock_tool_has_no_answer(client):
    cid = new_call(client)
    frame = turn(client, cid, "track ORD-12345").json()["slots"]["frames"][0]
    assert frame["tool_result"] is None and frame["answer"] is None


def test_lookup_falls_back_to_contacts(client):
    cid = new_call(client)
    frame = turn(client, cid, "look me up at contact@example.com").json()["slots"]["frames"][0]
    assert fake_zoho_state["searches"] == ["Leads", "Contacts"]
    assert frame["tool_result"]["status"] == "success"
    assert frame["tool_result"]["data"] == {"data": [{"Email": "contact@example.com", "Account_Name": {"name": "King"}}]}


def test_lookup_stops_at_first_match_or_error(client):
    cid = new_call(client)
    turn(client, cid, "look me up at jane@example.com")
    assert fake_zoho_state["searches"] == ["Leads"]  # found a lead: Contacts never searched
    fake_zoho_state.update(unauthorised=True, searches=[])
    frame = turn(client, new_call(client), "look me up at jane@example.com").json()["slots"]["frames"][0]
    assert frame["tool_result"]["status"] == "error" and fake_zoho_state["searches"] == ["Leads"]


def test_lookup_with_no_match_anywhere_is_no_results(client):
    frame = turn(client, new_call(client), "look me up at nobody@example.com").json()["slots"]["frames"][0]
    assert frame["tool_result"]["status"] == "no_results" and fake_zoho_state["searches"] == ["Leads", "Contacts"]


def test_tool_fallback_binding_validated(client):
    spec = client.get("/v1/tools/search_zoho_lead").json()
    body = {"description": spec["description"], "input_schema": spec["input_schema"]}
    bad = {"connection": "zoho", "tool": "ZohoCRM_searchRecords",
           "fallback": {"connection": "zoho", "tool": "ZohoCRM_searchRecords", "arguments": {"q": "{phone}"}}}
    assert_error(client.put("/v1/tools/search_zoho_lead", json={**body, "remote": bad}), 422, "invalid_remote")
    good = {**bad, "fallback": {"connection": "zoho", "tool": "ZohoCRM_searchRecords", "arguments": {"q": "{email}"}}}
    r = client.put("/v1/tools/search_zoho_lead", json={**body, "remote": good})
    assert r.status_code == 200 and r.json()["remote"]["fallback"]["arguments"] == {"q": "{email}"}


@pytest.mark.parametrize("slot, text, value", [
    ("phone", "my number is 555-555-5555", "555-555-5555"),
    ("phone", "call me on +1 (415) 555 0134 please", "+1 (415) 555 0134"),
    ("phone", "order ORD-48213 from 2026", None),
    ("company_name", "Hi, I'm calling from Truhlar And Truhlar about our account", "Truhlar And Truhlar"),
    ("company_name", "I'm with the Rangoni Of Florence team", "Rangoni Of Florence"),
    ("company_name", "can you look up our company account", None),
    ("deal_name", "where are we on the Commercial Press deal?", "Commercial Press"),
    ("deal_name", "The King deal, has it closed?", "King"),
    ("deal_name", "what stage is the deal with Truhlar And Truhlar Attys in", "Truhlar And Truhlar Attys"),
    ("deal_name", "how is our deal going", None),
])
def test_crm_slot_extractors(slot, text, value):
    from ciap.pipeline.slots import SLOT_EXTRACTORS
    assert SLOT_EXTRACTORS[slot](text) == value


def test_responder_prune_and_fallback_render():
    data = {"data": [
        {"id": "1", "$approval": {"x": 1}, "First_Name": "Jane", "Company": "", "Owner": {"name": "Priya"},
         "Tags": [], "leadStatus": "Contacted"},
        {"id": "2", "First_Name": "John"},
    ], "info": {"count": 2}}
    pruned = responder.prune(data)
    assert pruned["data"][0] == {"id": "1", "First_Name": "Jane", "Owner": {"name": "Priya"}, "leadStatus": "Contacted"}
    assert responder.render_fallback("search_zoho_lead", pruned) == (
        "Zoho CRM Customer Lookup found 2 results: (1) first name Jane, owner name Priya, lead status Contacted. "
        "(2) first name John."
    )
    assert responder.render_fallback("search_zoho_lead", {"text": "No such lead."}) == "Zoho CRM Customer Lookup: No such lead."


def test_responder_full_mode_limits():
    data = {"data": [{"First_Name": f"Lead{i}", "Notes": "x" * 500} for i in range(8)]}
    focused, full = responder.prune(data), responder.prune(data, "full")
    assert len(focused["data"]) == 5 and len(focused["data"][0]["Notes"]) == 303
    assert len(full["data"]) == 8 and full["data"][0]["Notes"] == "x" * 500
    assert "Showing the first 3." in responder.render_fallback("search_zoho_lead", full)
    assert "Showing the first" not in responder.render_fallback("search_zoho_lead", full, "full")
    frame = state_store.Frame(id=1, intent="lookup_crm_record", status="FIRED",
                              tool_call={"tool": "search_zoho_lead", "arguments": {}})
    assert "holds 8 records" in responder._prompt(frame, full, "full")
    assert "records" not in responder._prompt(frame, full, "focused").split("Rules:")[0]


@pytest.mark.parametrize("data, key, arguments, first", [
    # Zoho: {"data": [...]}, flat fields
    ({"data": [{"Email": f"lead{i}@example.com"} for i in range(8)]}, "data",
     {"email": "LEAD6@example.com"}, {"Email": "lead6@example.com"}),
    # HubSpot: {"results": [...]}, fields nested under "properties"
    ({"results": [{"id": str(i), "properties": {"email": f"c{i}@example.com"}} for i in range(12)]}, "results",
     {"email": "c11@example.com"}, {"id": "11", "properties": {"email": "c11@example.com"}}),
    # Salesforce: {"records": [...]}, phone formatted differently from the lookup
    ({"totalSize": 7, "records": [{"Name": f"Acct {i}", "Phone": f"(555) 010-12{i:02d}"} for i in range(7)]}, "records",
     {"phone": "+1 555 010 1206"}, {"Name": "Acct 6", "Phone": "(555) 010-1206"}),
])
def test_responder_exact_lookup_match_kept_first(data, key, arguments, first):
    relevance = responder.Relevance("can you pull up my account", arguments)
    pruned = responder.prune(data, "focused", relevance)
    assert pruned[key][0] == first and len(pruned[key]) == 5  # survives the focused list limit
    assert pruned[key].total == len(data[key]) and pruned[key].positions[0] == data[key].index(first) + 1


def test_responder_exact_match_beats_partial():
    data = {"data": [{"Name": "Rangoni Of Florence"}, {"Name": "Kidman"}, {"Name": "Rangoni"}, {"Name": "Acme"}]}
    pruned = responder.prune(data, "focused", responder.Relevance("look up the account", {"name": "rangoni"}))
    assert [r["Name"] for r in pruned["data"][:2]] == ["Rangoni", "Rangoni Of Florence"]
    assert pruned["data"].positions[:2] == [3, 1]


def test_responder_keeps_records_closest_in_meaning():
    # nothing in the question matches a value exactly: the closed deal is 8th of 8
    deals = [{"Deal_Name": f"Order {i}", "Stage": "Qualification", "Amount": 1000 + i} for i in range(7)]
    deals.append({"Deal_Name": "Order 7", "Stage": "Closed Won", "Amount": 5400})
    pruned = responder.prune({"data": deals}, "focused",
                             responder.Relevance("which of his deals did we close and win?", {"name": "Acme"}))
    assert pruned["data"][0]["Stage"] == "Closed Won" and pruned["data"].positions[0] == 8


def test_responder_long_text_keeps_passages_about_the_call():
    notes = ("Called about the spring catalogue and wanted samples of the oak range. " * 4
             + "He said the delivery must arrive before the 14th or he will cancel the order. "
             + "Prefers email over phone calls for follow ups. " * 4)
    relevance = responder.Relevance("what did he say about the delivery date?")
    excerpt = responder.prune({"Notes": notes}, "focused", relevance)["Notes"]
    assert "delivery must arrive before the 14th" in excerpt
    assert len(excerpt) <= 300 + 20 and "..." in excerpt
    assert responder.prune({"Notes": notes})["Notes"] == notes[:300] + "..."  # no question: the first 300


def test_responder_prompt_says_what_was_left_out():
    rows = [{"Product": f"Item {i}", "Status": "Shipped"} for i in range(8)]
    rows[6]["Status"] = "Backordered"
    pruned = responder.prune({"lines": rows}, "focused", responder.Relevance("which item is backordered?"))
    text = responder.as_text(pruned)
    assert "showing 5 of 8, most relevant to the call first" in text and "- item 7:" in text
    plain = responder.as_text(responder.prune({"lines": rows}))
    assert "showing 5 of 8, the first ones" in plain and "- item 7:" not in plain


def test_responder_finds_records_nested_deep_and_keeps_tag_order():
    data = {"response": {"result": {"items": [{"email": f"u{i}@example.com"} for i in range(9)]}},
            "tags": [f"tag{i}" for i in range(9)]}
    pruned = responder.prune(data, "focused", responder.Relevance("look me up", {"email": "u8@example.com"}))
    assert pruned["response"]["result"]["items"][0] == {"email": "u8@example.com"}
    assert pruned["tags"].positions == sorted(pruned["tags"].positions) and len(pruned["tags"]) == 5


def test_responder_fallback_counts_the_whole_result():
    data = {"data": [{"Email": f"lead{i}@example.com"} for i in range(8)]}
    ranked = responder.prune(data, "focused", responder.Relevance("find me", {"email": "lead7@example.com"}))
    rendered = responder.render_fallback("search_zoho_lead", ranked)
    assert rendered.startswith("Zoho CRM Customer Lookup found 8 results: (1) email lead7@example.com.")
    assert rendered.endswith("Showing the 3 most relevant.")
    assert responder.render_fallback("search_zoho_lead", responder.prune(data)).endswith("Showing the first 3.")


def test_responder_relevance_edge_cases():
    data = {"data": [{"Name": "Zoë Åström", "Note": "日本語のメモ"}, {"Name": "Ann"}]}
    for question in ("", "   ", "¿dónde está mi pedido?", "x" * 5000):
        pruned = responder.prune(data, "focused", responder.Relevance(question, {"name": "Ann"}))
        assert pruned["data"][0] == {"Name": "Ann"} and len(pruned["data"]) == 2
    assert responder.prune({"data": []}, "focused", responder.Relevance("hi")) is None
    assert responder.prune({"data": [{}, {"x": ""}]}, "focused", responder.Relevance("hi")) is None
    one_sentence = "word " * 200  # no sentence breaks: plain cut
    assert responder.prune({"t": one_sentence}, "focused", responder.Relevance("word"))["t"].endswith("...")
    # an exact match is found anywhere; only the first MAX_SCORED_ITEMS are compared by meaning
    big = {"data": [{"email": f"u{i}@example.com"} for i in range(1000)]}
    pruned = responder.prune(big, "focused", responder.Relevance("find me", {"email": "u999@example.com"}))
    assert pruned["data"].total == 1000 and pruned["data"].positions[0] == 1000
    assert max(pruned["data"].positions[1:]) <= responder.MAX_SCORED_ITEMS


def test_responder_prompt_sees_match_past_list_limit(monkeypatch):
    prompts = []
    monkeypatch.setattr(responder, "_ask_model", lambda prompt, mode="focused": prompts.append(prompt) or "ok")
    frame = state_store.Frame(id=1, intent="lookup_crm_record", status="FIRED",
                              tool_call={"tool": "search_zoho_lead", "arguments": {"email": "lead7@example.com"}})
    frame.tool_result = {"status": "success", "message": None,
                         "data": {"data": [{"Email": f"lead{i}@example.com"} for i in range(8)]}}
    assert responder.compose_answer(frame)["source"] == "llm"
    assert "lead7@example.com" in prompts[0].split("Result:")[1]


def _raw_frame(data):
    frame = state_store.Frame(id=1, intent="lookup_crm_record", status="FIRED",
                              tool_call={"tool": "search_zoho_lead", "arguments": {"email": "lead7@example.com"}},
                              utterances=[{"speaker": "customer", "text": "look me up"}])
    frame.tool_result = {"status": "success", "message": None, "data": data}
    return frame


def test_responder_unpruned_sends_the_raw_result(monkeypatch):
    prompts = []
    monkeypatch.setattr(responder, "_ask_model", lambda prompt, mode="focused": prompts.append(prompt) or "ok")
    monkeypatch.setattr(config, "RESPONDER_PRUNE_RESULTS", False)
    monkeypatch.setattr(config, "RESPONDER_NUM_CTX", 32768)
    data = {"data": [{"id": str(i), "$approval": {"delegate": False}, "Email": f"lead{i}@example.com",
                      "Notes": "x" * 2000, "Company": ""} for i in range(12)]}
    assert responder.compose_answer(_raw_frame(data))["source"] == "llm"
    assert prompts[0].split("Result:\n")[1].split("\n\nRespond")[0] == json.dumps(data)  # untouched
    assert "left out" not in prompts[0]
    assert responder._context_option() == {"num_ctx": 32768}
    # a bare list or string result is sent too
    responder.compose_answer(_raw_frame(["a", "b"]))
    assert '{"result": ["a", "b"]}' in prompts[1]


def test_responder_unpruned_result_too_big_for_context(monkeypatch):
    monkeypatch.setattr(responder, "_ask_model", lambda *a: pytest.fail("model must not be called"))
    monkeypatch.setattr(config, "RESPONDER_PRUNE_RESULTS", False)
    monkeypatch.setattr(config, "RESPONDER_NUM_CTX", 2048)
    data = {"data": [{"$approval": {"delegate": False}, "Email": f"lead{i}@example.com", "Notes": "y" * 900}
                     for i in range(12)]}
    answer = responder.compose_answer(_raw_frame(data))
    assert answer["source"] == "fallback" and "too large for the model's context" in answer["note"]
    assert answer["text"].startswith("Zoho CRM Customer Lookup found 12 results:") and "approval" not in answer["text"]


def test_responder_pruned_by_default_with_ollama_context(monkeypatch):
    monkeypatch.setattr(config, "RESPONDER_NUM_CTX", None)
    assert config.RESPONDER_PRUNE_RESULTS is True and responder._context_option() == {}
    assert responder._fits_context("z" * 10**6, "full")


def test_responder_grounding_check():
    data = {"amount": 1499.0, "email": "Jane@Example.com"}
    assert responder.ungrounded_values("Paid 1,499 by jane@example.com.", data) == []
    assert responder.ungrounded_values("Paid 2,000 by bob@example.com.", data) == ["bob@example.com", "2000"]


def test_zoho_failure_is_a_tool_error_not_a_500(client):
    fake_zoho_state["down"] = True
    cid = new_call(client)
    r = turn(client, cid, "look me up at jane@example.com")
    assert r.status_code == 200
    result = r.json()["slots"]["frames"][0]["tool_result"]
    assert result["status"] == "error" and result["data"] is None and "zoho down" in result["message"]


def test_zoho_no_match_is_no_results(client):
    cid = new_call(client)
    frame = turn(client, cid, "look me up at nobody@example.com").json()["slots"]["frames"][0]
    assert frame["tool_result"] == {"status": "no_results", "data": None, "message": mcp_tools.NO_RESULTS_MESSAGE}


def test_tool_returning_nothing_is_no_results(client):
    spec = client.get("/v1/tools/search_zoho_lead").json()
    client.put("/v1/tools/search_zoho_lead", json={
        "description": spec["description"], "input_schema": spec["input_schema"],
        "remote": {"connection": "zoho", "tool": "empty_tool"},
    })
    cid = new_call(client)
    frame = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]
    assert frame["tool_result"]["status"] == "no_results"
    assert frame["tool_result"]["message"] == mcp_tools.NO_RESULTS_MESSAGE


@pytest.mark.parametrize("value, empty", [
    (None, True), ({}, True), ("", True), ({"data": []}, True), ({"text": ""}, True),
    ({"data": [], "info": {}}, True), ({"result": []}, True),
    ({"data": [{"id": "1"}]}, False), ({"count": 0}, False), ({"ok": False}, False), ({"text": "hi"}, False),
])
def test_empty_results_detected(value, empty):
    assert mcp_tools._is_empty(value) is empty


def test_zoho_failure_status_is_a_tool_error(client):
    fake_zoho_state["unauthorised"] = True
    cid = new_call(client)
    frame = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]
    result = frame["tool_result"]
    assert result["status"] == "error" and result["data"] is None and "Connection not authorised" in result["message"]


# -- validation & errors -------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"utterance": ""}, {"utterance": "   "}, {"utterance": "x" * 2001},
    {"utterance": "hi", "speaker": "robot"}, {},
])
def test_turn_validation(client, payload):
    cid = new_call(client)
    r = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/turns", json=payload)
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
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["turn_count"] == 0


# -- slot override (archived) ----------------------------------------------------

def test_override_disabled_by_default(client, demo_ui):
    cid = new_call(client)
    turn(client, cid, "track my order")
    r = client.patch(f"/v1/tenants/{TENANT}/calls/{cid}/frames/1/slots/order_id", json={"value": "ORD-22222"})
    assert_error(r, 404, "endpoint_disabled")
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["turn_count"] == 1
    assert client.get("/demo/config").json() == {"slot_override_enabled": False}


def test_override_listed_as_archived(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/v1/tenants/{tenant_id}/calls/{call_id}/frames/{frame_id}/slots/{slot_name}"]["patch"]
    assert op["tags"] == ["archived"] and op["deprecated"] is True
    assert spec["tags"][-1]["name"] == "archived"


def test_override_refires_tool_with_corrected_value(client, slot_override):
    cid = new_call(client)
    turn(client, cid, "track my order")
    turn(client, cid, "ORD-11111")
    r = client.patch(f"/v1/tenants/{TENANT}/calls/{cid}/frames/1/slots/order_id", json={"value": "ORD-22222"})
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
    url = f"/v1/tenants/{TENANT}/calls/{cid}/frames/1/slots"
    assert_error(client.patch(f"/v1/tenants/{TENANT}/calls/{cid}/frames/99/slots/order_id", json={"value": "ORD-1234"}), 404, "frame_not_found")
    assert_error(client.patch(f"{url}/email", json={"value": "a@b.co"}), 422, "invalid_slot")
    assert_error(client.patch(f"{url}/order_id", json={"value": "no id here"}), 422, "invalid_slot_value")
    assert_error(client.patch(f"{url}/order_id", json={"value": ""}), 422, "validation_error")
    # Nothing above changed anything or consumed a turn index.
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["turn_count"] == 1


def test_override_on_reverted_frame_is_409(client, slot_override):
    cid = new_call(client)
    turn(client, cid, "cancel it")  # opens cancel_order, awaiting order_id
    assert turn(client, cid, "actually never mind").json()["intent"]["turn_type"] == "ambiguous_continuation"
    # Complete the frame first, then revert it with a correction.
    turn(client, cid, "ORD-12345")
    assert turn(client, cid, "actually never mind").json()["intent"]["turn_type"] == "correction"
    r = client.patch(f"/v1/tenants/{TENANT}/calls/{cid}/frames/1/slots/order_id", json={"value": "ORD-99999"})
    assert_error(r, 409, "frame_reverted")


# -- transcript ----------------------------------------------------------------

def test_transcript_text(client):
    cid = new_call(client)
    text = "Customer: please track my order\nAgent: sure, what is the order number?\nCustomer: ORD-99213\nnoise line"
    r = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/transcript", json={"transcript": text})
    assert r.status_code == 200
    body = r.json()
    assert [t["speaker"] for t in body["turns"]] == ["customer", "agent", "customer"]
    assert [t["turn_index"] for t in body["turns"]] == [1, 2, 3]
    assert body["frames"][0]["status"] == "FIRED"


def test_transcript_turns_list(client):
    cid = new_call(client)
    r = client.post(f"/v1/tenants/{TENANT}/calls/{cid}/transcript", json={"turns": [
        {"utterance": "track my order"}, {"utterance": "ORD-99213", "speaker": "customer"},
    ]})
    assert r.status_code == 200 and r.json()["frames"][0]["status"] == "FIRED"


@pytest.mark.parametrize("payload", [
    {}, {"transcript": "a", "turns": [{"utterance": "b"}]}, {"transcript": "no tags at all"},
    {"turns": [{"utterance": "ok"}, {"utterance": ""}]},
    {"transcript": "Customer: " + "x" * 2001},
])
def test_transcript_validation_processes_nothing(client, payload):
    cid = live_call(client)
    assert_error(client.post(f"/v1/tenants/{TENANT}/calls/{cid}/transcript", json=payload), 422, "validation_error")
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["turn_count"] == 1


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
    from ciap import catalog

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
    assert client.get("/v1/tools/search_zoho_lead").json()["label"] == "Zoho CRM Customer Lookup"


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

    state = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()
    assert state["frames"][0]["intent_label"] == "Track Order"

    blocked = turn(client, cid, "track and cancel both").json()
    assert blocked["intent"]["selected_intent_label"] == "Track Order + Cancel Order"

    unknown = turn(client, cid, "hello there").json()
    assert unknown["intent"]["selected_intent"] is None and unknown["intent"]["selected_intent_label"] is None


def test_label_create_update_persist_and_fallback(client):
    from ciap import catalog

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
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]["intent_label"] == "Order Tracking"


# -- renames ------------------------------------------------------------------

def test_intent_rename_moves_spec_mapping_and_live_frames(client, monkeypatch):
    from ciap import catalog
    from ciap.pipeline import embeddings

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
    frame = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]
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

    frame = client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]
    assert frame["tool_call"]["tool"] == "get_status_order"

    cid2 = new_call(client)
    v = turn(client, cid2, "track my order ORD-777777").json()
    assert v["mcp"]["fired_this_turn"][0]["tool"] == "get_status_order"


def test_renamed_live_tool_still_executes(client):
    r = client.post("/v1/tools/search_zoho_lead/rename", json={"new_name": "crm_lead_search"})
    assert r.json()["executes_live"] is True
    # A PUT that leaves out `remote` keeps it live too.
    spec = r.json()
    r = client.put("/v1/tools/crm_lead_search", json={"description": "Find a CRM lead.", "input_schema": spec["input_schema"]})
    assert r.json()["executes_live"] is True

    cid = new_call(client)
    v = turn(client, cid, "look me up, jane@example.com").json()
    frame = v["slots"]["frames"][0]
    assert frame["tool_call"]["tool"] == "crm_lead_search"
    assert frame["tool_result"] == {"status": "success", "data": {"data": [{"Email": "jane@example.com"}]}, "message": None}


def test_old_saved_catalog_keeps_zoho_live(client):
    from ciap import catalog

    # Saved before tools had a "backend" (live by name) ...
    tools = {n: {k: v for k, v in spec.items() if k != "remote"} for n, spec in mcp_tools.TOOLS.items()}
    Path(config.CATALOG_PATH).write_text(json.dumps({"intents": {}, "tools": tools}), encoding="utf-8")
    assert catalog.load()
    assert client.get("/v1/tools/search_zoho_lead").json()["remote"] == mcp_tools.ZOHO_LEAD_SEARCH_REMOTE

    # ... and with one, under a renamed tool.
    tools = {("crm_lookup" if n == "search_zoho_lead" else n): {k: v for k, v in spec.items() if k != "remote"}
             for n, spec in mcp_tools.TOOLS.items()}
    tools["crm_lookup"]["backend"] = "zoho_lead_search"
    Path(config.CATALOG_PATH).write_text(json.dumps({"intents": {}, "tools": tools}), encoding="utf-8")
    assert catalog.load()
    info = client.get("/v1/tools/crm_lookup").json()
    assert info["executes_live"] is True and info["remote"]["connection"] == "zoho"
    assert "backend" not in mcp_tools.TOOLS["crm_lookup"]


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
    from ciap.pipeline import embeddings

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
    assert client.get(f"/v1/tenants/{TENANT}/calls/{cid}").json()["frames"][0]["required"] == []


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
    assert client.get("/v1/slots").json() == ["company_name", "deal_name", "discount_code", "email", "new_date", "order_id", "phone", "product_name"]

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


def test_log_records_carry_tenant_call_and_user_ids(client):
    cid = pbx_id()
    client.post(f"/v1/tenants/{TENANT}/calls/{cid}/turns", json={"utterance": "track my order", "user_id": "agent-7"})
    client.post(f"/v1/tenants/{TENANT}/calls/{cid}/turns", json={"utterance": "ORD-12345", "user_id": "agent-9"})
    client.post(f"/v1/tenants/globex/calls/{cid}/transcript",
                json={"transcript": "Customer: cancel my order", "user_id": "agent-7"})
    client.post(f"/v1/tenants/{TENANT}/calls/{cid}/reset")

    recs = client.get("/v1/log", params={"tenant_id": TENANT}).json()["records"]
    assert [(r["event"], r.get("user_id")) for r in recs] == [
        ("call_created", "agent-7"), ("turn", "agent-7"), ("turn", "agent-9"), ("call_reset", None),
    ]
    assert all(r["tenant_id"] == TENANT and r["call_id"] == cid for r in recs)

    by_user = client.get("/v1/log", params={"user_id": "agent-7"}).json()["records"]
    assert [(r["tenant_id"], r["event"]) for r in by_user] == [
        (TENANT, "call_created"), (TENANT, "turn"), ("globex", "call_created"), ("globex", "turn"),
    ]
    both = client.get("/v1/log", params={"tenant_id": "globex", "call_id": cid}).json()
    assert both["total"] == 2


@pytest.mark.parametrize("user_id", ["", "   ", "x" * 129])
def test_invalid_user_id_is_rejected(client, user_id):
    r = client.post(f"/v1/tenants/{TENANT}/calls/{pbx_id()}/turns", json={"utterance": "hi", "user_id": user_id})
    assert_error(r, 422, "validation_error")


def test_health_ok_and_degraded(client, monkeypatch):
    from ciap.pipeline import llm

    monkeypatch.setattr(llm._client, "list", lambda: [])
    assert client.get("/health").json() == {"status": "ok", "ollama": True, "model": llm.MODEL_NAME}

    def down():
        raise ConnectionError
    monkeypatch.setattr(llm._client, "list", down)
    assert client.get("/health").json()["status"] == "degraded"


def test_demo_ui_disabled_by_default(client):
    for path in ["/demo", "/demo/config", "/demo/static/app.js"]:
        assert_error(client.get(path), 404, "not_found")


def test_demo_page_and_assets_and_no_legacy_routes(client, demo_ui):
    assert client.get("/demo").status_code == 200
    assert client.get("/demo/static/app.js").status_code == 200
    assert client.get("/demo/static/../server.py").status_code == 404
    assert client.post("/calls").status_code == 404
    assert client.post("/demo/calls").status_code == 404


def test_openapi_documents_all_v1_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/calls" not in paths
    for p in ["/v1/tenants/{tenant_id}/calls/{call_id}", "/v1/tenants/{tenant_id}/calls/{call_id}/turns", "/v1/tenants/{tenant_id}/calls/{call_id}/transcript",
              "/v1/tenants/{tenant_id}/calls/{call_id}/reset", "/v1/tenants/{tenant_id}/calls/{call_id}/frames/{frame_id}/slots/{slot_name}",
              "/v1/intents", "/v1/intents/{name}", "/v1/tools", "/v1/tools/{name}", "/v1/slots",
              "/v1/log", "/health"]:
        assert p in paths, p


# -- MCP connections and remote bindings ------------------------------------------

def test_unconfigured_connection_is_a_tool_error(client):
    spec = client.get("/v1/tools/search_zoho_lead").json()
    r = client.put("/v1/tools/search_zoho_lead", json={
        "description": spec["description"], "input_schema": spec["input_schema"],
        "remote": {"connection": "salesforce", "tool": "ZohoCRM_searchRecords"},
    })
    assert r.status_code == 200, r.text
    cid = new_call(client)
    frame = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]
    result = frame["tool_result"]
    assert result["status"] == "error" and result["data"] is None and "not configured" in result["message"]


def test_remote_binding_can_be_set_and_removed(client):
    body = {**NEW_TOOL, "remote": {"connection": "zoho", "tool": "ZohoCRM_searchRecords",
                                   "arguments": {"path_variables": {"module": "Orders"},
                                                 "query_params": {"criteria": "(Order_ID:equals:{order_id})"}}}}
    r = client.post("/v1/tools", json=body)
    assert r.status_code == 201, r.text
    assert r.json()["executes_live"] is True and r.json()["remote"]["tool"] == "ZohoCRM_searchRecords"

    r = client.put("/v1/tools/add_gift_wrap", json={**{k: NEW_TOOL[k] for k in ("description", "input_schema")},
                                                    "remote": None})
    assert r.json()["executes_live"] is False and r.json()["remote"] is None


@pytest.mark.parametrize("remote", [
    {"connection": "zoho", "tool": "ZohoCRM_searchRecords",
     "arguments": {"query_params": {"email": "{email}"}}},  # not an argument of this tool
    {"connection": "Bad Name", "tool": "ZohoCRM_searchRecords"},
    {"connection": "zoho", "tool": " "},
])
def test_invalid_remote_binding_rejected(client, remote):
    r = client.post("/v1/tools", json={**NEW_TOOL, "remote": remote})
    assert r.status_code == 422
    assert r.json()["error"]["code"] in ("invalid_remote", "validation_error")
    assert_error(client.get("/v1/tools/add_gift_wrap"), 404, "tool_not_found")


def test_remote_arguments_rendered_from_tool_arguments():
    template = {"module": "Leads", "criteria": "(Email:equals:{email})", "raw": "{email}", "limit": 1}
    assert mcp_tools.render_arguments(template, {"email": "a@b.co"}) == {
        "module": "Leads", "criteria": "(Email:equals:a@b.co)", "raw": "a@b.co", "limit": 1}


def test_connections_and_their_tools_listed(client):
    assert client.get("/v1/connections").json() == ["zoho"]
    tools = {t["name"]: t for t in client.get("/v1/connections/zoho/tools").json()}
    assert set(tools["ZohoCRM_searchRecords"]["input_schema"]["properties"]) == {"path_variables", "query_params"}
    assert tools["slow_tool"]["annotations"] == {"readOnlyHint": False}
    assert_error(client.get("/v1/connections/nope/tools"), 404, "connection_not_found")


def test_slow_remote_tool_times_out_as_a_tool_error(client):
    shipped = json.loads(Path(config.MCP_CONNECTIONS_PATH).read_text())["zoho"]
    write_connections({"zoho": {**shipped, "timeout_seconds": 0.2}})
    spec = client.get("/v1/tools/search_zoho_lead").json()
    client.put("/v1/tools/search_zoho_lead", json={
        "description": spec["description"], "input_schema": spec["input_schema"],
        "remote": {"connection": "zoho", "tool": "slow_tool"},
    })
    cid = new_call(client)
    start = time.perf_counter()
    r = turn(client, cid, "look me up at jane@example.com")
    assert r.status_code == 200 and time.perf_counter() - start < 1
    result = r.json()["slots"]["frames"][0]["tool_result"]
    assert result["status"] == "error" and result["data"] is None and "didn't answer" in result["message"]


def test_shipped_connections_are_valid_and_keep_secrets_in_env_vars(monkeypatch):
    # connections.json is committed: every URL, header and credential must be a
    # ${NAME} reference, and each entry must pass validation. (An mcp_oauth URL
    # can be literal: it carries no key, the saved sign-in is the secret.)
    monkeypatch.setattr(config, "MCP_CONNECTIONS_PATH", str(ROOT / "connections.json"))
    shipped = json.loads((ROOT / "connections.json").read_text())
    for name, spec in shipped.items():
        auth = spec.get("auth") or {}
        url = None if auth.get("type") == "mcp_oauth" else spec.get("url")
        secrets = [url, *spec.get("headers", {}).values(),
                   *(auth.get(k) for k in ("token", "client_secret", "refresh_token"))]
        for value in filter(None, secrets):
            assert "${" in value, (name, value)
        for ref in mcp_client._ENV_REF.findall(json.dumps(spec)):
            monkeypatch.setenv(ref, "x" * 8)
        mcp_client._connection_spec(name)


# -- connection config -----------------------------------------------------------

def write_connections(connections):
    Path(config.MCP_CONNECTIONS_PATH).write_text(json.dumps(connections), encoding="utf-8")


@pytest.mark.parametrize("spec, problem", [
    ({"url": "https://x/mcp", "timeout": 5}, "timeout: Extra inputs are not permitted"),
    ({}, 'needs "url"'),
    ({"transport": "stdio"}, 'needs "command"'),
    ({"transport": "stdio", "command": "srv", "url": "https://x"}, "not url/headers/auth"),
    ({"url": "https://x", "auth": {"type": "basic"}}, "auth"),
    ({"url": "https://x", "auth": {"type": "oauth2", "client_id": "c"}}, "auth.oauth2.token_url"),
    ({"url": "https://x", "timeout_seconds": 0}, "timeout_seconds"),
    ({"url": "${NOT_SET_ANYWHERE}"}, "NOT_SET_ANYWHERE is not set"),
])
def test_invalid_connection_config_is_reported(client, spec, problem):
    write_connections({"zoho": spec})
    assert problem in mcp_client.check_connections()["zoho"]
    cid = new_call(client)
    result = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]["tool_result"]
    assert result["status"] == "error" and problem in result["message"]


def test_valid_connection_configs_accepted(client, monkeypatch):
    monkeypatch.setenv("CRM_TOKEN", "secret-token")
    write_connections({
        "a": {"url": "https://a/mcp", "auth": {"type": "bearer", "token": "${CRM_TOKEN}"}},
        "b": {"url": "https://b/sse", "transport": "sse", "headers": {"X-Api-Key": "${CRM_TOKEN}"}},
        "c": {"transport": "stdio", "command": "npx", "args": ["-y", "some-crm-mcp"], "env": {"KEY": "${CRM_TOKEN}"}},
        "d": {"url": "https://d/mcp", "auth": {"type": "oauth2", "token_url": "https://d/token",
                                                "client_id": "id", "client_secret": "${CRM_TOKEN}"}},
    })
    assert mcp_client.check_connections() == {"a": None, "b": None, "c": None, "d": None}
    spec = mcp_client._connection_spec("a")
    assert spec.redact("failed for https://a/mcp?k=secret-token") == "failed for https://a/mcp?k=***"


def test_allowed_tools_limit_bindings_and_calls(client):
    shipped = json.loads(Path(config.MCP_CONNECTIONS_PATH).read_text())["zoho"]
    write_connections({"zoho": {**shipped, "allowed_tools": ["ZohoCRM_searchRecords"]}})
    tools = {t["name"]: t["allowed"] for t in client.get("/v1/connections/zoho/tools").json()}
    assert tools["ZohoCRM_searchRecords"] is True and tools["slow_tool"] is False

    spec = client.get("/v1/tools/search_zoho_lead").json()
    r = client.put("/v1/tools/search_zoho_lead", json={
        "description": spec["description"], "input_schema": spec["input_schema"],
        "remote": {"connection": "zoho", "tool": "slow_tool"},
    })
    assert_error(r, 422, "invalid_remote")
    with pytest.raises(mcp_client.ToolCallFailed, match="allowed_tools"):
        mcp_client.call_tool("zoho", "slow_tool", {"email": "a@b.co"})

    cid = new_call(client)
    frame = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]
    assert frame["tool_result"]["status"] == "success"


def test_without_response_config_the_envelope_passes_through(client):
    write_connections({"zoho": {"url": "http://zoho.invalid/mcp"}})
    cid = new_call(client)
    result = turn(client, cid, "look me up at jane@example.com").json()["slots"]["frames"][0]["tool_result"]
    assert result["status"] == "success"
    assert result["data"] == {"status": "success", "data": {"data": [{"Email": "jane@example.com"}]}}


def test_changed_connection_config_reconnects(client):
    args = {"path_variables": {"module": "Leads"}, "query_params": {"email": "a@b.co"}}
    mcp_client.call_tool("zoho", "ZohoCRM_searchRecords", args)
    first = mcp_client._sessions["zoho"]
    mcp_client.call_tool("zoho", "ZohoCRM_searchRecords", args)
    assert mcp_client._sessions["zoho"] is first  # unchanged config: session reused

    shipped = json.loads(Path(config.MCP_CONNECTIONS_PATH).read_text())["zoho"]
    write_connections({"zoho": {**shipped, "timeout_seconds": 7}})
    mcp_client.call_tool("zoho", "ZohoCRM_searchRecords", args)
    assert mcp_client._sessions["zoho"] is not first


def _mock_token_endpoint(monkeypatch, handler):
    import httpx2

    real_client = httpx2.AsyncClient
    monkeypatch.setattr(httpx2, "AsyncClient", lambda **kw: real_client(transport=httpx2.MockTransport(handler), **kw))


def test_oauth2_token_fetched_cached_and_refreshed(monkeypatch):
    import asyncio
    import httpx2

    token_requests = []

    def token_endpoint(request):
        token_requests.append(dict(httpx2.QueryParams(request.content.decode())))
        n = len(token_requests)
        return httpx2.Response(200, json={"access_token": f"t{n}", "expires_in": 3600, "refresh_token": f"r{n}"})

    _mock_token_endpoint(monkeypatch, token_endpoint)
    flow = mcp_client._oauth2(mcp_client.OAuth2Auth(
        type="oauth2", token_url="https://crm/token", client_id="id", client_secret="sec",
        refresh_token="r0", scope="crm.read"))

    async def send(status):
        gen = flow.async_auth_flow(httpx2.Request("POST", "https://crm/mcp"))
        request = await gen.__anext__()
        sent = [request.headers["Authorization"]]
        try:
            request = await gen.asend(httpx2.Response(status))
            sent.append(request.headers["Authorization"])
        except StopAsyncIteration:
            pass
        return sent

    assert asyncio.run(send(200)) == ["Bearer t1"]
    assert asyncio.run(send(200)) == ["Bearer t1"]  # cached
    assert asyncio.run(send(401)) == ["Bearer t1", "Bearer t2"]  # refreshed on a 401
    assert token_requests[0] == {"client_id": "id", "client_secret": "sec", "scope": "crm.read",
                                 "grant_type": "refresh_token", "refresh_token": "r0"}
    assert token_requests[1]["refresh_token"] == "r1"  # the rotated refresh token is used


def test_oauth2_token_error_reported(monkeypatch):
    import asyncio
    import httpx2

    _mock_token_endpoint(monkeypatch, lambda request: httpx2.Response(200, json={"error": "invalid_client"}))
    flow = mcp_client._oauth2(mcp_client.OAuth2Auth(type="oauth2", token_url="https://crm/token", client_id="id"))

    async def send():
        await flow.async_auth_flow(httpx2.Request("POST", "https://crm/mcp")).__anext__()

    with pytest.raises(mcp_client.ToolCallFailed, match="invalid_client"):
        asyncio.run(send())
