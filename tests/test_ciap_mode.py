# CIP mode: each tenant's catalog comes from ciap-api and each call's state is
# loaded from and saved back to it around every turn, so the process keeps
# nothing between turns. ciap-api is faked here (FakeCiap); its HTTP client is
# tested on its own at the end against an httpx mock transport.

import json

import httpx
import pytest

from ciap import catalog
from ciap import ciap_api
from ciap import config
from ciap.pipeline import llm
from ciap.pipeline import state_store

from test_api import assert_error, client  # noqa: F401  (client: the shared fixture)

TENANT_A, TENANT_B = "200", "208"

CLINIC = {
    "tenantCode": TENANT_A,
    "version": 3,
    "intents": [
        {"name": "book_appointment", "label": "Book Appointment", "description": None,
         "phrases": ["I want to book an appointment", "Can I see the doctor"],
         "slots": [
             {"name": "visit_time", "label": "Time", "type": "TIME", "required": True, "prompt": "What time suits you?",
              "description": None, "options": []},
             {"name": "clinic", "label": "Clinic", "type": "OPTION", "required": False, "prompt": None,
              "description": None, "options": ["Downtown", "Airport"]},
         ]},
        {"name": "update_name", "label": "Update Name", "description": None,
         "phrases": ["I need to change the name on my file"],
         "slots": [{"name": "full_name", "label": "Full name", "type": "TEXT", "required": True,
                    "prompt": "What's the new name?", "description": None, "options": []}]},
        {"name": "no_phrases", "label": "Never matched", "description": None, "phrases": [], "slots": []},
    ],
}
SHOP = {
    "tenantCode": TENANT_B,
    "version": 1,
    "intents": [{"name": "refund", "label": "Refund", "description": None,
                 "phrases": ["I want my money back"], "slots": []}],
}


class FakeCiap:
    def __init__(self):
        self.catalogs = {TENANT_A: CLINIC, TENANT_B: SHOP}
        self.states = {}  # (tenant, call) -> (state, version)
        self.catalog_loads = []
        self.saves = []
        self.down = False
        self.conflicts_left = 0
        self.correlation_ids = []

    def get_catalog(self, tenant):
        self.correlation_ids.append(ciap_api.correlation_id.get())
        if self.down:
            raise ciap_api.CiapApiError("ciap-api unreachable: ConnectError")
        self.catalog_loads.append(tenant)
        if tenant not in self.catalogs:
            raise ciap_api.NotFound("Unknown tenant", 404)
        return json.loads(json.dumps(self.catalogs[tenant]))

    def load_state(self, tenant, call):
        if self.down:
            raise ciap_api.CiapApiError("ciap-api unreachable: ConnectError")
        if (tenant, call) not in self.states:
            raise ciap_api.NotFound("No Agent Assist call", 404)
        state, version = self.states[(tenant, call)]
        return json.loads(json.dumps(state)), version

    def save_state(self, tenant, call, state, expected):
        if self.conflicts_left:
            self.conflicts_left -= 1
            raise ciap_api.StateConflict("moved on", 409)
        current = self.states.get((tenant, call), (None, 0))[1]
        if current != expected:
            raise ciap_api.StateConflict("moved on", 409)
        json.dumps(state)  # must be plain JSON
        self.states[(tenant, call)] = (state, current + 1)
        self.saves.append((tenant, call, expected))
        return current + 1


def overlap_retrieve(text, top_k=3):
    """Ranks the turn's catalog by word overlap with each intent's phrases."""
    words = set(text.lower().split())
    scores = []
    for name, spec in catalog.current().intents.items():
        best = max((len(words & set(p.lower().split())) / len(set(p.lower().split())) for p in spec["exemplars"]), default=0)
        scores.append((name, round(min(best, 0.95), 3)))
    return sorted(scores, key=lambda kv: kv[1], reverse=True)[:top_k]


@pytest.fixture
def fake(client, monkeypatch):  # noqa: F811
    fake = FakeCiap()
    monkeypatch.setattr(config, "CIAP_API_BASE_URL", "http://ciap-api.test")
    for name in ("get_catalog", "load_state", "save_state"):
        monkeypatch.setattr(ciap_api, name, getattr(fake, name))
    monkeypatch.setattr(state_store, "retrieve", overlap_retrieve)
    catalog.clear_cache()
    yield fake
    catalog.clear_cache()


def turn(client, tenant, call, text, **body):  # noqa: F811
    return client.post(f"/v1/tenants/{tenant}/calls/{call}/turns", json={"utterance": text, **body})


# -- catalog per tenant ---------------------------------------------------------

def test_each_tenant_gets_its_own_catalog(client, fake):  # noqa: F811
    v = turn(client, TENANT_A, "c1", "I want to book an appointment").json()
    assert v["intent"]["selected_intent"] == "book_appointment"
    assert v["intent"]["selected_intent_label"] == "Book Appointment"
    assert {c["intent"] for c in v["intent"]["candidates"]} <= {"book_appointment", "update_name"}

    other = turn(client, TENANT_B, "c1", "I want to book an appointment").json()
    assert other["intent"]["selected_intent"] != "book_appointment"
    assert turn(client, TENANT_B, "c2", "I want my money back").json()["intent"]["selected_intent"] == "refund"


def test_tenant_catalog_listed(client, fake):  # noqa: F811
    intents = {i["name"]: i for i in client.get(f"/v1/tenants/{TENANT_A}/intents").json()}
    assert set(intents) == {"book_appointment", "update_name"}  # an intent with no phrases can't be matched
    assert intents["book_appointment"]["required_slots"] == ["visit_time"]
    assert intents["book_appointment"]["optional_slots"] == ["clinic"]
    assert intents["book_appointment"]["executes_live"] is False


def test_catalog_cached_and_reloaded_for_a_newer_version(client, fake):  # noqa: F811
    turn(client, TENANT_A, "c1", "I want to book an appointment", catalog_version=3)
    turn(client, TENANT_A, "c1", "at 3pm", catalog_version=3)
    assert fake.catalog_loads == [TENANT_A]

    fake.catalogs[TENANT_A] = {**CLINIC, "version": 4, "intents": [
        {"name": "cancel_appointment", "label": "Cancel Appointment", "description": None,
         "phrases": ["cancel my appointment"], "slots": []}]}
    v = turn(client, TENANT_A, "c2", "please cancel my appointment", catalog_version=4).json()
    assert fake.catalog_loads == [TENANT_A, TENANT_A]
    assert v["intent"]["selected_intent"] == "cancel_appointment"


def test_catalog_reloaded_after_max_age(client, fake, monkeypatch):  # noqa: F811
    now = [100.0]
    monkeypatch.setattr(catalog, "_clock", lambda: now[0])
    turn(client, TENANT_A, "c1", "I want to book an appointment")
    now[0] += config.CATALOG_MAX_AGE_SECONDS + 1
    turn(client, TENANT_A, "c1", "at 3pm")
    assert fake.catalog_loads == [TENANT_A, TENANT_A]


def test_ciap_api_down_without_a_cached_catalog_is_503(client, fake):  # noqa: F811
    fake.down = True
    assert_error(turn(client, TENANT_A, "c1", "I want to book an appointment"), 503, "catalog_unavailable")


def test_ciap_api_down_with_a_cached_catalog_keeps_it(client, fake):  # noqa: F811
    turn(client, TENANT_A, "c1", "I want to book an appointment")
    fake.down = True
    catalog._cache[TENANT_A].loaded_at -= config.CATALOG_MAX_AGE_SECONDS + 1  # due for a reload
    # The catalog falls back to the cached copy; the call state can't be loaded.
    assert_error(turn(client, TENANT_A, "c1", "at 3pm"), 503, "state_unavailable")


# -- call state in ciap-api -------------------------------------------------------

def test_state_saved_after_each_turn_and_picked_up_by_any_instance(client, fake):  # noqa: F811
    from ciap.api import server

    v = turn(client, TENANT_A, "c1", "I want to book an appointment").json()
    assert v["slots"]["frames"][0]["status"] == "AWAITING_INFO"
    assert not server._calls  # nothing kept in this process

    v = turn(client, TENANT_A, "c1", "Downtown at 3pm please").json()
    frame = v["slots"]["frames"][0]
    assert v["turn_index"] == 2
    assert frame["status"] == "FIRED"
    assert frame["slots"] == {"visit_time": "15:00", "clinic": "Downtown"}
    assert fake.saves == [(TENANT_A, "c1", 0), (TENANT_A, "c1", 1)]

    state = client.get(f"/v1/tenants/{TENANT_A}/calls/c1").json()
    assert state["turn_count"] == 2 and state["frames"][0]["intent_label"] == "Book Appointment"


def test_optional_slot_said_up_front_is_kept(client, fake):  # noqa: F811
    v = turn(client, TENANT_A, "c1", "I want to book an appointment at the Airport clinic").json()
    frame = v["slots"]["frames"][0]
    assert frame["slots"] == {"clinic": "Airport"} and frame["missing"] == ["visit_time"]


def test_text_slot_read_by_the_model_only_when_waited_on(client, fake, monkeypatch):  # noqa: F811
    asked = []
    monkeypatch.setattr(llm, "extract_text_slot",
                        lambda utterance, name, spec: asked.append((name, utterance)) or ("Jane Roe" if "Jane" in utterance else None))
    v = turn(client, TENANT_A, "c1", "I need to change the name on my file").json()
    assert v["slots"]["frames"][0]["missing"] == ["full_name"]
    v = turn(client, TENANT_A, "c1", "It's Jane Roe now").json()
    assert v["slots"]["frames"][0]["slots"] == {"full_name": "Jane Roe"}
    assert v["slots"]["frames"][0]["status"] == "FIRED"
    assert [name for name, _ in asked] == ["full_name", "full_name"]


def test_model_text_slot_must_be_what_was_said(monkeypatch):
    monkeypatch.setattr(llm, "_ask", lambda prompt: {"value": "John Smith"})
    assert llm.extract_text_slot("my name is Jane", "full_name", {"label": "Full name"}) is None
    monkeypatch.setattr(llm, "_ask", lambda prompt: {"value": "Jane"})
    assert llm.extract_text_slot("my name is Jane", "full_name", {"label": "Full name"}) == "Jane"
    monkeypatch.setattr(llm, "_ask", lambda prompt: {"_error": "down"})
    assert llm.extract_text_slot("my name is Jane", "full_name", {}) is None


def test_save_conflict_reruns_the_turn_once(client, fake):  # noqa: F811
    turn(client, TENANT_A, "c1", "I want to book an appointment")
    fake.conflicts_left = 1
    r = turn(client, TENANT_A, "c1", "at 3pm")
    assert r.status_code == 200 and r.json()["slots"]["frames"][0]["status"] == "FIRED"
    fake.conflicts_left = 2
    assert_error(turn(client, TENANT_A, "c1", "at 4pm"), 409, "state_conflict")


def test_intent_removed_mid_call_doesnt_hold_later_turns(client, fake):  # noqa: F811
    turn(client, TENANT_A, "c1", "I want to book an appointment", catalog_version=3)
    fake.catalogs[TENANT_A] = {**CLINIC, "version": 4, "intents": [CLINIC["intents"][1]]}
    v = turn(client, TENANT_A, "c1", "I need to change the name on my file", catalog_version=4).json()
    assert v["intent"]["selected_intent"] == "update_name"
    assert [f["intent"] for f in v["slots"]["frames"]] == ["book_appointment", "update_name"]


def test_unknown_call_is_404_and_delete_and_reset(client, fake):  # noqa: F811
    assert_error(client.get(f"/v1/tenants/{TENANT_A}/calls/nope"), 404, "call_not_found")
    assert_error(client.delete(f"/v1/tenants/{TENANT_A}/calls/nope"), 404, "call_not_found")
    turn(client, TENANT_A, "c1", "I want to book an appointment")
    assert client.post(f"/v1/tenants/{TENANT_A}/calls/c1/reset").status_code == 200
    assert client.get(f"/v1/tenants/{TENANT_A}/calls/c1").json() == {
        "tenant_id": TENANT_A, "call_id": "c1", "turn_count": 0, "frames": []}
    assert client.delete(f"/v1/tenants/{TENANT_A}/calls/c1").status_code == 204


def test_correlation_id_passed_to_ciap_api(client, fake):  # noqa: F811
    client.post(f"/v1/tenants/{TENANT_A}/calls/c1/turns", json={"utterance": "hello"}, headers={"X-Correlation-Id": "corr-9"})
    assert fake.correlation_ids == ["corr-9"]


# -- inbound auth -------------------------------------------------------------------

def test_api_key_required_when_set(client, fake, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "API_KEY", "s3cret")
    assert_error(turn(client, TENANT_A, "c1", "hello"), 401, "unauthorized")
    r = client.post(f"/v1/tenants/{TENANT_A}/calls/c1/turns", json={"utterance": "hello"}, headers={"X-Api-Key": "wrong"})
    assert_error(r, 401, "unauthorized")
    r = client.post(f"/v1/tenants/{TENANT_A}/calls/c1/turns", json={"utterance": "hello"}, headers={"X-Api-Key": "s3cret"})
    assert r.status_code == 200
    assert client.get("/health").status_code == 200  # ops stays open


# -- the ciap-api client --------------------------------------------------------------

def test_client_paths_headers_and_errors(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/state") and request.method == "PUT":
            body = json.loads(request.content)
            if body["expectedVersion"] != 2:
                return httpx.Response(409, json={"message": "Call state is at version 2"})
            return httpx.Response(200, json={"version": 3})
        if request.url.path.endswith("/state"):
            return httpx.Response(404, json={"message": "No Agent Assist call"})
        return httpx.Response(200, json={"tenantCode": "200", "version": 7, "intents": []})

    monkeypatch.setattr(ciap_api, "_client", httpx.Client(
        base_url="http://ciap-api.test", headers={"X-Api-Key": "k"}, transport=httpx.MockTransport(handler)))
    token = ciap_api.correlation_id.set("corr-1")
    try:
        assert ciap_api.get_catalog("200")["version"] == 7
        with pytest.raises(ciap_api.NotFound):
            ciap_api.load_state("200", "1727690000.12")
        assert ciap_api.save_state("200", "1727690000.12", {"store": {}}, 2) == 3
        with pytest.raises(ciap_api.StateConflict):
            ciap_api.save_state("200", "1727690000.12", {"store": {}}, 1)
    finally:
        ciap_api.correlation_id.reset(token)
        monkeypatch.setattr(ciap_api, "_client", None)
    assert seen[0].url.path == "/api/internal/assist/tenants/200/catalog"
    assert seen[1].url.path == "/api/internal/assist/tenants/200/calls/1727690000.12/state"
    assert all(r.headers["X-Api-Key"] == "k" and r.headers["X-Correlation-Id"] == "corr-1" for r in seen)


def test_catalog_from_ciap_maps_the_response():
    cat = catalog.from_ciap(TENANT_A, CLINIC)
    assert cat.version == 3
    assert cat.intents["book_appointment"] == {
        "exemplars": ["I want to book an appointment", "Can I see the doctor"],
        "required_slots": ["visit_time"], "optional_slots": ["clinic"], "label": "Book Appointment"}
    assert cat.slot("clinic")["options"] == ["Downtown", "Airport"]
    assert cat.slot("unknown")["type"] == "TEXT"


# -- slots by CIP type -------------------------------------------------------------------

@pytest.mark.parametrize("name, spec, text, value", [
    ("visit_time", {"type": "TIME"}, "can you do 3pm", "15:00"),
    ("visit_time", {"type": "TIME"}, "10:30 a.m. works", "10:30"),
    ("visit_time", {"type": "TIME"}, "around noon", "12:00"),
    ("visit_time", {"type": "TIME"}, "order 4", None),
    ("quantity", {"type": "NUMBER"}, "call 555-555-5555 for 3 items", "3"),
    ("quantity", {"type": "NUMBER"}, "about 1,250.5 units", "1250.5"),
    ("amount", {"type": "CURRENCY"}, "I was charged $12.50 twice", "12.50 USD"),
    ("amount", {"type": "CURRENCY"}, "it cost 40 euros", "40 EUR"),
    ("plan", {"type": "OPTION", "options": ["Express", "Express Plus"]}, "express plus please", "Express Plus"),
    ("plan", {"type": "OPTION", "options": ["Express"]}, "expressway", None),
    ("contact", {"type": "EMAIL"}, "it's jane@example.com", "jane@example.com"),
    ("callback", {"type": "PHONE"}, "on 555-555-5555", "555-555-5555"),
    ("order_id", {"type": "NUMBER"}, "order ORD-48213", "ORD-48213"),  # a named slot keeps its extractor
    ("nickname", {"type": "TEXT"}, "call me Jay", None),  # free text: the model's job
])
def test_slot_extraction_by_type(name, spec, text, value):
    from ciap.pipeline import slots
    assert slots.extract(name, spec, text) == value
