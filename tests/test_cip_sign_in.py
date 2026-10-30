# A tenant connection signed in from CIP ("auth": {"type": "cip_sign_in"}):
# ciap-api keeps the sign-in and hands out access tokens; this service calls
# the server with the one it has and asks ciap-api for a current one when it's
# about to expire or the server refuses it. Against a real local MCP server
# that requires a bearer token (test_mcp_oauth's).

import time

import pytest

from ciap import catalog
from ciap import ciap_api
from ciap import config
from ciap.integrations import mcp_client
from test_mcp_oauth import AuthServer

TENANT = "200"


@pytest.fixture
def crm(monkeypatch):
    server = AuthServer()
    server.connection = {"transport": "streamable_http", "url": f"{server.base}/mcp",
                         "auth": {"type": "cip_sign_in", "access_token": None, "expires_at": None}}
    server.token_asks = []
    server.ciap_answer = None  # what ciap-api's token endpoint does: a callable(rejected)

    def connection_token(tenant, name, rejected=None):
        server.token_asks.append((tenant, name, rejected))
        return server.ciap_answer(rejected)

    monkeypatch.setattr(ciap_api, "enabled", lambda: True)
    monkeypatch.setattr(ciap_api, "get_connection", lambda tenant, name: server.connection)
    monkeypatch.setattr(ciap_api, "connection_token", connection_token)
    monkeypatch.setattr(config, "MCP_ALLOW_PRIVATE_HOSTS", True)
    mcp_client._tenant_specs.clear()
    yield server
    mcp_client.shutdown()
    mcp_client._tenant_specs.clear()
    server.stop()


def signed_in(server, token, expires_in=3600):
    server.access_tokens.add(token)
    server.connection["auth"].update(access_token=token, expires_at=time.time() + expires_in)


def renews_to(server, token):
    def answer(rejected):
        server.access_tokens.add(token)
        return {"access_token": token, "expires_at": time.time() + 3600}
    server.ciap_answer = answer


def whoami():
    with catalog.using(catalog.from_ciap(TENANT, {"version": 1, "intents": []})):
        return mcp_client.call_tool("crm", "whoami", {}, timeout=10)


def test_the_token_from_ciap_api_is_used_as_is(crm):
    signed_in(crm, "at-1")
    assert whoami() == {"user": "agent@crm"}
    assert crm.token_asks == []


def test_an_expiring_token_is_renewed_through_ciap_api(crm):
    signed_in(crm, "at-1", expires_in=10)
    renews_to(crm, "at-2")
    assert whoami() == {"user": "agent@crm"}
    assert crm.token_asks == [(TENANT, "crm", "at-1")]


def test_a_token_the_server_refuses_is_reported_and_renewed(crm):
    crm.connection["auth"].update(access_token="revoked", expires_at=time.time() + 3600)
    renews_to(crm, "at-2")
    assert whoami() == {"user": "agent@crm"}
    assert ("200", "crm", "revoked") in crm.token_asks


def test_not_signed_in_says_so(crm):
    def refuse(rejected):
        raise ciap_api.StateConflict("MCP connection 'crm' isn't signed in", 409)
    crm.ciap_answer = refuse
    with pytest.raises(mcp_client.ToolCallFailed, match="isn't signed in"):
        whoami()


def test_the_access_token_is_redacted(crm):
    signed_in(crm, "access-token-secret")
    with catalog.using(catalog.from_ciap(TENANT, {"version": 1, "intents": []})):
        _, spec = mcp_client._resolve("crm")
    assert "access-token-secret" not in spec.redact("failed with access-token-secret")


def test_cip_sign_in_isnt_for_the_connections_file(monkeypatch, tmp_path):
    import json
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"crm": {"url": "https://crm.example.com/mcp", "auth": {"type": "cip_sign_in"}}}))
    monkeypatch.setattr(config, "MCP_CONNECTIONS_PATH", str(path))
    with pytest.raises(mcp_client.ConnectionNotConfigured, match="only for connections configured in CIP"):
        mcp_client._connection_spec("crm")
