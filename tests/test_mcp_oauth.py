# The MCP standard sign-in ("auth": {"type": "mcp_oauth"}) end to end, against
# a real local MCP server protected by its own OAuth authorization server:
# discovery, registration, PKCE, the localhost redirect, the saved sign-in,
# and refreshing (with rotated refresh tokens) at call time.

import asyncio
import base64
import hashlib
import json
import secrets
import socket
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from ciap import config
from ciap.integrations import mcp_client


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AuthServer:
    """A CRM's MCP server and its authorization server, as one local app."""

    def __init__(self):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.clients = {"app": "app-secret"}  # pre-registered: client_id -> secret
        self.codes = {}  # code -> (client_id, code_challenge, redirect_uri)
        self.access_tokens = set()
        self.refresh_tokens = set()
        self.token_requests = []
        self.issued = 0

        mcp = MCPServer("fake-crm")

        @mcp.tool()
        def whoami() -> dict:
            return {"user": "agent@crm"}

        self.app = mcp.streamable_http_app()
        prm = Route("/.well-known/oauth-protected-resource{rest:path}", self.protected_resource)
        self.app.router.routes[:0] = [
            prm,
            Route("/.well-known/oauth-authorization-server", self.authorization_server),
            Route("/authorize", self.authorize),
            Route("/token", self.token, methods=["POST"]),
            Route("/register", self.register, methods=["POST"]),
        ]
        self.app.add_middleware(_RequireBearer, server=self)
        self.server = uvicorn.Server(uvicorn.Config(self.app, port=self.port, log_level="warning"))
        threading.Thread(target=self.server.run, daemon=True).start()
        while not self.server.started:
            time.sleep(0.01)

    def stop(self):
        self.server.should_exit = True

    async def protected_resource(self, request):
        return JSONResponse({"resource": f"{self.base}/mcp", "authorization_servers": [self.base]})

    async def authorization_server(self, request):
        return JSONResponse({
            "issuer": self.base,
            "authorization_endpoint": f"{self.base}/authorize",
            "token_endpoint": f"{self.base}/token",
            "registration_endpoint": f"{self.base}/register",
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
        })

    async def register(self, request):
        metadata = await request.json()
        client_id, secret = f"dyn-{len(self.clients)}", secrets.token_hex(8)
        self.clients[client_id] = secret
        return JSONResponse({**metadata, "client_id": client_id, "client_secret": secret,
                             "token_endpoint_auth_method": "client_secret_post"}, status_code=201)

    async def authorize(self, request):
        q = request.query_params
        assert q["client_id"] in self.clients and q["code_challenge_method"] == "S256"
        code = secrets.token_hex(8)
        self.codes[code] = (q["client_id"], q["code_challenge"], q["redirect_uri"])
        return RedirectResponse(f"{q['redirect_uri']}?{urlencode({'code': code, 'state': q['state']})}")

    def _issue(self):
        self.issued += 1
        access, refresh = f"access-{self.issued}", f"refresh-{self.issued}"
        self.access_tokens.add(access)
        self.refresh_tokens.add(refresh)
        return JSONResponse({"access_token": access, "token_type": "bearer", "expires_in": 3600,
                             "refresh_token": refresh})

    async def token(self, request):
        form = dict(await request.form())
        self.token_requests.append(form)
        if self.clients.get(form.get("client_id")) != form.get("client_secret"):
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        if form.get("resource") != f"{self.base}/mcp":
            return JSONResponse({"error": "invalid_target"}, status_code=400)
        if form["grant_type"] == "authorization_code":
            client_id, challenge, redirect_uri = self.codes.pop(form["code"])
            verifier_hash = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest())
            assert verifier_hash.rstrip(b"=").decode() == challenge and form["redirect_uri"] == redirect_uri
            return self._issue()
        if form["refresh_token"] not in self.refresh_tokens:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        self.refresh_tokens.discard(form["refresh_token"])  # single use: rotated
        return self._issue()


class _RequireBearer:
    def __init__(self, app, server):
        self.app, self.server = app, server

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            token = Request(scope).headers.get("authorization", "").removeprefix("Bearer ")
            if token not in self.server.access_tokens:
                metadata = f'Bearer resource_metadata="{self.server.base}/.well-known/oauth-protected-resource"'
                await Response(status_code=401, headers={"WWW-Authenticate": metadata})(scope, receive, send)
                return
        await self.app(scope, receive, send)


@pytest.fixture
def crm(monkeypatch, tmp_path):
    server = AuthServer()
    monkeypatch.setattr(config, "MCP_CONNECTIONS_PATH", str(tmp_path / "connections.json"))
    monkeypatch.setattr(config, "MCP_TOKENS_DIR", str(tmp_path / "oauth"))
    monkeypatch.setenv("CRM_CLIENT_SECRET", "app-secret")
    yield server
    mcp_client.shutdown()
    server.stop()


def connect(server, **auth):
    redirect_uri = f"http://127.0.0.1:{free_port()}/callback"
    connection = {"url": f"{server.base}/mcp", "auth": {"type": "mcp_oauth", "redirect_uri": redirect_uri, **auth}}
    Path(config.MCP_CONNECTIONS_PATH).write_text(json.dumps({"crm": connection}), encoding="utf-8")


def approve_in_browser(url):
    # Stands in for the person: the sign-in page approves at once and the
    # browser follows the redirect to the localhost callback. (In a thread:
    # sign_in's loop is serving that callback.)
    threading.Thread(target=httpx.get, args=(url,), kwargs={"follow_redirects": True}, daemon=True).start()


def saved(name="crm"):
    return json.loads((Path(config.MCP_TOKENS_DIR) / f"{name}.json").read_text())


def call_whoami():
    return mcp_client.call_tool("crm", "whoami", {}, timeout=10)


@pytest.mark.parametrize("auth, client_id", [
    ({"client_id": "app", "client_secret": "${CRM_CLIENT_SECRET}"}, "app"),  # pre-registered app (e.g. HubSpot)
    ({}, "dyn-1"),  # the client registers itself
])
def test_sign_in_then_calls(crm, auth, client_id):
    connect(crm, **auth)
    assert "isn't signed in: run python scripts/connect.py crm" in mcp_client.check_connections()["crm"]
    with pytest.raises(mcp_client.SignInRequired, match="connect.py crm"):
        call_whoami()

    assert mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10) == ["whoami"]
    tokens = saved()
    assert tokens["access_token"] == "access-1" and tokens["refresh_token"] == "refresh-1"
    assert tokens["client_id"] == client_id and tokens["token_endpoint"] == f"{crm.base}/token"
    assert tokens["resource"] == f"{crm.base}/mcp"
    # A configured secret stays in the environment; a self-registered one is saved.
    assert (tokens["client_secret"] is None) == bool(auth)
    assert mcp_client.check_connections() == {"crm": None}
    assert call_whoami() == {"user": "agent@crm"}


def test_expired_token_is_refreshed_and_rotation_saved(crm):
    connect(crm, client_id="app", client_secret="${CRM_CLIENT_SECRET}")
    mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10)
    tokens = saved()
    tokens["expires_at"] = time.time() - 1
    mcp_client._save_tokens("crm", tokens)

    assert call_whoami() == {"user": "agent@crm"}
    refresh = crm.token_requests[-1]
    assert refresh["grant_type"] == "refresh_token" and refresh["refresh_token"] == "refresh-1"
    assert refresh["client_secret"] == "app-secret"
    assert saved()["refresh_token"] == "refresh-2"  # the rotated token survives a restart

    mcp_client.shutdown()  # a restart: only the file is left
    mcp_client._sessions.clear()
    assert call_whoami() == {"user": "agent@crm"}
    assert len(crm.token_requests) == 2  # the saved access token was still good


def test_revoked_access_token_is_refreshed_on_401(crm):
    connect(crm, client_id="app", client_secret="${CRM_CLIENT_SECRET}")
    mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10)
    call_whoami()
    crm.access_tokens.clear()
    mcp_client.shutdown()
    mcp_client._sessions.clear()
    assert call_whoami() == {"user": "agent@crm"}
    assert saved()["access_token"] == "access-2"


def test_rejected_sign_in_asks_to_sign_in_again(crm):
    connect(crm, client_id="app", client_secret="${CRM_CLIENT_SECRET}")
    mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10)
    crm.access_tokens.clear()
    crm.refresh_tokens.clear()  # e.g. the app's access was revoked in the CRM
    mcp_client.shutdown()
    mcp_client._sessions.clear()
    with pytest.raises(mcp_client.SignInRequired, match=r"rejected by the server \(invalid_grant\).*connect.py crm"):
        call_whoami()
    before = saved()
    assert before["refresh_token"] == "refresh-1"  # left for inspection, not wiped


def test_refresh_by_another_process_is_picked_up(crm):
    connect(crm, client_id="app", client_secret="${CRM_CLIENT_SECRET}")
    mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10)
    stale = saved()
    stale["expires_at"] = time.time() - 1

    # Another worker refreshes first: refresh-1 is used up, refresh-2 saved.
    asyncio.run(mcp_client._refresh_tokens("crm", mcp_client._connection_spec("crm"), stale))
    # This one still holds refresh-1; the server rejects it, so it re-reads the file.
    latest = asyncio.run(mcp_client._refresh_tokens("crm", mcp_client._connection_spec("crm"), stale))
    assert latest["refresh_token"] == "refresh-2"


def test_unreachable_token_endpoint_keeps_the_sign_in(crm):
    connect(crm, client_id="app", client_secret="${CRM_CLIENT_SECRET}")
    mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10)
    tokens = saved()
    tokens["expires_at"] = time.time() - 1
    tokens["token_endpoint"] = f"http://127.0.0.1:{free_port()}/token"  # nothing listening
    mcp_client._save_tokens("crm", tokens)
    with pytest.raises(mcp_client.ToolCallFailed, match="couldn't refresh its sign-in: ConnectError") as exc:
        call_whoami()
    assert not isinstance(exc.value, mcp_client.SignInRequired)
    assert saved() == tokens


def test_sign_in_for_another_url_or_app_is_not_used(crm):
    connect(crm, client_id="app", client_secret="${CRM_CLIENT_SECRET}")
    mcp_client.sign_in("crm", approve_in_browser, timeout_seconds=10)
    connections = json.loads(Path(config.MCP_CONNECTIONS_PATH).read_text())
    connections["crm"]["auth"]["client_id"] = "other-app"
    Path(config.MCP_CONNECTIONS_PATH).write_text(json.dumps(connections))
    assert "signed in with a different client_id" in mcp_client.check_connections()["crm"]
    connections["crm"]["url"] = f"{crm.base}/other"
    Path(config.MCP_CONNECTIONS_PATH).write_text(json.dumps(connections))
    assert "signed in to a different url" in mcp_client.check_connections()["crm"]


@pytest.mark.parametrize("auth, problem", [
    ({"client_secret": "s"}, '"client_secret" needs "client_id"'),
    ({"redirect_uri": "https://ciap.com/callback"}, "must be http://localhost"),
    ({"redirect_uri": "http://localhost/callback"}, "must be http://localhost"),
    ({"scope": "crm.read"}, "Extra inputs are not permitted"),
])
def test_invalid_mcp_oauth_config(monkeypatch, tmp_path, auth, problem):
    monkeypatch.setattr(config, "MCP_CONNECTIONS_PATH", str(tmp_path / "connections.json"))
    (tmp_path / "connections.json").write_text(json.dumps(
        {"crm": {"url": "https://crm/mcp", "auth": {"type": "mcp_oauth", **auth}}}))
    assert problem in mcp_client.check_connections()["crm"]
