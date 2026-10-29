# Real MCP client for live tools: an intent whose catalog entry has a
# `remote` binding (intents.py, mcp_tools.py) has its tool call executed by
# calling a tool on a remote MCP server.
# Everything that differs between CRMs' MCP servers is per-connection config
# in config.MCP_CONNECTIONS_PATH (see ConnectionSpec below), so onboarding a
# CRM is a config change, not a code change:
#   transport        streamable_http (default), sse, or stdio (a local program)
#   auth             none, a fixed bearer token, OAuth 2 (client_credentials
#                    or refresh_token grant), or the MCP standard sign-in
#                    (mcp_oauth: once in a browser, scripts/connect.py);
#                    tokens are refreshed automatically
#   timeout_seconds  per-call limit
#   allowed_tools    optional allowlist of remote tools that may be called
#   response         how to unwrap a server's own result envelope
# The catalog only names a connection and never holds a secret.
#
# The MCP SDK is async and turn handling is sync (FastAPI runs our handlers in
# its thread pool), so one background thread runs an event loop that keeps a
# long-lived session open per connection. Sync callers submit work to it and
# wait with a timeout. A session that fails at the transport level is dropped
# and the next call reconnects; so is one whose connection config changed.

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from ciap import config

load_dotenv(Path(config.ROOT_DIR) / ".env")

logger = logging.getLogger("ciap.mcp")


class ConnectionNotConfigured(Exception):
    pass


class ToolCallFailed(Exception):
    pass


class SignInRequired(ToolCallFailed):
    """An mcp_oauth connection has no usable saved sign-in."""

    def __init__(self, name, reason):
        super().__init__(f"MCP connection {name!r} {reason}: run python scripts/connect.py {name}")


# -- connection config -----------------------------------------------------------

class _Strict(BaseModel):
    # Unknown keys are errors, so a typo in connections.json is reported
    # instead of silently ignored.
    model_config = ConfigDict(extra="forbid")


class BearerAuth(_Strict):
    """A fixed token, sent as "Authorization: Bearer <token>"."""
    type: Literal["bearer"]
    token: str


class OAuth2Auth(_Strict):
    """Server-to-server OAuth 2: an access token from `token_url`, cached and
    refreshed before it expires (and once on a 401). With `refresh_token` it
    uses the refresh_token grant (for a token obtained once, out of band, from
    an interactive sign-in); without it, the client_credentials grant."""
    type: Literal["oauth2"]
    token_url: str
    client_id: str
    client_secret: str | None = None
    refresh_token: str | None = None
    scope: str | None = None
    extra_params: dict[str, str] = Field(default_factory=dict)  # e.g. {"audience": "..."}


class McpOAuthAuth(_Strict):
    """The MCP standard sign-in (OAuth 2.1 authorization code with PKCE, the
    server's endpoints discovered from its metadata), as used by Claude,
    Cursor and other MCP clients. `python scripts/connect.py <name>` signs in
    once in a browser and saves the tokens to config.MCP_TOKENS_DIR; calls
    only ever read and refresh that file, never prompt. Without client_id the
    client registers itself with the server (dynamic client registration);
    servers that don't allow that need an app registered with them, e.g.
    HubSpot's MCP auth app. The server's scopes apply."""
    type: Literal["mcp_oauth"]
    client_id: str | None = None
    client_secret: str | None = None
    client_auth: Literal["client_secret_post", "client_secret_basic"] = "client_secret_post"
    redirect_uri: str = "http://localhost:8765/callback"  # add it to the app's redirect URLs

    @model_validator(mode="after")
    def _check(self):
        if self.client_secret and not self.client_id:
            raise ValueError('"client_secret" needs "client_id"')
        redirect = urlparse(self.redirect_uri)
        if redirect.scheme != "http" or redirect.hostname not in ("localhost", "127.0.0.1") or not redirect.port:
            raise ValueError('"redirect_uri" must be http://localhost:<port>/<path> (the sign-in catches it locally)')
        return self


class ResponseSpec(_Strict):
    """For servers that wrap every result in their own envelope, e.g. Zoho's
    {"status": "success", "data": {...}}. A result is treated as enveloped
    when it has `status_field`; anything else passes through unchanged."""
    status_field: str = "status"
    data_field: str = "data"
    message_field: str = "message"  # looked up in the data, then at the top level
    success_values: list[str] = Field(default_factory=lambda: ["success"])
    error_values: list[str] = Field(default_factory=lambda: ["failure", "error"])
    empty_message_pattern: str | None = None  # regex: a success with this message means no results


class ConnectionSpec(_Strict):
    transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    auth: Annotated[BearerAuth | OAuth2Auth | McpOAuthAuth, Field(discriminator="type")] | None = None
    timeout_seconds: float | None = Field(None, gt=0, le=120)
    allowed_tools: list[str] | None = None
    response: ResponseSpec | None = None

    # Values read from the environment, so messages can be redacted.
    _secrets: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _check_transport(self):
        if self.transport == "stdio":
            if not self.command:
                raise ValueError('a "stdio" connection needs "command"')
            if self.url or self.headers or self.auth:
                raise ValueError('a "stdio" connection takes command/args/env, not url/headers/auth')
        elif not self.url:
            raise ValueError(f'a "{self.transport}" connection needs "url"')
        return self

    def fingerprint(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def redact(self, text):
        for secret in sorted(self._secrets, key=len, reverse=True):
            if len(secret) >= 4:
                text = text.replace(secret, "***")
        return text


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value, secrets):
    # "${NAME}" in the connections file is read from the environment (or
    # .env), so the file itself can be kept free of secrets.
    if isinstance(value, str):
        def sub(match):
            name = match.group(1)
            if name not in os.environ:
                raise ConnectionNotConfigured(f"environment variable {name} is not set")
            secrets.append(os.environ[name])
            return os.environ[name]
        return _ENV_REF.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, secrets) for v in value]
    return value


def _raw_connections():
    path = Path(config.MCP_CONNECTIONS_PATH)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConnectionNotConfigured(f"{path.name} must be a JSON object of connections")
    return data


def connection_names():
    return sorted(_raw_connections())


def allowed_tools(name):
    """The connection's allowed_tools, or None when any tool may be called
    (or the connection isn't configured). Doesn't need its secrets."""
    raw = _raw_connections().get(name)
    return raw.get("allowed_tools") if isinstance(raw, dict) else None


def _connection_spec(name):
    connections = _raw_connections()
    if name not in connections:
        raise ConnectionNotConfigured(
            f"MCP connection {name!r} is not configured (see {Path(config.MCP_CONNECTIONS_PATH).name})")
    secrets = []
    expanded = _expand(connections[name], secrets)
    try:
        spec = ConnectionSpec.model_validate(expanded)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'connection'}: {err['msg']}" for err in exc.errors()
        )
        raise ConnectionNotConfigured(f"MCP connection {name!r} is invalid: {problems}") from None
    spec._secrets = secrets
    return spec


def check_connections():
    """{name: None if its config is valid, else why not}. Reads the config and
    environment only; doesn't connect."""
    problems = {}
    for name in connection_names():
        try:
            spec = _connection_spec(name)
            if isinstance(spec.auth, McpOAuthAuth):
                _load_tokens(name, spec)
            problems[name] = None
        except (ConnectionNotConfigured, SignInRequired) as exc:
            problems[name] = str(exc)
    return problems


# -- transports and auth ------------------------------------------------------------

def _oauth2(auth):
    """An httpx Auth that adds an OAuth 2 bearer token, fetching it on first
    use and again when it's about to expire or the server answers 401. A
    rotated refresh token is kept in memory only."""
    import httpx2

    state = {"token": None, "expires_at": 0.0, "refresh_token": auth.refresh_token}
    lock = asyncio.Lock()

    async def ensure_token(force=False):
        async with lock:
            if not force and state["token"] and time.time() < state["expires_at"]:
                return state["token"]
            data = {"client_id": auth.client_id, **auth.extra_params}
            if auth.client_secret:
                data["client_secret"] = auth.client_secret
            if auth.scope:
                data["scope"] = auth.scope
            if state["refresh_token"]:
                data.update(grant_type="refresh_token", refresh_token=state["refresh_token"])
            else:
                data["grant_type"] = "client_credentials"
            async with httpx2.AsyncClient(timeout=10) as http:
                response = await http.post(auth.token_url, data=data)
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.status_code >= 400 or "access_token" not in payload:
                # Some providers (e.g. Zoho) answer 200 with {"error": ...}.
                reason = payload.get("error_description") or payload.get("error") or f"HTTP {response.status_code}"
                raise ToolCallFailed(f"OAuth token request failed: {reason}")
            state["token"] = payload["access_token"]
            state["expires_at"] = time.time() + float(payload.get("expires_in", 3600)) - 60
            state["refresh_token"] = payload.get("refresh_token") or state["refresh_token"]
            return state["token"]

    class Flow(httpx2.Auth):
        async def async_auth_flow(self, request):
            request.headers["Authorization"] = f"Bearer {await ensure_token()}"
            response = yield request
            if response.status_code == 401:
                request.headers["Authorization"] = f"Bearer {await ensure_token(force=True)}"
                yield request

    return Flow()


# -- mcp_oauth: saved sign-in ------------------------------------------------------
# data/oauth/<name>.json holds what calls need to refresh without discovery:
# the token endpoint and how to authenticate to it, the RFC 8707 resource,
# the tokens, and the client credentials when the client registered itself.
# It's rewritten atomically on every refresh, so a rotated refresh token
# survives a restart. Each process re-reads it before refreshing, so several
# workers (or a new sign-in) don't trip over each other's tokens.

_REFRESH_MARGIN_SECONDS = 60


def _token_path(name):
    return Path(config.MCP_TOKENS_DIR) / f"{name}.json"


def _load_tokens(name, spec):
    path = _token_path(name)
    if not path.exists():
        raise SignInRequired(name, "isn't signed in")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or not saved.get("access_token") or not saved.get("token_endpoint"):
            raise ValueError
    except ValueError:
        raise SignInRequired(name, f"has an unreadable sign-in file ({path.name})") from None
    if saved.get("server_url") != spec.url:
        raise SignInRequired(name, "was signed in to a different url")
    if spec.auth.client_id and saved.get("client_id") != spec.auth.client_id:
        raise SignInRequired(name, "was signed in with a different client_id")
    return saved


def _save_tokens(name, saved):
    path = _token_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    os.replace(temp, path)


def _expires_at(token):
    expires_in = token.get("expires_in")
    return time.time() + float(expires_in) if expires_in else None


def _token_request_auth(spec, saved, data):
    """Client authentication for a token request: (data, headers)."""
    import base64
    from urllib.parse import quote

    client_id = spec.auth.client_id or saved["client_id"]
    secret = spec.auth.client_secret if spec.auth.client_id else saved.get("client_secret")
    method = saved.get("token_auth_method") or "none"
    data = {**data, "client_id": client_id}
    if secret and method == "client_secret_basic":
        pair = f"{quote(client_id, safe='')}:{quote(secret, safe='')}"
        return data, {"Authorization": f"Basic {base64.b64encode(pair.encode()).decode()}"}
    if secret and method == "client_secret_post":
        data["client_secret"] = secret
    return data, {}


async def _refresh_tokens(name, spec, saved):
    """Refresh the saved sign-in and save the result. Network trouble or a
    server error leaves the saved tokens as they are (the next call retries);
    a refresh token the server rejects means signing in again, unless another
    process already rotated it."""
    import httpx2

    if not saved.get("refresh_token"):
        raise SignInRequired(name, "sign-in has expired (the server issued no refresh token)")
    data = {"grant_type": "refresh_token", "refresh_token": saved["refresh_token"]}
    if saved.get("resource"):
        data["resource"] = saved["resource"]
    data, headers = _token_request_auth(spec, saved, data)
    try:
        async with httpx2.AsyncClient(timeout=10) as http:
            response = await http.post(saved["token_endpoint"], data=data, headers=headers)
    except httpx2.HTTPError as exc:
        raise ToolCallFailed(f"MCP connection {name!r} couldn't refresh its sign-in: {type(exc).__name__}") from None
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code == 200 and isinstance(payload, dict) and payload.get("access_token"):
        latest = {
            **saved,
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token") or saved["refresh_token"],
            "expires_at": _expires_at(payload),
            "scope": payload.get("scope") or saved.get("scope"),
        }
        _save_tokens(name, latest)
        logger.info("MCP connection %s sign-in refreshed", name)
        return latest
    error = payload.get("error") if isinstance(payload, dict) else None
    if response.status_code in (400, 401) and error in ("invalid_grant", "invalid_client", "unauthorized_client"):
        current = _load_tokens(name, spec)
        if current.get("refresh_token") != saved["refresh_token"]:
            return current  # another process refreshed (or someone signed in again) meanwhile
        raise SignInRequired(name, f"sign-in was rejected by the server ({error})")
    raise ToolCallFailed(
        f"MCP connection {name!r} couldn't refresh its sign-in: {error or f'HTTP {response.status_code}'}")


def _usable(saved):
    expires_at = saved.get("expires_at")
    return expires_at is None or time.time() < expires_at - _REFRESH_MARGIN_SECONDS


def _mcp_oauth(name, spec):
    """An httpx Auth for an mcp_oauth connection: the saved access token,
    refreshed when it's about to expire and once when the server answers 401."""
    import httpx2

    lock = asyncio.Lock()

    async def access_token(rejected=None):
        async with lock:
            saved = _load_tokens(name, spec)
            if rejected is None and _usable(saved):
                return saved["access_token"]
            if rejected is not None and saved["access_token"] != rejected and _usable(saved):
                return saved["access_token"]  # refreshed elsewhere since that request
            return (await _refresh_tokens(name, spec, saved))["access_token"]

    class Flow(httpx2.Auth):
        async def async_auth_flow(self, request):
            token = await access_token()
            request.headers["Authorization"] = f"Bearer {token}"
            response = yield request
            if response.status_code == 401:
                request.headers["Authorization"] = f"Bearer {await access_token(rejected=token)}"
                yield request

    return Flow()


def sign_in(name, show_sign_in_page, timeout_seconds=300):
    """The one-time browser sign-in for an mcp_oauth connection: the MCP SDK's
    standard flow (discovery, registration when there's no client_id, PKCE),
    with the redirect caught on redirect_uri. show_sign_in_page(url) shows
    the person the server's sign-in page. Saves the sign-in and returns the
    server's tool names. Run by scripts/connect.py."""
    spec = _connection_spec(name)
    if not isinstance(spec.auth, McpOAuthAuth):
        raise ConnectionNotConfigured(f'MCP connection {name!r} doesn\'t use "auth": {{"type": "mcp_oauth"}}')
    return asyncio.run(_sign_in(name, spec, show_sign_in_page, timeout_seconds))


class _MemoryStorage:
    """TokenStorage for the SDK's OAuth flow during sign-in."""

    def __init__(self, client_info):
        self.client_info = client_info
        self.tokens = None
        self.expires_at = None

    async def get_tokens(self):
        return None  # always a fresh sign-in

    async def set_tokens(self, tokens):
        self.tokens = tokens
        self.expires_at = time.time() + tokens.expires_in if tokens.expires_in else None

    async def get_client_info(self):
        return self.client_info

    async def set_client_info(self, client_info):
        self.client_info = client_info


async def _sign_in(name, spec, show_sign_in_page, timeout_seconds):
    from mcp import Client
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata

    auth = spec.auth
    redirect = urlparse(auth.redirect_uri)
    preregistered = None
    if auth.client_id:
        preregistered = OAuthClientInformationFull(
            client_id=auth.client_id,
            client_secret=auth.client_secret,
            redirect_uris=[auth.redirect_uri],
            token_endpoint_auth_method=auth.client_auth if auth.client_secret else "none",
        )
    storage = _MemoryStorage(preregistered)
    redirected = asyncio.get_running_loop().create_future()

    async def on_request(reader, writer):
        try:
            line = (await reader.readline()).decode("latin-1")
            target = urlparse(line.split(" ")[1]) if line.count(" ") >= 2 else None
            if target is None or target.path != (redirect.path or "/"):
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                return
            query = {k: v[0] for k, v in parse_qs(target.query).items()}
            done = "error" not in query and "code" in query
            body = ("Signed in. You can close this tab." if done else "Sign-in failed. See the terminal.").encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
                         + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            if not redirected.done():
                redirected.set_result(query)
        finally:
            await writer.drain()
            writer.close()

    callback_server = await asyncio.start_server(on_request, redirect.hostname, redirect.port)

    async def redirect_handler(url):
        show_sign_in_page(url)

    async def callback_handler():
        try:
            query = await asyncio.wait_for(redirected, timeout_seconds)
        except TimeoutError:
            raise ToolCallFailed(f"no sign-in within {timeout_seconds}s") from None
        if "error" in query:
            raise ToolCallFailed(f"sign-in failed: {query.get('error_description') or query['error']}")
        return AuthorizationCodeResult(code=query["code"], state=query.get("state"), iss=query.get("iss"))

    provider = OAuthClientProvider(
        server_url=spec.url,
        client_metadata=OAuthClientMetadata(
            client_name="CIAP Agent Assist",
            redirect_uris=[auth.redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    try:
        async with Client(_http_transport(spec, auth=provider), read_timeout_seconds=_timeout(spec)) as client:
            tools = (await client.list_tools()).tools
    finally:
        callback_server.close()
    if storage.tokens is None:
        raise ToolCallFailed("the server didn't ask for a sign-in (does it need mcp_oauth?)")

    context = provider.context
    client_info = context.client_info
    resource = (context.get_resource_url()
                if context.should_include_resource_param(context.protocol_version) else None)
    _save_tokens(name, {
        "server_url": spec.url,
        "token_endpoint": provider._get_token_endpoint(),
        "token_auth_method": client_info.token_endpoint_auth_method or "none",
        "resource": resource,
        "client_id": client_info.client_id,
        # Only a self-registered client's secret is kept; a configured one
        # stays in the environment.
        "client_secret": None if auth.client_id else client_info.client_secret,
        "access_token": storage.tokens.access_token,
        "refresh_token": storage.tokens.refresh_token,
        "expires_at": storage.expires_at,
        "scope": storage.tokens.scope,
        "signed_in_at": time.time(),
    })
    return [tool.name for tool in tools]


@asynccontextmanager
async def _http_transport(spec, name=None, auth=None):
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    headers = dict(spec.headers)
    if auth is None:  # (sign-in passes its own)
        if isinstance(spec.auth, BearerAuth):
            headers["Authorization"] = f"Bearer {spec.auth.token}"
        elif isinstance(spec.auth, OAuth2Auth):
            auth = _oauth2(spec.auth)
        elif isinstance(spec.auth, McpOAuthAuth):
            auth = _mcp_oauth(name, spec)
    # The SDK's default HTTP timeouts (long reads, for the server's event
    # stream); per-call limits are applied in call_tool / list_tools.
    if spec.transport == "sse":
        async with sse_client(spec.url, headers=headers or None, auth=auth) as streams:
            yield streams
        return
    async with create_mcp_http_client(headers=headers or None, auth=auth) as http:
        async with streamable_http_client(spec.url, http_client=http) as streams:
            yield streams


def _server_for(name, spec):
    """What mcp.Client connects to for this connection. Tests replace this
    with an in-process server."""
    if spec.transport == "stdio":
        from mcp import StdioServerParameters

        return StdioServerParameters(command=spec.command, args=spec.args, env={**os.environ, **spec.env})
    return _http_transport(spec, name=name)


def _describe(exc):
    """'Type: message' for an error, reporting the errors inside an
    ExceptionGroup (the SDK's task groups wrap the real cause in one)."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def _timeout(spec, override=None):
    return override or spec.timeout_seconds or config.MCP_CALL_TIMEOUT_SECONDS


# -- background loop and sessions -------------------------------------------------

_loop = None
_loop_lock = threading.Lock()
_sessions = {}  # connection name -> _Session; only touched on the loop thread


def _ensure_loop():
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="mcp-client", daemon=True).start()
        return _loop


class _Session:
    """One open MCP session. Requests are queued to the task that owns the
    Client (its context has to be entered and exited by the same task) and
    each runs as its own task, so a slow call doesn't hold up the others."""

    def __init__(self, name, spec):
        self.name = name
        self.spec = spec
        self.fingerprint = spec.fingerprint()
        self.queue = asyncio.Queue()
        self.ready = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        from mcp import Client

        try:
            async with Client(_server_for(self.name, self.spec), read_timeout_seconds=_timeout(self.spec)) as client:
                logger.info("MCP connection %s opened", self.name)
                self.ready.set_result(True)
                while True:
                    job = await self.queue.get()
                    if job is None:
                        return
                    asyncio.create_task(self._serve(client, *job))
        except BaseException as exc:
            while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
            if not self.ready.done():
                self.ready.set_exception(exc if isinstance(exc, Exception) else ToolCallFailed("connection closed"))
            logger.warning("MCP connection %s unavailable: %s", self.name, self.spec.redact(_describe(exc)))
            if not isinstance(exc, Exception):
                raise
        finally:
            if _sessions.get(self.name) is self:
                del _sessions[self.name]
            while not self.queue.empty():
                job = self.queue.get_nowait()
                if job is not None and not job[1].done():
                    job[1].set_exception(ToolCallFailed(f"MCP connection {self.name!r} closed"))

    async def _serve(self, client, operation, future):
        from mcp import MCPError

        try:
            result = await operation(client)
        except MCPError as exc:
            # A JSON-RPC error from the server: the session itself is fine.
            if not future.done():
                future.set_exception(ToolCallFailed(str(exc)))
            return
        except Exception as exc:
            # Transport trouble: reconnect on the next call.
            if not future.done():
                future.set_exception(exc)
            self.close()
            return
        if not future.done():
            future.set_result(result)

    def close(self):
        if _sessions.get(self.name) is self:
            del _sessions[self.name]
        self.queue.put_nowait(None)


async def _submit(name, spec, operation):
    session = _sessions.get(name)
    if session is not None and session.fingerprint != spec.fingerprint():
        # connections.json (or a secret it reads) changed: reconnect with the
        # new config instead of waiting for a restart.
        logger.info("MCP connection %s changed; reconnecting", name)
        session.close()
        session = None
    if session is None:
        session = _sessions[name] = _Session(name, spec)
    await session.ready
    future = asyncio.get_running_loop().create_future()
    session.queue.put_nowait((operation, future))
    return await future


def _run(name, spec, operation, timeout):
    future = asyncio.run_coroutine_threadsafe(_submit(name, spec, operation), _ensure_loop())
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        future.cancel()
        raise ToolCallFailed(f"MCP connection {name!r} didn't answer within {timeout:g}s") from None
    except (ToolCallFailed, ConnectionNotConfigured):
        raise
    except Exception as exc:
        # Transport errors can quote the URL, which may carry a key.
        raise ToolCallFailed(spec.redact(_describe(exc))) from None


def shutdown():
    """Close every open session (server shutdown)."""
    if _loop is None:
        return

    async def close_all():
        sessions = list(_sessions.values())
        for session in sessions:
            session.close()
        await asyncio.gather(*(s.task for s in sessions), return_exceptions=True)

    try:
        asyncio.run_coroutine_threadsafe(close_all(), _loop).result(timeout=5)
    except Exception:
        pass


# -- results -------------------------------------------------------------------------

def _normalize(result, response_spec=None):
    # tool_result data is a JSON object: prefer structured content, else parse
    # the text content as JSON, else return it as text. An empty result comes
    # back as {} (mcp_tools reports it as no_results).
    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    text = "\n".join(texts)
    if result.is_error:
        raise ToolCallFailed(text or "the tool reported an error")
    value = result.structured_content
    if value is None and text:
        try:
            value = json.loads(text)
        except ValueError:
            value = {"text": text}
    if value is None:
        value = {}
    if response_spec is not None:
        value = _unwrap(value, response_spec)
    return value if isinstance(value, dict) else {"result": value}


def _unwrap(value, spec):
    # A server's own envelope (ResponseSpec): an error status becomes a tool
    # error, a "no results" message becomes {}, and a success yields its data.
    if not isinstance(value, dict) or spec.status_field not in value:
        return value
    status = value[spec.status_field]
    data = value.get(spec.data_field)
    message = data.get(spec.message_field) if isinstance(data, dict) else None
    message = message or value.get(spec.message_field)
    if status in spec.error_values:
        raise ToolCallFailed(str(message) if message else json.dumps(value))
    if status not in spec.success_values:
        return value
    if spec.empty_message_pattern and isinstance(message, str) and re.search(spec.empty_message_pattern, message):
        return {}
    return data if data is not None else {}


# -- public calls ---------------------------------------------------------------

def call_tool(connection, tool, arguments, timeout=None):
    """Call `tool` on the named connection. Returns the result as a dict ({}
    when the server returned nothing); raises ConnectionNotConfigured or
    ToolCallFailed (including timeouts)."""
    spec = _connection_spec(connection)
    if spec.allowed_tools is not None and tool not in spec.allowed_tools:
        raise ToolCallFailed(f"tool {tool!r} isn't in connection {connection!r}'s allowed_tools")
    result = _run(connection, spec, lambda client: client.call_tool(tool, arguments), _timeout(spec, timeout))
    return _normalize(result, spec.response)


def list_tools(connection, timeout=None):
    """The tools the connection's server offers, as plain dicts. `allowed` is
    False for tools outside the connection's allowed_tools."""
    spec = _connection_spec(connection)
    result = _run(connection, spec, lambda client: client.list_tools(), _timeout(spec, timeout))
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
            "annotations": tool.annotations.model_dump(by_alias=True, exclude_none=True) if tool.annotations else {},
            "allowed": spec.allowed_tools is None or tool.name in spec.allowed_tools,
        }
        for tool in result.tools
    ]
