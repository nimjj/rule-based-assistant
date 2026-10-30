# Client for ciap-api, the only place this service reads tenant data from or
# keeps call state in (CIP mode, config.CIAP_API_BASE_URL set). The service
# never touches CIP's database: ciap-api serves each tenant's catalog and each
# call's conversation state under /api/internal/assist/**, protected by its
# ingestion API key (config.CIAP_API_KEY, sent as X-Api-Key).
#
#   GET /api/internal/assist/tenants/{tenant}/catalog
#       -> {"tenantCode", "version", "intents": [{"name", "label", "description",
#           "phrases", "slots": [{"name", "label", "type", "required", "prompt",
#           "description", "options"}]}]}
#   GET /api/internal/assist/tenants/{tenant}/calls/{call}/state
#       -> {"linkedId", "version", "state"}; 404 for a call ciap-api doesn't know
#   PUT /api/internal/assist/tenants/{tenant}/calls/{call}/state
#       {"expectedVersion", "state"} -> {"version"}; 409 when the call's state
#       moved on since it was loaded

import threading
from contextvars import ContextVar
from urllib.parse import quote

import httpx

from ciap import config

# The X-Correlation-Id of the request being served (ciap-kafka sends the
# call's), passed on to ciap-api so both sides' logs line up.
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CiapApiError(Exception):
    """ciap-api couldn't be reached or refused the request."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NotFound(CiapApiError):
    pass


class StateConflict(CiapApiError):
    """The call's state was saved by someone else since it was loaded."""


_client = None
_client_lock = threading.Lock()


def enabled():
    return bool(config.CIAP_API_BASE_URL)


def _http():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                headers = {"X-Api-Key": config.CIAP_API_KEY} if config.CIAP_API_KEY else {}
                _client = httpx.Client(
                    base_url=config.CIAP_API_BASE_URL, headers=headers, timeout=config.CIAP_API_TIMEOUT_SECONDS
                )
    return _client


def close():
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None


def _path(tenant_id, call_id=None):
    path = f"/api/internal/assist/tenants/{quote(tenant_id, safe='')}"
    return path if call_id is None else f"{path}/calls/{quote(call_id, safe='')}"


def _request(method, path, body=None):
    headers = {}
    corr = correlation_id.get()
    if corr:
        headers["X-Correlation-Id"] = corr
    try:
        response = _http().request(method, path, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise CiapApiError(f"ciap-api unreachable: {type(exc).__name__}: {exc}") from exc
    if response.status_code == 404:
        raise NotFound(_detail(response), 404)
    if response.status_code == 409:
        raise StateConflict(_detail(response), 409)
    if response.status_code >= 400:
        raise CiapApiError(f"ciap-api returned {response.status_code}: {_detail(response)}", response.status_code)
    return response.json()


def _detail(response):
    try:
        return response.json().get("message") or response.text[:200]
    except Exception:
        return response.text[:200]


def get_catalog(tenant_id):
    return _request("GET", _path(tenant_id) + "/catalog")


def load_state(tenant_id, call_id):
    """(state or None before the first save, version). NotFound for a call
    ciap-api has no record of."""
    body = _request("GET", _path(tenant_id, call_id) + "/state")
    return body.get("state"), int(body.get("version") or 0)


def save_state(tenant_id, call_id, state, expected_version):
    """Save the call's state if it's still at expected_version; returns the new
    version. StateConflict otherwise."""
    body = _request("PUT", _path(tenant_id, call_id) + "/state", {"expectedVersion": expected_version, "state": state})
    return int(body["version"])
