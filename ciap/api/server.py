# HTTP API around the intent pipeline (embedding retrieval, regex/library slot
# extraction, frame-based dialogue state, Gemma 3 1B LLM escalation).
#
# The pipeline is stateful within one call (frames persist turn to turn): the
# first turn for a new (tenant ID, PBXware call ID) pair starts a call, then
# every turn for that call continues the same ConversationStateStore.
#
# CIP mode (config.CIAP_API_BASE_URL set, how CIP runs it): the process keeps
# nothing between turns. Each turn loads the call's state from ciap-api, runs
# against the tenant's own catalog (catalog.py) and saves the state back, so
# any number of instances can serve a call. Otherwise (local development, the
# demo, the tests) calls live in this process's memory and every tenant gets
# the built-in catalog.
#
# All response shaping lives in views.py; all pipeline logic in state_store.py.
# This module is only routing, validation, session bookkeeping and errors.
#
# Run:  uvicorn ciap.api.server:app
# Docs: http://127.0.0.1:8000/docs   (OpenAPI JSON: /openapi.json)
#
# /v1 requires X-Api-Key when config.API_KEY (AGENT_ASSIST_API_KEY) is set.

import asyncio
import hmac
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from ciap import call_log
from ciap import catalog
from ciap import ciap_api
from ciap import config
from ciap.integrations import mcp_client
from ciap.api import views
from ciap.api.demo_web import router as demo_router
from ciap.pipeline.slots import SLOT_EXTRACTORS
from ciap.pipeline.state_store import ConversationStateStore
from ciap.pipeline.transcripts import parse_turns

logger = call_log.logger

MAX_UTTERANCE_CHARS = 2000
MAX_TRANSCRIPT_TURNS = 200


def warm_embeddings():
    """Load the sentence-embedding model and build the exemplar index, so the
    first real turn isn't a 2-5s cold start."""
    from ciap.pipeline.embeddings import retrieve

    start = time.perf_counter()
    retrieve("warm up")
    logger.info("embedding index warmed (%.1fs)", time.perf_counter() - start)


def warm_responder():
    """Load the responder model into Ollama, so the first live tool answer
    isn't spent loading it (and falling back on the timeout). A failure is
    logged, not fatal: answers then fall back until Ollama has the model."""
    from ciap.pipeline import responder

    start = time.perf_counter()
    try:
        responder.warm()
    except Exception as exc:
        logger.warning("responder model %s not warmed: %s: %s", responder.MODEL_NAME, type(exc).__name__, exc)
        return
    logger.info("responder model %s warmed (%.1fs)", responder.MODEL_NAME, time.perf_counter() - start)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Adds the model-load time to boot instead of the first turn. (The
    # escalation LLM is still lazy -- warm it with one escalating turn if the
    # first live one needs to be fast.) The responder model is warmed too:
    # its answers have a timeout a cold load would always blow.
    logger.info(
        "catalogs and call state: %s",
        f"ciap-api {config.CIAP_API_BASE_URL}" if ciap_api.enabled() else "built-in catalog, in-memory calls",
    )
    if not config.API_KEY:
        logger.warning("AGENT_ASSIST_API_KEY is not set: /v1 is open (local development only)")
    warm_embeddings()
    warm_responder()
    _check_connections()
    sweeper = asyncio.create_task(_sweep_idle_calls_forever())
    yield
    mcp_client.shutdown()
    ciap_api.close()
    sweeper.cancel()
    with suppress(asyncio.CancelledError):
        await sweeper


def _check_connections():
    # Report connection config problems at boot rather than on the first call.
    try:
        problems = mcp_client.check_connections()
    except Exception as exc:
        logger.warning("MCP connections file unreadable: %s", exc)
        return
    for name, problem in problems.items():
        if problem:
            logger.warning("MCP connection %s won't work: %s", name, problem)
        else:
            logger.info("MCP connection %s configured", name)


# -- errors -------------------------------------------------------------------
#
# Every non-2xx response has the same body:
#   {"error": {"code": "<machine_readable>", "message": "<human readable>"}}
# (validation errors add "details").


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


class ErrorBody(BaseModel):
    code: str = Field(..., examples=["call_not_found"])
    message: str
    details: list[dict] | None = Field(default=None, description="Present on 422 validation errors.")


class ErrorResponse(BaseModel):
    error: ErrorBody


def _error_response(status: int, code: str, message: str, details=None) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status, content={"error": body})


_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


# -- schemas ------------------------------------------------------------------

Utterance = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_UTTERANCE_CHARS)
]
Speaker = Literal["customer", "agent"]


# Tenant and call IDs come from the telephony system (PBXware), never
# generated here. Permissive enough for Asterisk-style IDs ("1727690000.12345")
# and UUIDs, and safe to use as a URL path segment. A call is identified by
# the pair: the same call ID under two tenants is two separate calls.
ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
TenantIdPath = Annotated[str, Path(pattern=ID_PATTERN, description="The tenant ID.")]
CallIdPath = Annotated[str, Path(pattern=ID_PATTERN, description="The PBXware call ID.")]

# The call agent handling the call. Only logged; never affects processing.
UserId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
_USER_ID_FIELD = Field(
    default=None, examples=["agent-042"],
    description="The call agent's user ID. Optional; only recorded in the logs, never affects processing.",
)

AnswerMode = Literal["focused", "full"]
_ANSWER_MODE_FIELD = Field(
    default=None,
    description=(
        "How any tool this request fires gets its result explained in `answer`. `focused`: one or two "
        "sentences answering what was asked. `full`: what was asked, then every record in the result and "
        "its key details, in a short paragraph (slower). Defaults to `RESPONDER_DEFAULT_MODE` (`focused`)."
    ),
)


_CATALOG_VERSION_FIELD = Field(
    default=None, ge=0,
    description=(
        "The tenant's catalog version when CIP queued the turn. CIP mode: a newer version than the "
        "cached catalog reloads it before the turn runs. Ignored otherwise."
    ),
)


class NewCallResponse(BaseModel):
    tenant_id: str = Field(..., examples=["acme"])
    call_id: str = Field(..., examples=["1727690000.12345"])


class TurnRequest(BaseModel):
    utterance: Utterance = Field(..., description='One turn, e.g. "Where is my order?"')
    speaker: Speaker = Field(
        default="customer",
        description=(
            "Who said this turn. Both speakers run through the exact same "
            "intent/slot/frame pipeline; `speaker` only affects provenance -- it "
            "is echoed back and becomes `initiated_by` on any frame this turn opens."
        ),
    )
    user_id: UserId | None = _USER_ID_FIELD
    answer_mode: AnswerMode | None = _ANSWER_MODE_FIELD
    catalog_version: int | None = _CATALOG_VERSION_FIELD


class TranscriptTurn(BaseModel):
    utterance: Utterance
    speaker: Speaker = "customer"


class TranscriptRequest(BaseModel):
    """Provide exactly one of `transcript` or `turns`."""

    transcript: str | None = Field(
        default=None,
        description=(
            'Raw transcript text with one turn per line, tagged "Customer: ..." or '
            '"Agent: ...". Untagged lines are ignored.'
        ),
    )
    turns: list[TranscriptTurn] | None = Field(
        default=None, max_length=MAX_TRANSCRIPT_TURNS, description="Pre-parsed turns, in order."
    )
    user_id: UserId | None = _USER_ID_FIELD
    answer_mode: AnswerMode | None = _ANSWER_MODE_FIELD
    catalog_version: int | None = _CATALOG_VERSION_FIELD

    @model_validator(mode="after")
    def _exactly_one(self):
        if (self.transcript is None) == (self.turns is None):
            raise ValueError("provide exactly one of 'transcript' or 'turns'")
        return self


class SlotOverrideRequest(BaseModel):
    value: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)] = Field(
        ...,
        description=(
            "Corrected raw value for the slot (e.g. the real order ID after a "
            "misread one). Parsed with the same extractor a normal turn uses, so "
            "it must look like a real value."
        ),
    )
    user_id: UserId | None = _USER_ID_FIELD


class Candidate(BaseModel):
    intent: str
    label: str = Field(..., description="Display name for `intent`.")
    score: float = Field(..., description="Cosine similarity, 0-1.")


class Escalation(BaseModel):
    reason: str | None = None
    model: str | None = None
    note: str | None = Field(default=None, description="The model's one-sentence reasoning.")
    resolved: Any = Field(
        default=None,
        description="Intent name, list of intent names, or a boolean (correction/reference verdict), depending on `reason`.",
    )


class IntentView(BaseModel):
    turn_type: str = Field(..., description="Which pipeline branch handled the turn. See API_DOCUMENTATION.md.")
    decision: str = Field(..., description="Coarser label over turn_type, for badges/summaries.")
    selected_intent: str | None = None
    selected_intent_label: str | None = Field(
        default=None, description='Display name for `selected_intent` (multi-intent: labels joined with " + ").'
    )
    top_score: float | None = None
    candidates: list[Candidate]
    heuristic_flags: list[str] = Field(..., description="Any of: correction, reference, interruption, override.")
    escalation: Escalation | None = None
    message: str | None = Field(
        default=None, description="Only on multi_intent_blocked: what to tell the customer."
    )


class FrameView(BaseModel):
    id: int
    intent: str
    intent_label: str = Field(..., description="Display name for `intent`.")
    status: Literal["OPEN", "AWAITING_INFO", "FIRED", "REVERTED"]
    initiated_by: Speaker
    required: list[str] = Field(..., description="Slots this intent needs before its tool call fires.")
    slots: dict = Field(..., description="Filled slots. `new_date` is the resolved ISO date.")
    missing: list[str]
    filled_this_turn: list[str] = Field(
        ..., description="Slots this turn filled on this frame (always [] from GET on the call)."
    )
    tool_call: dict | None = Field(
        default=None,
        description=(
            '`{"tool": ..., "tool_label": ..., "arguments": {...}}`, set once the frame is FIRED. '
            "`tool` is the intent's name and `arguments` its required slots."
        ),
    )
    tool_result: dict | None = Field(
        default=None,
        description=(
            "Outcome of a tool that actually executed (one with a `remote` MCP binding; "
            "the shipped Zoho CRM lookups and the product tools), always "
            "`{\"status\": \"success\" | \"no_results\" | \"error\", \"data\": ..., \"message\": ...}`: "
            "`data` is the server's response on success (null otherwise), `message` explains "
            "`no_results` and `error` (null on success). Null for every mock tool."
        ),
    )
    answer: dict | None = Field(
        default=None,
        description=(
            "`tool_result` as a short plain-English briefing for the agent, answering what was "
            "asked on the call: `{\"text\": ..., \"source\": \"llm\" | \"fallback\" | \"rule\", "
            "\"model\": ..., \"note\": ..., \"mode\": \"focused\" | \"full\"}`. `llm`: written by the "
            "responder model; `fallback`: a plain listing of the result because the model failed, timed "
            "out or mentioned a value not in the result (`note` says which); `rule`: a fixed sentence for "
            "`no_results` / `error`. `model` is set for `llm` only. `mode` is the request's `answer_mode` "
            "on the turn that fired the tool. Null whenever `tool_result` is."
        ),
    )


class SlotsView(BaseModel):
    frames: list[FrameView] = Field(..., description="Every frame in the call so far.")


class McpView(BaseModel):
    fired_this_turn: list[dict] = Field(..., description='`{"tool", "tool_label", "arguments"}` per call fired this turn.')
    all_calls: list[dict] = Field(
        ...,
        description='Every tool call built in this call so far: `{"frame", "intent", "intent_label", "tool", "tool_label", "arguments"}`.',
    )


class TurnView(BaseModel):
    turn_index: int = Field(..., description="1-based count of turns/overrides in this call.")
    utterance: str
    speaker: Speaker
    intent: IntentView
    slots: SlotsView
    mcp: McpView
    debug: dict | None = Field(default=None, description="Raw pipeline trace. Only with `?debug=true`.")


class CallStateResponse(BaseModel):
    tenant_id: str
    call_id: str
    turn_count: int
    frames: list[FrameView]


class TranscriptResponse(BaseModel):
    tenant_id: str
    call_id: str
    turns: list[TurnView] = Field(..., description="One view per processed turn, in order.")
    frames: list[FrameView] = Field(..., description="Final state of every frame.")


class IntentInfo(BaseModel):
    name: str = Field(..., description="Also the name of the tool call the intent fires.")
    label: str = Field(..., description="Display name. Show this in UIs; `name` is the ID.")
    description: str | None = Field(..., description="What the intent's tool call does, or null if not set.")
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list, description="Collected when said, but not needed to fire.")
    exemplar_count: int
    exemplars: list[str]
    executes_live: bool = Field(..., description="True if the intent has a `remote` binding and really runs; false if it's a mock.")
    remote: dict | None = Field(None, description="How the tool call executes on a remote MCP server, or null for a mock (see `RemoteBinding`).")


class LogResponse(BaseModel):
    records: list[dict]
    total: int = Field(..., description="Matching records before `limit` was applied.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: bool = Field(..., description="Whether the escalation LLM server is reachable.")
    model: str = Field(..., description="Escalation model name.")


# -- session store ------------------------------------------------------------
#
# CIP mode: ciap-api holds each call's state (see _locked_call); nothing below
# keeps anything between requests.
#
# Otherwise in memory: lost on restart, correct for a single process only.
# A call idle for config.CALL_IDLE_TIMEOUT_SECONDS is dropped -- on its next
# request, or by the background sweep for calls nobody comes back to.

# How often the background sweep looks for idle calls. An abandoned call can
# stay in memory up to this long past its timeout (requests to it still get
# `call_expired` on time).
SWEEP_INTERVAL_SECONDS = 60

# Remember this many expired calls, so a late request gets `call_expired`
# rather than `call_not_found`.
MAX_EXPIRED_IDS = 10_000

_now = time.monotonic  # patched in tests


class CallSession:
    def __init__(self):
        self.store = ConversationStateStore()
        self.turn_count = 0
        self.last_activity = _now()
        # One in-flight turn per call: process_utterance mutates the frame list.
        self.lock = threading.Lock()

    @classmethod
    def from_state(cls, state):
        """A session from the state ciap-api saved for the call (None: a new call)."""
        session = cls()
        if state:
            session.store = ConversationStateStore.from_dict(state.get("store"))
            session.turn_count = int(state.get("turn_count") or 0)
        return session

    def to_state(self):
        return {"store": views.to_jsonable(self.store.to_dict()), "turn_count": self.turn_count}

    def touch(self):
        self.last_activity = _now()

    def is_idle(self) -> bool:
        timeout = config.CALL_IDLE_TIMEOUT_SECONDS
        return timeout > 0 and _now() - self.last_activity > timeout


# Sessions are keyed by (tenant_id, call_id).
CallKey = tuple[str, str]

_calls: dict[CallKey, CallSession] = {}
_expired_ids: OrderedDict[CallKey, None] = OrderedDict()
# Serialises creating a call, so two first requests for the same new ID
# (e.g. two racing first turns) end up on one session.
_create_lock = threading.Lock()


def _ids(key: CallKey, user_id: str | None = None) -> dict:
    """The identifying fields every call log record carries."""
    ids = {"tenant_id": key[0], "call_id": key[1]}
    if user_id is not None:
        ids["user_id"] = user_id
    return ids


def _who(key: CallKey, user_id: str | None = None) -> str:
    """Prefix for human-readable log lines."""
    return f"tenant={key[0]} call={key[1]}" + (f" user={user_id}" if user_id else "")


def _expire(key: CallKey, session: CallSession) -> None:
    """Drop an idle call. Caller holds session.lock."""
    if _calls.get(key) is not session:
        return  # already deleted, reset or expired
    del _calls[key]
    _expired_ids[key] = None
    while len(_expired_ids) > MAX_EXPIRED_IDS:
        _expired_ids.popitem(last=False)
    idle = _now() - session.last_activity
    logger.info("%s expired after %.0fs idle", _who(key), idle)
    call_log.log_event("call_expired", **_ids(key), idle_seconds=round(idle), turn_count=session.turn_count)


def _expired_error(key: CallKey) -> ApiError:
    return ApiError(
        404, "call_expired",
        f"Call {key[1]!r} for tenant {key[0]!r} expired after {config.CALL_IDLE_TIMEOUT_SECONDS}s "
        "without activity. Send a turn to start it again.",
    )


def _get_session(key: CallKey, touch: bool = True) -> CallSession:
    """Look up a live call. `touch` counts this request as activity; pass
    False for read-only requests so they don't keep an idle call alive."""
    session = _calls.get(key)
    if session is None:
        if key in _expired_ids:
            raise _expired_error(key)
        raise ApiError(
            404, "call_not_found", f"No call {key[1]!r} for tenant {key[0]!r}. Send a turn to start one."
        )
    with session.lock:
        if session.is_idle():
            _expire(key, session)
            raise _expired_error(key)
        if touch:
            session.touch()
    return session


def _get_or_create_session(key: CallKey, source: str, user_id: str | None = None) -> CallSession:
    """The live call for this key, or a new one. An expired call starts over
    empty. `source` and `user_id` are logged."""
    try:
        return _get_session(key)
    except ApiError as exc:
        if exc.code not in ("call_not_found", "call_expired"):
            raise
    with _create_lock:
        session = _calls.get(key)
        if session is not None:  # created by a racing request
            return session
        after_expiry = key in _expired_ids
        _expired_ids.pop(key, None)
        session = _calls[key] = CallSession()
    logger.info("%s created (%s%s)", _who(key, user_id), source, ", after expiry" if after_expiry else "")
    call_log.log_event("call_created", **_ids(key, user_id), source=source, after_expiry=after_expiry)
    return session


def evict_idle_calls() -> int:
    """Drop every idle call; returns how many. Calls with a turn in flight
    are busy, not idle, and are skipped."""
    evicted = 0
    for key, session in list(_calls.items()):
        if not session.lock.acquire(blocking=False):
            continue
        try:
            if session.is_idle() and _calls.get(key) is session:
                _expire(key, session)
                evicted += 1
        finally:
            session.lock.release()
    return evicted


async def _sweep_idle_calls_forever():
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            evict_idle_calls()
        except Exception:
            logger.exception("idle-call sweep failed")


def _tenant_catalog(tenant_id: str, min_version: int | None = None) -> catalog.Catalog:
    try:
        return catalog.for_tenant(tenant_id, min_version)
    except catalog.CatalogUnavailable as exc:
        logger.error("tenant=%s %s", tenant_id, exc)
        raise ApiError(503, "catalog_unavailable", f"Tenant {tenant_id!r}'s intent catalog couldn't be loaded; retry.") from exc


@contextmanager
def _locked_call(key: CallKey, *, create: bool, source: str = "turn", user_id: str | None = None,
                 touch: bool = True, save: bool = True):
    """The call's session, held for one request. In memory: the live session
    under its lock. CIP mode: loaded from ciap-api and, when `save` and the
    request succeeds, saved back -- refused (ciap_api.StateConflict) if
    another request saved the call in between. `create` starts an unknown call
    instead of answering 404."""
    if not ciap_api.enabled():
        session = _get_or_create_session(key, source, user_id) if create else _get_session(key, touch)
        with session.lock:
            if _calls.get(key) is not session:
                # Deleted, reset or expired while this request waited for the
                # lock: applying it to the dropped session would lose it silently.
                raise ApiError(409, "call_replaced", f"Call {key[1]!r} was deleted or reset while this request waited; resend it.")
            yield session
        return
    try:
        state, version = ciap_api.load_state(*key)
    except ciap_api.NotFound:
        if not create:
            raise ApiError(404, "call_not_found", f"No call {key[1]!r} for tenant {key[0]!r}. Send a turn to start one.")
        state, version = None, 0
    except ciap_api.CiapApiError as exc:
        raise ApiError(503, "state_unavailable", f"Couldn't load call {key[1]!r}'s state; retry. ({exc})") from exc
    session = CallSession.from_state(state)
    yield session
    if save:
        try:
            ciap_api.save_state(*key, session.to_state(), version)
        except ciap_api.StateConflict:
            raise
        except ciap_api.CiapApiError as exc:
            raise ApiError(503, "state_unavailable", f"Couldn't save call {key[1]!r}'s state; retry. ({exc})") from exc


def _retrying(key: CallKey, work):
    """Run `work()` (which uses _locked_call), once more if ciap-api refused
    its save because the call moved on meanwhile. Turns for one call arrive
    in order, one at a time, so this is only a request seen twice."""
    try:
        return work()
    except ciap_api.StateConflict:
        logger.warning("%s state changed while a request ran; running it again on the new state", _who(key))
    try:
        return work()
    except ciap_api.StateConflict as exc:
        raise ApiError(409, "state_conflict", f"Call {key[1]!r} kept changing while this request ran; resend it.") from exc


def _run_turn(
    key: CallKey, session: CallSession, text: str, speaker: str, user_id: str | None = None, debug: bool = False,
    answer_mode: str | None = None,
) -> dict:
    """Process one turn. The caller holds the call (_locked_call) and has set
    its tenant's catalog (catalog.using)."""
    try:
        trace = session.store.process_utterance(text, speaker=speaker, answer_mode=answer_mode)
    except Exception as exc:
        logger.exception("%s process_utterance failed (%d chars)", _who(key, user_id), len(text))
        call_log.log_event(
            "turn_error", **_ids(key, user_id), utterance=text, error=f"{type(exc).__name__}: {exc}"
        )
        raise ApiError(500, "pipeline_error", f"Couldn't process that turn ({type(exc).__name__}).") from exc
    session.turn_count += 1
    session.touch()
    view = views.build_turn_view(trace, session.store, session.turn_count)
    # The plain-text log carries no caller speech or tool arguments (emails,
    # phone numbers); the full turn is in the call log (call_log.py).
    fired = view["mcp"]["fired_this_turn"]
    tools = ", ".join(c["tool"] for c in fired)
    logger.info(
        "%s turn=%d (%d chars) -> %s/%s intent=%s%s",
        _who(key, user_id), view["turn_index"], len(text),
        view["intent"]["decision"], view["intent"]["turn_type"], view["intent"]["selected_intent"],
        f" | MCP: {tools}" if fired else "",
    )
    raw = views.to_jsonable(trace)
    call_log.log_event(
        "turn", **_ids(key, user_id), turn_index=view["turn_index"], utterance=text, view=view, raw_trace=raw
    )
    if debug:
        view["debug"] = raw
    return view


def _call_state(key: CallKey, session: CallSession) -> dict:
    return {
        "tenant_id": key[0],
        "call_id": key[1],
        "turn_count": session.turn_count,
        "frames": views.frames_view(session.store),
    }


# -- app ----------------------------------------------------------------------

app = FastAPI(
    title="Online Sales Intent Pipeline API",
    description=(
        "Embedding-based intent retrieval, confidence-banded classification, "
        "deterministic slot extraction and a frame-based dialogue state machine, "
        "with LLM escalation (Gemma 3 1B via Ollama) for disambiguation, "
        "multi-intent splitting and correction/reference resolution.\n\n"
        "A call is identified by its tenant ID plus its PBXware call ID, under "
        "`/v1/tenants/{tenant_id}/calls/{call_id}`. Workflow: `POST .../turns` "
        "(the first turn for a new ID starts the call; repeat) -> `GET` the call -> "
        "`DELETE` it on hangup. Turns may carry the call agent's `user_id`, which is only logged.\n\n"
        "All errors use `{\"error\": {\"code\", \"message\"}}`. `/v1` needs `X-Api-Key` when "
        "`AGENT_ASSIST_API_KEY` is set.\n\n"
        "CIP mode (`CIAP_API_BASE_URL` set): each tenant's intents come from CIP, and each call's state "
        "is loaded from and saved to ciap-api around every turn, so any instance can serve any turn. "
        "Otherwise every tenant gets the built-in catalog and calls live in memory: a call idle for "
        f"{config.CALL_IDLE_TIMEOUT_SECONDS}s is dropped, and later requests to it return 404 `call_expired`."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "calls", "description": "Call lifecycle and turns."},
        {"name": "catalog", "description": "The intent catalog: what each intent recognises, collects and fires."},
        {"name": "ops", "description": "Logging and health."},
        {
            "name": "archived",
            "description": (
                "Out of scope for this release. Kept in the code but disabled unless "
                "`SLOT_OVERRIDE_ENABLED` is set; while disabled it returns 404 `endpoint_disabled`."
            ),
        },
    ],
)

# Browser demo page (/demo), served only when config.DEMO_UI_ENABLED; it
# talks to the /v1 API above.
app.include_router(demo_router)


@app.middleware("http")
async def _service_auth(request: Request, call_next):
    """/v1 needs X-Api-Key when config.API_KEY is set (ciap-kafka sends it).
    The request's X-Correlation-Id goes on to ciap-api with the call's state."""
    if config.API_KEY and request.url.path.startswith("/v1/"):
        given = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(given.encode(), config.API_KEY.encode()):
            return _error_response(401, "unauthorized", "Missing or wrong X-Api-Key.")
    token = ciap_api.correlation_id.set(request.headers.get("x-correlation-id"))
    try:
        return await call_next(request)
    finally:
        ciap_api.correlation_id.reset(token)


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError):
    return _error_response(exc.status, exc.code, exc.message)


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(request: Request, exc: StarletteHTTPException):
    return _error_response(exc.status_code, _HTTP_CODES.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    details = [
        {"loc": [str(p) for p in e.get("loc", ())], "message": e.get("msg", "")} for e in exc.errors()
    ]
    first = details[0] if details else {"loc": [], "message": "invalid request"}
    message = f"{'.'.join(first['loc'])}: {first['message']}" if first["loc"] else first["message"]
    return _error_response(422, "validation_error", message, details)


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return _error_response(500, "internal_error", "Internal server error.")


_NOT_FOUND = {404: {"model": ErrorResponse, "description": "No such call (`call_not_found`), or it expired from inactivity (`call_expired`)."}}
_INVALID = {422: {"model": ErrorResponse, "description": "Request failed validation."}}
_CATALOG_NOT_FOUND = {404: {"model": ErrorResponse, "description": "No such intent."}}
_CONFLICT = {409: {"model": ErrorResponse, "description": "Conflicts with the current catalog or live calls."}}


# -- calls --------------------------------------------------------------------


@app.get("/v1/tenants/{tenant_id}/calls/{call_id}", response_model=CallStateResponse, tags=["calls"], responses=_NOT_FOUND)
def get_call(tenant_id: str, call_id: str):
    """Current state of every frame opened in this call, including each frame's
    `tool_call` and (for tools that really execute) `tool_result`."""
    key = (tenant_id, call_id)
    with catalog.using(_tenant_catalog(tenant_id)), _locked_call(key, create=False, touch=False, save=False) as session:
        return _call_state(key, session)


@app.delete("/v1/tenants/{tenant_id}/calls/{call_id}", status_code=204, tags=["calls"], responses=_NOT_FOUND)
def delete_call(tenant_id: str, call_id: str):
    """End a call. In memory its state is dropped (calls idle for the
    configured timeout are dropped automatically). CIP mode: nothing to free --
    ciap-api keeps the call's last state with its record."""
    key = (tenant_id, call_id)
    if ciap_api.enabled():
        with _locked_call(key, create=False, save=False):
            pass
    else:
        session = _get_session(key, touch=False)
        with session.lock:  # let a turn in flight finish first
            if _calls.get(key) is session:
                del _calls[key]
    logger.info("%s deleted", _who(key))
    call_log.log_event("call_deleted", **_ids(key))


@app.post("/v1/tenants/{tenant_id}/calls/{call_id}/reset", response_model=NewCallResponse, tags=["calls"], responses=_NOT_FOUND)
def reset_call(tenant_id: str, call_id: str):
    """Wipe a call's frames and turn counter but keep the same IDs."""
    key = (tenant_id, call_id)
    if ciap_api.enabled():
        def work():
            with _locked_call(key, create=False) as session:
                session.store, session.turn_count = ConversationStateStore(), 0
        _retrying(key, work)
    else:
        session = _get_session(key)
        with session.lock:  # let a turn in flight finish first
            if _calls.get(key) is session:
                _calls[key] = CallSession()
    logger.info("%s reset", _who(key))
    call_log.log_event("call_reset", **_ids(key))
    return NewCallResponse(tenant_id=tenant_id, call_id=call_id)


@app.post(
    "/v1/tenants/{tenant_id}/calls/{call_id}/turns",
    response_model=TurnView,
    response_model_exclude_none=False,
    tags=["calls"],
    responses={**_INVALID, 500: {"model": ErrorResponse, "description": "Pipeline failure."}},
)
def post_turn(
    tenant_id: TenantIdPath,
    call_id: CallIdPath,
    turn: TurnRequest,
    debug: bool = Query(False, description="Include the raw pipeline trace under `debug`."),
):
    """
    Feed one utterance into a call and get back a **stable** view of how it was
    handled: `intent` (decision and candidates), `slots` (every frame's state)
    and `mcp` (tool calls fired). The first turn for an unknown (or expired)
    tenant/call ID pair starts the call. `user_id` (the call agent) is only
    recorded in the logs.

    ### `intent.turn_type`

    | `turn_type` | `decision` | Meaning |
    |---|---|---|
    | `new_intent` | accepted | Top candidate scored above the high-confidence threshold; a frame opened. |
    | `new_intent_escalated` | escalated | Close candidates or mid-confidence score; the LLM chose the intent; a frame opened. |
    | `multi_intent_blocked` | multi_intent | The LLM confirmed the utterance holds two asks. No frame opens; `intent.message` says what to tell the customer. |
    | `slot_fill` | slot_fill | A pending frame received one or more missing slots. |
    | `unresolved_continuation` | continuation | A frame is waiting on a slot and this turn didn't supply it. Frame stays waiting. |
    | `ambiguous_continuation` | escalated | Same, but a correction/reference keyword was present, so the LLM was consulted. **Its verdict is reported but not applied.** |
    | `correction` | correction | Low confidence plus the LLM agreed it refers back to the latest frame; that frame becomes `REVERTED`. |
    | `unknown_intent` | unknown | Nothing matched; no frame opens. |
    | `duplicate_suppressed` | duplicate | Would have reopened an intent whose frame already FIRED this call; nothing new opens or fires. |
    | `slot_override` | override | Produced by the archived `PATCH .../slots/{slot}`, never by this endpoint. |

    `slots.frames[].tool_call` is set when a frame's required slots are all
    filled. Intents with a `remote` binding really execute on their MCP server
    (the shipped Zoho CRM lookups and the product intents); the response is `tool_result`,
    and `answer` is that result as a briefing for the agent: `answer_mode`
    picks a `focused` one (just what was asked) or a `full` one (everything
    useful in the result). The turn waits for the call, up to
    `MCP_CALL_TIMEOUT_SECONDS`, then for the answer, up to
    `RESPONDER_TIMEOUT_SECONDS` (`RESPONDER_FULL_TIMEOUT_SECONDS` for `full`).
    """
    key = (tenant_id, call_id)
    cat = _tenant_catalog(tenant_id, turn.catalog_version)

    def work():
        with catalog.using(cat), _locked_call(key, create=True, user_id=turn.user_id) as session:
            return _run_turn(key, session, turn.utterance, turn.speaker, turn.user_id, debug, turn.answer_mode)

    return _retrying(key, work)


@app.post(
    "/v1/tenants/{tenant_id}/calls/{call_id}/transcript",
    response_model=TranscriptResponse,
    tags=["calls"],
    responses=_INVALID,
)
def post_transcript(tenant_id: TenantIdPath, call_id: CallIdPath, body: TranscriptRequest):
    """Feed a whole transcript through the call, in order, as if each turn had
    been posted individually. Continues from the call's current state (use
    `/reset` first for a clean run), or starts the call if the ID is unknown.
    `answer_mode` applies to every turn. Validated up front: nothing is
    processed (or created) if any turn is invalid."""

    if body.turns is not None:
        turns = [(t.speaker, t.utterance) for t in body.turns]
    else:
        parsed = parse_turns(body.transcript)
        if not parsed:
            raise ApiError(422, "validation_error", 'No "Customer: ..." / "Agent: ..." lines found in transcript.')
        if len(parsed) > MAX_TRANSCRIPT_TURNS:
            raise ApiError(422, "validation_error", f"Transcript has {len(parsed)} turns; the limit is {MAX_TRANSCRIPT_TURNS}.")
        too_long = next((i for i, (_, t) in enumerate(parsed, 1) if len(t) > MAX_UTTERANCE_CHARS), None)
        if too_long:
            raise ApiError(422, "validation_error", f"Turn {too_long} exceeds {MAX_UTTERANCE_CHARS} characters.")
        turns = [(speaker.lower(), text) for speaker, text in parsed]

    key = (tenant_id, call_id)
    cat = _tenant_catalog(tenant_id, body.catalog_version)

    def work():
        with catalog.using(cat), _locked_call(key, create=True, user_id=body.user_id) as session:
            results = [
                _run_turn(key, session, text, speaker, body.user_id, answer_mode=body.answer_mode)
                for speaker, text in turns
            ]
            return {"tenant_id": tenant_id, "call_id": call_id, "turns": results,
                    "frames": views.frames_view(session.store)}

    return _retrying(key, work)


# -- catalog ------------------------------------------------------------------
#
# Read-only: in CIP mode each tenant's catalog is managed in CIP (Settings ->
# Agent Assist) and read from ciap-api; otherwise every tenant gets the
# built-in one (intents.py). Every view that carries an intent or tool name
# carries its label next to it, so UIs never need to show the raw name.


def _intent_info(cat: catalog.Catalog, name: str) -> IntentInfo:
    spec = cat.intents.get(name)
    if spec is None:
        raise ApiError(404, "intent_not_found", f"No intent named {name!r}.")
    return IntentInfo(
        name=name,
        label=spec.get("label") or views.humanize(name),
        description=spec.get("description"),
        required_slots=spec["required_slots"],
        optional_slots=spec.get("optional_slots", []),
        exemplar_count=len(spec["exemplars"]),
        exemplars=spec["exemplars"],
        executes_live=spec.get("remote") is not None,
        remote=spec.get("remote"),
    )


@app.get("/v1/tenants/{tenant_id}/intents", response_model=list[IntentInfo], tags=["catalog"])
def list_intents(tenant_id: TenantIdPath):
    """Every intent the tenant's turns are matched against: its display label,
    what its tool call does, its slots, its phrases (exemplars), and whether it
    really executes (`remote`) or is a mock that only builds the call."""
    cat = _tenant_catalog(tenant_id)
    return [_intent_info(cat, name) for name in cat.intents]


@app.get("/v1/tenants/{tenant_id}/intents/{name}", response_model=IntentInfo, tags=["catalog"], responses=_CATALOG_NOT_FOUND)
def get_intent(tenant_id: TenantIdPath, name: str):
    return _intent_info(_tenant_catalog(tenant_id), name)


class SlotTypes(BaseModel):
    types: list[str] = Field(..., description="Slot types a tenant's slots can have (CIP's slot types).")
    named: list[str] = Field(..., description="Slot names with their own extractor, used whatever the slot's type.")


@app.get("/v1/slots", response_model=SlotTypes, tags=["catalog"])
def list_slots():
    """How slots are read. A slot named like one of `named` uses that
    extractor; any other goes by its type (`OPTION` matches the slot's options;
    `TEXT` is read by the LLM when the slot is the one a request is waiting on)."""
    return SlotTypes(types=list(catalog.SLOT_TYPES), named=sorted(SLOT_EXTRACTORS))


# -- MCP connections -------------------------------------------------------------


class RemoteToolInfo(BaseModel):
    name: str = Field(..., description="Use as `remote.tool` on an intent.")
    description: str
    input_schema: dict
    annotations: dict = Field(..., description="The server's hints, e.g. `readOnlyHint`, `destructiveHint`.")
    allowed: bool = Field(
        ..., description="False if the connection's `allowed_tools` leaves this tool out: it can't be bound or called."
    )


@app.get(
    "/v1/tenants/{tenant_id}/connections/{name}/tools", response_model=list[RemoteToolInfo], tags=["connections"],
    responses={
        404: {"model": ErrorResponse, "description": "No such connection for the tenant (`connection_not_found`)."},
        502: {"model": ErrorResponse, "description": "The MCP server couldn't be reached or refused (`connection_failed`)."},
    },
)
def list_tenant_connection_tools(tenant_id: TenantIdPath, name: str):
    """The tools one of the tenant's connections offers (CIP mode: a connection
    configured in CIP; otherwise the shared connections file). CIP's Settings
    use it to pick the tools intents may call."""
    with catalog.using(_tenant_catalog(tenant_id)):
        return list_connection_tools(name)


@app.get("/v1/connections", response_model=list[str], tags=["connections"])
def list_connections():
    """Names of the configured MCP connections (`config.MCP_CONNECTIONS_PATH`).
    URLs and credentials are never returned."""
    return mcp_client.connection_names()


@app.get(
    "/v1/connections/{name}/tools", response_model=list[RemoteToolInfo], tags=["connections"],
    responses={
        404: {"model": ErrorResponse, "description": "No such connection (`connection_not_found`)."},
        502: {"model": ErrorResponse, "description": "The MCP server couldn't be reached or refused (`connection_failed`)."},
    },
)
def list_connection_tools(name: str):
    """The tools the connection's MCP server offers. Use these to fill an
    intent's `remote` binding."""
    try:
        return mcp_client.list_tools(name)
    except mcp_client.ConnectionNotConfigured as exc:
        raise ApiError(404, "connection_not_found", str(exc))
    except Exception as exc:
        raise ApiError(502, "connection_failed", f"{type(exc).__name__}: {exc}")


# -- ops ----------------------------------------------------------------------


@app.get("/v1/log", response_model=LogResponse, tags=["ops"])
def get_log(
    limit: int = Query(200, ge=1, le=1000, description="Max records returned (the newest)."),
    tenant_id: str | None = Query(None, description="Only records for this tenant."),
    call_id: str | None = Query(None, description="Only records for this call ID."),
    user_id: str | None = Query(None, description="Only records for this call agent."),
):
    """Recent structured records (calls, turns with full view + raw trace,
    overrides, errors), newest last. Backed by `logs/turns.jsonl`, which
    contains raw customer utterances -- protect this endpoint in production."""
    records, total = call_log.read_events(limit=limit, tenant_id=tenant_id, call_id=call_id, user_id=user_id)
    return LogResponse(records=records, total=total)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """`ok` when the escalation LLM server answers; `degraded` when it doesn't.
    Degraded still serves turns: LLM escalations fall back to a heuristic."""
    from ciap.pipeline import llm

    try:
        llm._client.list()
        ollama_up = True
    except Exception:
        ollama_up = False
    return HealthResponse(status="ok" if ollama_up else "degraded", ollama=ollama_up, model=llm.MODEL_NAME)


# -- archived -------------------------------------------------------------------
#
# Out of scope for this release, kept in the code. Off unless
# config.SLOT_OVERRIDE_ENABLED; checked per request so the flag is the only switch.


@app.patch(
    "/v1/tenants/{tenant_id}/calls/{call_id}/frames/{frame_id}/slots/{slot_name}",
    response_model=TurnView,
    tags=["archived"],
    deprecated=True,
    responses={
        **_NOT_FOUND,
        **_INVALID,
        409: {"model": ErrorResponse, "description": "The frame was REVERTED."},
    },
)
def override_slot(tenant_id: str, call_id: str, frame_id: int, slot_name: str, body: SlotOverrideRequest):
    """**Archived: disabled unless `SLOT_OVERRIDE_ENABLED` is set** (otherwise
    404 `endpoint_disabled`).

    Agent-driven correction of one slot (e.g. a misheard order ID). The value
    goes through the same extractor a normal turn uses. If the frame's required
    slots are all filled afterward, its tool call is rebuilt and **re-dispatched
    with the corrected data**, even if it already fired once with the wrong one.
    Counts as a turn (`turn_type` `slot_override`, speaker `agent`)."""
    if not config.SLOT_OVERRIDE_ENABLED:
        raise ApiError(404, "endpoint_disabled", "Slot override is archived and disabled on this server.")
    key, user_id = (tenant_id, call_id), body.user_id
    cat = _tenant_catalog(tenant_id)
    with catalog.using(cat), _locked_call(key, create=False) as session:
        store = session.store
        frame = store.get_frame(frame_id)
        if frame is None:
            raise ApiError(404, "frame_not_found", f"No frame #{frame_id} in call {call_id!r}.")
        if frame.status == "REVERTED":
            raise ApiError(409, "frame_reverted", f"Frame #{frame_id} was reverted; nothing to override.")
        if slot_name not in cat.intents.get(frame.intent, {}).get("required_slots", []):
            raise ApiError(422, "invalid_slot", f"{slot_name!r} isn't a slot on {frame.intent}.")

        try:
            frame, refired = store.override_slot(frame_id, slot_name, body.value)
        except ValueError as exc:
            call_log.log_event(
                "override_rejected", **_ids(key, user_id), frame_id=frame_id, slot=slot_name,
                value=body.value, error=str(exc),
            )
            raise ApiError(422, "invalid_slot_value", str(exc)) from exc

        session.turn_count += 1
        session.touch()
        view = views.build_turn_view(views.build_override_trace(frame, slot_name, refired), store, session.turn_count)

    logger.info(
        "%s turn=%d override %s#%d.%s -> %r%s",
        _who(key, user_id), view["turn_index"], frame.intent, frame.id, slot_name, frame.slots[slot_name],
        " | MCP refired" if refired else "",
    )
    call_log.log_event(
        "slot_override", **_ids(key, user_id), turn_index=view["turn_index"], frame_id=frame.id,
        slot=slot_name, value=views.to_jsonable(frame.slots[slot_name]), refired=refired, view=view,
    )
    return view
