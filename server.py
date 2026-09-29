# HTTP API around the intent pipeline (embedding retrieval, regex/library slot
# extraction, frame-based dialogue state, Gemma 3 1B LLM escalation).
#
# The pipeline is stateful within one call (frames persist turn to turn), so
# this isn't a single stateless endpoint: POST /v1/calls starts a call and
# hands back a call_id, then every turn for that call is fed into the same
# ConversationStateStore.
#
# All response shaping lives in views.py; all pipeline logic in state_store.py.
# This module is only routing, validation, session bookkeeping and errors.
#
# Run:  uvicorn server:app
# Docs: http://127.0.0.1:8000/docs   (OpenAPI JSON: /openapi.json)
#
# Not production-ready as shipped -- no auth, in-memory sessions, no TTL.
# See API_DOCUMENTATION.md "Production handover items".

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

import call_log
import views
from demo import parse_turns
from demo_web import router as demo_router
from intents import INTENTS
from mcp_tools import INTENT_TO_TOOL, TOOLS
from state_store import ConversationStateStore

logger = call_log.logger

MAX_UTTERANCE_CHARS = 2000
MAX_TRANSCRIPT_TURNS = 200


def warm_embeddings():
    """Load the sentence-embedding model and build the exemplar index, so the
    first real turn isn't a 2-5s cold start."""
    from embeddings import retrieve

    start = time.perf_counter()
    retrieve("warm up")
    logger.info("embedding index warmed (%.1fs)", time.perf_counter() - start)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Adds the model-load time to boot instead of the first turn. (The
    # escalation LLM is still lazy -- warm it with one escalating turn if the
    # first live one needs to be fast.)
    warm_embeddings()
    yield


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


class NewCallResponse(BaseModel):
    call_id: str = Field(..., examples=["298cec6767e2"])


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


class Candidate(BaseModel):
    intent: str
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
    status: Literal["OPEN", "AWAITING_INFO", "FIRED", "REVERTED"]
    initiated_by: Speaker
    required: list[str] = Field(..., description="Slots this intent needs before its tool call fires.")
    slots: dict = Field(..., description="Filled slots. `new_date` is the resolved ISO date.")
    missing: list[str]
    filled_this_turn: list[str] = Field(
        ..., description="Slots this turn filled on this frame (always [] from GET /v1/calls/{id})."
    )
    tool_call: dict | None = Field(
        default=None, description='`{"tool": ..., "arguments": {...}}`, set once the frame is FIRED.'
    )
    tool_result: dict | None = Field(
        default=None,
        description=(
            "Response from a tool that actually executed. Only `search_zoho_lead` "
            "does today (a failure appears as `{\"error\": ...}`); null for every mock tool."
        ),
    )


class SlotsView(BaseModel):
    frames: list[FrameView] = Field(..., description="Every frame in the call so far.")


class McpView(BaseModel):
    fired_this_turn: list[dict]
    all_calls: list[dict] = Field(..., description="Every tool call built in this call so far.")


class TurnView(BaseModel):
    turn_index: int = Field(..., description="1-based count of turns/overrides in this call.")
    utterance: str
    speaker: Speaker
    intent: IntentView
    slots: SlotsView
    mcp: McpView
    debug: dict | None = Field(default=None, description="Raw pipeline trace. Only with `?debug=true`.")


class CallStateResponse(BaseModel):
    call_id: str
    turn_count: int
    frames: list[FrameView]


class TranscriptResponse(BaseModel):
    call_id: str
    turns: list[TurnView] = Field(..., description="One view per processed turn, in order.")
    frames: list[FrameView] = Field(..., description="Final state of every frame.")


class IntentInfo(BaseModel):
    name: str
    required_slots: list[str]
    tool: str | None
    exemplar_count: int
    exemplars: list[str]


class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict
    intents: list[str] = Field(..., description="Intents that fire this tool.")
    executes_live: bool = Field(..., description="True if the tool really runs against a backend; false if it's a mock.")


class LogResponse(BaseModel):
    records: list[dict]
    total: int = Field(..., description="Matching records before `limit` was applied.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: bool = Field(..., description="Whether the escalation LLM server is reachable.")
    model: str = Field(..., description="Escalation model name.")


# -- session store ------------------------------------------------------------
#
# In-memory only: lost on restart, correct for a single process only, no TTL.


class CallSession:
    def __init__(self):
        self.store = ConversationStateStore()
        self.turn_count = 0
        # One in-flight turn per call: process_utterance mutates the frame list.
        self.lock = threading.Lock()


_calls: dict[str, CallSession] = {}


def _get_session(call_id: str) -> CallSession:
    session = _calls.get(call_id)
    if session is None:
        raise ApiError(404, "call_not_found", f"No call with id {call_id!r}. POST /v1/calls to start one.")
    return session


def _run_turn(call_id: str, session: CallSession, text: str, speaker: str, debug: bool = False) -> dict:
    """Process one turn under the call's lock. Caller holds no lock."""
    with session.lock:
        try:
            trace = session.store.process_utterance(text, speaker=speaker)
        except Exception as exc:
            logger.exception("call=%s process_utterance failed on %r", call_id, text)
            call_log.log_event("turn_error", call_id=call_id, utterance=text, error=f"{type(exc).__name__}: {exc}")
            raise ApiError(500, "pipeline_error", f"Couldn't process that turn ({type(exc).__name__}).") from exc
        session.turn_count += 1
        view = views.build_turn_view(trace, session.store, session.turn_count)

    fired = view["mcp"]["fired_this_turn"]
    tools = ", ".join(f"{c['tool']}({c['arguments']})" for c in fired)
    logger.info(
        "call=%s turn=%d %r -> %s/%s intent=%s%s",
        call_id, view["turn_index"], text,
        view["intent"]["decision"], view["intent"]["turn_type"], view["intent"]["selected_intent"],
        f" | MCP: {tools}" if fired else "",
    )
    raw = views.to_jsonable(trace)
    call_log.log_event("turn", call_id=call_id, turn_index=view["turn_index"], utterance=text, view=view, raw_trace=raw)
    if debug:
        view["debug"] = raw
    return view


def _call_state(call_id: str, session: CallSession) -> dict:
    return {
        "call_id": call_id,
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
        "Workflow: `POST /v1/calls` -> `POST /v1/calls/{call_id}/turns` (repeat) -> "
        "`GET /v1/calls/{call_id}` -> `DELETE /v1/calls/{call_id}`.\n\n"
        "All errors use `{\"error\": {\"code\", \"message\"}}`. See API_DOCUMENTATION.md "
        "for the full reference, including production handover items "
        "(no auth, in-memory sessions, no TTL)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "calls", "description": "Call lifecycle and turns."},
        {"name": "catalog", "description": "Intent and tool catalogs."},
        {"name": "ops", "description": "Logging and health."},
    ],
)

# Browser demo page only (/demo); it talks to the /v1 API above.
app.include_router(demo_router)


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


_NOT_FOUND = {404: {"model": ErrorResponse, "description": "No such call."}}
_INVALID = {422: {"model": ErrorResponse, "description": "Request failed validation."}}


# -- calls --------------------------------------------------------------------


@app.post("/v1/calls", response_model=NewCallResponse, status_code=201, tags=["calls"])
def create_call():
    """Start a new call. Pass the returned `call_id` to every later request."""
    call_id = uuid.uuid4().hex[:12]
    _calls[call_id] = CallSession()
    logger.info("call=%s created", call_id)
    call_log.log_event("call_created", call_id=call_id)
    return NewCallResponse(call_id=call_id)


@app.get("/v1/calls/{call_id}", response_model=CallStateResponse, tags=["calls"], responses=_NOT_FOUND)
def get_call(call_id: str):
    """Current state of every frame opened in this call, including each frame's
    `tool_call` and (for `search_zoho_lead`) `tool_result`."""
    session = _get_session(call_id)
    with session.lock:
        return _call_state(call_id, session)


@app.delete("/v1/calls/{call_id}", status_code=204, tags=["calls"], responses=_NOT_FOUND)
def delete_call(call_id: str):
    """Drop a call's state. Not required -- nothing expires calls automatically."""
    _get_session(call_id)
    del _calls[call_id]
    logger.info("call=%s deleted", call_id)
    call_log.log_event("call_deleted", call_id=call_id)


@app.post("/v1/calls/{call_id}/reset", response_model=NewCallResponse, tags=["calls"], responses=_NOT_FOUND)
def reset_call(call_id: str):
    """Wipe a call's frames and turn counter but keep the same `call_id`."""
    _get_session(call_id)
    _calls[call_id] = CallSession()
    logger.info("call=%s reset", call_id)
    call_log.log_event("call_reset", call_id=call_id)
    return NewCallResponse(call_id=call_id)


@app.post(
    "/v1/calls/{call_id}/turns",
    response_model=TurnView,
    response_model_exclude_none=False,
    tags=["calls"],
    responses={**_NOT_FOUND, **_INVALID, 500: {"model": ErrorResponse, "description": "Pipeline failure."}},
)
def post_turn(
    call_id: str,
    turn: TurnRequest,
    debug: bool = Query(False, description="Include the raw pipeline trace under `debug`."),
):
    """
    Feed one utterance into an existing call and get back a **stable** view of
    how it was handled: `intent` (decision and candidates), `slots` (every
    frame's state) and `mcp` (tool calls fired).

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
    | `slot_override` | override | Produced by `PATCH .../slots/{slot}`, never by this endpoint. |

    `slots.frames[].tool_call` is set when a frame's required slots are all
    filled. Every tool is a mock except `search_zoho_lead`, which really
    executes; its response is `tool_result`.
    """
    session = _get_session(call_id)
    return _run_turn(call_id, session, turn.utterance, turn.speaker, debug)


@app.post(
    "/v1/calls/{call_id}/transcript",
    response_model=TranscriptResponse,
    tags=["calls"],
    responses={**_NOT_FOUND, **_INVALID},
)
def post_transcript(call_id: str, body: TranscriptRequest):
    """Feed a whole transcript through the call, in order, as if each turn had
    been posted individually. Continues from the call's current state (use
    `/reset` first for a clean run). Validated up front: nothing is processed
    if any turn is invalid."""
    session = _get_session(call_id)

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

    results = [_run_turn(call_id, session, text, speaker) for speaker, text in turns]
    with session.lock:
        frames = views.frames_view(session.store)
    return {"call_id": call_id, "turns": results, "frames": frames}


@app.patch(
    "/v1/calls/{call_id}/frames/{frame_id}/slots/{slot_name}",
    response_model=TurnView,
    tags=["calls"],
    responses={
        **_NOT_FOUND,
        **_INVALID,
        409: {"model": ErrorResponse, "description": "The frame was REVERTED."},
    },
)
def override_slot(call_id: str, frame_id: int, slot_name: str, body: SlotOverrideRequest):
    """Agent-driven correction of one slot (e.g. a misheard order ID). The value
    goes through the same extractor a normal turn uses. If the frame's required
    slots are all filled afterward, its tool call is rebuilt and **re-dispatched
    with the corrected data**, even if it already fired once with the wrong one.
    Counts as a turn (`turn_type` `slot_override`, speaker `agent`)."""
    session = _get_session(call_id)
    with session.lock:
        store = session.store
        frame = store.get_frame(frame_id)
        if frame is None:
            raise ApiError(404, "frame_not_found", f"No frame #{frame_id} in call {call_id!r}.")
        if frame.status == "REVERTED":
            raise ApiError(409, "frame_reverted", f"Frame #{frame_id} was reverted; nothing to override.")
        if slot_name not in INTENTS[frame.intent]["required_slots"]:
            raise ApiError(422, "invalid_slot", f"{slot_name!r} isn't a slot on {frame.intent}.")

        try:
            frame, refired = store.override_slot(frame_id, slot_name, body.value)
        except ValueError as exc:
            call_log.log_event(
                "override_rejected", call_id=call_id, frame_id=frame_id, slot=slot_name,
                value=body.value, error=str(exc),
            )
            raise ApiError(422, "invalid_slot_value", str(exc)) from exc

        session.turn_count += 1
        view = views.build_turn_view(views.build_override_trace(frame, slot_name, refired), store, session.turn_count)

    logger.info(
        "call=%s turn=%d override %s#%d.%s -> %r%s",
        call_id, view["turn_index"], frame.intent, frame.id, slot_name, frame.slots[slot_name],
        " | MCP refired" if refired else "",
    )
    call_log.log_event(
        "slot_override", call_id=call_id, turn_index=view["turn_index"], frame_id=frame.id,
        slot=slot_name, value=views.to_jsonable(frame.slots[slot_name]), refired=refired, view=view,
    )
    return view


# -- catalog ------------------------------------------------------------------


@app.get("/v1/intents", response_model=list[IntentInfo], tags=["catalog"])
def list_intents():
    """Every intent the pipeline can recognise, its required slots, the tool it
    fires, and its retrieval exemplars."""
    return [
        IntentInfo(
            name=name,
            required_slots=spec["required_slots"],
            tool=INTENT_TO_TOOL.get(name),
            exemplar_count=len(spec["exemplars"]),
            exemplars=spec["exemplars"],
        )
        for name, spec in INTENTS.items()
    ]


@app.get("/v1/tools", response_model=list[ToolInfo], tags=["catalog"])
def list_tools():
    """The MCP-shaped tool catalog. `executes_live` marks the tools that really
    run against a backend; all others are mocks that only build the call."""
    return [
        ToolInfo(
            name=name,
            description=spec["description"],
            input_schema=spec["inputSchema"],
            intents=[i for i, t in INTENT_TO_TOOL.items() if t == name],
            executes_live=name == "search_zoho_lead",
        )
        for name, spec in TOOLS.items()
    ]


# -- ops ----------------------------------------------------------------------


@app.get("/v1/log", response_model=LogResponse, tags=["ops"])
def get_log(
    limit: int = Query(200, ge=1, le=1000, description="Max records returned (the newest)."),
    call_id: str | None = Query(None, description="Only records for this call."),
):
    """Recent structured records (calls, turns with full view + raw trace,
    overrides, errors), newest last. Backed by `logs/turns.jsonl`, which
    contains raw customer utterances -- protect this endpoint in production."""
    records, total = call_log.read_events(limit=limit, call_id=call_id)
    return LogResponse(records=records, total=total)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """`ok` when the escalation LLM server answers; `degraded` when it doesn't.
    Degraded still serves turns: LLM escalations fall back to a heuristic."""
    import llm

    try:
        llm._client.list()
        ollama_up = True
    except Exception:
        ollama_up = False
    return HealthResponse(status="ok" if ollama_up else "degraded", ollama=ollama_up, model=llm.MODEL_NAME)
