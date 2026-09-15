# Demo HTTP wrapper around the stage4 pipeline (embedding retrieval,
# regex/library slot extraction, frame-based dialogue state, Gemma 3 1B LLM
# escalation). Not a production service -- see README.md's "Known
# limitations" for what that would need. This exists so the pipeline can be
# poked at over HTTP (curl, Postman, or FastAPI's own /docs UI) instead of
# only through demo.py's CLI.
#
# The pipeline is stateful within one call (frames persist turn to turn), so
# this isn't a single stateless endpoint: POST /calls starts one and hands
# back a call_id, then every /calls/{call_id}/turns request feeds one more
# customer utterance into that same ConversationStateStore.
#
# Run: uvicorn server:app --reload
# Docs: http://127.0.0.1:8000/docs

import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from demo_web import router as demo_router, warm_embeddings
from state_store import ConversationStateStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the sentence-embedding model and build the exemplar index at
    # startup, so the first real turn isn't a 2-5s cold start. Adds that
    # time to boot instead. (The escalation LLM is still lazy -- warm it
    # with one escalating turn if the first live one needs to be fast.)
    warm_embeddings()
    yield


app = FastAPI(
    title="Online Sales Intent Pipeline (Demo API)",
    description=(
        "Embedding-based intent retrieval, confidence-banded classification, "
        "deterministic slot extraction, and a frame-based dialogue state "
        "machine -- with LLM escalation (Gemma 3 1B) for disambiguation, "
        "multi-intent splitting, and correction/reference resolution. "
        "See the root CLAUDE.md and stage4/README.md for the design. "
        "See `POST /calls/{call_id}/turns` below for the full `turn_type` "
        "reference and a note on which MCP tool call actually executes "
        "against a real backend (`search_zoho_lead`) versus the rest, "
        "which are mocks."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Focused browser demo (self-contained router, own /demo routes and call
# store -- does not touch the API below). Open http://127.0.0.1:8000/demo
# See demo_web.py and static_demo/.
app.include_router(demo_router)

# In-memory only: call_id -> ConversationStateStore. Lost on restart, and
# only correct for a single server process -- the same "no persistence
# across calls" limitation stage4's README already flags, just made visible
# at the process level too. Fine for a demo; a real deployment would swap
# this for Redis or a DB keyed the same way.
_calls: dict[str, ConversationStateStore] = {}


class NewCallResponse(BaseModel):
    call_id: str


class TurnRequest(BaseModel):
    utterance: str = Field(..., min_length=1, description="One turn, e.g. \"Where is my order?\"")
    speaker: Literal["customer", "agent"] = Field(
        default="customer",
        description=(
            "Who said this turn. Both run through the exact same intent/slot/frame "
            "pipeline below -- an agent can ask for something specific or state that "
            "an action is being taken, just as validly as a customer can. `speaker` "
            "only affects provenance: it's echoed back in the trace and becomes "
            "`initiated_by` on any frame this turn opens."
        ),
    )


class FrameOut(BaseModel):
    id: int
    intent: str
    status: str
    slots: dict
    missing_slots: list[str]
    tool_call: dict | None = None
    initiated_by: str = "customer"


class CallStateResponse(BaseModel):
    call_id: str
    frames: list[FrameOut]


def _get_store(call_id: str) -> ConversationStateStore:
    store = _calls.get(call_id)
    if store is None:
        raise HTTPException(
            status_code=404,
            detail=f"No call with id {call_id!r}. POST /calls to start one.",
        )
    return store


@app.post("/calls", response_model=NewCallResponse, status_code=201)
def create_call():
    """Start a new call. Returns a call_id -- pass it to every subsequent
    /calls/{call_id}/turns request so turns land in the same conversation."""
    call_id = uuid.uuid4().hex[:12]
    _calls[call_id] = ConversationStateStore()
    return NewCallResponse(call_id=call_id)


@app.post("/calls/{call_id}/turns")
def post_turn(call_id: str, turn: TurnRequest):
    """
    Feed one customer utterance into an existing call. Returns the same
    per-turn trace `demo.py` prints: retrieval candidates, heuristic flags,
    `turn_type`, any LLM escalation (with the model's reasoning), slots
    filled, and any tool call fired.

    ### `turn_type` reference

    Which branch of `ConversationStateStore.process_utterance()` handled the
    turn, and which extra fields ride along with it:

    | `turn_type` | What it means | Extra fields |
    |---|---|---|
    | `new_intent` | Top candidate scored above `HIGH_CONFIDENCE` -- a fresh frame opened for it. | `frame`, `intent`, `filled_slots`, `tool_calls` (if it fired immediately) |
    | `new_intent_escalated` | Either two candidates were close enough to trigger the multi-intent check but the LLM decided it's really one intent, or the top score fell in the `MID_CONFIDENCE` band and the LLM picked between candidates. | `frame`, `intent`, `filled_slots`, `llm_escalation`, `tool_calls` |
    | `multi_intent_blocked` | Two candidates both scored above `HIGH_CONFIDENCE`, close together, and the LLM confirmed the utterance really contains both asks -- blocked outright, no frame opened for either intent. | `intents` (list of names), `message` (ask the customer to resubmit one at a time), `llm_escalation` |
    | `slot_fill` | A frame was `AWAITING_INFO` and this turn supplied at least one of its missing slots. | `frame`, `filled_slots`, `tool_calls` (if that completed the frame) |
    | `unresolved_continuation` | A frame was `AWAITING_INFO`, this turn didn't fill anything, and it didn't look like a correction/reference either -- treated as a non-answer, frame stays waiting. | `frame` |
    | `ambiguous_continuation` | A frame was `AWAITING_INFO`, nothing got filled, but a correction/reference keyword was present -- escalated to the LLM to interpret. | `llm_escalation` |
    | `correction` | Low confidence, but a correction/reference keyword was present and the LLM agreed it refers back to the most recent frame -- that frame flips to `REVERTED`. | `frame`, `llm_escalation` |
    | `unknown_intent` | Nothing matched: low confidence, no pending frame, no correction/reference keyword (or the LLM decided a keyword hit wasn't one). No frame opens. | `llm_escalation` (only if the correction/reference path came back negative) |
    | `duplicate_suppressed` | Would otherwise have opened a fresh frame (via any of the three rows above), but a `FIRED` frame for that same intent already exists this call -- e.g. an agent's "I'll flag it" scoring against an intent whose frame already fired earlier. No new frame opens, no tool call fires; `frame` points at the existing one instead. | `intent`, `frame` (the existing frame's id), `llm_escalation` (if the pre-empted branch would have escalated) |

    Every trace also carries `speaker` (`"customer"` or `"agent"`, from the
    request) -- both run through the table above identically; `speaker` only
    determines `initiated_by` on any frame the turn opens.

    `filled_slots` is always a list, even for one slot. `tool_calls` only
    appears on turn types that can fire a frame this turn, and only when one
    actually did.

    ### MCP tool calls

    Every intent mapped in `mcp_tools.INTENT_TO_TOOL` fires a tool-call
    envelope (`mcp_tools.build_tool_call`) once its frame's required slots
    are all filled. **Almost every tool is a mock** -- the envelope is just
    constructed and returned, nothing executes it. The one exception is
    `search_zoho_lead` (intent `lookup_crm_record`, slot `email`):
    `mcp_tools.dispatch_tool_call` actually executes it against the real
    Zoho CRM API (see `zoho_client.py`), using credentials from the
    environment (`.env`). Its result is *not* included in this endpoint's
    response today -- it's only surfaced by the `/demo` browser UI (as
    `frame.tool_result`), so calling this HTTP API directly for
    `lookup_crm_record` still fires the real Zoho request, you just won't
    see the response back here.
    """
    store = _get_store(call_id)
    return store.process_utterance(turn.utterance, speaker=turn.speaker)


@app.get("/calls/{call_id}", response_model=CallStateResponse)
def get_call(call_id: str):
    """Current state of every frame opened in this call so far. `initiated_by`
    is `"customer"` or `"agent"` depending on which turn opened it. `tool_call`
    is the envelope built once a frame's slots are all filled (see
    `POST /calls/{call_id}/turns` for the `turn_type` reference this ties
    into) -- note this does not include `tool_result`, the response from the
    one live-executing tool (`search_zoho_lead`); that's only in the `/demo`
    browser UI today."""
    store = _get_store(call_id)
    frames = [FrameOut(**asdict(frame)) for frame in store.frames]
    return CallStateResponse(call_id=call_id, frames=frames)


@app.delete("/calls/{call_id}", status_code=204)
def delete_call(call_id: str):
    """Drop a call's state. Not required -- there's no TTL or cleanup here,
    so long-running processes will accumulate calls until restarted."""
    if call_id not in _calls:
        raise HTTPException(status_code=404, detail=f"No call with id {call_id!r}.")
    del _calls[call_id]


@app.get("/health")
def health():
    return {"status": "ok"}
