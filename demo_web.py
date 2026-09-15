# Focused browser demo for the intent pipeline.
#
# One page, one input box, three panels mapped 1:1 to what the demo has to
# show:
#   1. Intent identification
#   2. Slot extraction across multiple turns
#   3. The MCP tool call that fires once a frame's slots are all filled
#
# Built for a live walkthrough where arbitrary text gets typed in, so every
# pipeline path (accept / escalate / unknown / slot-fill / correction) renders
# as a tidy panel, and a bad turn shows a one-line message instead of a stack
# trace.
#
# Self-contained: its own in-memory call store, its own routes under /demo,
# its own static files in static_demo/. It reuses the real pipeline
# (ConversationStateStore) unchanged. server.py mounts it with:
#
#     from demo_web import router as demo_router, warm_embeddings
#     app.include_router(demo_router)
#     # warm_embeddings() from the startup lifespan
#
# Logging: every call/turn/error is written two ways --
#   logs/demo.log        human-readable, one line per event (also to console)
#   logs/demo_turns.jsonl one JSON object per event, the full view + raw trace
# and GET /demo/log serves recent structured records back.
#
# The MCP step is still mcp_tools.build_tool_call()'s dict -- when a real MCP
# server is available, the only change is at the "fired_this_turn" seam below.

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from demo import parse_turn  # same "Speaker: text" tagging convention as demo.py / real transcripts
from intents import INTENTS
from state_store import ConversationStateStore

router = APIRouter()

_STATIC = Path(__file__).parent / "static_demo"
_ASSETS = {"app.js": "text/javascript", "style.css": "text/css"}

MAX_UTTERANCE_CHARS = 2000

# -- logging --------------------------------------------------------------

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_TURN_LOG = _LOG_DIR / "demo_turns.jsonl"

logger = logging.getLogger("demo")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s")
    for _handler in (
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "demo.log", encoding="utf-8"),
    ):
        _handler.setFormatter(_fmt)
        logger.addHandler(_handler)
    logger.propagate = False


def _log_event(event: str, **fields):
    """Append one structured record to logs/demo_turns.jsonl."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    try:
        with _TURN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("failed to append to %s", _TURN_LOG.name)


def warm_embeddings():
    """Load the sentence-embedding model and build the exemplar index.
    Called from server.py's startup so the first real turn -- from either
    speaker -- isn't a 2-5s cold start."""
    from embeddings import retrieve

    start = time.perf_counter()
    retrieve("warm up")
    logger.info("embedding index warmed (%.1fs)", time.perf_counter() - start)


# -- call store ---------------------------------------------------------------

# Demo-local, separate from server.py's _calls. Same caveats (in-memory,
# single process, no TTL) -- fine for one presenter, one page.
_calls: dict[str, ConversationStateStore] = {}
_turn_counts: dict[str, int] = {}


class TurnRequest(BaseModel):
    utterance: str = Field(
        default="",
        description=(
            'One turn. Tag it "Customer: ..." or "Agent: ..." the way a real '
            "transcript does -- untagged text is treated as the customer."
        ),
    )


class SlotOverrideRequest(BaseModel):
    value: str = Field(
        default="",
        description=(
            "Corrected raw value for this slot (e.g. the real order ID after "
            "a misread one). Parsed with the same extractor a normal turn "
            "would use, so it has to look like a real value, not just any text."
        ),
    )


def _get_store(call_id: str) -> ConversationStateStore:
    store = _calls.get(call_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"No demo call {call_id!r}. Start a new call.")
    return store


# -- static page --------------------------------------------------------------

@router.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse(_STATIC / "index.html")


@router.get("/demo/static/{name}", include_in_schema=False)
def demo_asset(name: str):
    media_type = _ASSETS.get(name)
    if media_type is None:  # allowlist only -- no path traversal
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(_STATIC / name, media_type=media_type)


# -- call lifecycle ---------------------------------------------------------

@router.post("/demo/calls", include_in_schema=False)
def demo_create_call():
    call_id = uuid.uuid4().hex[:12]
    _calls[call_id] = ConversationStateStore()
    _turn_counts[call_id] = 0
    logger.info("call=%s created", call_id)
    _log_event("call_created", call_id=call_id)
    return {"call_id": call_id}


@router.post("/demo/calls/{call_id}/reset", include_in_schema=False)
def demo_reset_call(call_id: str):
    _get_store(call_id)
    _calls[call_id] = ConversationStateStore()
    _turn_counts[call_id] = 0
    logger.info("call=%s reset", call_id)
    _log_event("call_reset", call_id=call_id)
    return {"call_id": call_id, "ok": True}


@router.post("/demo/calls/{call_id}/turns", include_in_schema=False)
def demo_turn(call_id: str, turn: TurnRequest):
    store = _get_store(call_id)

    raw = (turn.utterance or "").strip()
    if not raw:
        msg = 'Empty input -- type a turn ("Customer: ..." or "Agent: ...") and send again.'
        _log_event("turn_rejected", call_id=call_id, utterance=turn.utterance, error=msg)
        return {"error": msg}
    if len(raw) > MAX_UTTERANCE_CHARS:
        msg = f"That's {len(raw)} characters; keep a single turn under {MAX_UTTERANCE_CHARS}."
        _log_event("turn_rejected", call_id=call_id, utterance=raw[:200] + " ...", error=msg)
        return {"error": msg}

    # Untagged text (no "Customer: "/"Agent: " prefix) is treated as the
    # customer, same as before this tagging convention existed.
    parsed = parse_turn(raw)
    speaker, text = parsed if parsed else ("Customer", raw)

    try:
        trace = store.process_utterance(text, speaker=speaker.lower())
    except Exception as exc:  # never let a raw traceback reach the page
        logger.exception("call=%s process_utterance failed on %r", call_id, text)
        _log_event("turn_error", call_id=call_id, utterance=text, error=str(exc))
        return {"error": f"Couldn't process that turn ({type(exc).__name__}: {exc})."}

    _turn_counts[call_id] = _turn_counts.get(call_id, 0) + 1
    turn_index = _turn_counts[call_id]
    view = _build_view(trace, store, turn_index)

    fired = view["mcp"]["fired_this_turn"]
    tools = ", ".join(f"{c['tool']}({c['arguments']})" for c in fired)
    logger.info(
        "call=%s turn=%d %r -> %s/%s intent=%s%s",
        call_id, turn_index, text,
        view["intent"]["decision"], view["intent"]["turn_type"],
        view["intent"]["selected_intent"],
        f" | MCP: {tools}" if fired else "",
    )
    _log_event(
        "turn",
        call_id=call_id,
        turn_index=turn_index,
        utterance=text,
        view=view,
        raw_trace=trace,
    )
    return view


# -- agent-driven correction: override a frame's slot ------------------------
#
# For when a slot got the wrong value -- most commonly a misheard/mistyped
# order ID -- and the agent needs to fix it directly rather than re-saying
# the whole turn. Reuses the normal turn-view shape (_build_view) so the
# existing 3-panel renderer needs no special case: this shows up in the
# transcript and panels exactly like any other turn, tagged "override". If
# the frame's slots are now all filled, its MCP tool call is rebuilt and
# redispatched with the corrected data -- including re-firing a call that
# had already gone out once with the wrong value.

@router.post("/demo/calls/{call_id}/frames/{frame_id}/slots/{slot_name}", include_in_schema=False)
def demo_override_slot(call_id: str, frame_id: int, slot_name: str, body: SlotOverrideRequest):
    store = _get_store(call_id)

    raw = (body.value or "").strip()
    if not raw:
        msg = "Enter a corrected value before saving."
        _log_event("override_rejected", call_id=call_id, frame_id=frame_id, slot=slot_name, error=msg)
        return {"error": msg}

    try:
        frame, refired = store.override_slot(frame_id, slot_name, raw)
    except ValueError as exc:
        _log_event(
            "override_rejected", call_id=call_id, frame_id=frame_id, slot=slot_name,
            value=raw, error=str(exc),
        )
        return {"error": str(exc)}

    _turn_counts[call_id] = _turn_counts.get(call_id, 0) + 1
    turn_index = _turn_counts[call_id]

    trace = {
        "utterance": f"Corrected {slot_name} to {frame.slots[slot_name]!r}",
        "speaker": "agent",
        "candidates": [],
        "heuristic_flags": ["override"],
        "turn_type": "slot_override",
        "intent": frame.intent,
        "frame": frame.id,
        "filled_slots": [slot_name],
    }
    if refired:
        trace["tool_calls"] = [frame.tool_call]
    view = _build_view(trace, store, turn_index)

    logger.info(
        "call=%s turn=%d override %s#%d.%s -> %r%s",
        call_id, turn_index, frame.intent, frame.id, slot_name, frame.slots[slot_name],
        " | MCP refired" if refired else "",
    )
    _log_event(
        "slot_override",
        call_id=call_id, turn_index=turn_index, frame_id=frame.id,
        slot=slot_name, value=frame.slots[slot_name], refired=refired, view=view,
    )
    return view


@router.get("/demo/log", include_in_schema=False)
def demo_log(limit: int = 200, call_id: str | None = None):
    """Recent structured records from logs/demo_turns.jsonl, newest last.
    Optionally filtered to one call_id."""
    if not _TURN_LOG.exists():
        return {"records": [], "total": 0}
    records = []
    with _TURN_LOG.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if call_id and rec.get("call_id") != call_id:
                continue
            records.append(rec)
    return {"records": records[-limit:], "total": len(records)}


# -- trace -> stable 3-panel view -----------------------------------------

# turn_type (from state_store.process_utterance) -> a short decision label for
# the INTENT panel headline. Keeps the panel readable regardless of which
# pipeline branch handled the turn.
_DECISION_BY_TURN_TYPE = {
    "new_intent": "accepted",
    "new_intent_escalated": "escalated",
    "multi_intent_blocked": "multi_intent",
    "slot_fill": "slot_fill",
    "unresolved_continuation": "continuation",
    "ambiguous_continuation": "escalated",
    "correction": "correction",
    "unknown_intent": "unknown",
    "duplicate_suppressed": "duplicate",
    "slot_override": "override",
}


def _frame_by_id(store, frame_id):
    for frame in store.frames:
        if frame.id == frame_id:
            return frame
    return None


def _slot_display(value):
    # new_date is stored as {"text": ..., "date": ...}; show the resolved date.
    if isinstance(value, dict) and "date" in value:
        return value["date"]
    return value


def _frames_view(store, filled_this_turn, focus_frame_id):
    frames_view = []
    for frame in store.frames:
        required = INTENTS.get(frame.intent, {}).get("required_slots", [])
        frames_view.append({
            "id": frame.id,
            "intent": frame.intent,
            "status": frame.status,
            "initiated_by": frame.initiated_by,
            "required": required,
            "slots": {k: _slot_display(v) for k, v in frame.slots.items()},
            "missing": frame.missing_slots,
            "filled_this_turn": filled_this_turn if frame.id == focus_frame_id else [],
            "tool_call": frame.tool_call,
            "tool_result": frame.tool_result,
        })
    return frames_view


def _all_calls(store):
    return [
        {"frame": f.id, "intent": f.intent, **f.tool_call}
        for f in store.frames
        if f.tool_call
    ]


def _build_view(trace, store, turn_index):
    turn_type = trace.get("turn_type", "unknown_intent")

    candidates = [
        {"intent": name, "score": round(float(score), 3)}
        for name, score in trace.get("candidates", [])
    ]

    # Which intent(s) the turn settled on, for the panel headline.
    selected = trace.get("intent")
    if selected is None and turn_type == "multi_intent_blocked":
        selected = " + ".join(trace.get("intents", [])) or None
    elif selected is None and "frame" in trace:
        frame = _frame_by_id(store, trace["frame"])
        selected = frame.intent if frame else None

    escalation = None
    esc = trace.get("llm_escalation")
    if esc:
        escalation = {
            "reason": esc.get("reason"),
            "model": esc.get("model"),
            "note": esc.get("note"),
            "resolved": esc.get("resolved_intent")
            or esc.get("resolved_intents")
            or esc.get("is_correction_or_reference"),
        }

    filled_this_turn = trace.get("filled_slots", []) or []
    focus_frame_id = trace.get("frame")

    return {
        "turn_index": turn_index,
        "utterance": trace.get("utterance", ""),
        "speaker": trace.get("speaker", "customer"),
        "intent": {
            "turn_type": turn_type,
            "decision": _DECISION_BY_TURN_TYPE.get(turn_type, turn_type),
            "selected_intent": selected,
            "top_score": candidates[0]["score"] if candidates else None,
            "candidates": candidates,
            "heuristic_flags": trace.get("heuristic_flags", []),
            "escalation": escalation,
            "message": trace.get("message"),
        },
        "slots": {"frames": _frames_view(store, filled_this_turn, focus_frame_id)},
        "mcp": {
            # The seam: today this is build_tool_call()'s dict. A real MCP
            # client call would slot in right here without touching anything
            # above.
            "fired_this_turn": trace.get("tool_calls", []),
            "all_calls": _all_calls(store),
        },
        "error": None,
    }
