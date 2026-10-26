# Turn/frame "views": the stable, presentation-ready shape built from the
# pipeline's raw per-turn trace and the store's frames. Used by the HTTP API
# (server.py) -- and through it the browser demo -- so there's exactly one
# place that decides what a turn looks like to a consumer. Pure functions, no
# I/O and no FastAPI, so they're testable without a server.

from ciap.pipeline.intents import INTENTS
from ciap.integrations.mcp_tools import TOOLS

# turn_type (from state_store.process_utterance) -> a short decision label.
# Coarser than turn_type, meant for badges / at-a-glance summaries.
DECISION_BY_TURN_TYPE = {
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


def humanize(name):
    """Fallback label for a catalog entry with none set: "check_refund_status"
    -> "Check Refund Status"."""
    return name.replace("_", " ").title()


# Labels are looked up at view time (not stored on frames), so a label edited
# through the catalog API shows up immediately, including on live calls. A
# name that's no longer in the catalog (deleted mid-call) falls back too.
def intent_label(name):
    if name is None:
        return None
    return INTENTS.get(name, {}).get("label") or humanize(name)


def tool_label(name):
    if name is None:
        return None
    return TOOLS.get(name, {}).get("label") or humanize(name)


def tool_call_view(tool_call):
    """A tool call with its display label alongside the tool name."""
    if not tool_call:
        return tool_call
    return {**tool_call, "tool_label": tool_label(tool_call.get("tool"))}


def slot_display(value):
    # new_date is stored as {"text": ..., "date": ...}; consumers get the resolved date.
    if isinstance(value, dict) and "date" in value:
        return value["date"]
    return value


def to_jsonable(obj):
    """Recursively make a raw trace JSON-safe (tuples -> lists, numpy scalars
    -> Python numbers). Anything else unknown becomes its string form."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return to_jsonable(obj.item())
    return str(obj)


def frame_view(frame, filled_this_turn=()):
    return {
        "id": frame.id,
        "intent": frame.intent,
        "intent_label": intent_label(frame.intent),
        "status": frame.status,
        "initiated_by": frame.initiated_by,
        "required": INTENTS.get(frame.intent, {}).get("required_slots", []),
        "slots": {k: slot_display(v) for k, v in frame.slots.items()},
        "missing": list(frame.missing_slots),
        "filled_this_turn": list(filled_this_turn),
        "tool_call": tool_call_view(frame.tool_call),
        "tool_result": frame.tool_result,
        "answer": frame.answer,
    }


def frames_view(store, filled_this_turn=(), focus_frame_id=None):
    return [
        frame_view(f, filled_this_turn if f.id == focus_frame_id else ())
        for f in store.frames
    ]


def all_tool_calls(store):
    return [
        {"frame": f.id, "intent": f.intent, "intent_label": intent_label(f.intent), **tool_call_view(f.tool_call)}
        for f in store.frames
        if f.tool_call
    ]


def build_turn_view(trace, store, turn_index):
    turn_type = trace.get("turn_type", "unknown_intent")

    candidates = [
        {"intent": name, "label": intent_label(name), "score": round(float(score), 3)}
        for name, score in trace.get("candidates", [])
    ]

    # Which intent(s) the turn settled on.
    selected = trace.get("intent")
    selected_label = intent_label(selected)
    if selected is None and turn_type == "multi_intent_blocked":
        names = trace.get("intents", [])
        selected = " + ".join(names) or None
        selected_label = " + ".join(intent_label(n) for n in names) or None
    elif selected is None and "frame" in trace:
        frame = store.get_frame(trace["frame"])
        selected = frame.intent if frame else None
        selected_label = intent_label(selected)

    escalation = None
    esc = trace.get("llm_escalation")
    if esc:
        resolved = (
            esc.get("resolved_intent")
            or esc.get("resolved_intents")
            or esc.get("is_correction_or_reference")
        )
        escalation = {
            "reason": esc.get("reason"),
            "model": esc.get("model"),
            "note": esc.get("note"),
            "resolved": to_jsonable(resolved),
        }

    filled = trace.get("filled_slots", []) or []

    return {
        "turn_index": turn_index,
        "utterance": trace.get("utterance", ""),
        "speaker": trace.get("speaker", "customer"),
        "intent": {
            "turn_type": turn_type,
            "decision": DECISION_BY_TURN_TYPE.get(turn_type, turn_type),
            "selected_intent": selected,
            "selected_intent_label": selected_label,
            "top_score": candidates[0]["score"] if candidates else None,
            "candidates": candidates,
            "heuristic_flags": trace.get("heuristic_flags", []),
            "escalation": escalation,
            "message": trace.get("message"),
        },
        "slots": {"frames": frames_view(store, filled, trace.get("frame"))},
        "mcp": {
            "fired_this_turn": [tool_call_view(c) for c in to_jsonable(trace.get("tool_calls", []))],
            "all_calls": all_tool_calls(store),
        },
    }


def build_override_trace(frame, slot_name, refired):
    """A synthetic trace for an agent slot override, shaped so it renders
    through build_turn_view like any other turn (turn_type 'slot_override')."""
    trace = {
        "utterance": f"Corrected {slot_name} to {slot_display(frame.slots[slot_name])!r}",
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
    return trace
