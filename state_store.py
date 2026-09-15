# Stage 3: the conversational state store.
#
# Everything before this stage classified and slot-filled one utterance in
# isolation. That's not enough for a real call: a customer asks about their
# order, then updates their address, then circles back to dispute a charge,
# then corrects something they said two turns ago. This module is what
# remembers all of that across turns.
#
# Frames exactly as specified in the root CLAUDE.md: OPEN, AWAITING_INFO,
# FIRED, REVERTED. Each detected intent becomes a frame that tracks which of
# its required slots are still missing; later turns either fill those slots,
# open a new frame (interruption), or revert a previous one (correction).

from dataclasses import dataclass, field

import config
import llm
import mcp_tools
from embeddings import retrieve
from intents import INTENTS
from slots import SLOT_EXTRACTORS


@dataclass
class Frame:
    id: int
    intent: str
    status: str  # OPEN, AWAITING_INFO, FIRED, REVERTED
    slots: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    tool_call: dict = None  # set once status becomes FIRED - see mcp_tools.build_tool_call
    tool_result: dict = None  # set for tool_calls actually dispatched - see mcp_tools.dispatch_tool_call
    initiated_by: str = "customer"  # "customer" or "agent" -- who opened this frame


class ConversationStateStore:
    """Holds every frame opened during a call and decides, turn by turn,
    whether an utterance fills a pending frame, opens a new one, splits into
    several, or reverts something already in play."""

    def __init__(self):
        self.frames = []
        self._next_id = 1

    # -- frame helpers --------------------------------------------------

    def _new_frame(self, intent, initiated_by="customer"):
        frame = Frame(id=self._next_id, intent=intent, status="OPEN", initiated_by=initiated_by)
        self._next_id += 1
        self.frames.append(frame)
        return frame

    def _pending_frame(self):
        for frame in reversed(self.frames):
            if frame.status == "AWAITING_INFO":
                return frame
        return None

    def _last_frame(self):
        return self.frames[-1] if self.frames else None

    def get_frame(self, frame_id):
        for frame in self.frames:
            if frame.id == frame_id:
                return frame
        return None

    def override_slot(self, frame_id, slot_name, raw_value):
        """Agent-driven correction for one slot on an existing frame (e.g. a
        misheard order ID) -- run through the same extractor a normal turn
        would use, so a corrected value is validated the same way. If the
        frame's required slots are all filled afterward, its MCP tool call is
        rebuilt and redispatched with the corrected slots, whether or not it
        had already fired once with the wrong data. Raises ValueError for
        anything that doesn't make sense to apply, for the caller to surface.
        Returns (frame, refired)."""
        frame = self.get_frame(frame_id)
        if frame is None:
            raise ValueError(f"No frame #{frame_id} in this call.")
        if frame.status == "REVERTED":
            raise ValueError(f"Frame #{frame_id} was reverted; nothing to override.")

        required = INTENTS[frame.intent]["required_slots"]
        if slot_name not in required:
            raise ValueError(f"{slot_name!r} isn't a slot on {frame.intent}.")

        extractor = SLOT_EXTRACTORS.get(slot_name)
        value = extractor(raw_value) if extractor else raw_value
        if not value:
            raise ValueError(f"Couldn't read a valid {slot_name} from {raw_value!r}.")

        frame.slots[slot_name] = value
        frame.missing_slots = [s for s in required if s not in frame.slots]

        refired = False
        if not frame.missing_slots:
            frame.tool_call = mcp_tools.build_tool_call(frame.intent, frame.slots)
            frame.tool_result = mcp_tools.dispatch_tool_call(frame.tool_call)
            frame.status = "FIRED"
            refired = True
        else:
            frame.status = "AWAITING_INFO"
        return frame, refired

    def _fill_slots(self, frame, utterance):
        required = INTENTS[frame.intent]["required_slots"]
        filled_now = []
        for slot_name in required:
            if slot_name in frame.slots:
                continue
            extractor = SLOT_EXTRACTORS.get(slot_name)
            if not extractor:
                continue
            value = extractor(utterance)
            if value:
                frame.slots[slot_name] = value
                filled_now.append(slot_name)
        frame.missing_slots = [s for s in required if s not in frame.slots]
        frame.status = "FIRED" if not frame.missing_slots else "AWAITING_INFO"
        if frame.status == "FIRED" and frame.tool_call is None:
            frame.tool_call = mcp_tools.build_tool_call(frame.intent, frame.slots)
            frame.tool_result = mcp_tools.dispatch_tool_call(frame.tool_call)
        return filled_now

    def _open_and_fill(self, intent, utterance, initiated_by="customer"):
        frame = self._new_frame(intent, initiated_by=initiated_by)
        filled = self._fill_slots(frame, utterance)
        return frame, filled

    def _fired_frame_for_intent(self, intent):
        """Most recent FIRED frame for this intent, if any -- lets a stray
        remark about something already completed ("I'll flag it",
        "connecting you now") get recognized without reopening it and
        re-firing its tool call."""
        for frame in reversed(self.frames):
            if frame.intent == intent and frame.status == "FIRED":
                return frame
        return None

    def _open_new_intent(self, intent, utterance, speaker, trace, turn_type):
        """Shared tail for every branch below that opens a frame for a
        freshly detected intent: reuse an already-FIRED frame for the same
        intent instead of duplicating it, otherwise open and fill a new
        one."""
        duplicate = self._fired_frame_for_intent(intent)
        if duplicate is not None:
            trace["turn_type"] = "duplicate_suppressed"
            trace["intent"] = intent
            trace["frame"] = duplicate.id
            return trace

        frame, filled = self._open_and_fill(intent, utterance, initiated_by=speaker)
        trace["turn_type"] = turn_type
        trace["frame"] = frame.id
        trace["intent"] = intent
        trace["filled_slots"] = filled
        if frame.status == "FIRED" and frame.tool_call:
            trace["tool_calls"] = [frame.tool_call]
        return trace

    # -- main entry point -------------------------------------------------

    def process_utterance(self, utterance, speaker="customer"):
        """Feed one utterance into the store, in speaking order. `speaker`
        is "customer" (default) or "agent" -- both run through the exact
        same intent-retrieval/confidence-band/frame logic below (an agent
        can be the one asking for something specific, or stating that an
        action is being taken, just as validly as a customer can). `speaker`
        only affects provenance: it rides along in the trace and becomes
        `Frame.initiated_by` on any frame this turn opens. Returns a trace
        dict describing how the turn was handled, for the caller to display."""
        candidates = retrieve(utterance, top_k=config.RETRIEVAL_TOP_K)
        top_intent, top_score = candidates[0] if candidates else (None, 0.0)
        second_intent, second_score = candidates[1] if len(candidates) > 1 else (None, 0.0)

        lowered = utterance.lower()
        correction_flag = any(kw in lowered for kw in config.CORRECTION_KEYWORDS)
        reference_flag = any(kw in lowered for kw in config.REFERENCE_KEYWORDS)

        trace = {
            "utterance": utterance,
            "speaker": speaker,
            "candidates": candidates,
            "heuristic_flags": [
                name for name, present in
                (("correction", correction_flag), ("reference", reference_flag)) if present
            ],
        }

        pending = self._pending_frame()

        # Multiple intents in one utterance: both top candidates score high
        # and disagree. Computed here because the pending-frame check right
        # below needs it too -- an utterance that's ambiguous between two
        # intents is much more likely noise around a pending question (e.g.
        # an agent's "what's the order number?" scoring close across two
        # unrelated intents by vocabulary overlap alone) than a real new ask,
        # so it shouldn't count as confidently abandoning the pending frame.
        multi_intent = (
            second_intent is not None
            and top_score >= config.HIGH_CONFIDENCE
            and second_score >= config.HIGH_CONFIDENCE
            and (top_score - second_score) <= config.MULTI_INTENT_GAP
            and top_intent != second_intent
        )

        # A frame is waiting on a slot, and this turn doesn't confidently and
        # unambiguously look like a brand new intent -> treat it as the
        # answer (or a non-answer) to the pending frame first, before ever
        # opening/blocking on a new one.
        if pending is not None and (
            top_score < config.HIGH_CONFIDENCE
            or top_intent == pending.intent
            or multi_intent
        ):
            filled = self._fill_slots(pending, utterance)
            if filled:
                trace["turn_type"] = "slot_fill"
                trace["frame"] = pending.id
                trace["filled_slots"] = filled
                if pending.status == "FIRED" and pending.tool_call:
                    trace["tool_calls"] = [pending.tool_call]
            elif correction_flag or reference_flag:
                trace["llm_escalation"] = llm.escalate(
                    "unresolved_continuation_possible_correction", utterance,
                    context={"pending_frame": pending.id, "intent": pending.intent},
                )
                trace["turn_type"] = "ambiguous_continuation"
            else:
                trace["turn_type"] = "unresolved_continuation"
                trace["frame"] = pending.id
            return trace

        # Rather than splitting into two frames, a confirmed multi-intent is
        # blocked outright -- no frame opens for either intent, and the
        # customer is asked to send them one at a time. (Unreachable with a
        # pending frame still in play -- see the check above.)
        if multi_intent:
            llm_result = llm.escalate(
                "multiple_intents_in_utterance", utterance,
                context=[top_intent, second_intent],
            )
            trace["llm_escalation"] = llm_result
            resolved_intents = llm_result["resolved_intents"]
            if len(resolved_intents) > 1:
                trace["turn_type"] = "multi_intent_blocked"
                trace["intents"] = resolved_intents
                trace["message"] = config.MULTI_INTENT_BLOCK_MESSAGE.format(
                    intents=" and ".join(name.replace("_", " ") for name in resolved_intents)
                )
                return trace
            # Model decided this was single-intent after all.
            return self._open_new_intent(
                resolved_intents[0], utterance, speaker, trace, "new_intent_escalated"
            )

        # Confident new intent.
        if top_score >= config.HIGH_CONFIDENCE:
            if pending is not None and pending.intent != top_intent:
                trace["heuristic_flags"].append("interruption")
            return self._open_new_intent(top_intent, utterance, speaker, trace, "new_intent")

        # Mid-confidence: a real system would ask an LLM to pick between the
        # top candidates. Stubbed here, falling back to the top candidate.
        if top_score >= config.MID_CONFIDENCE:
            llm_result = llm.escalate(
                "mid_confidence_ambiguous_intent", utterance,
                context=candidates, fallback_intent=top_intent,
            )
            trace["llm_escalation"] = llm_result
            chosen_intent = llm_result["resolved_intent"]
            return self._open_new_intent(chosen_intent, utterance, speaker, trace, "new_intent_escalated")

        # Low confidence but looks like a correction/reference to something
        # already in play -> revert the most recent frame.
        if (correction_flag or reference_flag) and self.frames:
            last = self._last_frame()
            llm_result = llm.escalate(
                "correction_or_reference", utterance,
                context={"last_frame": last.id, "intent": last.intent},
            )
            trace["llm_escalation"] = llm_result
            if llm_result["is_correction_or_reference"]:
                last.status = "REVERTED"
                trace["turn_type"] = "correction"
                trace["frame"] = last.id
            else:
                trace["turn_type"] = "unknown_intent"
            return trace

        trace["turn_type"] = "unknown_intent"
        return trace
