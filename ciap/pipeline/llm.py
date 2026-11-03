# Stage 4: LLM escalation.
#
# By design, the LLM is only invoked for:
#   - disambiguating between close-scoring intent candidates
#   - multiple intents in one utterance
#   - references ("that one", "the previous order")
#   - corrections
#   - interruptions
#   - conditional requests
#   - free-text (TEXT) slots no pattern can read, e.g. a customer's name
#
# Everything else stays deterministic (embeddings + regex + state machine)
# because it's cheaper and more reliable. This module used to be a stub
# (llm_stub.py) that recorded an escalation and returned a best-effort
# fallback without calling a model. It now calls a real local model served by
# Ollama -- by default the same one responder.py answers with
# (config.LLM_MODEL_NAME) -- for the reasons above, and falls back to
# the same heuristic behavior only if the call fails or returns something
# unusable, so the rest of the pipeline keeps working either way.

import json

import ollama

from ciap import catalog
from ciap import config

MODEL_NAME = config.LLM_MODEL_NAME

# Escalations run while the call's lock is held, so a stalled model must not
# hold the call (or a rename waiting on every call's lock) forever.
_client = ollama.Client(timeout=config.LLM_ESCALATION_TIMEOUT_SECONDS)


def _shares_answer_model():
    return MODEL_NAME == config.RESPONDER_MODEL_NAME


def _shared_model_options():
    # Sharing the answer model: the same num_ctx as responder.py on every
    # call, or Ollama reloads the model each time the two kinds of call
    # alternate.
    if _shares_answer_model() and config.RESPONDER_NUM_CTX:
        return {"num_ctx": config.RESPONDER_NUM_CTX}
    return {}


def _shared_keep_alive():
    # ...and the same keep_alive, or an escalation resets the answer model's
    # "loaded for good" to Ollama's 5-minute default.
    return {"keep_alive": config.RESPONDER_KEEP_ALIVE_SECONDS} if _shares_answer_model() else {}


def _intent_brief(intent_name, n_exemplars=3):
    exemplars = catalog.intents().get(intent_name, {}).get("exemplars", [])[:n_exemplars]
    return f"- {intent_name}: " + "; ".join(f'"{e}"' for e in exemplars)


def _ask(prompt):
    """Call the model, expecting a single JSON object back. Returns the
    parsed dict, or {"_error": ...} if the call failed or didn't return a
    JSON object."""
    try:
        response = _client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0, **_shared_model_options()},
            # Models that can reason first (Gemma 4) do by default; a pick from
            # a short list doesn't need it, and it would cost seconds per turn.
            think=False,
            **_shared_keep_alive(),
        )
        result = json.loads(response["message"]["content"])
    except Exception as exc:
        return {"_error": str(exc)}
    if not isinstance(result, dict):
        return {"_error": f"expected a JSON object, got {type(result).__name__}"}
    return result


# -- one handler per escalation reason ---------------------------------------

def _resolve_ambiguous_intent(utterance, candidates, fallback_intent):
    candidate_names = [intent for intent, _score in candidates]
    prompt = (
        "A customer support message scored close to several possible intents. "
        "Pick the single best match.\n\n"
        f'Customer message: "{utterance}"\n\n'
        "Candidate intents (name: example phrasings):\n"
        + "\n".join(_intent_brief(name) for name in candidate_names)
        + "\n\nRespond with a JSON object: "
        '{"resolved_intent": "<one of the candidate names above, exactly>", '
        '"reasoning": "<one short sentence>"}.'
    )
    result = _ask(prompt)
    resolved = result.get("resolved_intent")
    if resolved not in candidate_names:
        resolved = fallback_intent
        result["reasoning"] = result.get("reasoning") or "model call failed or returned an unlisted intent; used top candidate"
    return {
        "escalated": True,
        "reason": "mid_confidence_ambiguous_intent",
        "model": MODEL_NAME,
        "utterance": utterance,
        "context": candidates,
        "resolved_intent": resolved,
        "note": result.get("reasoning", "resolved by model"),
    }


def _resolve_multi_intent(utterance, two_intents, fallback_intent):
    top_intent, second_intent = two_intents
    prompt = (
        "A customer support message scored highly for two different intents. "
        "Decide whether the customer is really asking about both, or just one.\n\n"
        f'Customer message: "{utterance}"\n\n'
        "Candidate intents:\n"
        f"{_intent_brief(top_intent)}\n{_intent_brief(second_intent)}\n\n"
        "Respond with a JSON object: "
        '{"resolved_intents": [<one or both of the two intent names above>], '
        '"reasoning": "<one short sentence>"}.'
    )
    result = _ask(prompt)
    resolved = result.get("resolved_intents")
    if isinstance(resolved, list):
        resolved = list(dict.fromkeys(resolved))  # ["a", "a"] is one intent, not two
    valid = (
        isinstance(resolved, list)
        and 1 <= len(resolved) <= 2
        and all(name in two_intents for name in resolved)
    )
    if not valid:
        resolved = list(two_intents)
        result["reasoning"] = result.get("reasoning") or "model call failed or returned invalid intents; treated as genuine multi-intent"
    return {
        "escalated": True,
        "reason": "multiple_intents_in_utterance",
        "model": MODEL_NAME,
        "utterance": utterance,
        "context": two_intents,
        "resolved_intents": resolved,
        "note": result.get("reasoning", "resolved by model"),
    }


def _resolve_correction(reason, utterance, ref_context, fallback_intent):
    referenced_intent = ref_context.get("intent") if ref_context else None
    prompt = (
        "A customer said something that might be a correction or a reference "
        "back to an earlier request, or it might just be a new, unrelated "
        "statement.\n\n"
        f'Customer message: "{utterance}"\n'
        f'Earlier request in play: "{referenced_intent}"\n\n'
        "Respond with a JSON object: "
        '{"is_correction_or_reference": true or false, '
        '"reasoning": "<one short sentence>"}.'
    )
    result = _ask(prompt)
    is_correction = result.get("is_correction_or_reference")
    if not isinstance(is_correction, bool):
        # Reverting a frame (possibly one whose tool already ran) on no verdict
        # does more harm than leaving the turn unresolved.
        is_correction = False
        result["reasoning"] = result.get("reasoning") or "model call failed or returned no verdict; not treated as a correction"
    return {
        "escalated": True,
        "reason": reason,
        "model": MODEL_NAME,
        "utterance": utterance,
        "context": ref_context,
        "resolved_intent": fallback_intent,
        "is_correction_or_reference": is_correction,
        "note": result.get("reasoning", "resolved by model"),
    }


def escalate(reason, utterance, context=None, fallback_intent=None):
    """Resolve one LLM escalation. `reason` picks which prompt/handler runs;
    `context` and `fallback_intent` carry whatever the caller (state_store.py)
    already knows so a failed/unparseable model call can fall back to the
    same heuristic the old stub used."""
    if reason == "mid_confidence_ambiguous_intent":
        return _resolve_ambiguous_intent(utterance, context, fallback_intent)
    if reason == "multiple_intents_in_utterance":
        return _resolve_multi_intent(utterance, context, fallback_intent)
    if reason in ("correction_or_reference", "unresolved_continuation_possible_correction"):
        return _resolve_correction(reason, utterance, context, fallback_intent)
    # No specific prompt for this reason (shouldn't happen given the call
    # sites in state_store.py) -- record it without calling the model.
    return {
        "escalated": True,
        "reason": reason,
        "model": MODEL_NAME,
        "utterance": utterance,
        "context": context,
        "resolved_intent": fallback_intent,
        "note": "no handler for this escalation reason; no model call made",
    }


def extract_text_slot(utterance, name, spec):
    """Read a free-text slot (TEXT, no pattern) from one turn, e.g. the
    caller's name or the product they mean, or None when the turn doesn't say
    it. Only asked for the slot a frame is waiting on, so at most one model
    call per turn."""
    spec = spec or {}
    about = "; ".join(x for x in (spec.get("label"), spec.get("description")) if x) or name.replace("_", " ")
    asked = f'The agent asked: "{spec["prompt"]}"\n' if spec.get("prompt") else ""
    prompt = (
        "Extract one piece of information from a line said on a customer support call.\n\n"
        f"Information wanted: {about}\n{asked}"
        f'Line: "{utterance}"\n\n'
        "Copy the value exactly as said. If the line doesn't contain it, use null. "
        'Respond with a JSON object: {"value": "<the value>" or null}.'
    )
    value = _ask(prompt).get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    # Only what was actually said: a model that paraphrases or invents doesn't fill the slot.
    return value if value.lower() in utterance.lower() else None

