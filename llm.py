# Stage 4: LLM escalation.
#
# Per design (root CLAUDE.md), the LLM is only invoked for:
#   - disambiguating between close-scoring intent candidates
#   - multiple intents in one utterance
#   - references ("that one", "the previous order")
#   - corrections
#   - interruptions
#   - conditional requests
#
# Everything else stays deterministic (embeddings + regex + state machine)
# because it's cheaper and more reliable. This module used to be a stub
# (llm_stub.py) that recorded an escalation and returned a best-effort
# fallback without calling a model. It now calls a real local model --
# Gemma 3 1B, served by Ollama -- for the reasons above, and falls back to
# the same heuristic behavior only if the call fails or returns something
# unusable, so the rest of the pipeline keeps working either way.

import json

import ollama

from intents import INTENTS

MODEL_NAME = "gemma3:1b"

_client = ollama.Client()


def _intent_brief(intent_name, n_exemplars=3):
    exemplars = INTENTS[intent_name]["exemplars"][:n_exemplars]
    return f"- {intent_name}: " + "; ".join(f'"{e}"' for e in exemplars)


def _ask(prompt):
    """Call the model, expecting a single JSON object back. Returns the
    parsed dict, or {} if the call failed or didn't return valid JSON."""
    try:
        response = _client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        return json.loads(response["message"]["content"])
    except Exception as exc:
        return {"_error": str(exc)}


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
        is_correction = True
        result["reasoning"] = result.get("reasoning") or "model call failed or returned no verdict; defaulted to treating it as a correction"
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
