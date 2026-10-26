# Stage 5: an agent-facing answer for a live tool's result.
#
# A tool that really executes (see mcp_tools.dispatch_tool_call) hands back
# its server's response as JSON. This turns that into a short plain-English
# briefing for the call agent, answering what was asked on the call -- with
# no per-tool setup, so a newly bound MCP tool gets answers on day one:
#   - no_results / error: a fixed sentence, no model call
#   - success: the LLM writes 1-2 sentences from the data (records matching
#     the lookup first, then pruned to the mode's limits); if the
#     call fails, times out, or the answer mentions an email or number that
#     isn't in the data or the question, a generic renderer lists the
#     record's fields instead.
# Each request picks a mode: "focused" answers just what was asked in 1-2
# sentences; "full" briefs the agent on everything useful in the result, so
# the model (and the fallback) sees more of it and gets longer to answer.
# Every answer is {"text", "source", "model", "note", "mode"}; `source` says
# which path produced it ("rule", "llm" or "fallback") and `note` why a
# fallback ran.

import json
import re

import ollama

from ciap import config
from ciap.integrations.mcp_tools import TOOLS

MODEL_NAME = config.RESPONDER_MODEL_NAME

_clients = {
    "focused": ollama.Client(timeout=config.RESPONDER_TIMEOUT_SECONDS),
    "full": ollama.Client(timeout=config.RESPONDER_FULL_TIMEOUT_SECONDS),
}

# Loading the model from disk can take far longer than an answer may.
WARM_TIMEOUT_SECONDS = 120

MODES = ("focused", "full")

# Per mode: what the model sees of a result (lists cut to max_list_items,
# strings to max_string_chars, the whole payload to max_payload_chars) and
# what the fallback renderer lists (max_records records, max_fields each).
LIMITS = {
    "focused": {"max_list_items": 5, "max_string_chars": 300, "max_payload_chars": 4000,
                "max_records": 3, "max_fields": 8},
    "full": {"max_list_items": 10, "max_string_chars": 600, "max_payload_chars": 8000,
             "max_records": 10, "max_fields": 20},
}

# What each mode asks the model to write.
INSTRUCTIONS = {
    "focused": "Brief the agent on the result in one or two plain sentences, answering what was asked.",
    "full": (
        "Answer what was asked in the first sentence, then brief the agent on "
        "everything else useful in the result: every record and its key details, "
        "in a short paragraph of plain sentences."
    ),
}


def _tool_label(name):
    return TOOLS.get(name, {}).get("label") or name.replace("_", " ").title()


def _answer(text, source, mode, note=None):
    return {
        "text": text, "source": source, "model": MODEL_NAME if source == "llm" else None, "note": note, "mode": mode,
    }


# -- preparing the data -------------------------------------------------------

def prune(value, mode="focused"):
    """Drop what doesn't help answer a question: empty values, "$"-prefixed
    server metadata keys (Zoho's $approval, $editable, ...), and anything past
    the mode's list and string limits. Returns None if nothing is left."""
    limits = LIMITS[mode]
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if str(key).startswith("$"):
                continue
            item = prune(item, mode)
            if item is not None:
                kept[key] = item
        return kept or None
    if isinstance(value, list):
        kept = [item for item in (prune(v, mode) for v in value[:limits["max_list_items"]]) if item is not None]
        return kept or None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        cap = limits["max_string_chars"]
        return value if len(value) <= cap else value[:cap] + "..."
    return value


def _scalars(value):
    # Every non-container value in a record, however deeply nested.
    if isinstance(value, dict):
        for item in value.values():
            yield from _scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from _scalars(item)
    elif value is not None:
        yield value


def _match_score(record, wanted):
    """How well a record matches the lookup arguments: per argument, 2 for a
    field equal to it, 1 for a field containing it. Case-insensitive; values
    with 7+ digits (phone numbers) compare on digits alone, ignoring a
    country code or formatting on either side."""
    fields = [str(v).strip().casefold() for v in _scalars(record)]
    digits = [d for d in (re.sub(r"\D", "", f) for f in fields) if len(d) >= 7]
    score = 0
    for arg in wanted:
        arg_digits = re.sub(r"\D", "", arg)
        if len(arg_digits) >= 7 and any(d.endswith(arg_digits) or arg_digits.endswith(d) for d in digits):
            score += 2
        elif arg in fields:
            score += 2
        elif len(arg) >= 3 and any(arg in f for f in fields):
            score += 1
    return score


def rank_records(data, arguments):
    """The result with its records reordered so those matching the lookup
    arguments come first (otherwise in the server's order), so the list
    limits below never drop the record that was asked about in favor of
    ones that weren't. Uses only the values, never field names, so it works
    for any server's result shape."""
    if not isinstance(data, dict) or not arguments:
        return data
    wanted = [str(v).strip().casefold() for v in _scalars(arguments) if str(v).strip()]
    for key, value in data.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            ranked = sorted(value, key=lambda record: -_match_score(record, wanted))
            return {**data, key: ranked}
    return data


def as_text(value, indent=0):
    """The result as indented "key: value" lines for the prompt. Same content
    as JSON at far fewer tokens (no quotes, braces or commas) -- and on a CPU
    reading the prompt is most of the answer's time. "id" keys are left out:
    record IDs are noise to a briefing."""
    pad = " " * indent
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "id":
                continue
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(as_text(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {item}")
    elif isinstance(value, list):
        for n, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}- item {n}:")
                lines.extend(as_text(item, indent + 2))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{value}")
    return lines if indent else "\n".join(lines)


# -- the LLM path ----------------------------------------------------------------

def _question(frame):
    return "\n".join(f'{u["speaker"].capitalize()}: "{u["text"]}"' for u in frame.utterances)


def _prompt(frame, data, mode="focused"):
    tool = frame.tool_call["tool"]
    payload = as_text(data)[:LIMITS[mode]["max_payload_chars"]]
    instructions = INSTRUCTIONS[mode]
    if mode == "full":
        # Small models otherwise brief the first record and stop.
        count = len(_records(data)) if data else 0
        if count > 1:
            instructions += f" The result holds {count} records; cover each one."
    return (
        "You are helping a call-center agent during a live call. Because of what "
        "was said on the call, a lookup was run in a backend system. "
        f"{instructions}\n\n"
        "Rules: write to the agent and refer to the customer in the third person, "
        "never as \"you\". Use only facts from the result below; never guess or add details. "
        "Don't mention field names or the lookup itself. If the result "
        "doesn't answer what was asked, say what it does show.\n\n"
        f"What was said on the call:\n{_question(frame)}\n\n"
        f"Lookup: {_tool_label(tool)} -- {TOOLS.get(tool, {}).get('description', '')}\n"
        f"Lookup arguments: {json.dumps(frame.tool_call['arguments'], ensure_ascii=False)}\n"
        f"Result:\n{payload}\n\n"
        'Respond with a JSON object: {"answer": "<your briefing for the agent>"}.'
    )


def _ask_model(prompt, mode="focused"):
    """The model's answer text, or None if it returned nothing usable.
    Raises on a failed or timed-out call."""
    response = _clients[mode].chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
        # Models that can reason first (Gemma 4) do by default, which costs
        # several seconds per answer; a 1-2 sentence summary doesn't need it.
        think=False,
        keep_alive=config.RESPONDER_KEEP_ALIVE_SECONDS,
    )
    answer = json.loads(response["message"]["content"]).get("answer")
    return answer.strip() if isinstance(answer, str) and answer.strip() else None


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DIGITS = re.compile(r"\d{3,}")


def ungrounded_values(answer, *sources):
    """Emails and 3+ digit numbers in `answer` that appear in none of
    `sources` -- a cheap check for a value the model made up. Thousands
    separators are ignored, so "1,499" matches 1499.0."""
    source_text = " ".join(json.dumps(s, ensure_ascii=False, default=str) for s in sources).lower()
    source_digits = source_text.replace(",", "")
    missing = [e for e in _EMAIL.findall(answer) if e.lower().rstrip(".") not in source_text]
    missing += [d for d in _DIGITS.findall(answer.replace(",", "")) if d not in source_digits]
    return missing


# -- the fallback renderer -------------------------------------------------------

def _humanize_key(key):
    # "Lead_Status" / "leadStatus" -> "lead status"
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key))
    return re.sub(r"[_\s]+", " ", key).strip().lower()


def _fields(record, prefix=""):
    """(label, value) pairs for a record, nested dicts flattened into
    "owner name"-style labels. Bare "id" keys are skipped."""
    for key, value in record.items():
        if str(key).lower() == "id":
            continue
        label = f"{prefix} {_humanize_key(key)}".strip()
        if isinstance(value, dict):
            yield from _fields(value, label)
        elif isinstance(value, list):
            scalars = [str(v) for v in value if not isinstance(v, (dict, list))]
            if scalars:
                yield label, ", ".join(scalars)
        else:
            yield label, value


def _records(data):
    # The first list of records in the result (e.g. Zoho's {"data": [...]}),
    # else the result itself as one record.
    for value in data.values():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            return value
    return [data]


def render_fallback(tool, data, mode="focused"):
    """A plain listing of the result, for when the model can't be used."""
    label = _tool_label(tool)
    if set(data) == {"text"}:
        return f"{label}: {data['text']}"
    limits = LIMITS[mode]
    records = _records(data)
    described = []
    for record in records[:limits["max_records"]]:
        # Who/what the record is comes first: its own name fields lead.
        named = sorted(record, key=lambda k: not ("name" in str(k).lower() and not isinstance(record[k], (dict, list))))
        fields = list(_fields({k: record[k] for k in named}))[:limits["max_fields"]]
        described.append(", ".join(f"{name} {value}" for name, value in fields))
    if len(records) == 1:
        return f"{label} found 1 result: {described[0]}."
    listed = " ".join(f"({i}) {d}." for i, d in enumerate(described, 1))
    more = f" Showing the first {len(described)}." if len(records) > len(described) else ""
    return f"{label} found {len(records)} results: {listed}{more}"


# -- public entry points ---------------------------------------------------------

def warm():
    """Load the model into Ollama (a generate call with no prompt), so the
    first live answer doesn't spend its timeout on the load. Raises if
    Ollama or the model isn't available."""
    ollama.Client(timeout=WARM_TIMEOUT_SECONDS).generate(
        model=MODEL_NAME, keep_alive=config.RESPONDER_KEEP_ALIVE_SECONDS,
    )


def compose_answer(frame, mode="focused"):
    """The agent-facing answer for a fired frame's tool_result, or None when
    there's no result to explain (a mock tool). `mode` is one of MODES."""
    result = frame.tool_result
    if result is None:
        return None
    tool = frame.tool_call["tool"]
    label = _tool_label(tool)
    if result["status"] == "no_results":
        searched = ", ".join(f"{_humanize_key(k)} {v}" for k, v in frame.tool_call["arguments"].items())
        return _answer(f"{label}: nothing found" + (f" for {searched}." if searched else "."), "rule", mode)
    if result["status"] == "error":
        return _answer(f"{label} couldn't be completed ({result['message']}).", "rule", mode)

    data = prune(rank_records(result["data"], frame.tool_call["arguments"]), mode) or {}
    try:
        text = _ask_model(_prompt(frame, data, mode), mode)
    except Exception as exc:
        return _answer(
            render_fallback(tool, data, mode), "fallback", mode, f"model call failed: {type(exc).__name__}: {exc}"
        )
    if text is None:
        return _answer(render_fallback(tool, data, mode), "fallback", mode, "model returned no answer")
    missing = ungrounded_values(text, data, frame.tool_call["arguments"], _question(frame))
    if missing:
        return _answer(
            render_fallback(tool, data, mode), "fallback", mode,
            "model answer mentioned values not in the result: " + ", ".join(missing),
        )
    return _answer(text, "llm", mode)
