# Stage 5: an agent-facing answer for a live tool's result.
#
# A tool that really executes (see mcp_tools.dispatch_tool_call) hands back
# its server's response as JSON. This turns that into a short plain-English
# briefing for the call agent, answering what was asked on the call -- with
# no per-tool setup, so a newly bound MCP tool gets answers on day one:
#   - no_results / error: a fixed sentence, no model call
#   - success: the LLM writes 1-2 sentences from the (pruned) data; if the
#     call fails, times out, or the answer mentions an email or number that
#     isn't in the data or the question, a generic renderer lists the
#     record's fields instead.
# Every answer is {"text", "source", "model", "note"}; `source` says which
# path produced it ("rule", "llm" or "fallback") and `note` why a fallback ran.

import json
import re

import ollama

from ciap import config
from ciap.integrations.mcp_tools import TOOLS

MODEL_NAME = config.RESPONDER_MODEL_NAME

_client = ollama.Client(timeout=config.RESPONDER_TIMEOUT_SECONDS)

# Loading the model from disk can take far longer than an answer may.
WARM_TIMEOUT_SECONDS = 120

# What the model sees of a result: lists cut to this many items, strings to
# this many characters, and the whole payload to this many characters.
MAX_LIST_ITEMS = 5
MAX_STRING_CHARS = 300
MAX_PAYLOAD_CHARS = 4000

# What the fallback renderer lists: this many records, this many fields each.
FALLBACK_MAX_RECORDS = 3
FALLBACK_MAX_FIELDS = 8


def _tool_label(name):
    return TOOLS.get(name, {}).get("label") or name.replace("_", " ").title()


def _answer(text, source, note=None):
    return {"text": text, "source": source, "model": MODEL_NAME if source == "llm" else None, "note": note}


# -- preparing the data -------------------------------------------------------

def prune(value):
    """Drop what doesn't help answer a question: empty values, "$"-prefixed
    server metadata keys (Zoho's $approval, $editable, ...), and anything past
    MAX_LIST_ITEMS / MAX_STRING_CHARS. Returns None if nothing is left."""
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if str(key).startswith("$"):
                continue
            item = prune(item)
            if item is not None:
                kept[key] = item
        return kept or None
    if isinstance(value, list):
        kept = [item for item in (prune(v) for v in value[:MAX_LIST_ITEMS]) if item is not None]
        return kept or None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value if len(value) <= MAX_STRING_CHARS else value[:MAX_STRING_CHARS] + "..."
    return value


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


def _prompt(frame, data):
    tool = frame.tool_call["tool"]
    payload = as_text(data)[:MAX_PAYLOAD_CHARS]
    return (
        "You are helping a call-center agent during a live call. Because of what "
        "was said on the call, a lookup was run in a backend system. Brief the "
        "agent on the result in one or two plain sentences, answering what was "
        "asked.\n\n"
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


def _ask_model(prompt):
    """The model's answer text, or None if it returned nothing usable.
    Raises on a failed or timed-out call."""
    response = _client.chat(
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


def render_fallback(tool, data):
    """A plain listing of the result, for when the model can't be used."""
    label = _tool_label(tool)
    if set(data) == {"text"}:
        return f"{label}: {data['text']}"
    records = _records(data)
    described = []
    for record in records[:FALLBACK_MAX_RECORDS]:
        # Who/what the record is comes first: its own name fields lead.
        named = sorted(record, key=lambda k: not ("name" in str(k).lower() and not isinstance(record[k], (dict, list))))
        fields = list(_fields({k: record[k] for k in named}))[:FALLBACK_MAX_FIELDS]
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


def compose_answer(frame):
    """The agent-facing answer for a fired frame's tool_result, or None when
    there's no result to explain (a mock tool)."""
    result = frame.tool_result
    if result is None:
        return None
    tool = frame.tool_call["tool"]
    label = _tool_label(tool)
    if result["status"] == "no_results":
        searched = ", ".join(f"{_humanize_key(k)} {v}" for k, v in frame.tool_call["arguments"].items())
        return _answer(f"{label}: nothing found" + (f" for {searched}." if searched else "."), "rule")
    if result["status"] == "error":
        return _answer(f"{label} couldn't be completed ({result['message']}).", "rule")

    data = prune(result["data"]) or {}
    try:
        text = _ask_model(_prompt(frame, data))
    except Exception as exc:
        return _answer(render_fallback(tool, data), "fallback", f"model call failed: {type(exc).__name__}: {exc}")
    if text is None:
        return _answer(render_fallback(tool, data), "fallback", "model returned no answer")
    missing = ungrounded_values(text, data, frame.tool_call["arguments"], _question(frame))
    if missing:
        return _answer(
            render_fallback(tool, data), "fallback",
            "model answer mentioned values not in the result: " + ", ".join(missing),
        )
    return _answer(text, "llm")
