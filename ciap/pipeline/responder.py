# Stage 5: an agent-facing answer for a live tool's result.
#
# A tool that really executes (see mcp_tools.dispatch_tool_call) hands back
# its server's response as JSON. This turns that into a short plain-English
# briefing for the call agent, answering what was asked on the call -- with
# no per-tool setup, so a newly bound MCP tool gets answers on day one:
#   - no_results / error: a fixed sentence, no model call
#   - success: the LLM writes 1-2 sentences from the data, pruned to the
#     mode's limits toward what the call is about (the records and passages
#     closest to what was said, exact lookup matches first); if the
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
from ciap.pipeline import embeddings

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

# Finding what a long result's call is about (see Relevance): exact lookup
# matches are checked on every item, but only a list's first MAX_SCORED_ITEMS
# are compared by meaning (see config), each on its first ITEM_TEXT_CHARS
# characters (the embedding model reads ~1,000 at most); long strings are
# split into sentences.
MAX_SCORED_ITEMS = config.RESPONDER_MAX_SCORED_ITEMS

# Tokens set aside for the answer when checking a prompt fits the context
# window (see _fits_context): a focused answer is a sentence or two, a full
# one a paragraph per few records.
ANSWER_TOKENS = {"focused": 256, "full": 1024}
ITEM_TEXT_CHARS = 1000
_PASSAGE = re.compile(r"(?<=[.!?])\s+|\n+")

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

class Picked(list):
    """A list prune() chose items from or reordered: `positions` are the kept
    items' 1-based places in the server's list, `total` its length, so the
    prompt can say what was left out and "item 7" still means the 7th.
    `ranked`: chosen by relevance, not just the first ones."""

    def __init__(self, items, positions, total, ranked):
        super().__init__(items)
        self.positions = positions
        self.total = total
        self.ranked = ranked


def prune(value, mode="focused", relevance=None):
    """Drop what doesn't help answer a question: empty values, "$"-prefixed
    server metadata keys (Zoho's $approval, $editable, ...), and anything past
    the mode's list and string limits. With a Relevance, lists keep the items
    closest to what the call is about (most relevant first) and long strings
    the passages closest to it; without one, the first items and characters.
    Returns None if nothing is left."""
    limits = LIMITS[mode]
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if str(key).startswith("$"):
                continue
            item = prune(item, mode, relevance)
            if item is not None:
                kept[key] = item
        return kept or None
    if isinstance(value, list):
        return _prune_list(value, mode, relevance)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        cap = limits["max_string_chars"]
        if len(value) <= cap:
            return value
        return relevance.excerpt(value, cap) if relevance else value[:cap] + "..."
    return value


def _prune_list(items, mode, relevance):
    cap = LIMITS[mode]["max_list_items"]
    # (position, light-pruned item) for every non-empty item
    candidates = [(n, light) for n, light in ((n, prune(v, mode)) for n, v in enumerate(items, 1)) if light is not None]
    if not candidates:
        return None
    records = all(isinstance(light, dict) for _, light in candidates)
    ranked = bool(relevance) and len(candidates) > 1 and (records or len(candidates) > cap)
    if ranked:
        scores = relevance.score_items([light for _, light in candidates])
        best = sorted(range(len(candidates)), key=lambda i: -scores[i])[:cap]
        if not records:
            best.sort()  # plain values (tags, ...) keep the server's order
        chosen = [candidates[i] for i in best]
    else:
        chosen = candidates[:cap]
    kept = [(n, prune(items[n - 1], mode, relevance)) for n, _ in chosen]
    positions = [n for n, _ in kept]
    values = [v for _, v in kept]
    if len(items) == len(values) and positions == list(range(1, len(values) + 1)):
        return values
    return Picked(values, positions, len(items), ranked)


class Relevance:
    """What a result is pruned toward: what was said on the call (by meaning,
    with the intent model's embeddings) and the lookup's arguments (by exact
    value). Uses only values, never field names, so it works for any server's
    result shape."""

    def __init__(self, question, arguments=None):
        self.question = question.strip()
        self.wanted = [str(v).strip().casefold() for v in _scalars(arguments or {}) if str(v).strip()]
        self._query = None

    def _similarity(self, texts):
        if not self.question:
            return [0.0] * len(texts)
        if self._query is None:
            self._query = embeddings.encode([self.question])[0]
        return [float(s) for s in embeddings.encode(texts) @ self._query]

    def score_items(self, items):
        """Per item, how likely it's what the call is about: an exact match on
        a lookup argument outranks any similarity in meaning."""
        texts = [as_text(item)[:ITEM_TEXT_CHARS] if isinstance(item, (dict, list)) else str(item)
                 for item in items[:MAX_SCORED_ITEMS]]
        similarities = self._similarity(texts) + [0.0] * (len(items) - len(texts))
        return [_match_score(item, self.wanted) * 10 + similarity for item, similarity in zip(items, similarities)]

    def excerpt(self, text, cap):
        """Up to `cap` characters of `text`: its sentences closest to the call,
        in their original order, with "..." where text was left out."""
        passages = [p.strip() for p in _PASSAGE.split(text) if p.strip()]
        if len(passages) < 2:
            return text[:cap] + "..."
        scores = self._similarity(passages)
        chosen, size = [], 0
        for i in sorted(range(len(passages)), key=lambda i: -scores[i]):
            if size + len(passages[i]) <= cap:
                chosen.append(i)
                size += len(passages[i]) + 1
        if not chosen:
            best = max(range(len(passages)), key=lambda i: scores[i])
            return passages[best][:cap] + "..."
        chosen.sort()
        parts = ["..."] if chosen[0] > 0 else []
        for previous, i in zip([None] + chosen, chosen):
            if previous is not None and i > previous + 1:
                parts.append("...")
            parts.append(passages[i])
        if chosen[-1] < len(passages) - 1:
            parts.append("...")
        return " ".join(parts)


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
        positions = getattr(value, "positions", range(1, len(value) + 1))
        if isinstance(value, Picked):
            shown = f"showing {len(value)} of {value.total}" if value.total > len(value) else f"all {value.total}"
            order = "most relevant to the call first" if value.ranked else "the first ones"
            lines.append(f"{pad}({shown}, {order}; item numbers are positions in the full list)")
        for n, item in zip(positions, value):
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


def _prompt(frame, data, mode="focused", pruned=True):
    """The answer model's prompt. `pruned`: `data` came from prune() (shown as
    text, capped); otherwise it's the raw result, shown as JSON in full."""
    tool = frame.tool_call["tool"]
    if pruned:
        payload = as_text(data)[:LIMITS[mode]["max_payload_chars"]]
        left_out = (" Parts of a long result may be left out (marked \"...\" or \"showing N of M\"); "
                    "if the answer could be in what was left out, say that, not that it doesn't exist.")
    else:
        payload, left_out = json.dumps(data, ensure_ascii=False, default=str), ""
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
        f"doesn't answer what was asked, say what it does show.{left_out}\n\n"
        f"What was said on the call:\n{_question(frame)}\n\n"
        f"Lookup: {_tool_label(tool)} -- {TOOLS.get(tool, {}).get('description', '')}\n"
        f"Lookup arguments: {json.dumps(frame.tool_call['arguments'], ensure_ascii=False)}\n"
        f"Result:\n{payload}\n\n"
        'Respond with a JSON object: {"answer": "<your briefing for the agent>"}.'
    )


def _context_option():
    # The same num_ctx on every call (and on warm()): Ollama reloads the model
    # whenever it changes.
    return {"num_ctx": config.RESPONDER_NUM_CTX} if config.RESPONDER_NUM_CTX else {}


def _fits_context(prompt, mode):
    """Whether `prompt` plus room for the answer fits the context window, at
    a conservative ~3 characters per token (JSON runs short tokens). True
    when no window is configured (Ollama's default applies)."""
    if not config.RESPONDER_NUM_CTX:
        return True
    return len(prompt) // 3 + ANSWER_TOKENS[mode] <= config.RESPONDER_NUM_CTX


def _ask_model(prompt, mode="focused"):
    """The model's answer text, or None if it returned nothing usable.
    Raises on a failed or timed-out call."""
    response = _clients[mode].chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, **_context_option()},
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
    total = getattr(records, "total", len(records))  # before prune left any out
    if total == 1:
        return f"{label} found 1 result: {described[0]}."
    listed = " ".join(f"({i}) {d}." for i, d in enumerate(described, 1))
    if len(described) == total:
        more = ""
    elif getattr(records, "ranked", False):
        more = f" Showing the {len(described)} most relevant."
    else:
        more = f" Showing the first {len(described)}."
    return f"{label} found {total} results: {listed}{more}"


# -- public entry points ---------------------------------------------------------

def warm():
    """Load the model into Ollama (a generate call with no prompt), so the
    first live answer doesn't spend its timeout on the load. Raises if
    Ollama or the model isn't available."""
    ollama.Client(timeout=WARM_TIMEOUT_SECONDS).generate(
        model=MODEL_NAME, keep_alive=config.RESPONDER_KEEP_ALIVE_SECONDS, options=_context_option(),
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

    if config.RESPONDER_PRUNE_RESULTS:
        relevance = Relevance(" ".join(u["text"] for u in frame.utterances), frame.tool_call["arguments"])
        data = listing = prune(result["data"], mode, relevance) or {}
    else:
        # As the server returned it; the fallback listing still drops the noise.
        data = result["data"] if isinstance(result["data"], dict) else {"result": result["data"]}
        listing = prune(data, mode) or {}
    prompt = _prompt(frame, data, mode, pruned=config.RESPONDER_PRUNE_RESULTS)
    if not _fits_context(prompt, mode):
        return _answer(
            render_fallback(tool, listing, mode), "fallback", mode,
            f"result too large for the model's context ({len(prompt):,} characters; "
            f"RESPONDER_NUM_CTX is {config.RESPONDER_NUM_CTX:,} tokens)",
        )
    try:
        text = _ask_model(prompt, mode)
    except Exception as exc:
        return _answer(
            render_fallback(tool, listing, mode), "fallback", mode, f"model call failed: {type(exc).__name__}: {exc}"
        )
    if text is None:
        return _answer(render_fallback(tool, listing, mode), "fallback", mode, "model returned no answer")
    missing = ungrounded_values(text, data, frame.tool_call["arguments"], _question(frame))
    if missing:
        return _answer(
            render_fallback(tool, listing, mode), "fallback", mode,
            "model answer mentioned values not in the result: " + ", ".join(missing),
        )
    return _answer(text, "llm", mode)
