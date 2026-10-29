# Step 6: turning a fired intent frame into an MCP tool call, and executing
# it. Each intent in intents.py is its own tool: the call is named after the
# intent and passes its required slots as arguments, the way an MCP-aware LLM
# client would hand off a tool call, so no call site re-derives "which
# action, which arguments" by hand. Most intents are mocks; one with a
# `remote` binding really executes, as a call to a tool on a remote MCP
# server (see the binding notes below, and mcp_client.py).

import re

from ciap.integrations import mcp_client
from ciap.pipeline.intents import INTENTS


def build_tool_call(intent, slots):
    """Resolve a fired frame's intent + extracted slots to a tool call: the
    intent's name, and its required slots as arguments. Returns None if the
    intent is no longer in the catalog (deleted mid-call).
    """
    spec = INTENTS.get(intent)
    if spec is None:
        return None
    arguments = {}
    for name in spec["required_slots"]:
        value = slots.get(name)
        if value is None:
            continue
        # new_date is stored as {"text": ..., "date": ...}; the tool call
        # only wants the resolved ISO date.
        arguments[name] = value["date"] if isinstance(value, dict) and "date" in value else value
    return {"tool": intent, "arguments": arguments}


# An intent's `remote` binding says how its tool call executes:
#   {"connection": "zoho",              -- a name from config.MCP_CONNECTIONS_PATH
#    "tool": "ZohoCRM_searchRecords",   -- the tool's name on that MCP server
#    "arguments": {"path_variables": {"module": "Leads"},   -- what to send it; optional
#                  "query_params": {"email": "{email}"}},
#    "fallback": {...another binding...}}  -- tried if this one finds nothing; optional
# `{name}` in an argument string is replaced by the call's argument of that
# name (i.e. the slot); an argument that is exactly "{name}" passes the value
# through as-is; anything else is sent unchanged. Without "arguments" the
# call's own arguments are sent as they are. Intents without a binding are mocks.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def placeholders(template):
    """Every {name} used anywhere in a binding's arguments."""
    if isinstance(template, str):
        return set(_PLACEHOLDER.findall(template))
    if isinstance(template, dict):
        return set().union(*(placeholders(v) for v in template.values())) if template else set()
    if isinstance(template, list):
        return set().union(*(placeholders(v) for v in template)) if template else set()
    return set()


def render_arguments(template, arguments):
    if isinstance(template, str):
        whole = _PLACEHOLDER.fullmatch(template)
        if whole:
            return arguments.get(whole.group(1))
        return _PLACEHOLDER.sub(lambda m: str(arguments.get(m.group(1), "")), template)
    if isinstance(template, dict):
        return {k: render_arguments(v, arguments) for k, v in template.items()}
    if isinstance(template, list):
        return [render_arguments(v, arguments) for v in template]
    return template


def remote_calls(tool_call):
    """The (connection, remote tool, arguments) a tool call executes as --
    its binding, then each fallback in turn -- or [] for a mock tool."""
    calls = []
    remote = INTENTS.get(tool_call["tool"], {}).get("remote")
    while remote is not None:
        template = remote.get("arguments")
        arguments = tool_call["arguments"] if template is None else render_arguments(template, tool_call["arguments"])
        calls.append((remote["connection"], remote["tool"], arguments))
        remote = remote.get("fallback")
    return calls


def executes_live(tool_name):
    return INTENTS.get(tool_name, {}).get("remote") is not None


NO_RESULTS_MESSAGE = "The tool ran but found no results."


def _is_empty(value):
    # Nothing came back: no content at all, or only empty containers, e.g. {},
    # {"data": []}, {"text": ""}. A result with any real value isn't empty.
    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return all(_is_empty(v) for v in value.values())
    if isinstance(value, list):
        return all(_is_empty(v) for v in value)
    return False


def tool_result(status, data=None, message=None):
    """The fixed shape of every executed tool's frame.tool_result:
      {"status": "success",    "data": {...server response...}, "message": None}
      {"status": "no_results", "data": None, "message": NO_RESULTS_MESSAGE}
      {"status": "error",      "data": None, "message": "<ExceptionType>: <why>"}"""
    return {"status": status, "data": data, "message": message}


def dispatch_tool_call(tool_call):
    """Execute a fired frame's tool call on its remote MCP tool -- trying each
    fallback in turn while nothing is found -- and return its tool_result (see
    tool_result above). Returns None for mock tools (no remote binding), which
    the demo just displays.
    """
    if tool_call is None:
        return None
    calls = remote_calls(tool_call)
    if not calls:
        return None
    # The binding, then its fallbacks while each finds nothing.
    for connection, remote_tool, arguments in calls:
        try:
            result = mcp_client.call_tool(connection, remote_tool, arguments)
        except Exception as exc:
            # An unconfigured connection, a network/auth failure, a timeout or an
            # error from the tool shouldn't take down the whole conversation --
            # surface it as a tool error instead.
            return tool_result("error", message=f"{type(exc).__name__}: {exc}")
        if not _is_empty(result):
            return tool_result("success", data=result)
    return tool_result("no_results", message=NO_RESULTS_MESSAGE)
