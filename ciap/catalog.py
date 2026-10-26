# Admin edits to the intent and tool catalogs (the /v1/intents and /v1/tools
# write endpoints in server.py).
#
# intents.py and mcp_tools.py stay the shipped defaults. Once anything is
# edited, the whole catalog is written to config.CATALOG_PATH and loaded over
# those defaults at startup, so edits survive a restart.
#
# INTENTS, TOOLS and INTENT_TO_TOOL are updated in place because every other
# module imported them by reference. Specs are always replaced whole, never
# mutated, so a reader mid-turn sees either the old spec or the new one.
#
# Validation that needs the HTTP layer's context (live calls, request shape)
# lives in server.py; this module checks the catalog's own consistency and
# raises CatalogError for anything that would leave it broken.

import copy
import json
import os
import re
import threading
from pathlib import Path

from ciap import config
from ciap.pipeline import embeddings
from ciap.pipeline.intents import INTENTS
from ciap.integrations import mcp_client
from ciap.integrations.mcp_tools import INTENT_TO_TOOL, TOOLS, ZOHO_LEAD_SEARCH_REMOTE, placeholders
from ciap.pipeline.slots import SLOT_EXTRACTORS

# Serialises edits and the file write. Turns don't take it.
lock = threading.RLock()


class CatalogError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# -- persistence --------------------------------------------------------------


def load():
    """Replace the shipped defaults with the saved catalog, if one exists."""
    path = Path(config.CATALOG_PATH)
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    with lock:
        INTENTS.clear()
        INTENT_TO_TOOL.clear()
        TOOLS.clear()
        TOOLS.update(data["tools"])
        # Older catalogs marked the live Zoho tool with "backend":
        # "zoho_lead_search", or (before that) only by its name. Either way it
        # now runs through the Zoho MCP connection.
        if not any(t.get("remote") for t in TOOLS.values()):
            legacy = [n for n, t in TOOLS.items() if t.get("backend") == "zoho_lead_search"]
            if not legacy and "search_zoho_lead" in TOOLS:
                legacy = ["search_zoho_lead"]
            for name in legacy:
                TOOLS[name] = {**TOOLS[name], "remote": copy.deepcopy(ZOHO_LEAD_SEARCH_REMOTE)}
        for name, spec in list(TOOLS.items()):
            if "backend" in spec:
                TOOLS[name] = {k: v for k, v in spec.items() if k != "backend"}
        for name, spec in data["intents"].items():
            INTENTS[name] = _intent_spec(spec.get("label"), spec["exemplars"], spec["required_slots"])
            if spec.get("tool"):
                INTENT_TO_TOOL[name] = spec["tool"]
    embeddings.invalidate_index()
    return True


# Labels are optional: an entry without one (including catalogs saved before
# labels existed) is shown under its humanized name -- see views.intent_label.
def _intent_spec(label, exemplars, required_slots):
    spec = {"exemplars": list(exemplars), "required_slots": list(required_slots)}
    if label:
        spec["label"] = label
    return spec


def _tool_spec(label, description, input_schema, remote=None):
    spec = {"description": description, "inputSchema": input_schema}
    if label:
        spec["label"] = label
    if remote:
        spec["remote"] = remote
    return spec


def _save():
    data = {
        "intents": {
            name: {**spec, "tool": INTENT_TO_TOOL.get(name)} for name, spec in INTENTS.items()
        },
        "tools": TOOLS,
    }
    path = Path(config.CATALOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# -- validation ---------------------------------------------------------------


def _check_intent(name, required_slots, tool):
    unknown = [s for s in required_slots if s not in SLOT_EXTRACTORS]
    if unknown:
        raise CatalogError(
            422, "unknown_slot",
            f"No extractor for slot(s) {unknown}; available: {sorted(SLOT_EXTRACTORS)}.",
        )
    if tool is None:
        return
    if tool not in TOOLS:
        raise CatalogError(422, "tool_not_found", f"No tool named {tool!r}.")
    _check_tool_covered(tool, TOOLS[tool]["inputSchema"], name, required_slots)


def _check_tool_covered(tool, schema, intent, required_slots):
    # build_tool_call fills arguments from slots by property name, so a tool
    # argument the intent never collects would always be sent missing.
    uncovered = [p for p in schema.get("required", []) if p not in required_slots]
    if uncovered:
        raise CatalogError(
            422, "tool_slot_mismatch",
            f"Tool {tool!r} requires {uncovered}, which intent {intent!r} doesn't collect "
            f"(its required_slots are {required_slots}).",
        )


_CONNECTION_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _check_remote(remote, schema):
    connection, remote_tool = remote.get("connection"), remote.get("tool")
    if not isinstance(connection, str) or not _CONNECTION_NAME.match(connection):
        raise CatalogError(422, "invalid_remote", "remote.connection must be a connection name like \"zoho\".")
    if not isinstance(remote_tool, str) or not remote_tool.strip():
        raise CatalogError(422, "invalid_remote", "remote.tool must be the tool's name on the MCP server.")
    allowed = mcp_client.allowed_tools(connection)
    if allowed is not None and remote_tool not in allowed:
        raise CatalogError(
            422, "invalid_remote",
            f"remote.tool {remote_tool!r} isn't in connection {connection!r}'s allowed_tools ({allowed}).",
        )
    arguments = remote.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        raise CatalogError(422, "invalid_remote", "remote.arguments must be an object.")
    unknown = sorted(placeholders(arguments) - set(schema["properties"]))
    if unknown:
        raise CatalogError(
            422, "invalid_remote",
            f"remote.arguments uses {unknown}, which aren't arguments of this tool "
            f"(its input_schema has {sorted(schema['properties'])}).",
        )
    fallback = remote.get("fallback")
    if fallback is not None:
        if not isinstance(fallback, dict):
            raise CatalogError(422, "invalid_remote", "remote.fallback must be a binding object.")
        _check_remote(fallback, schema)


def _check_schema(schema):
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise CatalogError(422, "invalid_input_schema", 'input_schema needs "type": "object" and a "properties" object.')
    required = schema.get("required", [])
    if not isinstance(required, list) or any(r not in schema["properties"] for r in required):
        raise CatalogError(422, "invalid_input_schema", '"required" must list names from "properties".')


def _changed():
    _save()
    embeddings.invalidate_index()


# -- intents ------------------------------------------------------------------


def put_intent(name, label, exemplars, required_slots, tool, create):
    """Create (create=True) or replace an intent."""
    with lock:
        exists = name in INTENTS
        if create and exists:
            raise CatalogError(409, "intent_exists", f"Intent {name!r} already exists.")
        if not create and not exists:
            raise CatalogError(404, "intent_not_found", f"No intent named {name!r}.")
        _check_intent(name, required_slots, tool)
        INTENTS[name] = _intent_spec(label, exemplars, required_slots)
        if tool is None:
            INTENT_TO_TOOL.pop(name, None)
        else:
            INTENT_TO_TOOL[name] = tool
        _changed()


def rename_intent(old, new):
    """Move an intent, and its tool mapping, to a new name. The caller moves
    live frames over (server.py owns the calls)."""
    with lock:
        if old not in INTENTS:
            raise CatalogError(404, "intent_not_found", f"No intent named {old!r}.")
        if new in INTENTS:
            raise CatalogError(409, "intent_exists", f"Intent {new!r} already exists.")
        INTENTS[new] = INTENTS[old]
        del INTENTS[old]
        if old in INTENT_TO_TOOL:
            INTENT_TO_TOOL[new] = INTENT_TO_TOOL.pop(old)
        _changed()


def delete_intent(name):
    with lock:
        if name not in INTENTS:
            raise CatalogError(404, "intent_not_found", f"No intent named {name!r}.")
        del INTENTS[name]
        INTENT_TO_TOOL.pop(name, None)
        _changed()


# -- tools --------------------------------------------------------------------


KEEP = object()


def put_tool(name, label, description, input_schema, create, remote=KEEP):
    """Create (create=True) or replace a tool. `remote` is its MCP binding
    (None makes it a mock); an update that leaves it as KEEP keeps the
    current one."""
    with lock:
        exists = name in TOOLS
        if create and exists:
            raise CatalogError(409, "tool_exists", f"Tool {name!r} already exists.")
        if not create and not exists:
            raise CatalogError(404, "tool_not_found", f"No tool named {name!r}.")
        _check_schema(input_schema)
        for intent, mapped in INTENT_TO_TOOL.items():
            if mapped == name:
                _check_tool_covered(name, input_schema, intent, INTENTS[intent]["required_slots"])
        if remote is KEEP:
            remote = TOOLS[name].get("remote") if exists else None
        if remote is not None:
            _check_remote(remote, input_schema)
        TOOLS[name] = _tool_spec(label, description, input_schema, remote)
        _save()


def rename_tool(old, new):
    """Move a tool to a new name and repoint every intent mapped to it. The
    caller updates live frames' tool calls."""
    with lock:
        if old not in TOOLS:
            raise CatalogError(404, "tool_not_found", f"No tool named {old!r}.")
        if new in TOOLS:
            raise CatalogError(409, "tool_exists", f"Tool {new!r} already exists.")
        TOOLS[new] = TOOLS[old]
        for intent, mapped in list(INTENT_TO_TOOL.items()):
            if mapped == old:
                INTENT_TO_TOOL[intent] = new
        del TOOLS[old]
        _save()


def delete_tool(name):
    with lock:
        if name not in TOOLS:
            raise CatalogError(404, "tool_not_found", f"No tool named {name!r}.")
        users = sorted(i for i, t in INTENT_TO_TOOL.items() if t == name)
        if users:
            raise CatalogError(
                409, "tool_in_use", f"Tool {name!r} is still mapped from intent(s) {users}; remap them first."
            )
        del TOOLS[name]
        _save()
