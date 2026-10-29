# Admin edits to the intent catalog (the /v1/intents write endpoints in
# server.py).
#
# intents.py stays the shipped defaults. Once anything is edited, the whole
# catalog is written to config.CATALOG_PATH and loaded over those defaults at
# startup, so edits survive a restart.
#
# INTENTS is updated in place because every other module imported it by
# reference. Specs are always replaced whole, never mutated, so a reader
# mid-turn sees either the old spec or the new one.
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
from ciap.integrations import mcp_client, zoho
from ciap.integrations.mcp_tools import placeholders
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
    saved = data["intents"]
    if "tools" in data:
        saved = _merge_saved_tools(saved, data["tools"])
    with lock:
        INTENTS.clear()
        for name, spec in saved.items():
            INTENTS[name] = _intent_spec(
                spec.get("label"), spec.get("description"), spec["exemplars"], spec["required_slots"], spec.get("remote")
            )
    embeddings.invalidate_index()
    return True


def _merge_saved_tools(intents, tools):
    """Catalogs saved before intents carried their own tool kept tools in a
    separate map, each intent naming the one it fired: the intent takes over
    that tool's description and binding."""
    # Older still, the live Zoho tool was marked "backend": "zoho_lead_search",
    # or (before that) only by its name. Either way it now runs through the
    # Zoho MCP connection.
    if not any(t.get("remote") for t in tools.values()):
        legacy = [n for n, t in tools.items() if t.get("backend") == "zoho_lead_search"]
        if not legacy and "search_zoho_lead" in tools:
            legacy = ["search_zoho_lead"]
        tools = {**tools, **{n: {**tools[n], "remote": copy.deepcopy(zoho.LEAD_SEARCH)} for n in legacy}}
    merged = {}
    for name, spec in intents.items():
        tool = tools.get(spec.get("tool")) or {}
        merged[name] = {**spec, "description": tool.get("description"), "remote": tool.get("remote")}
    return merged


# Labels are optional: an entry without one (including catalogs saved before
# labels existed) is shown under its humanized name -- see views.intent_label.
# An intent without a description is briefed to the model by its label alone,
# and one without a remote binding is a mock.
def _intent_spec(label, description, exemplars, required_slots, remote=None):
    spec = {"exemplars": list(exemplars), "required_slots": list(required_slots)}
    if label:
        spec["label"] = label
    if description:
        spec["description"] = description
    if remote:
        spec["remote"] = remote
    return spec


def _save():
    path = Path(config.CATALOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"intents": INTENTS}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# -- validation ---------------------------------------------------------------


def _check_slots(required_slots):
    unknown = [s for s in required_slots if s not in SLOT_EXTRACTORS]
    if unknown:
        raise CatalogError(
            422, "unknown_slot",
            f"No extractor for slot(s) {unknown}; available: {sorted(SLOT_EXTRACTORS)}.",
        )


_CONNECTION_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _check_remote(remote, required_slots):
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
    # The call's arguments are the intent's required slots, so a placeholder
    # for any other slot would always render empty.
    unknown = sorted(placeholders(arguments) - set(required_slots))
    if unknown:
        raise CatalogError(
            422, "invalid_remote",
            f"remote.arguments uses {unknown}, which this intent doesn't collect "
            f"(its required_slots are {list(required_slots)}).",
        )
    fallback = remote.get("fallback")
    if fallback is not None:
        if not isinstance(fallback, dict):
            raise CatalogError(422, "invalid_remote", "remote.fallback must be a binding object.")
        _check_remote(fallback, required_slots)


def _changed():
    _save()
    embeddings.invalidate_index()


# -- intents ------------------------------------------------------------------


KEEP = object()


def put_intent(name, label, description, exemplars, required_slots, create, remote=KEEP):
    """Create (create=True) or replace an intent. `remote` is its MCP binding
    (None makes it a mock); an update that leaves it as KEEP keeps the
    current one."""
    with lock:
        exists = name in INTENTS
        if create and exists:
            raise CatalogError(409, "intent_exists", f"Intent {name!r} already exists.")
        if not create and not exists:
            raise CatalogError(404, "intent_not_found", f"No intent named {name!r}.")
        _check_slots(required_slots)
        if remote is KEEP:
            remote = INTENTS[name].get("remote") if exists else None
        if remote is not None:
            _check_remote(remote, required_slots)
        INTENTS[name] = _intent_spec(label, description, exemplars, required_slots, remote)
        _changed()


def rename_intent(old, new):
    """Move an intent to a new name. The caller moves live frames and their
    tool calls over (server.py owns the calls)."""
    with lock:
        if old not in INTENTS:
            raise CatalogError(404, "intent_not_found", f"No intent named {old!r}.")
        if new in INTENTS:
            raise CatalogError(409, "intent_exists", f"Intent {new!r} already exists.")
        INTENTS[new] = INTENTS[old]
        del INTENTS[old]
        _changed()


def delete_intent(name):
    with lock:
        if name not in INTENTS:
            raise CatalogError(404, "intent_not_found", f"No intent named {name!r}.")
        del INTENTS[name]
        _changed()
