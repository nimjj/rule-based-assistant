# The intent catalog a turn runs against: which intents a tenant has, the
# phrases that express each one (its exemplars), the slots it collects and,
# for the built-in catalog, the remote tool it calls.
#
# CIP mode (config.CIAP_API_BASE_URL set): every tenant has its own catalog,
# managed in CIP (Settings -> Agent Assist) and read from ciap-api -- this
# service never stores or edits one. It's cached per tenant and reloaded when a
# turn carries a newer catalog_version (ciap-api bumps it on every save), or
# after config.CATALOG_MAX_AGE_SECONDS for turns sent without one.
#
# Otherwise the built-in catalog (intents.py) serves every tenant: local
# development, the demo and the tests.
#
# The pipeline modules read the catalog of the turn in progress through
# current() / intents(); server.py sets it around each turn with using().
# Specs keep the built-in shape (`remote`: the MCP tool the intent calls, on a
# connection of the tenant's own in CIP mode -- see mcp_client):
#   {"label", "description", "exemplars", "required_slots", "optional_slots", "remote"}
# and slots(name) is the slot's {"type", "label", "options", "prompt", "description"}.

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ciap import call_log
from ciap import ciap_api
from ciap import config
from ciap.pipeline.intents import INTENTS as BUILTIN_INTENTS

logger = call_log.logger

# Slot types CIP offers (app.assist_slots.type). Extraction picks by type
# unless the slot's name has its own extractor (slots.SLOT_EXTRACTORS).
SLOT_TYPES = ("TEXT", "NUMBER", "DATE", "TIME", "PHONE", "EMAIL", "CURRENCY", "OPTION")

# The built-in catalog's slots: each has a named extractor; the type is what
# CIP would call it.
_BUILTIN_SLOT_TYPES = {"email": "EMAIL", "phone": "PHONE", "new_date": "DATE"}


class CatalogUnavailable(Exception):
    """The tenant's catalog couldn't be loaded and none is cached."""


@dataclass(eq=False)
class Catalog:
    tenant_id: str | None
    version: object
    intents: dict
    slot_specs: dict = field(default_factory=dict)
    loaded_at: float = 0.0

    def slot(self, name):
        return self.slot_specs.get(name) or {"type": "TEXT", "label": name, "options": []}


def _builtin():
    slot_names = {s for spec in BUILTIN_INTENTS.values() for s in spec["required_slots"]}
    slots = {n: {"type": _BUILTIN_SLOT_TYPES.get(n, "TEXT"), "label": n.replace("_", " "), "options": []} for n in slot_names}
    intents = {name: {**spec, "optional_slots": []} for name, spec in BUILTIN_INTENTS.items()}
    return Catalog(tenant_id=None, version="builtin", intents=intents, slot_specs=slots)


BUILTIN = _builtin()

_current: ContextVar[Catalog | None] = ContextVar("catalog", default=None)


def current():
    """The catalog of the turn in progress (the built-in one outside a turn)."""
    return _current.get() or BUILTIN


def intents():
    return current().intents


def slot(name):
    return current().slot(name)


@contextmanager
def using(catalog):
    token = _current.set(catalog)
    try:
        yield catalog
    finally:
        _current.reset(token)


# -- per-tenant catalogs from ciap-api ------------------------------------------------

_cache: dict[str, Catalog] = {}
_tenant_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()
_clock = time.monotonic  # patched in tests


def from_ciap(tenant_id, body):
    """A Catalog from ciap-api's catalog response."""
    intents, slots = {}, {}
    for i in body.get("intents") or []:
        phrases = [p for p in i.get("phrases") or [] if p and p.strip()]
        if not phrases:
            continue  # nothing to recognise it by
        required, optional = [], []
        for s in i.get("slots") or []:
            (required if s.get("required", True) else optional).append(s["name"])
            slots.setdefault(s["name"], {
                "type": (s.get("type") or "TEXT").upper(),
                "label": s.get("label") or s["name"],
                "options": s.get("options") or [],
                "prompt": s.get("prompt"),
                "description": s.get("description"),
            })
        spec = {"exemplars": phrases, "required_slots": required, "optional_slots": optional}
        if i.get("label"):
            spec["label"] = i["label"]
        if i.get("description"):
            spec["description"] = i["description"]
        if i.get("remote"):
            spec["remote"] = i["remote"]  # the tool it calls on one of the tenant's connections (mcp_client)
        intents[i["name"]] = spec
    return Catalog(tenant_id=tenant_id, version=body.get("version", 0), intents=intents, slot_specs=slots,
                   loaded_at=_clock())


def _fresh(cached, min_version):
    if cached is None:
        return False
    if min_version is not None and cached.version < min_version:
        return False
    return _clock() - cached.loaded_at < config.CATALOG_MAX_AGE_SECONDS


def for_tenant(tenant_id, min_version=None):
    """The catalog for a tenant's turn. CIP mode: its own, from ciap-api, at
    least min_version (the turn's catalog_version). Otherwise the built-in one.
    Raises CatalogUnavailable when ciap-api can't be reached and nothing is
    cached; with a cached copy it logs and carries on with that."""
    if not ciap_api.enabled():
        return BUILTIN
    cached = _cache.get(tenant_id)
    if _fresh(cached, min_version):
        return cached
    with _locks_lock:
        lock = _tenant_locks.setdefault(tenant_id, threading.Lock())
    with lock:  # one reload per tenant at a time; the rest wait for it
        cached = _cache.get(tenant_id)
        if _fresh(cached, min_version):
            return cached
        try:
            loaded = from_ciap(tenant_id, ciap_api.get_catalog(tenant_id))
        except ciap_api.CiapApiError as exc:
            if cached is not None:
                logger.warning("tenant=%s catalog reload failed, using version %s: %s", tenant_id, cached.version, exc)
                return cached
            raise CatalogUnavailable(f"Couldn't load tenant {tenant_id!r}'s catalog: {exc}") from exc
        _cache[tenant_id] = loaded
        logger.info("tenant=%s catalog version %s loaded (%d intents)", tenant_id, loaded.version, len(loaded.intents))
        return loaded


def clear_cache():
    _cache.clear()
