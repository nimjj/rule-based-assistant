# Structured logging for the API. Every call/turn/override/error is written
# two ways, tagged with tenant_id and call_id (plus the agent's user_id when
# the request carried one):
#   logs/api.log     human-readable, one line per event (also to console)
#   logs/turns.jsonl one JSON object per event: the full view + raw trace
# and read_events() serves the jsonl back (GET /v1/log).
#
# Note: turns.jsonl holds raw customer utterances. Treat it as sensitive and
# do not commit it (it's in .gitignore).

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from ciap import config

LOG_DIR = Path(config.ROOT_DIR) / "logs"
LOG_DIR.mkdir(exist_ok=True)
TURN_LOG = LOG_DIR / "turns.jsonl"

logger = logging.getLogger("api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s")
    for _handler in (
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
    ):
        _handler.setFormatter(_fmt)
        logger.addHandler(_handler)
    logger.propagate = False

_write_lock = threading.Lock()


def log_event(event, **fields):
    """Append one structured record to logs/turns.jsonl."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    try:
        with _write_lock, TURN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("failed to append to %s", TURN_LOG.name)


def read_events(limit=200, **filters):
    """Most recent records (newest last). Keyword filters (tenant_id, call_id,
    user_id, ...) keep only records whose field equals the value; None means
    no filter. Returns (records, total_matching)."""
    filters = {k: v for k, v in filters.items() if v is not None}
    if not TURN_LOG.exists():
        return [], 0
    records = []
    with TURN_LOG.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if any(rec.get(k) != v for k, v in filters.items()):
                continue
            records.append(rec)
    return records[-limit:], len(records)
