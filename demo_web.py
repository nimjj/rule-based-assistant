# Browser demo page. Serves the static UI in static_demo/ and nothing else --
# the page talks to the /v1 API in server.py (calls, turns, transcripts, and
# slot overrides when enabled), so there is no demo-only backend logic to keep
# in sync. /demo/config only tells the page which optional features are on.
#
# Mounted by server.py: app.include_router(demo_router). Open /demo.
# Off unless config.DEMO_UI_ENABLED: every route here answers 404 otherwise.

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

import config


def _require_enabled():
    if not config.DEMO_UI_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")


router = APIRouter(dependencies=[Depends(_require_enabled)])

_STATIC = Path(__file__).parent / "static_demo"
_ASSETS = {"app.js": "text/javascript", "style.css": "text/css"}


@router.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse(_STATIC / "index.html")


@router.get("/demo/config", include_in_schema=False)
def demo_config():
    return {"slot_override_enabled": config.SLOT_OVERRIDE_ENABLED}


@router.get("/demo/static/{name}", include_in_schema=False)
def demo_asset(name: str):
    media_type = _ASSETS.get(name)
    if media_type is None:  # allowlist only -- no path traversal
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(_STATIC / name, media_type=media_type)
