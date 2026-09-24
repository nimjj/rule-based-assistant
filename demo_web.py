# Browser demo page. Serves the static UI in static_demo/ and nothing else --
# the page talks to the /v1 API in server.py (calls, turns, slot overrides,
# transcripts), so there is no demo-only backend logic to keep in sync.
#
# Mounted by server.py: app.include_router(demo_router). Open /demo.

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_STATIC = Path(__file__).parent / "static_demo"
_ASSETS = {"app.js": "text/javascript", "style.css": "text/css"}


@router.get("/demo", include_in_schema=False)
def demo_page():
    return FileResponse(_STATIC / "index.html")


@router.get("/demo/static/{name}", include_in_schema=False)
def demo_asset(name: str):
    media_type = _ASSETS.get(name)
    if media_type is None:  # allowlist only -- no path traversal
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(_STATIC / name, media_type=media_type)
