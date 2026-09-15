# Demo pre-flight checklist

Run this **before** the session, not live. Order matters.

## T-30 min — environment

1. `cd api_version`
2. `pip install -r requirements.txt` (only if the machine isn't already set up)
3. Ollama running, escalation model pulled:
   - `ollama serve` (or the desktop app is open)
   - `ollama pull gemma3:1b`
4. `python demo_check.py` — must print **all checks passed**. It:
   - imports every dependency
   - loads the embedding model (downloads from HuggingFace if not cached — this must happen now, not on stage)
   - confirms `gemma3:1b` is pulled and warms it
   - runs intent -> slot -> MCP call end to end

If `demo_check.py` fails on the embedding model, the machine has no cached
weights and no internet — resolve before continuing.

## T-10 min — start the server

```
uvicorn server:app
```

Leave `--reload` **off** for the demo (a stray file save mid-demo would
restart the server and drop every call).

Startup now **loads the embedding model before it accepts requests** — the
terminal prints `embedding index warmed (~8s)` and only then
`Application startup complete`. Wait for that line. The first turn is fast
after it.

Open **http://127.0.0.1:8000/demo** in the browser. Full-screen it.

## T-5 min — warm the escalation LLM

The embedding model is already warm from startup. The escalation LLM
(`gemma3:1b`) is still lazy, so warm it once:

1. Type `Can you tell me where my package is` → send. This escalates to the
   LLM — you're warming the Ollama round-trip so the first *live* escalation
   isn't a cold start.
2. Click **New call** to clear the board.

You're ready.

## Logs — everything is recorded

While the server runs, every call, turn, rejection, and error is written to:

- **`logs/demo.log`** — one human-readable line per event (also echoed to the
  server's console).
- **`logs/demo_turns.jsonl`** — one JSON object per event: the full 3-panel
  view plus the raw pipeline trace (candidates, scores, escalation, frames,
  tool calls).
- **`GET /demo/log`** — recent JSONL records over HTTP; `?call_id=<id>` to
  filter to one call, `?limit=<n>` to cap.

Files are appended across restarts. To start the demo with a clean log,
delete both files first.

## Fallbacks if something breaks live

- **A turn shows a red error banner** — the pipeline caught something; read
  the banner, click New call, move on. No traceback reaches the screen.
- **Escalation hangs (Ollama down)** — the pipeline falls back to the top
  candidate on its own after a moment; the turn still completes. Say "the
  local LLM isn't responding so it fell back to the retriever's pick" and
  continue. Don't wait on it.
- **Server unresponsive** — Ctrl+C, `uvicorn server:app` again, reopen
  /demo. Calls are in memory so they're gone; just restart the scenario.
- **Wrong/odd result on the manager's input** — see "Known sharp edges" in
  DEMO_SCRIPT.md; every common one has a one-line explanation.
