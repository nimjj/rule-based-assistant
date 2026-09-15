# Online Sales: Conversational Intent Pipeline (HTTP Demo API)

Same pipeline as `stage4/` (embedding retrieval, confidence-banded
classification, deterministic slot extraction, frame-based dialogue state,
Gemma 3 1B LLM escalation) -- wrapped in a FastAPI app so it can be poked
at over HTTP instead of only through `demo.py`'s CLI. Forked from `stage4/`;
the only addition is `server.py`. This is a **testing/demo API**, not a
production service -- see "Known limitations" for what's missing before it
could be one.

## Why an API, and why this shape

The pipeline is stateful *within* one call -- frames persist turn to turn in
a `ConversationStateStore` (see `stage4/README.md`). That rules out a single
stateless endpoint: there's no way to hand the server one utterance and get
back a self-contained answer, because "what does this utterance mean" often
depends on frames opened by earlier turns in the same call (a bare
`"ORD-99213"` is meaningless without knowing there's a pending
`track_order` frame waiting on that slot). So the API has a session concept:
`POST /calls` starts one and returns a `call_id`; every following
`POST /calls/{call_id}/turns` feeds one more utterance into that same
`ConversationStateStore` instance, exactly the way `demo.py call` feeds a
transcript's lines into one store, turn by turn.

## Setup

```
pip install -r requirements.txt
ollama pull gemma3:1b   # if not already pulled for stage4/
```

## Run

```
uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000/docs** -- FastAPI's interactive Swagger
UI, generated from the endpoints below. Every route can be tried directly in
the browser (fill in the request body, click "Execute") without curl or
Postman.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/calls` | -- | `{"call_id": "..."}` -- start a new call |
| `POST` | `/calls/{call_id}/turns` | `{"utterance": "..."}` | The same per-turn trace `demo.py` prints: retrieval candidates, heuristic flags, `turn_type`, any LLM escalation (with the model's reasoning), slots filled, any tool call fired |
| `GET` | `/calls/{call_id}` | -- | Every frame opened in the call so far: `id`, `intent`, `status`, `slots`, `missing_slots`, `tool_call` |
| `DELETE` | `/calls/{call_id}` | -- | Drops the call's state (204, no body) |
| `GET` | `/health` | -- | `{"status": "ok"}` |

A `call_id` that doesn't exist returns `404` on any of the `/calls/{call_id}...` routes.

## Example (curl)

```
curl -s -X POST http://127.0.0.1:8000/calls
# {"call_id":"298cec6767e2"}

curl -s -X POST http://127.0.0.1:8000/calls/298cec6767e2/turns \
  -H "Content-Type: application/json" \
  -d '{"utterance":"Hi, I want to track my order"}'
# turn_type "new_intent_escalated" -- candidates were close enough to
# trigger multi-intent escalation; Gemma decided it's just track_order.

curl -s -X POST http://127.0.0.1:8000/calls/298cec6767e2/turns \
  -H "Content-Type: application/json" \
  -d '{"utterance":"ORD-99213"}'
# turn_type "slot_fill" -- fills order_id on the pending frame, frame
# reaches FIRED, tool_calls includes get_order_status(order_id='ORD-99213')

curl -s http://127.0.0.1:8000/calls/298cec6767e2
# frames: [{"id":1,"intent":"track_order","status":"FIRED", ...}]
```

## Turn types

Every trace (`POST /calls/{call_id}/turns`, and the demo's `view.intent.turn_type`)
carries a `turn_type` saying which branch of `ConversationStateStore.process_utterance()`
handled the turn. The demo UI's badge (`view.intent.decision`) is a coarser label
over the same value -- shown in the right column below.

| `turn_type` | UI badge | What it means | Extra trace fields |
|---|---|---|---|
| `new_intent` | accepted | Top candidate scored above `HIGH_CONFIDENCE` -- a fresh frame opened for it. | `frame`, `intent`, `filled_slots`, `tool_calls` (if it fired immediately) |
| `new_intent_escalated` | escalated | Either (a) two candidates were close enough to trigger the multi-intent check but the LLM decided it's really one intent, or (b) the top score fell in the `MID_CONFIDENCE` band and the LLM picked between the top candidates. | `frame`, `intent`, `filled_slots`, `llm_escalation`, `tool_calls` |
| `multi_intent_blocked` | multi_intent | Two candidates both scored above `HIGH_CONFIDENCE`, close together, and the LLM confirmed the utterance really contains both asks -- blocked outright, no frame opened for either intent. | `intents` (list of names), `message` (ask the customer to resubmit one at a time), `llm_escalation` |
| `slot_fill` | slot_fill | A frame was `AWAITING_INFO` and this turn supplied at least one of its missing slots. | `frame`, `filled_slots`, `tool_calls` (if that completed the frame) |
| `unresolved_continuation` | continuation | A frame was `AWAITING_INFO`, this turn didn't fill anything, and it didn't look like a correction/reference either -- treated as a non-answer, frame stays waiting. | `frame` |
| `ambiguous_continuation` | escalated | A frame was `AWAITING_INFO`, this turn didn't fill anything, but it *did* contain a correction/reference keyword -- escalated to the LLM to decide what it means (result isn't applied automatically; see "Known sharp edges" in `DEMO_SCRIPT.md`). | `llm_escalation` |
| `correction` | correction | Low confidence, but a correction/reference keyword was present and the LLM agreed it refers back to the most recent frame -- that frame flips to `REVERTED`. | `frame`, `llm_escalation` |
| `unknown_intent` | unknown | Nothing matched: low confidence, no pending frame to continue, and no correction/reference keyword (or the LLM decided a keyword hit wasn't actually one). No frame opens. | `llm_escalation` (only on the correction/reference path that came back negative) |
| `duplicate_suppressed` | duplicate | Would otherwise have opened a fresh frame, but a `FIRED` frame for that same intent already exists this call (e.g. an agent's "I'll flag it" landing on an intent whose frame already fired). No new frame, no tool call -- `frame` points at the existing one. | `intent`, `frame` (existing frame's id), `llm_escalation` (if the pre-empted branch would have escalated) |

`filled_slots` is always a list, even if just one slot was newly filled this
turn. `tool_calls` only appears on turn types that can fire a frame this
turn, and only when one actually did (see "MCP tool calls" below).

## Customer and agent turns

Every turn is tagged `speaker` -- `"customer"` (default) or `"agent"`
(`POST /calls/{call_id}/turns`'s `speaker` field; the CLI/browser demos infer
it from a `"Customer: ..."` / `"Agent: ..."` prefix). Both run through the
*exact same* pipeline above -- an agent can ask for something specific or
state that an action is being taken just as validly as a customer can (e.g.
"I'll go ahead and process that refund" scores against `check_refund_status`
the same way a customer saying it would). `speaker` only affects provenance:
it rides along in the trace and becomes `Frame.initiated_by` on whichever
frame the turn opens, so `GET /calls/{call_id}` can show who started each one.

## MCP tool calls

Every `INTENT_TO_TOOL`-mapped intent fires an MCP-shaped tool call
(`mcp_tools.build_tool_call`) the moment its frame's required slots are all
filled -- for the trace/demo UI's benefit, that's `tool_calls` on the turn
that completed the frame, and `frame.tool_call` from then on via
`GET /calls/{call_id}`.

**Almost every tool in `mcp_tools.TOOLS` is a mock** -- `build_tool_call`
constructs the call envelope (tool name + arguments), nothing executes it,
and the demo UI's "CALL SENT TO MCP" panel is just displaying that envelope.

**One exception: `search_zoho_lead`** (intent `lookup_crm_record`, slot
`email`). `mcp_tools.dispatch_tool_call` actually executes this one against
the real Zoho CRM API (`zoho_client.py` -- OAuth refresh-token exchange
against `accounts.zoho.com`, then `Leads/search` on `zohoapis.com`), reading
`ZOHO_CLIENT_ID` / `ZOHO_CLIENT_SECRET` / `ZOHO_REFRESH_TOKEN` from the
environment (`.env`, via `python-dotenv`). The result lands in
`frame.tool_result` and, in the demo UI, renders as a second "RESPONSE FROM
MCP" block under the call. Everything else's `tool_result` is `None`.

This means: if `.env` has real Zoho credentials, typing an email into the
demo actually queries your live CRM -- not a simulated response. There's no
mock fallback for this one tool, so a missing/invalid credential surfaces as
a caught exception (red error banner in the demo, per `demo_web.py`'s
catch-all around `process_utterance`), not a fake result.

## Design notes

- **In-memory session store**: `server.py` keeps `_calls: dict[str,
  ConversationStateStore]` in process memory -- no database, no Redis, no
  TTL/eviction. Fine for a demo hit from one client; restarting the server
  loses every call, and it only works correctly with a single server
  process (a second `uvicorn` worker would have its own empty `_calls`).
- **Trace responses aren't schema-locked**: `POST /calls/{call_id}/turns`
  returns whatever `ConversationStateStore.process_utterance()` returns --
  the exact same dict `demo.py` prints, which varies by `turn_type` (e.g.
  `intents` (list) + `message` on a blocked multi-intent turn vs. `frame`
  (single id) otherwise).
  This is deliberate: the trace shape is stage3/4's contract already, tested
  via `demo.py`, and duplicating it into a strict Pydantic response model
  here would just be a second place for that shape to drift out of sync.
  `GET /calls/{call_id}`'s `FrameOut` model is stricter because frame shape
  doesn't vary the same way.
- **No auth, no rate limiting, no HTTPS**: this binds to `127.0.0.1` by
  design for local testing; nothing here is safe to expose past localhost
  as-is.

## Known limitations

- **In-memory state only** -- calls don't survive a restart, and this can't
  run behind a load balancer with more than one worker without every call's
  turns landing on the same process. A real deployment would move `_calls`
  to Redis (or similar) keyed by `call_id`, same as `stage4/README.md`
  already flags for the underlying pipeline.
- **No cleanup** -- `DELETE /calls/{call_id}` exists but nothing calls it
  automatically; a long-running server accumulates every call ever started
  until it's restarted. No TTL/expiry is implemented.
- **No auth** -- anyone who can reach the port can start calls, read call
  state, or delete any call_id they can guess (`call_id`s are random
  hex, not sequential, but this is still not access control).
- **Everything `stage4/README.md` already flags still applies** -- Gemma
  3 1B's reliability envelope, no cross-call persistence, slot-extraction
  regex limitations -- none of that changed by adding HTTP on top.
