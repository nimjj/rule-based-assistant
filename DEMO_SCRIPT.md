# Demo script

Goal: show, on arbitrary typed input, (1) intent identification, (2) slot
extraction across multiple turns, (3) a triggered MCP tool call.

The page has three panels, numbered to match those three things. Each
customer turn you type updates all three. Click any past turn in the
transcript to bring its panels back.

**Click "New call" between scenarios** — it clears all frames. Several
behaviors below depend on starting with no pending frame.

All utterances here are rehearsed and verified. Type them exactly for the
scripted spine; improvise afterward when you hand the keyboard over.

---

## Scenario A — intent identification + immediate MCP call (30s)

Purpose: the simplest possible end-to-end, one turn.

| Type | What to point at |
|---|---|
| `What's your return policy?` | Panel 1: `return_policy_info`, **accepted**, top score ~0.98, runner-ups far below. Panel 2: "no slots required → fires immediately". Panel 3: **CALL SENT TO MCP** — `get_return_policy`, empty arguments. |

Line: "One turn: it identified the intent, saw the backend action needs no
data, and emitted the MCP call."

---

## Scenario B — intent + slot extracted from the same turn (30s)

New call.

| Type | What to point at |
|---|---|
| `Is the Aurora Desk Lamp in stock?` | Panel 1: `check_product_availability`, accepted. Panel 2: slot `product_name` = **Aurora Desk Lamp**, "filled this turn", frame FIRED. Panel 3: MCP call `get_product_availability(product_name="Aurora Desk Lamp")`. |

Line: "Here the required data was already in the sentence — a product name —
so it pulled it out and fired immediately."

---

## Scenario C — slot filled across multiple turns (the main one, 90s)

New call. This is the capability the manager specifically named.

| Turn | Type | What to point at |
|---|---|---|
| 1 | `Where is my order?` | Panel 1: `track_order`, accepted, ~0.97. Panel 2: frame #1 **AWAITING_INFO**, `order_id` = *waiting…*. Panel 3: no call — "fires only once every required slot is filled." |
| 2 | `let me find it, one sec` | Panel 1: turn_type `unresolved_continuation` — not a new intent, not an answer. Panel 2: **unchanged** — frame #1 still waiting. This shows state is held across turns. |
| 3 | `ORD-48213` | Panel 1: low scores, turn_type `slot_fill`. Panel 2: `order_id` = **ORD-48213**, "filled this turn", frame #1 → **FIRED**. Panel 3: **CALL SENT TO MCP** — `get_order_status(order_id="ORD-48213")`. |

Line: "The order number arrived two turns later, on its own, with no intent
words around it. It still landed in the frame that was waiting for it, and
that completed the frame and triggered the call."

---

## Scenario D — two slots, filled on separate turns (90s)

New call. Shows a frame that needs more than one piece of data.

| Turn | Type | What to point at |
|---|---|---|
| 1 | `Can I change my delivery date?` | `change_delivery_date`, AWAITING_INFO, **two** missing slots: `order_id`, `new_date`. |
| 2 | `ORD-77120` | `order_id` filled. Frame **still AWAITING_INFO** — `new_date` still *waiting…*. Point at the frame being half-complete. |
| 3 | `move it to next Friday` | `new_date` filled — parsed to an ISO date (`2026-09-04...`). Frame → FIRED. Panel 3: `reschedule_delivery(order_id="ORD-77120", new_date="2026-09-04T00:00:00")` — **both** arguments. |

Line: "Two separate facts, two separate turns, natural language for the date.
The frame tracks what's still outstanding and only fires when nothing is."

---

## Scenario E — "unknown intent" is a feature (20s)

**New call** (must have no pending frame for this to read cleanly).

| Type | What to point at |
|---|---|
| `do you sell car insurance` | Panel 1: **unknown**, no intent selected, all candidate scores below the accept threshold. Panel 2: "no frame opened." Panel 3: nothing. |

Line: "It's not guessing. Below the confidence threshold it declines rather
than routing to the closest wrong thing."

---

## Optional extras (if there's time / interest)

### Interruption — two frames at once
New call.
1. `Where is my order?` → frame #1 waiting on `order_id`.
2. `actually first, is the Marlow Sofa in stock?` → frame #2 opens, fires
   `get_product_availability` — **and frame #1 is still there, still
   waiting.** Panel 2 shows both frames.
3. `ORD-30012` → fills frame #1, fires `get_order_status`.

### Correction
New call.
1. `Where is my order?`
2. `ORD-30012` → fires.
3. `actually never mind that` → frame flips to **REVERTED** (dimmed in
   panel 2). turn_type `correction`.

### Escalation to the local LLM
New call.
1. `Can you tell me where my package is` → two intents score almost equal;
   turn_type `new_intent_escalated`, panel 1 shows the **escalated to LLM**
   block with `gemma3:1b`'s one-line reasoning. (Brief pause while the model
   answers — narrate it.)

---

## Handing the keyboard over

Invite the manager to type. Watch panel 1's decision badge and narrate.
Likely outcomes:

| He types something like | You'll see | Say |
|---|---|---|
| A clear request ("I want to return this", "my card was declined") | **accepted** + intent | "identified, threshold cleared" |
| A request + the data ("cancel order ORD-1234") | accepted + slot filled + MCP call | "intent and data in one turn" |
| A vague or borderline line | **escalated** — LLM picks | "retriever wasn't sure, local LLM broke the tie" |
| Off-topic ("book me a flight") **on a fresh call** | **unknown** | "declined, below threshold" |
| A bare number / ID with a frame waiting | `slot_fill` into that frame | "no intent words, still routed to the open frame" |
| Two asks in one line ("track my order and cancel another") | `multi_intent_blocked` — no frame opens | "blocked, asked to send one at a time" |

If a result looks off, check **Known sharp edges** below — say the one-liner
and move on. Don't debug live.

---

## Slot extractor cheat-sheet (so you can explain any extraction)

- **order_id** — `ORD-` + digits, *or* any bare run of 4+ digits. So `5551234`
  becomes an order_id too — that's the deterministic rule, an LLM pass would
  disambiguate.
- **product_name** — a Title-Case run after "the/my/this/that"
  ("the Aurora Desk Lamp"). No cue word, or lowercase, and the regex won't
  catch it — by design; that's the LLM's job in production.
- **discount_code** — ALL-CAPS token 4–12 chars mixing letters+digits
  (`SAVE20`). Pure letters (`FREESHIP`) only counts if "code/coupon/promo/
  discount" is also in the turn.
- **new_date** — `dateparser` on free text, biased to future dates
  ("next Friday", "the 15th", "September 12th").

---

## Known sharp edges (say the line, move on)

- **"order ORD-1234" while a *different* intent's frame is pending** may open
  a fresh `track_order` frame instead of filling the pending one — the word
  "order" + an ID scores just over the accept threshold for `track_order`.
  Line: "It read 'order' plus a number as a new tracking request; a stricter
  mid-band, which is a config threshold, folds that back into a slot fill."
  *In the scripted scenarios we feed bare IDs, which never hit this.*
- **Off-topic input while a frame is still waiting** shows as
  `unresolved_continuation`, not `unknown` — the pipeline assumes it might be
  a reply to the pending question. Line: "With an open frame it treats an
  unrecognized line as a non-answer to the open question, not a new topic."
  *Run Scenario E on a fresh call.*
- **A capitalized non-product phrase after "the/my"** ("the Best Buy price")
  can be grabbed as a product_name. Line: "Title-case heuristic; the LLM
  layer is where that ambiguity gets resolved in production."
- **First *escalation* after a server start** has a longer pause while
  `gemma3:1b` cold-starts. The embedding model is warmed at startup; the LLM
  is not. The T-5 warm-up step covers it — don't skip it.
