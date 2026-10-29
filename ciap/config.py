# Stage 3 tunable knobs: the full config, now that the mid-confidence band
# and correction/reference heuristics actually have somewhere to plug in
# (the state store's turn-handling logic in state_store.py).

import os

# The repository root: runtime files (data/, logs/, .env, connections.json)
# live here, outside the ciap package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sentence embedding model used for intent retrieval (embeddings.py). Small
# and CPU-friendly; swap this to retune retrieval quality/speed without
# touching embeddings.py itself.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many top-scoring intent candidates retrieve() returns. scripts/gate.py and
# state_store.py both consume this many; raising it gives multi-intent
# detection more candidates to compare, at the cost of scoring more exemplars.
RETRIEVAL_TOP_K = 3

# Cosine similarity thresholds (all-MiniLM-L6-v2 embeddings, normalized).
HIGH_CONFIDENCE = 0.5  # accept top intent outright
MID_CONFIDENCE = 0.35  # ambiguous band -> would ask an LLM to pick between candidates

# Multiple-intents-in-one-utterance trigger: both candidates need to be
# confident AND close together. A dominant top score with a moderately
# scoring second choice (e.g. two intents that just share vocabulary, like
# "payment status" vs "payment due date") is normal single-intent noise, not
# two separate asks -- so we also require a small gap between them.
MULTI_INTENT_GAP = 0.16

# Customer-facing message returned when a confirmed multi-intent utterance is
# blocked (state_store.py) -- no frames are opened for either intent; the
# customer is asked to resubmit them one at a time. `{intents}` is filled
# in with the detected intents, human-readable and joined with "and".
MULTI_INTENT_BLOCK_MESSAGE = (
    "It looks like you're asking about more than one thing at once ({intents}). "
    "Could you send those one at a time so I can help with each?"
)

# Cheap keyword cues that a turn is a correction or a reference to something
# said earlier, rather than a new instruction. Real disambiguation of these
# is left to the LLM (see llm.py) -- this is just a trigger.
CORRECTION_KEYWORDS = [
    "actually",
    "instead",
    "never mind",
    "changed my mind",
    "scratch that",
    "on second thought",
    "leave it as",
    "leave the",
]

REFERENCE_KEYWORDS = [
    "that one",
    "the previous",
    "the same one",
    "back to",
    "about the",
]

# -- slots.py: deterministic slot extraction --------------------------------

# Order IDs look like "ORD-48213": this literal prefix, followed by digits.
ORDER_ID_PREFIX = "ORD-"

# Fallback when no ORD-prefixed ID is found: a bare number this long or
# longer is treated as an order/account-style ID.
BARE_NUMBER_MIN_DIGITS = 4

# Discount/promo codes are all-caps tokens in this length range (SAVE20,
# WELCOME15, FREESHIP).
DISCOUNT_CODE_MIN_LEN = 4
DISCOUNT_CODE_MAX_LEN = 12

# A pure-letters token (no digits, e.g. FREESHIP) is only accepted as a
# discount code when one of these words also appears in the same turn --
# otherwise it's indistinguishable from any other capitalized word.
DISCOUNT_CONTEXT_KEYWORDS = ["code", "coupon", "promo", "promotion", "promotions", "discount"]

# Product names are assumed Title Case, cued by one of these determiners
# ("the Aurora Desk Lamp", "my Aurora Desk Lamp"). "a"/"an" are deliberately
# excluded -- they precede generic nouns ("a Tuesday delivery") far more
# often than a specific product name.
PRODUCT_NAME_CUE_WORDS = ["the", "my", "this", "that"]

# Each captured word must be at least this many characters, so a trailing
# capitalized pronoun ("...that Marlow Sofa I ordered") doesn't get swept
# into the match.
PRODUCT_NAME_MIN_WORD_LEN = 2

# Longest run of Title-Case words captured as a single product name.
PRODUCT_NAME_MAX_WORDS = 5

# Phone numbers: a run of digits, optionally with +, spaces, dashes, dots
# or parentheses, holding this many digits in all ("555-555-5555",
# "+1 (415) 555 0134"). Fewer is an order number or a year; more isn't a
# phone number (E.164 caps at 15).
PHONE_MIN_DIGITS = 7
PHONE_MAX_DIGITS = 15

# Company names are Title Case words cued by one of these phrases ("I'm
# calling from Truhlar And Truhlar", "I work for King"). Lowercase joining
# words in NAME_JOIN_WORDS may sit between the capitalized ones.
COMPANY_NAME_CUE_PHRASES = [
    "calling from", "calling on behalf of", "i'm with", "i am with", "we're with",
    "i work at", "i work for", "company is", "company called", "account for", "from",
]

# Deal names are Title Case words right before "deal" ("the Commercial
# Press deal") or after "deal for/with/called/named".
DEAL_NAME_CUE_PHRASES = ["deal for", "deal with", "deal called", "deal named"]

# Lowercase words allowed inside a company or deal name ("Truhlar and
# Truhlar", "Bank of the West").
NAME_JOIN_WORDS = ["and", "&", "of", "the"]

# Longest run of words captured as a single company or deal name.
ORG_NAME_MAX_WORDS = 6

# -- catalog.py: admin-edited intent/tool catalog -----------------------------

# Where edits made through the /v1/intents and /v1/tools write endpoints are
# saved. Loaded over the defaults in intents.py / mcp_tools.py at startup when
# present; delete it to go back to those defaults.
CATALOG_PATH = os.path.join(ROOT_DIR, "data", "catalog.json")

# -- mcp_client.py: remote MCP servers for live tools -------------------------

# Named MCP connections a tool's `remote` binding can use, e.g.
#   {"zoho": {"url": "${ZOHO_MCP_URL}"},
#    "other": {"url": "https://...", "headers": {"Authorization": "Bearer ${OTHER_TOKEN}"}}}
# "${NAME}" is read from the environment (or .env), so secrets can stay out of
# the file -- it's committed, so URLs with keys and tokens must always be
# "${NAME}" references. The shipped connections.json has Zoho's hosted MCP
# server, whose URL is a secret (ZOHO_MCP_URL). No file means no connections:
# live tools then return a tool error instead of executing.
# Override with the env var of the same name (e.g. an environment that needs
# a different set of connections).
MCP_CONNECTIONS_PATH = os.environ.get(
    "MCP_CONNECTIONS_PATH",
    os.path.join(ROOT_DIR, "connections.json"),
)

# Saved sign-ins for connections with "auth": {"type": "mcp_oauth"}, one
# <name>.json per connection, written by `python scripts/connect.py <name>`
# and refreshed by the server. They hold tokens: keep the folder out of git
# (data/ is ignored) and readable only by the server's user. Sign in on any
# machine with a browser and copy the file to the server.
# Override with the env var of the same name.
MCP_TOKENS_DIR = os.environ.get("MCP_TOKENS_DIR", os.path.join(ROOT_DIR, "data", "oauth"))

# The longest a turn waits for a remote tool call (or a tool listing) before
# giving up with a tool error. The turn response is held until then.
# Override with the env var of the same name.
MCP_CALL_TIMEOUT_SECONDS = float(os.environ.get("MCP_CALL_TIMEOUT_SECONDS", "5"))

# -- responder.py: agent-facing answers for live tool results -----------------

# The Ollama model that turns a live tool's JSON result into a short briefing
# for the agent. Bigger than the escalation model (llm.py): summarizing
# nested JSON faithfully needs more than picking from a list. Gemma 4 E2B
# (quantization-aware build) beat gemma3:1b/4b on both accuracy and speed
# for this, with thinking off (see responder._ask_model).
# Override with the env var of the same name.
RESPONDER_MODEL_NAME = os.environ.get("RESPONDER_MODEL_NAME", "gemma4:e2b-it-qat")

# The longest a turn waits for that answer before falling back to a plain
# listing of the result. On top of the tool call's own wait. 20s covers a
# full Zoho lead record (~900 prompt tokens, ~14s) on a CPU-only box; on a
# GPU an answer takes about a second, so this can come down there.
# Override with the env var of the same name.
RESPONDER_TIMEOUT_SECONDS = float(os.environ.get("RESPONDER_TIMEOUT_SECONDS", "20"))

# The same wait for a "full" answer (see responder.MODES), which reads more of
# the result and writes a paragraph rather than a sentence or two.
# Override with the env var of the same name.
RESPONDER_FULL_TIMEOUT_SECONDS = float(os.environ.get("RESPONDER_FULL_TIMEOUT_SECONDS", "40"))

# The answer mode a turn gets when its request doesn't name one: "focused" or
# "full" (see responder.MODES).
# Override with the env var of the same name.
RESPONDER_DEFAULT_MODE = os.environ.get("RESPONDER_DEFAULT_MODE", "focused")

# How long Ollama keeps the responder model loaded after its last use, in
# seconds; negative keeps it loaded for good. Ollama's own default unloads it
# after 5 idle minutes, and reloading costs the next answer its timeout.
# Override with the env var of the same name.
RESPONDER_KEEP_ALIVE_SECONDS = float(os.environ.get("RESPONDER_KEEP_ALIVE_SECONDS", "-1"))

# How many of a result list's items are compared by meaning to what was said
# on the call, to pick what the answer model sees (see responder.Relevance);
# exact lookup matches are found in every item regardless. Each item costs
# ~15ms to embed on a CPU-only laptop (i7-1355U), so 50 caps it near 1s;
# raise it on faster hardware.
# Override with the env var of the same name.
RESPONDER_MAX_SCORED_ITEMS = int(os.environ.get("RESPONDER_MAX_SCORED_ITEMS", "50"))

# Whether a tool's result is pruned before the answer model sees it (the
# items and passages closest to the call, within responder.LIMITS). "false"
# sends the result exactly as the server returned it -- nothing dropped,
# reordered or reformatted -- which is slower (every token is read) and needs
# a large enough RESPONDER_NUM_CTX.
# Override with the env var of the same name.
RESPONDER_PRUNE_RESULTS = os.environ.get("RESPONDER_PRUNE_RESULTS", "true").strip().lower() not in ("0", "false", "no")

# The answer model's context window in tokens (Ollama's num_ctx); a prompt
# longer than it would be silently cut from the start, so a result that
# can't fit gets the plain listing instead. Unset: Ollama's own default when
# pruning (pruned prompts are small), 32768 when not. gemma4:e2b reads up
# to 131072; a bigger window costs memory, not speed.
# Override with the env var of the same name.
RESPONDER_NUM_CTX = int(os.environ.get("RESPONDER_NUM_CTX") or (0 if RESPONDER_PRUNE_RESULTS else 32768)) or None

# -- server.py: call memory ----------------------------------------------------

# A call with no activity (turn, transcript, slot override, reset) for this
# many seconds is dropped from memory; reads of it then get 404
# `call_expired`, and a new turn starts it over empty. Reading state
# (GET /v1/tenants/{tenant_id}/calls/{call_id}) doesn't count as activity.
# 30 min leaves room for long holds; PBXware's hangup should still DELETE the
# call. 0 disables expiry.
# Override with the env var of the same name.
CALL_IDLE_TIMEOUT_SECONDS = int(os.environ.get("CALL_IDLE_TIMEOUT_SECONDS", "1800"))

# -- server.py: archived endpoints --------------------------------------------

# PATCH /v1/tenants/{tenant_id}/calls/{call_id}/frames/{frame_id}/slots/{slot_name}
# (agent slot correction) is out of scope for the current release but kept in
# the code.
# Off by default: the route answers 404 `endpoint_disabled` and the demo hides
# its slot edit button. Set the env var of the same name to 1/true to turn it on.
def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


SLOT_OVERRIDE_ENABLED = _env_flag("SLOT_OVERRIDE_ENABLED")

# -- demo_web.py: browser demo UI -----------------------------------------------

# The /demo page (static_demo/) is a dev/sales tool, not part of the product.
# Off by default: /demo and its assets answer 404. Set the env var of the same
# name to 1/true to serve it (e.g. on a dev or demo box).
# DEMO_UI_ENABLED = _env_flag("DEMO_UI_ENABLED")
DEMO_UI_ENABLED = 1
