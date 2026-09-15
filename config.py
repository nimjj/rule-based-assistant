# Stage 3 tunable knobs: the full config, now that the mid-confidence band
# and correction/reference heuristics actually have somewhere to plug in
# (the state store's turn-handling logic in state_store.py).

# Sentence embedding model used for intent retrieval (embeddings.py). Small
# and CPU-friendly; swap this to retune retrieval quality/speed without
# touching embeddings.py itself.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# How many top-scoring intent candidates retrieve() returns. gate.py and
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
