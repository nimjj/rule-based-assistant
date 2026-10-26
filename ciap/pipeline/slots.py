# Stage 2: Deterministic slot extraction.
#
# Cheapest, most reliable method first: regex for structured IDs, a
# dedicated library for dates. Anything still ambiguous is out of scope
# here -- that's a later stage's LLM escalation.

import re

import dateparser.search

from ciap import config

# Every regex below is built from config.py's knobs rather than hardcoded,
# so tuning the accepted shapes (ID prefix, code length, cue words, ...)
# never requires touching this file. See config.py for what each knob means.

_ORDER_ID_RE = re.compile(rf"\b{re.escape(config.ORDER_ID_PREFIX)}\d+\b", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(rf"\b\d{{{config.BARE_NUMBER_MIN_DIGITS},}}\b")

# Promo codes are conventionally all-caps and mix letters with digits
# (SAVE20, WELCOME15) -- matched directly, no context needed. A pure-letters
# code like FREESHIP is indistinguishable from any other capitalized word on
# its own, so it's only accepted as a fallback when one of
# config.DISCOUNT_CONTEXT_KEYWORDS appears somewhere in the same turn (true
# across every sample transcript). Still just a heuristic -- a code-shaped
# word with no such keyword nearby (or a false one that happens to sit next
# to "code") is exactly the "still ambiguous" case the design scopes to the
# LLM step.
_code_len = rf"{config.DISCOUNT_CODE_MIN_LEN},{config.DISCOUNT_CODE_MAX_LEN}"
_DISCOUNT_CODE_RE = re.compile(rf"\b(?=[A-Z0-9]{{{_code_len}}}\b)[A-Z]+[0-9]+[A-Z0-9]*\b")
_DISCOUNT_CODE_LETTERS_RE = re.compile(rf"\b[A-Z]{{{_code_len}}}\b")
_DISCOUNT_CONTEXT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in config.DISCOUNT_CONTEXT_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Product names in this MVP are assumed Title Case in the transcript (as a
# real UI/catalog would render them, e.g. "the Aurora Desk Lamp") -- the
# regex captures the run of capitalized words after a determiner
# (config.PRODUCT_NAME_CUE_WORDS) and stops at the first word that isn't
# Title Case. Free-form product references without one of those cues (or
# without consistent capitalization) still won't be caught here; that's the
# LLM's job in a production system, not a regex's.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_cue_alternation = "|".join(re.escape(w) for w in config.PRODUCT_NAME_CUE_WORDS)
_word_min_extra_chars = max(config.PRODUCT_NAME_MIN_WORD_LEN - 1, 0)  # 1st char is [A-Z]
_PRODUCT_NAME_RE = re.compile(
    rf"\b(?:(?i:{_cue_alternation}))\s+"
    rf"((?:[A-Z][A-Za-z0-9'\-]{{{_word_min_extra_chars},}}\s*){{1,{config.PRODUCT_NAME_MAX_WORDS}}})"
)


_PHONE_RE = re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]*\d(?![\w-])")

# A company or deal name: a Title Case word, then more of them, optionally
# joined by config.NAME_JOIN_WORDS ("Truhlar And Truhlar", "Bank of the West").
_NAME_WORD = r"[A-Z][\w'&.-]*"
_JOIN = "|".join(re.escape(w) for w in config.NAME_JOIN_WORDS)
_ORG_NAME = rf"{_NAME_WORD}(?:\s+(?:(?i:{_JOIN})\s+)*{_NAME_WORD}){{0,{config.ORG_NAME_MAX_WORDS - 1}}}"


def _cued(phrases):
    return "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))


_COMPANY_NAME_RE = re.compile(rf"\b(?i:{_cued(config.COMPANY_NAME_CUE_PHRASES)})\s+(?:(?i:the)\s+)?({_ORG_NAME})")
_DEAL_NAME_BEFORE_RE = re.compile(rf"({_ORG_NAME})\s+(?i:deal)\b")
_DEAL_NAME_AFTER_RE = re.compile(rf"\b(?i:{_cued(config.DEAL_NAME_CUE_PHRASES)})\s+(?:(?i:the)\s+)?({_ORG_NAME})")


def extract_order_id(text):
    """ORD-NNNN if present, otherwise a bare order/account-style number."""
    match = _ORDER_ID_RE.search(text)
    if match:
        return match.group().upper()
    match = _BARE_NUMBER_RE.search(text)
    return match.group() if match else None


def extract_discount_code(text):
    match = _DISCOUNT_CODE_RE.search(text)
    if match:
        return match.group().upper()
    if _DISCOUNT_CONTEXT_RE.search(text):
        match = _DISCOUNT_CODE_LETTERS_RE.search(text)
        if match:
            return match.group()
    return None


def extract_product_name(text):
    match = _PRODUCT_NAME_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip().rstrip(",.?!")
    return name or None


def extract_email(text):
    match = _EMAIL_RE.search(text)
    return match.group() if match else None


def extract_phone(text):
    """A phone number as said ("555-555-5555"); Zoho's phone search accepts
    any format. Digit runs too short or long to be a phone number are skipped."""
    for match in _PHONE_RE.finditer(text):
        digits = sum(c.isdigit() for c in match.group())
        if config.PHONE_MIN_DIGITS <= digits <= config.PHONE_MAX_DIGITS:
            return match.group().strip()
    return None


def _clean_name(name):
    # A sentence-initial "The" isn't part of the name.
    name = re.sub(r"^The\s+", "", name.strip().rstrip(",.?!"))
    return name or None


def extract_company_name(text):
    match = _COMPANY_NAME_RE.search(text)
    return _clean_name(match.group(1)) if match else None


def extract_deal_name(text):
    match = _DEAL_NAME_BEFORE_RE.search(text) or _DEAL_NAME_AFTER_RE.search(text)
    return _clean_name(match.group(1)) if match else None


# dateparser's fuzzy search can match ordinary short words as dates (e.g.
# "we" -> "Wednesday" in "...can we move it..."). Real date mentions in this
# domain ("next Friday", "tomorrow", "March 3rd") are never this short, so
# matches below this length are almost certainly false positives, not a
# genuine date the customer said.
_MIN_DATE_MATCH_LEN = 3


def extract_date(text):
    results = dateparser.search.search_dates(
        text, languages=["en"], settings={"PREFER_DATES_FROM": "future"}
    )
    for matched_text, parsed in results or []:
        if len(matched_text.strip()) >= _MIN_DATE_MATCH_LEN:
            return {"text": matched_text, "date": parsed.isoformat()}
    return None


# Maps slot name -> extractor function. Used by the pipeline to pull only the
# slots a given intent actually needs.
SLOT_EXTRACTORS = {
    "order_id": extract_order_id,
    "product_name": extract_product_name,
    "discount_code": extract_discount_code,
    "new_date": extract_date,
    "email": extract_email,
    "phone": extract_phone,
    "company_name": extract_company_name,
    "deal_name": extract_deal_name,
}
