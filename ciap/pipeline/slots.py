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


_MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _phone_spans(text):
    """Spans of separated phone numbers ("555-555-5555", "(555) 555 5555"),
    whose digit groups aren't order numbers. An unseparated digit run is
    left alone: it may well be an order or account number."""
    spans = []
    for match in _PHONE_RE.finditer(text):
        raw = match.group()
        digits = sum(c.isdigit() for c in raw)
        if config.PHONE_MIN_DIGITS <= digits <= config.PHONE_MAX_DIGITS and not raw.strip().isdigit():
            spans.append(match.span())
    return spans


def extract_order_id(text):
    """ORD-NNNN if present, otherwise a bare order/account-style number --
    but not a digit group of a phone number, a date's year ("March 15 2025")
    or a part of a dashed/slashed date ("2025-03-15")."""
    match = _ORDER_ID_RE.search(text)
    if match:
        return match.group().upper()
    phones = _phone_spans(text)
    has_month = bool(_MONTH_RE.search(text))
    for match in _BARE_NUMBER_RE.finditer(text):
        start, end = match.span()
        if any(a <= start and end <= b for a, b in phones):
            continue
        if (start > 0 and text[start - 1] in "-/.") or text[end:end + 1] in ("-", "/"):
            continue
        if has_month and _YEAR_RE.fullmatch(match.group()):
            continue
        return match.group()
    return None


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


# -- by CIP slot type ----------------------------------------------------------
#
# A tenant's slots (catalog.py) have a type rather than a fixed name. A slot
# named like one above uses that extractor (order_id, email, ...); any other
# goes by its type. TEXT slots without a named extractor can't be read with a
# pattern: state_store asks the LLM for the one a frame is waiting on.

_TIME_RE = re.compile(
    r"\b(?:(?P<h12>1[0-2]|0?[1-9])(?::(?P<m12>[0-5]\d))?\s*(?P<ampm>[ap])\.?\s?m\b\.?"
    r"|(?P<h24>[01]?\d|2[0-3]):(?P<m24>[0-5]\d)\b"
    r"|(?P<word>noon|midnight))",
    re.IGNORECASE,
)


def extract_time(text):
    """A time of day as "HH:MM" ("3pm" -> "15:00", "10:30 a.m." -> "10:30",
    "14:45", "noon")."""
    match = _TIME_RE.search(text)
    if not match:
        return None
    if match.group("word"):
        return "12:00" if match.group("word").lower() == "noon" else "00:00"
    if match.group("h24") is not None:
        return f"{int(match.group('h24')):02d}:{match.group('m24')}"
    hour = int(match.group("h12")) % 12 + (12 if match.group("ampm").lower() == "p" else 0)
    return f"{hour:02d}:{match.group('m12') or '00'}"


_NUMBER_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def extract_number(text):
    """The first number said that isn't part of a phone number ("2", "1,250",
    "3.5"), without thousands separators."""
    phones = _phone_spans(text)
    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        if any(a <= start and end <= b for a, b in phones):
            continue
        return match.group().replace(",", "")
    return None


_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
_CURRENCY_WORDS = {
    "dollar": "USD", "dollars": "USD", "usd": "USD", "bucks": "USD", "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "rupee": "INR", "rupees": "INR", "inr": "INR",
}
_AMOUNT = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_CURRENCY_BEFORE_RE = re.compile(rf"(?P<sym>[$€£₹])\s?(?P<amt>{_AMOUNT})")
_CURRENCY_AFTER_RE = re.compile(
    rf"(?P<amt>{_AMOUNT})\s?(?P<word>" + "|".join(sorted(_CURRENCY_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def extract_currency(text):
    """An amount of money as "<amount> <ISO code>" ("$12.50" -> "12.50 USD",
    "40 euros" -> "40 EUR")."""
    match = _CURRENCY_BEFORE_RE.search(text)
    if match:
        return f"{match.group('amt').replace(',', '')} {_CURRENCY_SYMBOLS[match.group('sym')]}"
    match = _CURRENCY_AFTER_RE.search(text)
    if match:
        return f"{match.group('amt').replace(',', '')} {_CURRENCY_WORDS[match.group('word').lower()]}"
    return None


def extract_option(text, options):
    """The slot's option mentioned in the turn, as the option is written. The
    longest match wins ("Express Plus" over "Express")."""
    lowered = text.lower()
    for option in sorted((o for o in options if o and o.strip()), key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(option.strip().lower())}(?!\w)", lowered):
            return option
    return None


TYPE_EXTRACTORS = {
    "EMAIL": extract_email,
    "PHONE": extract_phone,
    "DATE": extract_date,
    "TIME": extract_time,
    "NUMBER": extract_number,
    "CURRENCY": extract_currency,
}


def extract(name, spec, text):
    """A slot's value from one turn, or None. `spec` is the catalog's slot
    ({"type", "options", ...}). A TEXT slot with no named extractor is None
    here (see above)."""
    named = SLOT_EXTRACTORS.get(name)
    if named:
        return named(text)
    slot_type = ((spec or {}).get("type") or "TEXT").upper()
    if slot_type == "OPTION":
        return extract_option(text, (spec or {}).get("options") or [])
    extractor = TYPE_EXTRACTORS.get(slot_type)
    return extractor(text) if extractor else None


def has_pattern(name, spec):
    """Whether extract() can read this slot without the LLM."""
    return name in SLOT_EXTRACTORS or ((spec or {}).get("type") or "TEXT").upper() != "TEXT"

