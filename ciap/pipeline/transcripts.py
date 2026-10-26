# "Customer: ..." / "Agent: ..." transcript parsing, shared by the
# /v1/tenants/{tenant_id}/calls/{call_id}/transcript endpoint (server.py) and
# the CLI replay (scripts/demo.py).

import re

_TURN_RE = re.compile(r"^(Customer|Agent):\s*(.*)$", re.IGNORECASE)


def parse_turn(line):
    """Parse one "Customer: ..." / "Agent: ..." line -- the tagging real
    transcripts always use. Returns (speaker, utterance) with speaker
    normalized to "Customer"/"Agent", or None if the line doesn't match that
    shape."""
    match = _TURN_RE.match(line.strip())
    if not match:
        return None
    speaker, utterance = match.group(1).capitalize(), match.group(2).strip()
    return (speaker, utterance) if utterance else None


def parse_turns(text):
    """Parse every Customer/Agent line from a transcript, in order."""
    turns = []
    for line in text.splitlines():
        turn = parse_turn(line)
        if turn:
            turns.append(turn)
    return turns
