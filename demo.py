# Stage 3 demo: the conversational state store, on top of the Stage 1 gate
# and Stage 2 slot extraction.
#
#   python demo.py call            # replay every samples/convo*.txt as a simulated live call
#   python demo.py call convo2.txt # replay just one sample
#   python demo.py live            # type turns yourself, multi-turn, until Ctrl+C
#   python demo.py test            # score intent_test_dataset.xlsx (intent only, single-turn)

import argparse
import glob
import os
import re
import sys
import time

from gate import classify
from state_store import ConversationStateStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(__file__)
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
DATASET_PATH = os.path.join(SCRIPT_DIR, "intent_test_dataset.xlsx")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "intent_test_results.xlsx")

# How long a turn "takes to speak" before its text appears, and how long the
# pipeline takes to "think" once a customer turn has finished -- rough stand-
# ins for call audio and inference latency, just enough to make the replay
# feel like a live call instead of a text dump.
CUSTOMER_SPEAK_SECONDS = 2.0
AGENT_SPEAK_SECONDS = 1.0
THINKING_SECONDS = 0.4

_TURN_RE = re.compile(r"^(Customer|Agent):\s*(.*)$", re.IGNORECASE)


def parse_turn(line):
    """Parse one "Customer: ..." / "Agent: ..." line -- the tagging real
    transcripts (and this demo's live mode) always use. Returns
    (speaker, utterance) with speaker normalized to "Customer"/"Agent", or
    None if the line doesn't match that shape."""
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


def warm_up():
    # Loads the embedding model up front, so no timed measurement (and no
    # "live call" pacing) ever includes the one-time model-load cost.
    print("Loading models...", end=" ", flush=True)
    ConversationStateStore().process_utterance("warm up")
    print("done.\n")


# -- call mode: simulate a live call over a sample transcript ---------------

def print_trace(trace):
    cand_str = ", ".join(f"{intent}={score:.2f}" for intent, score in trace["candidates"])
    print(f"    candidates: {cand_str}")
    if trace.get("heuristic_flags"):
        print(f"    flags: {', '.join(trace['heuristic_flags'])}")
    print(f"    turn_type: {trace['turn_type']}")
    if "llm_escalation" in trace:
        esc = trace["llm_escalation"]
        print(f"    -> escalated to LLM ({esc['reason']}): {esc['note']}")
    if trace.get("frames"):
        print(f"    frames opened: {trace['frames']}")
    elif "frame" in trace:
        extra = f" filled: {trace['filled_slots']}" if trace.get("filled_slots") else ""
        print(f"    frame #{trace['frame']}{extra}")
    for call in trace.get("tool_calls", []):
        args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
        print(f"    -> tool call: {call['tool']}({args})")


def print_frames(store):
    print("\n  Final conversation state:")
    if not store.frames:
        print("    (no frames opened)")
    for frame in store.frames:
        origin = f" ({frame.initiated_by})" if frame.initiated_by != "customer" else ""
        print(
            f"    #{frame.id} {frame.intent}{origin} [{frame.status}] "
            f"slots={frame.slots} missing={frame.missing_slots}"
        )
        if frame.tool_call:
            args = ", ".join(f"{k}={v!r}" for k, v in frame.tool_call["arguments"].items())
            print(f"        tool call: {frame.tool_call['tool']}({args})")


def run_call(path):
    with open(path, "r", encoding="utf-8") as f:
        turns = parse_turns(f.read())

    print("=" * 70)
    print(f"Incoming call: {os.path.basename(path)}")
    print("=" * 70)

    store = ConversationStateStore()
    for speaker, utterance in turns:
        delay = CUSTOMER_SPEAK_SECONDS if speaker == "Customer" else AGENT_SPEAK_SECONDS
        print(f"\n[{speaker} speaking...]", end="", flush=True)
        time.sleep(delay)
        print(f"\r{speaker}: {utterance}" + " " * 20)

        print("  [analyzing...]", end="", flush=True)
        time.sleep(THINKING_SECONDS)
        print("\r" + " " * 20 + "\r", end="")
        trace = store.process_utterance(utterance, speaker=speaker.lower())
        print_trace(trace)

    print_frames(store)
    print()


def run_calls(names):
    if names:
        paths = [n if os.path.dirname(n) else os.path.join(SAMPLES_DIR, n) for n in names]
    else:
        paths = sorted(glob.glob(os.path.join(SAMPLES_DIR, "convo*.txt")))
    if not paths:
        print("No sample transcripts found (expected samples/convo*.txt).")
        return
    for path in paths:
        run_call(path)


# -- live mode: type turns yourself, state persists across the session -----

def interactive_loop():
    print(
        'Type turns tagged the same way a transcript is: "Customer: ..." or '
        '"Agent: ...". State persists across turns. Ctrl+C to quit.\n'
    )
    store = ConversationStateStore()
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        turn = parse_turn(line)
        if turn is None:
            print('  (ignored -- prefix the line with "Customer: " or "Agent: ")')
            continue
        speaker, utterance = turn

        start = time.perf_counter()
        trace = store.process_utterance(utterance, speaker=speaker.lower())
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  ({elapsed_ms:.1f} ms)")
        print_trace(trace)
        print()
    print_frames(store)


# -- test mode: single-turn intent accuracy against the labeled dataset ----

def run_test(dataset_path=DATASET_PATH, results_path=RESULTS_PATH):
    import pandas as pd  # only needed for this mode

    df = pd.read_excel(dataset_path)

    predicted_intents = []
    correct_flags = []
    elapsed_times_ms = []

    for utterance, expected in zip(df["Utterance"], df["Expected Intent"]):
        start = time.perf_counter()
        result = classify(utterance)
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted = result["intent"]
        # dataset labels the reject case "unknown", the gate reports "unknown_intent"
        expected_norm = "unknown_intent" if expected == "unknown" else expected
        is_correct = int(predicted == expected_norm)

        predicted_intents.append(predicted)
        correct_flags.append(is_correct)
        elapsed_times_ms.append(elapsed_ms)

        mark = "correct" if is_correct else "WRONG"
        print(f'  [{mark}] "{utterance}" -> {predicted} (expected {expected}, {elapsed_ms:.1f} ms)')

    df["Predicted Intent"] = predicted_intents
    df["Correct"] = correct_flags
    df.to_excel(results_path, index=False)

    total_ms = sum(elapsed_times_ms)
    accuracy = sum(correct_flags) / len(correct_flags)

    print(f"\nResults written to {os.path.basename(results_path)}")
    print(f"Accuracy: {sum(correct_flags)}/{len(correct_flags)} ({accuracy:.1%})")
    print(f"Avg classification time: {total_ms / len(correct_flags):.1f} ms")
    print(f"Total classification time: {total_ms:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Conversational State Store demo")
    parser.add_argument(
        "mode", choices=["call", "live", "test"],
        help="'call' to replay a sample transcript as a simulated live call, "
             "'live' to type turns yourself, 'test' to score intent_test_dataset.xlsx",
    )
    parser.add_argument(
        "samples", nargs="*",
        help="'call' mode only: specific sample file(s) under samples/ (default: all of them)",
    )
    args = parser.parse_args()

    warm_up()

    if args.mode == "call":
        run_calls(args.samples)
    elif args.mode == "live":
        interactive_loop()
    else:
        run_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
