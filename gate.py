# Stage 1: Embedding Gate.
#
# Wraps embeddings.retrieve() with an accept/reject decision: the top
# candidate is accepted as the intent if its cosine similarity clears
# HIGH_CONFIDENCE, otherwise the utterance is reported as unknown_intent.
# Mid-confidence disambiguation (LLM-assisted) is out of scope here — that's
# Stage 2, Intent Classification.

from embeddings import retrieve
from config import HIGH_CONFIDENCE, RETRIEVAL_TOP_K


def classify(utterance, top_k=RETRIEVAL_TOP_K):
    candidates = retrieve(utterance, top_k=top_k)

    if not candidates:
        return {
            "utterance": utterance,
            "intent": "unknown_intent",
            "confidence": 0.0,
            "candidates": [],
        }

    top_intent, top_score = candidates[0]
    intent = top_intent if top_score >= HIGH_CONFIDENCE else "unknown_intent"

    return {
        "utterance": utterance,
        "intent": intent,
        "confidence": top_score,
        "candidates": candidates,
    }
