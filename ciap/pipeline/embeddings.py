# Step 1: Intent retrieval via sentence embeddings.
#
# Each intent is represented by a handful of exemplar utterances (the
# tenant's catalog, see catalog.py). We embed a catalog's exemplars once, the
# first time a turn uses it, then for each
# incoming utterance we embed it and compare against all exemplars with
# cosine similarity. An intent's score is the similarity to its single
# closest exemplar (max, not average) so one great exemplar match is enough.
#
# At this scale (dozens of exemplars) a plain numpy matrix multiply is
# simpler and just as fast as standing up FAISS/Qdrant. Swap in a proper
# vector index (FAISS locally, Qdrant if distributed) if the exemplar set
# grows into the thousands.
#
# Used for both customer and agent turns (see state_store.py) -- an agent
# can express an intent just as validly as a customer can ("I'll submit a
# refund request" scores against check_refund_status/exchange_item the same
# way a customer saying it would), so there's one shared index rather than a
# second exemplar catalogue to keep in sync.

import threading
import weakref

from sentence_transformers import SentenceTransformer

from ciap import catalog
from ciap import config

MODEL_NAME = config.EMBEDDING_MODEL_NAME  # small, fast, CPU-friendly

# stage2 uses BAAI/bge-small-en-v1.5 (better at finding the intent, but
# misfires -- scores bunch up close together across unrelated intents). That
# bunching is harmless for stage2's single standalone utterances, but here in
# stage3 most turns are short slot answers ("It's 48291.", "INV-10382.") in a
# multi-turn call -- exactly where BAAI's bunched scores clear both
# HIGH_CONFIDENCE and MULTI_INTENT_GAP together and make process_utterance()
# treat a plain slot-fill answer as two brand-new intents. MiniLM scores
# these short answers low across the board, which is what lets the pending
# AWAITING_INFO frame's slot actually get filled instead.

_model = None
# Catalog -> (exemplar_intents, exemplar_matrix). Catalogs are never edited in
# place (a tenant's new version is a new Catalog), so an index stays valid for
# its catalog's lifetime and goes with it once the cache drops it.
_indexes = weakref.WeakKeyDictionary()
_index_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _build_index(cat):
    texts, intents = [], []
    for intent, spec in cat.intents.items():
        for text in spec["exemplars"]:
            texts.append(text)
            intents.append(intent)
    if not texts:
        return [], None
    matrix = _get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return intents, matrix


def _get_index(cat):
    index = _indexes.get(cat)
    if index is None:
        with _index_lock:
            index = _indexes.get(cat)
            if index is None:
                index = _indexes[cat] = _build_index(cat)
    return index


def encode(texts):
    """Normalized embeddings for `texts`, one row each, from the same model --
    responder.py reuses it to find the parts of a tool result a call is about."""
    return _get_model().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)


def retrieve(utterance, top_k=config.RETRIEVAL_TOP_K):
    """Return up to top_k (intent, score) pairs, best first, deduplicated by
    intent, from the catalog of the turn in progress."""
    exemplar_intents, exemplar_matrix = _get_index(catalog.current())
    if exemplar_matrix is None:
        return []  # a tenant with no intents

    model = _get_model()
    query_vec = model.encode([utterance], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = exemplar_matrix @ query_vec  # cosine similarity (vectors are normalized)

    best_per_intent = {}
    for score, intent in zip(sims, exemplar_intents):
        if intent not in best_per_intent or score > best_per_intent[intent]:
            best_per_intent[intent] = float(score)

    ranked = sorted(best_per_intent.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]
