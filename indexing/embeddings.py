"""
Multilingual meaning search (Person 3: Anvi).

build_embeddings() turns every translation of every verse into a vector.
search_dense() embeds the question and finds the verses whose translations are
closest in meaning (cosine similarity), in any language: a Hindi question can
match an English translation and the other way round.

Model: intfloat/multilingual-e5-base (about 1.1 GB, downloaded once on first use).
e5 models expect "query: " in front of questions and "passage: " in front of documents.

Usage (from the repo root):
    python indexing/embeddings.py            # builds the index (a few minutes on CPU)
    python indexing/embeddings.py "how do I deal with failure?"
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import SearchResult, make_result  # noqa: E402

MODEL_NAME = "intfloat/multilingual-e5-base"
VERSES_PATH = os.path.join("data", "verses.json")
SAMPLE_PATH = os.path.join("data", "sample_verses.json")
VECTORS_PATH = os.path.join("data", "processed", "embeddings.npy")
META_PATH = os.path.join("data", "processed", "embeddings_meta.json")

_model = None
_vectors = None
_meta = None


def get_model():
    """Load the embedding model once (first call downloads it)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_verses() -> list[dict]:
    """Full verses.json if it exists, otherwise the 50-verse sample."""
    path = VERSES_PATH if os.path.exists(VERSES_PATH) else SAMPLE_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_embeddings(verses: list[dict] | None = None, batch_size: int = 32) -> None:
    """Embed every translation of every verse and save vectors + metadata."""
    verses = verses if verses is not None else load_verses()

    texts, meta = [], []
    for verse in verses:
        for key, text in verse["translations"].items():
            if text.strip():
                texts.append("passage: " + text.strip())
                meta.append({"id": verse["id"], "translation": key, "lang": key.split("_")[0]})

    print(f"Embedding {len(texts)} translations of {len(verses)} verses with {MODEL_NAME} ...")
    vectors = get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )

    os.makedirs(os.path.dirname(VECTORS_PATH), exist_ok=True)
    np.save(VECTORS_PATH, vectors.astype(np.float32))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"Saved {vectors.shape} vectors to {VECTORS_PATH}")


def _load_index():
    global _vectors, _meta
    if _vectors is None:
        if not os.path.exists(VECTORS_PATH):
            raise FileNotFoundError("No embeddings yet: run  python indexing/embeddings.py")
        _vectors = np.load(VECTORS_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    return _vectors, _meta


def search_dense(query: str, k: int = 20, lang: str | None = None) -> list[SearchResult]:
    """Return the k verses closest in meaning to the query, best first.

    Each verse scores the similarity of its best-matching translation.
    lang: None = search all translations; "en" or "hi" = only that language
          (used in the multilingual experiment).
    """
    vectors, meta = _load_index()
    q = get_model().encode(["query: " + query.strip()], normalize_embeddings=True)[0]
    sims = vectors @ q  # cosine similarity, since all vectors are normalised

    best: dict[str, float] = {}
    for sim, m in zip(sims, meta):
        if lang and m["lang"] != lang:
            continue
        if sim > best.get(m["id"], -1.0):
            best[m["id"]] = float(sim)

    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:k]
    return [make_result(verse_id, score, "dense") for verse_id, score in ranked]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for r in search_dense(" ".join(sys.argv[1:]), k=5):
            print(r)
    else:
        build_embeddings()
