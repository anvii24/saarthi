"""
Multilingual meaning search (Person 3: Anvi).

build_embeddings() turns every translation of every verse into a vector;
search_dense() finds the verses closest in meaning to a question, in any language.

STUB: search_dense() returns dummy results from the sample file until the
embedding model is added.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import SearchResult, make_result  # noqa: E402

SAMPLE_PATH = os.path.join("data", "sample_verses.json")
MODEL_NAME = "intfloat/multilingual-e5-base"  # planned model


def build_embeddings(verses: list[dict]) -> None:
    """Embed every translation of every verse and save the vectors.

    TODO: load MODEL_NAME with sentence-transformers, embed each translation,
    save vectors + verse ids to data/processed/embeddings.npy / .json.
    """
    pass  # stub


def search_dense(query: str, k: int = 20) -> list[SearchResult]:
    """Return the k verses closest in meaning to the query, best first.

    STUB: returns the first k sample verses with made-up decreasing scores.
    """
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        verses = json.load(f)
    return [make_result(v["id"], 1.0 - i * 0.01, "dense") for i, v in enumerate(verses[:k])]


if __name__ == "__main__":
    print(search_dense("how do I deal with failure?", k=5))
