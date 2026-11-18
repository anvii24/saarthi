"""Reciprocal Rank Fusion: merge ranked lists by rank, not raw score, because
BM25 and cosine scores are on different scales.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import SearchResult, make_result  # noqa: E402


def rrf(result_lists: list[list[SearchResult]], k: int = 60, top_n: int = 20) -> list[SearchResult]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Args:
        result_lists: each list sorted best first
        k: smoothing constant (60 is the value from Cormack et al., 2009)
        top_n: how many fused results to return
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result["id"]] = scores.get(result["id"], 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [make_result(verse_id, score, "rrf") for verse_id, score in fused[:top_n]]


if __name__ == "__main__":
    bm25 = [make_result("2.47", 9.1, "bm25"), make_result("3.19", 7.4, "bm25"), make_result("2.48", 6.0, "bm25")]
    dense = [make_result("2.48", 0.91, "dense"), make_result("2.47", 0.88, "dense"), make_result("18.66", 0.80, "dense")]
    for r in rrf([bm25, dense]):
        print(r)
