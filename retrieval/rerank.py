import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import SearchResult, make_result  # noqa: E402

WEIGHTS = {"fused": 0.8, "pagerank": 0.1, "theme": 0.1}


def normalise(scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale to 0-1 (all 1.0 if every score is equal)."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def rerank(
    results: list[SearchResult],
    pagerank_scores: dict[str, float] | None = None,
    theme_scores: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Re-order results with PageRank and topic PageRank.

    Args:
        results: fused results, best first
        pagerank_scores: {verse id: PageRank} for all verses (None = not used)
        theme_scores: {verse id: topic PageRank for the predicted theme} (None = not used)
        weights: {"fused", "pagerank", "theme"}; defaults to WEIGHTS
    """
    weights = {**WEIGHTS, **(weights or {})}
    fused = normalise({r["id"]: r["score"] for r in results})
    pagerank = normalise(pagerank_scores or {})
    theme = normalise(theme_scores or {})

    reranked = []
    for result in results:
        vid = result["id"]
        score = weights["fused"] * fused[vid]
        if pagerank:
            score += weights["pagerank"] * pagerank.get(vid, 0.0)
        if theme:
            score += weights["theme"] * theme.get(vid, 0.0)
        reranked.append(make_result(vid, score, "rerank"))

    return sorted(reranked, key=lambda r: r["score"], reverse=True)


if __name__ == "__main__":
    fused = [make_result("2.47", 0.033, "rrf"), make_result("2.48", 0.032, "rrf"), make_result("18.66", 0.016, "rrf")]
    print(rerank(fused, pagerank_scores={"2.47": 0.001, "2.48": 0.004, "18.66": 0.002}))