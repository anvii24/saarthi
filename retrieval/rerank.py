"""
Re-ranking (Person 3: Anvi).

rerank() adjusts the fused results using extra signals:
  - fused relevance score
  - zone score (translation matches count more than commentary matches)
  - PageRank from Antara's link graph
  - theme boost if a verse's predicted theme matches the question's (Kashvi's classifier)

STUB: combines the fused score with PageRank and a theme boost using equal
weights; zone scores are added once Antara's index exposes zones.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import SearchResult, make_result  # noqa: E402

WEIGHTS = {"fused": 1.0, "pagerank": 1.0, "theme": 1.0}  # tune on q001-q020
THEME_BOOST = 0.1


def rerank(
    results: list[SearchResult],
    theme: str | None = None,
    pagerank_scores: dict[str, float] | None = None,
    verse_themes: dict[str, list[str]] | None = None,
) -> list[SearchResult]:
    """Re-order results using PageRank and theme match.

    Until Antara's PageRank is ready, pass nothing and every verse gets
    the same PageRank, so the order only changes through the theme boost.
    """
    pagerank_scores = pagerank_scores or {}
    verse_themes = verse_themes or {}

    reranked = []
    for result in results:
        pr = pagerank_scores.get(result["id"], 1.0)
        boost = THEME_BOOST if theme and theme in verse_themes.get(result["id"], []) else 0.0
        score = (
            WEIGHTS["fused"] * result["score"]
            + WEIGHTS["pagerank"] * pr * 0.01
            + WEIGHTS["theme"] * boost
        )
        reranked.append(make_result(result["id"], score, "rerank"))

    return sorted(reranked, key=lambda r: r["score"], reverse=True)


if __name__ == "__main__":
    fused = [make_result("2.47", 0.032, "rrf"), make_result("2.48", 0.031, "rrf")]
    print(rerank(fused, theme="action_and_results", verse_themes={"2.48": ["action_and_results"]}))
