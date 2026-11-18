"""Saarthi pipeline: a question in English, Hindi or Hinglish in; verses and a checked answer out.

Steps: language -> theme -> Gita concepts -> BM25 + meaning search -> RRF fusion
-> PageRank re-rank -> LLM answer -> fact-check. retrieve() runs only the search
steps, with switches for the ablation table (evaluation/run_retrieval.py).

The spell checker is not used here: it turns everyday words into Gita words
("exam" -> "enam"), so it is used on the Advanced Search page instead.

Try it: python pipeline.py "exam mein fail ho gaya, kya karun?"
"""

import json
import os
import sys
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generation.fact_check import CITATION, keep_supported, verify  # noqa: E402
from generation.rag import explain  # noqa: E402
from indexing.embeddings import search_dense  # noqa: E402
from indexing.link_analysis import load_graph, pagerank, topic_pagerank  # noqa: E402
from indexing.theme_seeds import THEME_SEEDS  # noqa: E402
from preprocessing.clean_en import clean_en  # noqa: E402
from query.classifier import classify_theme  # noqa: E402
from query.language import process_query  # noqa: E402
from query.rewrite import rewrite  # noqa: E402
from retrieval.bm25 import search_bm25  # noqa: E402
from retrieval.fusion import rrf  # noqa: E402
from retrieval.rerank import rerank  # noqa: E402

VERSES_PATH = os.path.join("data", "verses.json")

BM25_K1 = 8.0  # tuned on the tuning questions (evaluation/tune_bm25.py)
BM25_B = 0.75
CANDIDATES = 20  # results taken from each search before fusing
TOP_VERSES = 5   # verses given to the LLM


# ---------------------------------------------------------------------------
# Data loaded once
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def verses_by_id() -> dict:
    """All verses as {id: verse}, loaded once."""
    with open(VERSES_PATH, encoding="utf-8") as f:
        return {v["id"]: v for v in json.load(f)}


@lru_cache(maxsize=1)
def _graph():
    return load_graph()


@lru_cache(maxsize=1)
def global_pagerank() -> dict:
    """PageRank of every verse, computed once."""
    return pagerank(_graph())


@lru_cache(maxsize=None)
def theme_pagerank(theme: str) -> dict:
    """Topic PageRank started from the theme's seed verses ({} for an unknown theme)."""
    if theme not in THEME_SEEDS:
        return {}
    return topic_pagerank(_graph(), THEME_SEEDS[theme])


# ---------------------------------------------------------------------------
# Search (steps 4-7)
# ---------------------------------------------------------------------------

def retrieve(
    english: str,
    concepts: list[str] | None = None,
    theme: str | None = None,
    use_bm25: bool = True,
    use_dense: bool = True,
    use_pagerank: bool = True,
    use_theme: bool = True,
    weights: dict | None = None,
    k: int = CANDIDATES,
) -> dict:
    """Search steps for an English question. Returns {"results", "bm25", "dense", "fused"}, each best first.

    concepts=None skips query rewriting; the use_* switches turn parts off for the ablation table.
    """
    extra = " " + " ".join(concepts) if concepts else ""
    lists = {"bm25": [], "dense": []}

    if use_bm25:
        tokens = clean_en(english + extra)
        lists["bm25"] = search_bm25(tokens, k=k, lang="en", zone="translation", k1=BM25_K1, b=BM25_B)
    if use_dense:
        lists["dense"] = search_dense(english + extra, k=k)

    ranked = [lst for lst in lists.values() if lst]
    if len(ranked) == 1:
        fused = ranked[0]  # one method only: nothing to fuse
    else:
        fused = rrf(ranked, top_n=k)

    pr = global_pagerank() if use_pagerank else None
    tp = theme_pagerank(theme) if use_theme and theme else None
    final = rerank(fused, pr, tp, weights) if (pr or tp) else fused

    return {"results": final, **lists, "fused": fused}


# ---------------------------------------------------------------------------
# The whole pipeline (steps 1-9)
# ---------------------------------------------------------------------------

def check_answer(question_en: str, verses: list[dict]) -> dict:
    """Answer from the given verses, then drop unsupported sentences.

    Returns {"text_en", "citations", "removed", "safety_note"}.
    """
    answer = explain(question_en, verses)
    checks = verify(answer["text"], verses)
    text_en = keep_supported(checks)
    return {
        "text_en": text_en,
        "citations": list(dict.fromkeys(CITATION.findall(text_en))),
        "removed": [c for c in checks if c["supported"] is False],
        "safety_note": answer["safety_note"],
    }


def ask(question: str, with_answer: bool = True, top_n: int = TOP_VERSES) -> dict:
    """Run every step. Returns the language, English and Hindi versions, theme, concepts,
    the top verses, every intermediate ranking ("search") and the checked "answer".
    """
    query = process_query(question)                       # 1
    english = query["english"]
    theme = classify_theme(english)["predicted_theme"]    # 2
    concepts = rewrite(english)                           # 3
    search = retrieve(english, concepts, theme)           # 4-7

    verses = verses_by_id()
    top = [verses[r["id"]] for r in search["results"][:top_n] if r["id"] in verses]

    return {
        "question": question,
        "lang": query["lang"],
        "english": english,
        "hindi": query["hindi"],
        "theme": theme,
        "concepts": concepts,
        "verses": top,
        "search": search,
        "answer": check_answer(english, top) if with_answer and top else None,  # 8-9
    }


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "I studied hard but still failed my exam. How do I cope?"
    out = ask(text)
    print(f"Language: {out['lang']}   English: {out['english']}")
    print(f"Theme: {out['theme']}   Concepts: {out['concepts']}")
    print("Top verses:", ", ".join(v["id"] for v in out["verses"]))
    if out["answer"]:
        print("\nAnswer:\n" + out["answer"]["text_en"])
        if out["answer"]["removed"]:
            print(f"\n({len(out['answer']['removed'])} sentence(s) removed by the fact-checker)")
        if out["answer"]["safety_note"]:
            print("\n" + out["answer"]["safety_note"])