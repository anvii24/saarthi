"""Plain RAG (meaning search -> LLM, nothing checked) vs full Saarthi on the 80 holdout
questions: verse quality, citation relevance, and unsupported sentences shown to users.

The fact-checker scores both; it agreed with hand checks on 97% of sentences
(evaluation/results/rag_checker_review.csv). Run: python evaluation/baseline.py
"""

import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import ndcg_at_k, precision_at_k  # noqa: E402
from generation.fact_check import CITATION, keep_supported, verify  # noqa: E402
from generation.rag import explain  # noqa: E402
from indexing.embeddings import search_dense  # noqa: E402
from pipeline import retrieve, verses_by_id  # noqa: E402
from query.classifier import classify_theme  # noqa: E402
from query.rewrite import rewrite  # noqa: E402

QUESTIONS = os.path.join("data", "test_set", "questions_holdout.json")
QRELS = os.path.join("data", "test_set", "qrels_holdout.json")
OUT_DIR = os.path.join("evaluation", "results")
TOP_N = 5


def plain_rag(question: str, verses: dict) -> dict:
    """Meaning search on the raw question, v1 answer, no fact-check."""
    ids = [r["id"] for r in search_dense(question, k=TOP_N)]
    given = [verses[v] for v in ids]
    answer = explain(question, given, prompt_version="v1")
    checks = verify(answer["text"], given)
    return {"retrieved": ids, "answer": answer["text"], "checks": checks,
            "shown": checks}  # nothing is removed: the user sees every sentence


def saarthi(question: str, verses: dict) -> dict:
    """Full Saarthi search and answer (the English question, so no translation step)."""
    theme = classify_theme(question)["predicted_theme"]
    concepts = rewrite(question)
    ids = [r["id"] for r in retrieve(question, concepts, theme)["results"][:TOP_N]]
    given = [verses[v] for v in ids]
    answer = explain(question, given, prompt_version="v2")
    checks = verify(answer["text"], given)
    shown = [c for c in checks if c["supported"] is not False]
    return {"retrieved": ids, "theme": theme, "concepts": concepts, "answer": answer["text"],
            "checks": checks, "shown": shown, "shown_text": keep_supported(checks)}


def pct(part: float, whole: float) -> float:
    """Percentage rounded to 1 decimal (0 if whole is 0)."""
    return round(100 * part / whole, 1) if whole else 0.0


def summarise(runs: list[dict], qrels: dict) -> dict:
    """Average every measure over all questions for one system."""
    cited = [c for r in runs for c in r["checks"] if c["verse_ids"]]
    supported = [c for c in cited if c["supported"]]
    cited_verses = [(r["id"], v) for r in runs for v in dict.fromkeys(CITATION.findall(r["answer"]))]
    unsupported_shown = [sum(1 for c in r["shown"] if c["supported"] is False) for r in runs]
    n = len(runs)
    return {
        "P@5": round(sum(precision_at_k(r["retrieved"], set(qrels[r["id"]]), TOP_N) for r in runs) / n, 3),
        "nDCG@5": round(sum(ndcg_at_k(r["retrieved"], qrels[r["id"]], TOP_N) for r in runs) / n, 3),
        "citation_relevance_pct": pct(sum(1 for q, v in cited_verses if v in qrels[q]), len(cited_verses)),
        "cited_sentences": len(cited),
        "supported_pct": pct(len(supported), len(cited)),
        "unsupported_shown_per_answer": round(sum(unsupported_shown) / n, 2),
        "answers_with_no_unsupported_shown_pct": pct(sum(1 for u in unsupported_shown if u == 0), n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="q000")
    parser.add_argument("--to", dest="end", default="q999")
    args = parser.parse_args()

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)
    with open(QRELS, encoding="utf-8") as f:
        qrels = json.load(f)
    verses = verses_by_id()

    ids = [q for q in sorted(qrels) if args.start <= q <= args.end and q in questions]
    if not ids:
        sys.exit("No holdout questions found: run python evaluation/import_test_set.py first")
    print(f"Plain RAG vs Saarthi on {len(ids)} holdout questions")

    runs = {"plain_rag": [], "saarthi": []}
    for n, qid in enumerate(ids, start=1):
        question = questions[qid]["en"]
        print(f"[{n}/{len(ids)}] {qid}")
        runs["plain_rag"].append({"id": qid, "question": question, **plain_rag(question, verses)})
        runs["saarthi"].append({"id": qid, "question": question, **saarthi(question, verses)})

    table = {name: summarise(r, qrels) for name, r in runs.items()}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "baseline.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "plain_rag", "saarthi"])
        for metric in table["plain_rag"]:
            writer.writerow([metric, table["plain_rag"][metric], table["saarthi"][metric]])
    with open(os.path.join(OUT_DIR, "baseline_answers.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)

    print(f"\n{'metric':<40}{'plain RAG':>12}{'Saarthi':>12}")
    for metric in table["plain_rag"]:
        print(f"{metric:<40}{table['plain_rag'][metric]:>12}{table['saarthi'][metric]:>12}")
    print(f"\nSaved {OUT_DIR}/baseline.csv and baseline_answers.json")


if __name__ == "__main__":
    main()