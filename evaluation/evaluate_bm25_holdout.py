import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math

from preprocessing.clean_en import clean_en
from retrieval.bm25 import search_bm25


QUESTIONS_PATH = "data/test_set/questions_holdout.json"
QRELS_PATH = "data/test_set/qrels_holdout.json"

K = 20
K1 = 8.0  # provisional: update after re-running tune_bm25.py with the wider grid
B = 0.75


def ndcg_at_k(ranked_ids, grades, k=20):
    dcg = sum(
        (2 ** grades.get(vid, 0) - 1) / math.log2(rank + 1)
        for rank, vid in enumerate(ranked_ids[:k], start=1)
    )

    ideal = sorted(grades.values(), reverse=True)[:k]

    idcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, start=1)
    )

    return dcg / idcg if idcg else 0.0


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels = json.load(f)

    ids = sorted(set(questions) & set(qrels))

    if not ids:
        raise ValueError("No holdout questions found: run python evaluation/import_test_set.py")

    scores = []
    empty = 0

    print(f"Evaluating BM25: k1={K1}, b={B}, nDCG@{K}\n")

    for number, qid in enumerate(ids, start=1):
        tokens = clean_en(questions[qid]["en"])

        results = search_bm25(
            tokens,
            k=K,
            lang="en",
            zone="translation",
            k1=K1,
            b=B,
        )

        ranked_ids = [r["id"] for r in results]

        if not ranked_ids:
            empty += 1

        score = ndcg_at_k(ranked_ids, qrels[qid], K)
        scores.append(score)

        print(f"[{number}/{len(ids)}] {qid}: nDCG@20={score:.4f}")

    average = sum(scores) / len(scores)

    print("\n--- Holdout Results ---")
    print(f"Questions evaluated: {len(scores)}")
    print(f"Empty result sets: {empty}")
    print(f"Average nDCG@20: {average:.4f}")
    print(f"Parameters: k1={K1}, b={B}")


if __name__ == "__main__":
    main()
