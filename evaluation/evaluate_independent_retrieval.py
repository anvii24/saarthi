import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import math

from preprocessing.clean_en import clean_en
from retrieval.bm25 import search_bm25
from retrieval.tfidf import search_tfidf


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


def evaluate(method, questions, qrels):
    scores = []
    empty = 0

    for qid in sorted(set(questions) & set(qrels)):
        tokens = clean_en(questions[qid]["en"])

        if method == "TF-IDF":
            results = search_tfidf(
                tokens, k=K, lang="en", zone="translation"
            )
        else:
            results = search_bm25(
                tokens,
                k=K,
                lang="en",
                zone="translation",
                k1=K1,
                b=B,
            )

        ranked_ids = [result["id"] for result in results]

        if not ranked_ids:
            empty += 1

        scores.append(ndcg_at_k(ranked_ids, qrels[qid], K))

    return scores, empty



def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels = json.load(f)

    ids = set(questions) & set(qrels)

    if not ids:
        raise ValueError("No matching questions and qrels.")

    print(f"Independent questions with relevance labels: {len(ids)}")
    print(f"Metric: nDCG@{K}\n")

    rows = []

    for method in ("TF-IDF", "BM25"):
        scores, empty = evaluate(method, questions, qrels)
        average = sum(scores) / len(scores)

        print(f"--- {method} ---")
        print(f"Questions evaluated: {len(scores)}")
        print(f"Empty result sets: {empty}")
        print(f"Average nDCG@{K}: {average:.4f}\n")

        rows.append({
            "method": method,
            "questions": len(scores),
            "empty_results": empty,
            "nDCG@20": round(average, 4),
            "k1": K1 if method == "BM25" else "",
            "b": B if method == "BM25" else "",
        })

    output_path = "evaluation/results/independent_retrieval.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"BM25 parameters: k1={K1}, b={B}")
    print(f"Results saved to: {output_path}")



if __name__ == "__main__":
    main()
