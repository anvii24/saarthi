"""Choose BM25's k1 and b on the 20 tuning questions (nDCG@20).

Run: python evaluation/tune_bm25.py
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json

from evaluation.metrics import ndcg_at_k
from preprocessing.clean_en import clean_en
from retrieval.bm25 import search_bm25


QUESTIONS_PATH = "data/test_set/questions_tuning.json"
QRELS_PATH = "data/test_set/qrels_tuning.json"
OUTPUT_PATH = "evaluation/results/bm25_tuning.csv"

K = 20

# Parameter combinations to compare
K1_VALUES = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
B_VALUES = [0.5, 0.75, 1.0]


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    with open(QRELS_PATH, encoding="utf-8") as f:
        qrels = json.load(f)

    question_ids = sorted(set(questions) & set(qrels))

    if len(question_ids) != 20:
        raise ValueError(
            f"Expected 20 labelled tuning questions, found {len(question_ids)}"
        )

    results = []

    total = len(K1_VALUES) * len(B_VALUES)
    experiment = 0

    for k1 in K1_VALUES:
        for b in B_VALUES:
            experiment += 1
            scores = []
            empty_results = 0

            for qid in question_ids:
                query = questions[qid]["en"]
                tokens = clean_en(query)

                ranked = search_bm25(
                    tokens,
                    k=K,
                    lang="en",
                    zone="translation",
                    k1=k1,
                    b=b,
                )

                ranked_ids = [result["id"] for result in ranked]

                if not ranked_ids:
                    empty_results += 1

                scores.append(
                    ndcg_at_k(ranked_ids, qrels[qid], K)
                )

            average_score = sum(scores) / len(scores)

            results.append({
                "k1": k1,
                "b": b,
                "nDCG@20": round(average_score, 6),
                "questions": len(scores),
                "empty_results": empty_results,
            })

            print(
                f"[{experiment}/{total}] "
                f"k1={k1}, b={b}, "
                f"nDCG@20={average_score:.4f}, "
                f"empty={empty_results}"
            )

    results.sort(
        key=lambda row: (-row["nDCG@20"], row["k1"], row["b"])
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 5 BM25 configurations:")

    for result in results[:5]:
        print(
            f"k1={result['k1']}, "
            f"b={result['b']}, "
            f"nDCG@20={result['nDCG@20']:.4f}"
        )

    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
