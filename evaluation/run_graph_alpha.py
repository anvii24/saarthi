
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.embeddings import search_dense
from indexing.link_analysis import load_graph, pagerank, topic_pagerank
from indexing.theme_seeds import THEME_SEEDS
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
)

ALPHAS = [0.5, 0.7, 0.8, 0.9, 1.0]
K = 5


def normalize_scores(scores):
    if not scores:
        return {}

    minimum = min(scores.values())
    maximum = max(scores.values())

    if maximum == minimum:
        return {verse_id: 1.0 for verse_id in scores}

    return {
        verse_id: (score - minimum) / (maximum - minimum)
        for verse_id, score in scores.items()
    }


def combine_scores(dense_scores, graph_scores, alpha):
    dense_normalized = normalize_scores(dense_scores)
    graph_normalized = normalize_scores(graph_scores)

    return {
        verse_id: (
            alpha * dense_normalized[verse_id]
            + (1 - alpha) * graph_normalized.get(verse_id, 0.0)
        )
        for verse_id in dense_scores
    }


def evaluate(ranked, relevant):
    return {
        "P@5": precision_at_k(ranked, relevant.keys(), K),
        "Recall@5": recall_at_k(ranked, relevant.keys(), K),
        "MAP": mean_average_precision([ranked], [relevant.keys()]),
        "MRR": mrr(ranked, relevant.keys()),
        "nDCG@5": ndcg_at_k(ranked, relevant, K),
    }


def main():
    with open("data/test_set/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    with open("data/test_set/qrels.json", encoding="utf-8") as f:
        qrels = json.load(f)

    graph = load_graph()
    global_pr = pagerank(graph)

    topic_pr = {
        theme: topic_pagerank(graph, seeds)
        for theme, seeds in THEME_SEEDS.items()
    }

    results = []

    for alpha in ALPHAS:
        print(f"\nEvaluating alpha={alpha}")

        all_metrics = []

        for question_id, relevant in qrels.items():
            question = questions[question_id]

            dense_results = search_dense(question["en"], k=20)
            dense_scores = {
                result["id"]: result["score"]
                for result in dense_results
            }

            graph_scores = topic_pr.get(
                question["theme"], global_pr
            )

            combined = combine_scores(
                dense_scores, graph_scores, alpha
            )

            ranked = sorted(
                combined,
                key=lambda verse_id: combined[verse_id],
                reverse=True,
            )

            all_metrics.append(evaluate(ranked, relevant))

        row = {"alpha": alpha}

        for metric in all_metrics[0]:
            row[metric] = sum(
                result[metric] for result in all_metrics
            ) / len(all_metrics)

        results.append(row)
        print(row)

    os.makedirs("evaluation/results", exist_ok=True)

    output_path = "evaluation/results/graph_alpha.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved alpha comparison to {output_path}")


if __name__ == "__main__":
    main()
