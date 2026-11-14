
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.embeddings import search_dense
from indexing.link_analysis import load_graph, pagerank, topic_pagerank, hits
from indexing.theme_seeds import THEME_SEEDS
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
)

DATA_DIR = "data"
TEST_DIR = "data/test_set"
RESULTS_DIR = "evaluation/results"
GRAPH_SCORES_PATH = "data/processed/graph_scores.json"
RESULTS_PATH = "evaluation/results/graph.csv"

ALPHA = 0.8
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


def combine_scores(dense_scores, graph_scores):
    dense_normalized = normalize_scores(dense_scores)
    graph_normalized = normalize_scores(graph_scores)

    combined = {}

    for verse_id in dense_scores:
        combined[verse_id] = (
            ALPHA * dense_normalized[verse_id]
            + (1 - ALPHA) * graph_normalized.get(verse_id, 0.0)
        )

    return combined


def rank_scores(scores):
    return sorted(scores, key=lambda verse_id: scores[verse_id], reverse=True)


def evaluate_ranking(ranked, relevant):
    return {
        "P@5": precision_at_k(ranked, relevant.keys(), K),
        "Recall@5": recall_at_k(ranked, relevant.keys(), K),
        "MAP": mean_average_precision([ranked], [relevant.keys()]),
        "MRR": mrr(ranked, relevant.keys()),
        "nDCG@5": ndcg_at_k(ranked, relevant, K),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    with open(f"{TEST_DIR}/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    with open(f"{TEST_DIR}/qrels.json", encoding="utf-8") as f:
        qrels = json.load(f)

    graph = load_graph()
    print(f"Graph loaded: {len(graph)} verses")

    print("Computing graph scores...")
    global_pr = pagerank(graph)
    hubs, authorities = hits(graph)

    theme_pr = {
        theme: topic_pagerank(graph, seeds)
        for theme, seeds in THEME_SEEDS.items()
    }

    graph_scores = {
        "pagerank": global_pr,
        "hits_authority": authorities,
        "hits_hub": hubs,
        "topic_pagerank": theme_pr,
    }

    with open(GRAPH_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_scores, f, ensure_ascii=False, indent=2)

    print(f"Saved graph scores to {GRAPH_SCORES_PATH}")

    methods = ["dense", "pagerank", "topic_pagerank", "hits_authority"]
    all_results = {method: [] for method in methods}

    for question_id, relevant in qrels.items():
        question = questions[question_id]
        query = question["en"]
        theme = question["theme"]

        print(f"Evaluating {question_id}: {query}")

        dense_results = search_dense(query, k=20)
        dense_scores = {result["id"]: result["score"] for result in dense_results}

        rankings = {
            "dense": rank_scores(dense_scores),
            "pagerank": rank_scores(
                combine_scores(dense_scores, global_pr)
            ),
            "topic_pagerank": rank_scores(
                combine_scores(dense_scores, theme_pr.get(theme, global_pr))
            ),
            "hits_authority": rank_scores(
                combine_scores(dense_scores, authorities)
            ),
        }

        for method, ranked in rankings.items():
            all_results[method].append(
                evaluate_ranking(ranked, relevant)
            )

    rows = []

    for method, results in all_results.items():
        row = {"method": method}

        for metric in results[0]:
            row[metric] = sum(result[metric] for result in results) / len(results)

        rows.append(row)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nGraph evaluation complete")
    print(f"Results saved to {RESULTS_PATH}")

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
