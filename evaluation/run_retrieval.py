"""
Main retrieval results table (Person 2: Kashvi).

Runs 8 setups, each adding one part of Saarthi, and scores them on the holdout
questions with P@5, R@10, MRR, MAP and nDCG@10 (evaluation/metrics.py).

    1 TF-IDF, question only
    2 BM25, question only
    3 BM25 + Gita concepts (query rewriting)
    4 Meaning search (dense), question only
    5 Meaning search + concepts
    6 BM25 + meaning search fused with RRF + concepts
    7 + PageRank
    8 + topic PageRank = full Saarthi

Two steps, from the repo root:

    python evaluation/run_retrieval.py --tune   # 1. pick re-rank weights on the TUNING questions
    # copy the printed WEIGHTS line into retrieval/rerank.py
    python evaluation/run_retrieval.py          # 2. results table on the HOLDOUT questions

Before running:
    python indexing/embeddings.py   (once: builds data/processed/embeddings.npy)
    the LLM key in .env             (query rewriting; answers are cached in data/processed/llm_cache.json)

Outputs (evaluation/results/):
    retrieval.csv                  the main table
    retrieval_per_question.csv     nDCG@10 of every setup for every question
    retrieval.png                  bar chart of nDCG@10
    retrieval_weight_tuning.csv    every weight combination tried (--tune)
"""

import argparse
import csv
import itertools
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import average_precision, mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from indexing.embeddings import VECTORS_PATH  # noqa: E402
from pipeline import CANDIDATES, global_pagerank, retrieve, theme_pagerank  # noqa: E402
from preprocessing.clean_en import clean_en  # noqa: E402
from query.classifier import classify_theme  # noqa: E402
from query.rewrite import rewrite  # noqa: E402
from retrieval.rerank import WEIGHTS, rerank  # noqa: E402
from retrieval.tfidf import search_tfidf  # noqa: E402

TEST_DIR = os.path.join("data", "test_set")
RESULTS_DIR = os.path.join("evaluation", "results")

METRICS = ["P@5", "R@10", "MRR", "MAP", "nDCG@10"]

# Weight grid for the re-ranker (tuned on the tuning questions only)
FUSED_VALUES = [0.6, 0.7, 0.8, 0.9, 1.0]
PAGERANK_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3]
THEME_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(name: str) -> list[dict]:
    """Questions of one split ('tuning' or 'holdout') that have relevance labels."""
    with open(os.path.join(TEST_DIR, f"questions_{name}.json"), encoding="utf-8") as f:
        questions = json.load(f)
    with open(os.path.join(TEST_DIR, f"qrels_{name}.json"), encoding="utf-8") as f:
        qrels = json.load(f)

    items = [
        {"qid": qid, "english": questions[qid]["en"], "relevant": relevant}
        for qid, relevant in qrels.items()
        if qid in questions and relevant
    ]
    if not items:
        raise ValueError(f"No labelled {name} questions found in {TEST_DIR}")
    return items


def prepare(items: list[dict]) -> list[dict]:
    """Add the Gita concepts (query rewriting) and the predicted theme to each question."""
    print(f"Preparing {len(items)} questions (query rewriting + theme classifier) ...")
    for item in items:
        item["concepts"] = rewrite(item["english"])
        item["theme"] = classify_theme(item["english"])["predicted_theme"]

    empty = sum(1 for item in items if not item["concepts"])
    if empty:
        print(
            f"WARNING: query rewriting returned no concepts for {empty} of {len(items)} questions "
            "(LLM key missing or call failed). Rows 3, 5, 6, 7 and 8 will be wrong. "
            "Fix the key in .env and run again."
        )
    return items


def check_embeddings() -> None:
    if not os.path.exists(VECTORS_PATH):
        sys.exit("No embeddings yet. Run:  python indexing/embeddings.py   (a few minutes on CPU), then try again.")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(ranked_ids: list[str], relevant: dict) -> dict:
    """All five metrics for one question. `relevant` is {verse id: grade}."""
    return {
        "P@5": precision_at_k(ranked_ids, relevant.keys(), 5),
        "R@10": recall_at_k(ranked_ids, relevant.keys(), 10),
        "MRR": mrr(ranked_ids, relevant.keys()),
        "MAP": average_precision(ranked_ids, relevant.keys()),
        "nDCG@10": ndcg_at_k(ranked_ids, relevant, 10),
    }


def mean_scores(per_question: list[dict]) -> dict:
    return {m: sum(s[m] for s in per_question) / len(per_question) for m in METRICS}


def ids(results: list[dict]) -> list[str]:
    return [r["id"] for r in results]


# ---------------------------------------------------------------------------
# The 8 setups. Each takes a prepared question and returns a ranked result list.
# ---------------------------------------------------------------------------

def _retrieve(q: dict, concepts: bool, **switches) -> list[dict]:
    return retrieve(q["english"], q["concepts"] if concepts else None, q["theme"], **switches)["results"]


SETUPS = [
    ("1 TF-IDF, question only",
     lambda q: search_tfidf(clean_en(q["english"]), k=CANDIDATES)),
    ("2 BM25, question only",
     lambda q: _retrieve(q, False, use_dense=False, use_pagerank=False, use_theme=False)),
    ("3 BM25 + Gita concepts",
     lambda q: _retrieve(q, True, use_dense=False, use_pagerank=False, use_theme=False)),
    ("4 Meaning search, question only",
     lambda q: _retrieve(q, False, use_bm25=False, use_pagerank=False, use_theme=False)),
    ("5 Meaning search + concepts",
     lambda q: _retrieve(q, True, use_bm25=False, use_pagerank=False, use_theme=False)),
    ("6 BM25 + meaning (RRF) + concepts",
     lambda q: _retrieve(q, True, use_pagerank=False, use_theme=False)),
    ("7 + PageRank",
     lambda q: _retrieve(q, True, use_theme=False)),
    ("8 + topic PageRank (full Saarthi)",
     lambda q: _retrieve(q, True)),
]


# ---------------------------------------------------------------------------
# Step 1: tune the re-rank weights on the tuning questions
# ---------------------------------------------------------------------------

def tune_weights() -> None:
    items = prepare(load_split("tuning"))
    print(f"Tuning the re-rank weights on {len(items)} tuning questions (nDCG@10) ...")

    # Search once per question with re-ranking off, then re-rank for every weight combination.
    pr = global_pagerank()
    prepared = []
    for q in items:
        fused = retrieve(q["english"], q["concepts"], q["theme"], use_pagerank=False, use_theme=False)["fused"]
        prepared.append((q, fused, theme_pagerank(q["theme"]) or None))

    def mean_ndcg(weights: dict | None) -> float:
        total = 0.0
        for q, fused, tp in prepared:
            ranked = ids(rerank(fused, pr, tp, weights)) if weights else ids(fused)
            total += ndcg_at_k(ranked, q["relevant"], 10)
        return total / len(prepared)

    rows = []
    for a, b, c in itertools.product(FUSED_VALUES, PAGERANK_VALUES, THEME_VALUES):
        rows.append({"fused": a, "pagerank": b, "theme": c,
                     "nDCG@10": mean_ndcg({"fused": a, "pagerank": b, "theme": c})})

    # Best nDCG first. Only the ratios between the weights change the ranking, so many
    # combinations tie: among ties prefer the smaller re-ranking influence, then fused = 1.0.
    rows.sort(key=lambda r: (-r["nDCG@10"], r["pagerank"] + r["theme"], -r["fused"]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "retrieval_weight_tuning.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fused", "pagerank", "theme", "nDCG@10"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWithout PageRank or theme (fused only): nDCG@10 = {mean_ndcg(None):.4f}")
    print(f"Current WEIGHTS in retrieval/rerank.py {WEIGHTS}: nDCG@10 = {mean_ndcg(dict(WEIGHTS)):.4f}")
    print("\nTop 5 weight combinations on the tuning questions:")
    for r in rows[:5]:
        print(f"  fused={r['fused']}, pagerank={r['pagerank']}, theme={r['theme']}  nDCG@10={r['nDCG@10']:.4f}")

    best = rows[0]
    print(f"\nAll {len(rows)} combinations saved to {path}")
    print("\nPut this line into retrieval/rerank.py, then run the table without --tune:")
    print(f'WEIGHTS = {{"fused": {best["fused"]}, "pagerank": {best["pagerank"]}, "theme": {best["theme"]}}}')
    print("\nThe differences between the top combinations are small on 20 questions; "
          "if the top rows are within about 0.01, treat them as equal.")


# ---------------------------------------------------------------------------
# Step 2: the results table on the holdout questions
# ---------------------------------------------------------------------------

def save_chart(rows: list[dict], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: skipped the chart")
        return

    names = [r["setup"] for r in rows][::-1]
    values = [r["nDCG@10"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color="#3b6ea5")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3)
    ax.set_xlabel("nDCG@10 (holdout questions)")
    ax.set_title("What each part of Saarthi adds")
    ax.set_xlim(0, max(values) * 1.15 if max(values) > 0 else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_table() -> None:
    check_embeddings()
    items = prepare(load_split("holdout"))
    print(f"\nHoldout questions: {len(items)}")
    print(f"Re-rank weights in use (rows 7 and 8): {WEIGHTS}\n")

    summary, per_question_rows = [], {q["qid"]: {"qid": q["qid"]} for q in items}

    for name, run in SETUPS:
        per_question = []
        for q in items:
            ranked = ids(run(q))
            s = score(ranked, q["relevant"])
            per_question.append(s)
            per_question_rows[q["qid"]][name] = round(s["nDCG@10"], 4)

        row = {"setup": name, **mean_scores(per_question)}
        row["questions_with_zero_ndcg"] = sum(1 for s in per_question if s["nDCG@10"] == 0)
        summary.append(row)
        print(f"{name:<38} " + "  ".join(f"{m}={row[m]:.4f}" for m in METRICS)
              + f"  zero={row['questions_with_zero_ndcg']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    table_path = os.path.join(RESULTS_DIR, "retrieval.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setup", *METRICS, "questions_with_zero_ndcg"])
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    per_question_path = os.path.join(RESULTS_DIR, "retrieval_per_question.csv")
    with open(per_question_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", *[name for name, _ in SETUPS]])
        writer.writeheader()
        writer.writerows(per_question_rows.values())

    chart_path = os.path.join(RESULTS_DIR, "retrieval.png")
    save_chart(summary, chart_path)

    print(f"\nSaved {table_path}, {per_question_path} and {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval ablation table for the report")
    parser.add_argument("--tune", action="store_true", help="tune the re-rank weights on the tuning questions")
    args = parser.parse_args()

    if args.tune:
        check_embeddings()
        tune_weights()
    else:
        run_table()


if __name__ == "__main__":
    main()
