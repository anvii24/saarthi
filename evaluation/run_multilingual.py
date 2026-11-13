import argparse
import csv
import json
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexing.embeddings import search_dense  
from query.language import process_query  

QUESTIONS = os.path.join("data", "test_set", "questions.json")
QRELS = os.path.join("data", "test_set", "qrels.json")
OUT_DIR = os.path.join("evaluation", "results")
K = 10
LANGS = ["en", "hi", "hinglish"]
STRATEGIES = ["direct", "translate_en", "hindi_side"]

# Metrics (stand-in for evaluation/metrics.py)

def precision_at_k(ranked, rel, k):
    return sum(1 for v in ranked[:k] if v in rel) / k


def recall_at_k(ranked, rel, k):
    return sum(1 for v in ranked[:k] if v in rel) / len(rel) if rel else 0.0


def reciprocal_rank(ranked, rel):
    return next((1 / i for i, v in enumerate(ranked, start=1) if v in rel), 0.0)


def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, v in enumerate(ranked, start=1):
        if v in rel:
            hits += 1
            total += hits / i
    return total / len(rel) if rel else 0.0


def ndcg_at_k(ranked, grades, k):
    dcg = sum((2 ** grades.get(v, 0) - 1) / math.log2(i + 1) for i, v in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def score(ranked, grades):
    rel = set(grades)
    return {
        "P@5": precision_at_k(ranked, rel, 5),
        "R@10": recall_at_k(ranked, rel, K),
        "MRR": reciprocal_rank(ranked, rel),
        "MAP": average_precision(ranked, rel),
        "nDCG@10": ndcg_at_k(ranked, grades, K),
    }


METRICS = ["P@5", "R@10", "MRR", "MAP", "nDCG@10"]

# Experiment

def run_query(text, lang, strategy):
    """Return (ranked verse ids, the text actually searched)."""
    if strategy == "direct":
        searched, side = text, None
    elif strategy == "translate_en":
        searched, side = (text if lang == "en" else process_query(text)["english"]), "en"
    else:  # hindi_side
        searched, side = (text if lang == "hi" else process_query(text)["hindi"]), "hi"
    return [r["id"] for r in search_dense(searched, k=K, lang=side)], searched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="q000")
    parser.add_argument("--to", dest="end", default="q999")
    args = parser.parse_args()

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)
    with open(QRELS, encoding="utf-8") as f:
        qrels = json.load(f)

    ids = [q for q in sorted(qrels) if args.start <= q <= args.end and q in questions]
    if not ids:
        sys.exit("No labelled questions found: run python evaluation/import_test_set.py first")
    print(f"Evaluating {len(ids)} labelled questions x {len(LANGS)} languages x {len(STRATEGIES)} strategies")

    per_query, failed_conversions = [], 0
    for n, qid in enumerate(ids, start=1):
        print(f"[{n}/{len(ids)}] {qid}")
        for lang in LANGS:
            text = questions[qid].get(lang, "").strip()
            if not text:
                continue
            for strategy in STRATEGIES:
                ranked, searched = run_query(text, lang, strategy)
                # if the LLM conversion failed, process_query returns the text unchanged
                if strategy != "direct" and searched == text and lang != {"translate_en": "en", "hindi_side": "hi"}[strategy]:
                    failed_conversions += 1
                row = {"id": qid, "lang": lang, "strategy": strategy, "top5": " ".join(ranked[:5])}
                row.update({m: round(v, 4) for m, v in score(ranked, qrels[qid]).items()})
                per_query.append(row)

    # average per language x strategy
    summary = []
    for lang in LANGS:
        for strategy in STRATEGIES:
            rows = [r for r in per_query if r["lang"] == lang and r["strategy"] == strategy]
            if rows:
                avg = {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS}
                summary.append({"lang": lang, "strategy": strategy, "n": len(rows), **avg})

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, rows in [(os.path.join(OUT_DIR, "multilingual.csv"), summary),
                       (os.path.join(OUT_DIR, "multilingual_per_query.csv"), per_query)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{'language':<10}{'strategy':<14}" + "".join(f"{m:>9}" for m in METRICS))
    for s in summary:
        print(f"{s['lang']:<10}{s['strategy']:<14}" + "".join(f"{s[m]:>9.3f}" for m in METRICS))
    if failed_conversions:
        print(f"\nWARNING: {failed_conversions} LLM conversions failed (rate limit or no key); "
              f"those rows used the untranslated text. Rerun later: answers are cached.")

    make_chart(summary)
    print(f"\nSaved {OUT_DIR}/multilingual.csv, multilingual_per_query.csv and multilingual.png")


def make_chart(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"direct": "Direct (all translations)", "translate_en": "Translate to English first",
              "hindi_side": "Convert to Hindi first"}
    colors = {"direct": "#2a6f97", "translate_en": "#8ab17d", "hindi_side": "#e9c46a"}
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.26
    for i, strategy in enumerate(STRATEGIES):
        values = [next((s["nDCG@10"] for s in summary if s["lang"] == l and s["strategy"] == strategy), 0)
                  for l in LANGS]
        xs = [x + (i - 1) * width for x in range(len(LANGS))]
        bars = ax.bar(xs, values, width, label=labels[strategy], color=colors[strategy])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(range(len(LANGS)), ["English", "Hindi", "Hinglish"])
    ax.set_ylabel("nDCG@10 (higher is better)")
    ax.set_title("Meaning search by question language and strategy")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "multilingual.png"), dpi=200)


if __name__ == "__main__":
    main()