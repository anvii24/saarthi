import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from query.classifier import (
    calculate_theme_scores,
    create_vectorizer,
    load_training_data,
    predict_theme,
    train_model,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "test_set" / "questions.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "classifiers.csv"


def load_test_questions():
    """The 100 test questions (English) and their themes."""
    with open(TEST_PATH, encoding="utf-8") as file:
        questions = json.load(file)

    pairs = [
        (q.get("en", "").strip(), q.get("theme", "").strip())
        for q in questions.values()
    ]
    pairs = [(text, theme) for text, theme in pairs if text and theme]
    return [text for text, _ in pairs], [theme for _, theme in pairs]


def predict_similarity(train_texts, train_labels, test_texts, method):
    """TF-IDF mean / max: score each theme by similarity to its training questions."""
    vectorizer = create_vectorizer()
    matrix = vectorizer.fit_transform(train_texts)

    predictions = []
    for text in test_texts:
        similarities = cosine_similarity(vectorizer.transform([text]), matrix).flatten()
        scores = calculate_theme_scores(similarities, train_labels, method=method)
        predictions.append(max(scores, key=scores.get))
    return predictions


def main():
    train_texts, train_labels = load_training_data()
    test_texts, test_labels = load_test_questions()

    print(f"Training questions: {len(train_texts)}   Test questions: {len(test_texts)}   "
          f"Themes: {len(set(test_labels))}\n")

    runs = {
        "TF-IDF mean": predict_similarity(train_texts, train_labels, test_texts, "mean"),
        "TF-IDF max": predict_similarity(train_texts, train_labels, test_texts, "max"),
    }
    for name, method, k in (("Naive Bayes", "naive_bayes", 3), ("Rocchio", "rocchio", 3),
                            ("kNN k=1", "knn", 1), ("kNN k=3", "knn", 3),
                            ("kNN k=5", "knn", 5), ("kNN k=7", "knn", 7)):
        model = train_model(train_texts, train_labels, method=method, k=k)
        runs[name] = [predict_theme(model, text) for text in test_texts]

    rows = []
    for name, predicted in runs.items():
        rows.append({
            "method": name,
            "correct": sum(a == p for a, p in zip(test_labels, predicted)),
            "questions": len(test_labels),
            "accuracy": round(accuracy_score(test_labels, predicted), 3),
            "macro_f1": round(f1_score(test_labels, predicted, average="macro", zero_division=0), 3),
        })

    print(f"{'method':<14}{'correct':>10}{'accuracy':>10}{'macro F1':>10}")
    for row in rows:
        print(f"{row['method']:<14}{row['correct']:>6}/{row['questions']:<3}"
              f"{row['accuracy']:>10.3f}{row['macro_f1']:>10.3f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {OUTPUT_PATH.relative_to(ROOT)}")

    best = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))["method"]
    print(f"\nPer-theme report for the best method ({best}):")
    print(classification_report(test_labels, runs[best], zero_division=0))

    wrong = [(t, a, p) for t, a, p in zip(test_texts, test_labels, runs["TF-IDF mean"]) if a != p]
    print(f"Questions TF-IDF mean got wrong ({len(wrong)}):")
    for text, actual, predicted in wrong:
        print(f"  - {text}\n      actual: {actual}   predicted: {predicted}")


if __name__ == "__main__":
    main()