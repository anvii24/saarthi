import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, classification_report

from query.classifier import load_training_data
from query.classifiers import train_classifier, predict_theme


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data/test_set/questions.json"
)


def load_questions():
    """Load labelled English questions."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    texts = []
    labels = []

    for question in questions.values():
        text = question.get("en", "").strip()
        theme = question.get("theme", "").strip()

        if text and theme:
            texts.append(text)
            labels.append(theme)

    return texts, labels


def evaluate_method(train_texts, train_labels, test_texts, test_labels, method, k=3):
    """Train once on the separate training questions, score the test questions."""
    model = train_classifier(
        train_texts,
        train_labels,
        method=method,
        k=k
    )

    predicted = [predict_theme(model, text) for text in test_texts]

    return list(test_labels), predicted


def main():
    train_texts, train_labels = load_training_data()
    texts, labels = load_questions()

    print("Training questions:", len(train_texts))
    print("Test questions:", len(texts))
    print("Total themes:", len(set(labels)))

    # Keep the existing Naive Bayes and Rocchio results as references.
    for method in ("naive_bayes", "rocchio"):
        actual, predicted = evaluate_method(
            train_texts, train_labels, texts, labels, method
        )

        print(f"\n{method.upper()}")
        print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")

        print(classification_report(
            actual, predicted, zero_division=0
        ))

    # Compare different k values for kNN.
    print("\nKNN EXPERIMENT")
    for k in (1, 3, 5, 7):
        actual, predicted = evaluate_method(
            train_texts, train_labels, texts, labels, "knn", k=k
        )

        print(
            f"k={k} | "
            f"Accuracy={accuracy_score(actual, predicted):.4f}"
        )

        print(classification_report(
            actual, predicted, zero_division=0
        ))


if __name__ == "__main__":
    main()
