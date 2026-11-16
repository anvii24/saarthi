
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, classification_report

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


def evaluate_method(texts, labels, method, k=3):
    """Evaluate a classifier using leave-one-out cross-validation."""
    actual = []
    predicted = []

    for held_out in range(len(texts)):
        training_texts = [
            text for i, text in enumerate(texts)
            if i != held_out
        ]

        training_labels = [
            label for i, label in enumerate(labels)
            if i != held_out
        ]

        model = train_classifier(
            training_texts,
            training_labels,
            method=method,
            k=k
        )

        prediction = predict_theme(model, texts[held_out])

        actual.append(labels[held_out])
        predicted.append(prediction)

    return actual, predicted


def main():
    texts, labels = load_questions()

    print("Total questions:", len(texts))
    print("Total themes:", len(set(labels)))

    # Keep the existing Naive Bayes and Rocchio results as references.
    for method in ("naive_bayes", "rocchio"):
        actual, predicted = evaluate_method(texts, labels, method)

        print(f"\n{method.upper()}")
        print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")

        print(classification_report(
            actual, predicted, zero_division=0
        ))

    # Compare different k values for kNN.
    print("\nKNN EXPERIMENT")
    for k in (1, 3, 5, 7):
        actual, predicted = evaluate_method(
            texts, labels, "knn", k=k
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
