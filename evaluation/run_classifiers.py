import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.clean_en import clean_en
from query.classifier import calculate_theme_scores, load_training_data


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


def evaluate_method(train_texts, train_labels, test_texts, test_labels, method):
    """Train on the separate training questions, score the test questions."""
    actual = []
    predicted = []
    misclassified = []

    vectorizer = TfidfVectorizer(
        tokenizer=clean_en,
        token_pattern=None,
        lowercase=False
    )

    training_matrix = vectorizer.fit_transform(train_texts)

    for text, label in zip(test_texts, test_labels):
        query_vector = vectorizer.transform([text])

        similarities = cosine_similarity(
            query_vector,
            training_matrix
        ).flatten()

        scores = calculate_theme_scores(
            similarities,
            train_labels,
            method=method
        )

        predicted_theme = max(scores, key=scores.get)

        actual.append(label)
        predicted.append(predicted_theme)

        if predicted_theme != label:
            top_scores = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            misclassified.append({
                "question": text,
                "actual": label,
                "predicted": predicted_theme,
                "top_scores": top_scores
            })

    return actual, predicted, misclassified


def main():
    train_texts, train_labels = load_training_data()
    texts, labels = load_questions()

    print("Training questions:", len(train_texts))
    print("Test questions:", len(texts))
    print("Total themes:", len(set(labels)))

    mean_misclassified = []

    for method in ("mean", "max"):
        actual, predicted, misclassified = evaluate_method(
            train_texts,
            train_labels,
            texts,
            labels,
            method
        )

        correct = sum(
            a == p for a, p in zip(actual, predicted)
        )

        print("\n" + "=" * 50)
        print("SCORING METHOD:", method.upper())
        print("=" * 50)

        print(f"Correct: {correct}/{len(actual)}")
        print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")

        print("\nPer-theme report:")
        print(
            classification_report(
                actual,
                predicted,
                zero_division=0
            )
        )

        if method == "mean":
            mean_misclassified = misclassified

    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS (MEAN SCORING)")
    print("=" * 60)

    print("Total misclassified:", len(mean_misclassified))

    for index, item in enumerate(mean_misclassified, start=1):
        print(f"\n{index}. Question: {item['question']}")
        print("   Actual theme:   ", item["actual"])
        print("   Predicted theme:", item["predicted"])
        print("   Top 3 theme scores:")

        for theme, score in item["top_scores"]:
            print(f"      {theme}: {score:.4f}")


if __name__ == "__main__":
    main()
