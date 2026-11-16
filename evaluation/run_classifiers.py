
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.clean_en import clean_en
from query.classifier import calculate_theme_scores


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


def evaluate_method(texts, labels, method):
    """Evaluate using leave-one-out cross-validation."""
    actual = []
    predicted = []
    misclassified = []

    for held_out in range(len(texts)):
        training_texts = [
            text for i, text in enumerate(texts)
            if i != held_out
        ]

        training_labels = [
            label for i, label in enumerate(labels)
            if i != held_out
        ]

        vectorizer = TfidfVectorizer(
            tokenizer=clean_en,
            token_pattern=None,
            lowercase=False
        )

        training_matrix = vectorizer.fit_transform(training_texts)
        query_vector = vectorizer.transform([texts[held_out]])

        similarities = cosine_similarity(
            query_vector,
            training_matrix
        ).flatten()

        scores = calculate_theme_scores(
            similarities,
            training_labels,
            method=method
        )

        predicted_theme = max(scores, key=scores.get)

        actual.append(labels[held_out])
        predicted.append(predicted_theme)

        if predicted_theme != labels[held_out]:
            top_scores = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            misclassified.append({
                "question": texts[held_out],
                "actual": labels[held_out],
                "predicted": predicted_theme,
                "top_scores": top_scores
            })

    return actual, predicted, misclassified


def main():
    texts, labels = load_questions()

    print("Total questions:", len(texts))
    print("Total themes:", len(set(labels)))

    mean_misclassified = []

    for method in ("mean", "max"):
        actual, predicted, misclassified = evaluate_method(
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
