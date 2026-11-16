
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "test_set" / "questions.json"

    with open(data_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    texts = [q["en"] for q in questions.values()]
    labels = [q["theme"] for q in questions.values()]

    return texts, labels


def predict_theme(query, train_texts, train_labels):
    """Predict a theme using only the training questions."""

    train_embeddings = model.encode(
        train_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(query_embedding, train_embeddings)[0]

    theme_scores = {}

    for theme in set(train_labels):
        scores = [
            similarities[i].item()
            for i in range(len(train_labels))
            if train_labels[i] == theme
        ]

        theme_scores[theme] = sum(scores) / len(scores)

    return max(theme_scores, key=theme_scores.get)


def evaluate():
    texts, labels = load_data()

    predictions = []
    actual = []

    print(f"Running leave-one-out evaluation on {len(texts)} questions...")

    for i in range(len(texts)):
        # Hold out one question for testing.
        test_text = texts[i]
        test_label = labels[i]

        # Train using all remaining questions.
        train_texts = texts[:i] + texts[i + 1:]
        train_labels = labels[:i] + labels[i + 1:]

        predicted = predict_theme(
            test_text,
            train_texts,
            train_labels
        )

        predictions.append(predicted)
        actual.append(test_label)

        if predicted != test_label:
            print("\nMisclassified question:")
            print("Question:", test_text)
            print("Actual:", test_label)
            print("Predicted:", predicted)

    accuracy = accuracy_score(actual, predictions)

    print("\n" + "=" * 55)
    print("SENTENCE EMBEDDINGS - LEAVE-ONE-OUT EVALUATION")
    print("=" * 55)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {sum(a == p for a, p in zip(actual, predictions))}/{len(actual)}")

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate()
