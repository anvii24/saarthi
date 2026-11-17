
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from query.embedding_classifier import (
    model,
    train_classifier,
    calculate_theme_scores
)
from sentence_transformers import util


def main():
    project_root = Path(__file__).resolve().parent

    test_path = (
        project_root / "data" / "test_set"
        / "theme_training_questions.json"
    )

    with open(test_path, "r", encoding="utf-8") as file:
        test_questions = json.load(file)

    # Load the frozen training data and encode it once.
    train_embeddings, train_labels, _ = train_classifier()

    # Use only English questions for evaluation.
    test_texts = [q["en"] for q in test_questions.values()]
    actual_labels = [q["theme"] for q in test_questions.values()]

    # Encode all independent questions.
    test_embeddings = model.encode(
        test_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Calculate similarities against the training examples.
    similarities = util.cos_sim(test_embeddings, train_embeddings)

    predicted_labels = []

    for i in range(len(test_texts)):
        theme_scores = calculate_theme_scores(
            similarities[i],
            train_labels,
            method="mean"
        )

        predicted_labels.append(
            max(theme_scores, key=theme_scores.get)
        )

    themes = sorted(set(actual_labels))

    print("\nINDEPENDENT EVALUATION RESULTS")
    print("=" * 45)

    print(f"Total questions: {len(test_texts)}")
    print(
        f"Correct predictions: "
        f"{sum(a == p for a, p in zip(actual_labels, predicted_labels))}"
    )
    print(f"Accuracy: {accuracy_score(actual_labels, predicted_labels):.2%}")

    print("\nCLASSIFICATION REPORT")
    print("=" * 45)

    print(classification_report(
        actual_labels,
        predicted_labels,
        labels=themes,
        target_names=themes,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print("=" * 45)
    print("Theme order:", themes)
    print(confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=themes
    ))

    print("\nINCORRECT PREDICTIONS")
    print("=" * 45)

    for question_id, question, actual, predicted in zip(
        test_questions.keys(),
        test_texts,
        actual_labels,
        predicted_labels
    ):
        if actual != predicted:
            print(f"\nID: {question_id}")
            print(f"Question: {question}")
            print(f"Actual: {actual}")
            print(f"Predicted: {predicted}")


if __name__ == "__main__":
    main()
