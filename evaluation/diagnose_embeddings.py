
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "test_set" / "questions.json"

    with open(data_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    return [
        (q["en"], q["theme"])
        for q in questions.values()
    ]


def diagnose_theme(target_theme="anger_and_desire", top_k=3):
    data = load_data()

    print(f"\nDIAGNOSTICS FOR: {target_theme}")
    print("=" * 65)

    for i, (question, actual_theme) in enumerate(data):

        if actual_theme != target_theme:
            continue

        # Exclude the current question to avoid data leakage.
        train_data = data[:i] + data[i + 1:]

        train_texts = [item[0] for item in train_data]
        train_labels = [item[1] for item in train_data]

        train_embeddings = model.encode(
            train_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        query_embedding = model.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarities = util.cos_sim(
            query_embedding, train_embeddings
        )[0]

        theme_scores = {}

        for theme in set(train_labels):
            scores = [
                similarities[j].item()
                for j in range(len(train_labels))
                if train_labels[j] == theme
            ]

            theme_scores[theme] = sum(scores) / len(scores)

        predicted = max(theme_scores, key=theme_scores.get)

        top_themes = sorted(
            theme_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:3]

        top_indices = similarities.argsort(
            descending=True
        )[:top_k]

        print("\nQUESTION:", question)
        print("ACTUAL:", actual_theme)
        print("PREDICTED:", predicted)

        print("\nTop theme scores:")
        for theme, score in top_themes:
            print(f"  {theme}: {score:.4f}")

        print("\nClosest training examples:")
        for index in top_indices:
            j = index.item()
            print(
                f"  [{train_labels[j]}] "
                f"{train_texts[j]} "
                f"(similarity: {similarities[j].item():.4f})"
            )

        print("-" * 65)


if __name__ == "__main__":
    diagnose_theme()
