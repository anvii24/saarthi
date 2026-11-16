
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


# Load the sentence embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def load_training_data():
    """Load English questions and their corresponding themes."""

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "test_set" / "questions.json"

    with open(data_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    texts = []
    labels = []

    for question in questions.values():
        texts.append(question["en"])
        labels.append(question["theme"])

    return texts, labels


def train_classifier():
    """Encode all training questions into sentence embeddings."""

    texts, labels = load_training_data()

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return embeddings, labels, texts


def calculate_theme_scores(similarities, labels, method="mean"):
    """Calculate similarity scores for each theme."""

    if method not in ("mean", "max"):
        raise ValueError("Method must be 'mean' or 'max'.")

    theme_scores = {}

    for theme in set(labels):
        scores = [
            similarities[i].item()
            for i in range(len(labels))
            if labels[i] == theme
        ]

        if method == "mean":
            theme_scores[theme] = sum(scores) / len(scores)
        else:
            theme_scores[theme] = max(scores)

    return theme_scores


def classify_theme(query, top_k=3, method="mean"):
    """Predict the most relevant theme for an English query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    embeddings, labels, texts = train_classifier()

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(query_embedding, embeddings)[0]

    # Return unclassified if the query has no meaningful similarity.
    # A validated confidence threshold will be added after evaluation.
    if len(similarities) == 0:
        return {
            "predicted_theme": "unclassified",
            "theme_scores": {},
            "closest_examples": [],
            "is_classified": False
        }

    theme_scores = calculate_theme_scores(
        similarities, labels, method
    )

    predicted_theme = max(theme_scores, key=theme_scores.get)

    # Find the closest matching training questions.
    top_indices = similarities.argsort(descending=True)[:top_k]

    closest_examples = []

    for index in top_indices:
        i = index.item()

        closest_examples.append({
            "question": texts[i],
            "theme": labels[i],
            "similarity": similarities[i].item()
        })

    return {
        "predicted_theme": predicted_theme,
        "theme_scores": theme_scores,
        "closest_examples": closest_examples,
        "is_classified": True
    }
