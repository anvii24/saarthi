import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.clean_en import clean_en


DATA_PATH = Path(__file__).resolve().parent.parent / "data/test_set/questions.json"


def load_training_data():
    """Load labelled English questions and their themes."""
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

    if not texts:
        raise ValueError("No valid training questions found.")

    return texts, labels


def train_classifier():
    """Build the TF-IDF matrix from the labelled questions."""
    texts, labels = load_training_data()

    vectorizer = TfidfVectorizer(
        tokenizer=clean_en,
        token_pattern=None,
        lowercase=False
    )

    training_matrix = vectorizer.fit_transform(texts)

    return vectorizer, training_matrix, labels, texts


def calculate_theme_scores(similarities, labels, method="mean"):
    """Aggregate question similarities into theme scores."""
    if method not in ("mean", "max"):
        raise ValueError("Method must be 'mean' or 'max'")

    theme_scores = {}

    for theme in sorted(set(labels)):
        theme_similarities = [
            similarities[i]
            for i, label in enumerate(labels)
            if label == theme
        ]

        if method == "mean":
            score = sum(theme_similarities) / len(theme_similarities)
        else:
            score = max(theme_similarities)

        theme_scores[theme] = float(score)

    return theme_scores


def classify_theme(query, top_k=3, method="mean"):
    """
    Predict the theme of an English query using cosine similarity.

    Returns the predicted theme, scores for all themes,
    and the closest matching training examples.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    vectorizer, training_matrix, labels, texts = train_classifier()

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        training_matrix
    ).flatten()

    theme_scores = calculate_theme_scores(
        similarities,
        labels,
        method
    )
     # Check whether the query has any vocabulary overlap
    if not any(similarities > 0):
        return {
            "predicted_theme": "unclassified",
            "theme_scores": theme_scores,
            "closest_examples": [],
            "is_classified": False
        }

    predicted_theme = max(
        theme_scores,
        key=theme_scores.get
    )

    closest_indices = similarities.argsort()[::-1][:top_k]

    closest_examples = [
        {
            "question": texts[i],
            "theme": labels[i],
            "similarity": float(similarities[i])
        }
        for i in closest_indices
    ]

    return {
        "predicted_theme": predicted_theme,
        "theme_scores": theme_scores,
        "closest_examples": closest_examples,
        "is_classified": True
    }