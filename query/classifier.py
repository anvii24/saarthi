import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

from preprocessing.clean_en import clean_en


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data/test_set/theme_training_questions.json"
)


def load_training_data():
    """Load the separate theme-training questions (never the test questions)."""
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


def create_vectorizer():
    """TF-IDF vectorizer that cleans text the same way as the search index."""
    return TfidfVectorizer(
        tokenizer=clean_en,
        token_pattern=None,
        lowercase=False
    )


def train_classifier():
    """Build the TF-IDF matrix from the training questions."""
    texts, labels = load_training_data()

    vectorizer = create_vectorizer()

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


# ---------------------------------------------------------------------------
# Naive Bayes, Rocchio and kNN (compared in evaluation/run_classifiers.py)
# ---------------------------------------------------------------------------

METHODS = ("naive_bayes", "rocchio", "knn")


def train_model(texts, labels, method="rocchio", k=3):
    """Train one of: naive_bayes, rocchio, knn (k = number of neighbours)."""
    if method not in METHODS:
        raise ValueError(f"Method must be one of {METHODS}")

    if not texts or len(texts) != len(labels):
        raise ValueError("Texts and labels must have equal nonzero lengths.")

    if k <= 0:
        raise ValueError("k must be positive.")

    vectorizer = create_vectorizer()
    matrix = vectorizer.fit_transform(texts)

    model = {
        "method": method,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "labels": labels,
        "k": k
    }

    if method == "naive_bayes":
        model["classifier"] = MultinomialNB().fit(matrix, labels)

    elif method == "rocchio":
        # One centroid per theme: the average TF-IDF vector of its questions.
        model["centroids"] = {
            theme: np.asarray(
                matrix[[i for i, label in enumerate(labels) if label == theme]].mean(axis=0)
            ).ravel()
            for theme in sorted(set(labels))
        }

    return model


def predict_theme(model, query):
    """Predict a theme with a model from train_model(); "unclassified" if no word matches."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = model["vectorizer"].transform([query])

    if query_vector.nnz == 0:
        return "unclassified"

    if model["method"] == "naive_bayes":
        return model["classifier"].predict(query_vector)[0]

    if model["method"] == "rocchio":
        themes = list(model["centroids"])
        similarities = cosine_similarity(
            query_vector, [model["centroids"][theme] for theme in themes]
        ).flatten()
        return themes[similarities.argmax()]

    # kNN: majority vote among the k most similar training questions,
    # ties broken by the highest similarity, then alphabetically.
    similarities = cosine_similarity(query_vector, model["matrix"]).flatten()
    nearest = similarities.argsort()[::-1][:min(model["k"], len(similarities))]

    votes = {}
    best = {}
    for i in nearest:
        theme = model["labels"][i]
        votes[theme] = votes.get(theme, 0) + 1
        best[theme] = max(best.get(theme, 0.0), similarities[i])

    return sorted(votes, key=lambda theme: (-votes[theme], -best[theme], theme))[0]