
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from preprocessing.clean_en import clean_en


def create_vectorizer():
    """Create the shared TF-IDF vectorizer."""
    return TfidfVectorizer(
        tokenizer=clean_en,
        token_pattern=None,
        lowercase=False
    )


def train_classifier(texts, labels, method="knn", k=3):
    """
    Train a theme classifier.

    Supported methods:
    - naive_bayes
    - rocchio
    - knn
    """
    if method not in ("naive_bayes", "rocchio", "knn"):
        raise ValueError("Unknown classifier method.")

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
        classifier = MultinomialNB()
        classifier.fit(matrix, labels)
        model["classifier"] = classifier

    elif method == "rocchio":
        centroids = {}

        for theme in sorted(set(labels)):
            indices = [
                i for i, label in enumerate(labels)
                if label == theme
            ]

            centroid = np.asarray(
              matrix[indices].mean(axis=0)
            ).ravel()

            centroids[theme] = centroid

        model["centroids"] = centroids

    return model


def predict_theme(model, query):
    """Predict a theme using the selected classifier."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    method = model["method"]
    vectorizer = model["vectorizer"]
    query_vector = vectorizer.transform([query])

    if query_vector.nnz == 0:
        return "unclassified"

    if method == "naive_bayes":
        return model["classifier"].predict(query_vector)[0]

    if method == "rocchio":
        themes = list(model["centroids"].keys())

        centroids = [
            model["centroids"][theme]
            for theme in themes
        ]

        similarities = cosine_similarity(
            query_vector,
            centroids
        ).flatten()

        return themes[similarities.argmax()]

    if method == "knn":
        similarities = cosine_similarity(
            query_vector,
            model["matrix"]
        ).flatten()

        k = min(model["k"], len(similarities))

        nearest_indices = similarities.argsort()[::-1][:k]

        votes = {}

        for i in nearest_indices:
            theme = model["labels"][i]
            votes[theme] = votes.get(theme, 0) + 1

        return sorted(
            votes,
            key=lambda theme: (
                -votes[theme],
                -max(
                    similarities[i]
                    for i in nearest_indices
                    if model["labels"][i] == theme
                ),
                theme
            )
        )[0]

    raise ValueError("Unknown classifier method.")
