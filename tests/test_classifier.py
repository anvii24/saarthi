
from query.classifier import (
    load_training_data,
    train_classifier,
    calculate_theme_scores,
    classify_theme,
)


def test_training_data_loads():
    texts, labels = load_training_data()

    assert len(texts) == 55
    assert len(labels) == 55
    assert len(set(labels)) == 11


def test_classifier_trains():
    vectorizer, matrix, labels, texts = train_classifier()

    assert matrix.shape[0] == len(texts)
    assert len(labels) == len(texts)
    assert matrix.shape[1] > 0


def test_mean_theme_scores():
    similarities = [0.8, 0.6, 0.4, 0.2]
    labels = ["duty", "duty", "devotion", "devotion"]

    scores = calculate_theme_scores(
        similarities,
        labels,
        method="mean"
    )

    assert scores["duty"] == 0.7
    assert abs(scores["devotion"] - 0.3) < 1e-9


def test_max_theme_scores():
    similarities = [0.8, 0.6, 0.4, 0.2]
    labels = ["duty", "duty", "devotion", "devotion"]

    scores = calculate_theme_scores(
        similarities,
        labels,
        method="max"
    )

    assert scores["duty"] == 0.8
    assert scores["devotion"] == 0.4


def test_invalid_scoring_method():
    similarities = [0.8, 0.4]
    labels = ["duty", "devotion"]

    try:
        calculate_theme_scores(
            similarities,
            labels,
            method="invalid"
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_classify_theme_returns_expected_structure():
    result = classify_theme(
        "I am worried about my upcoming exams and cannot sleep"
    )

    assert result["predicted_theme"] in load_training_data()[1]
    assert len(result["theme_scores"]) == 11
    assert len(result["closest_examples"]) == 3


def test_empty_query_is_rejected():
    try:
        classify_theme("   ")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_top_k_is_rejected():
    try:
        classify_theme("How can I control my anger?", top_k=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_unclassified_query():
    result = classify_theme("xyzqv blorf snarg")

    assert result["predicted_theme"] == "unclassified"
    assert result["is_classified"] is False
    assert result["closest_examples"] == []


def test_classified_query():
    result = classify_theme(
        "I am worried about my upcoming exams and cannot sleep"
    )

    assert result["predicted_theme"] != "unclassified"
    assert result["is_classified"] is True