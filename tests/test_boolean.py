
import pytest

from retrieval.boolean import search_boolean


def test_empty_query():
    assert search_boolean("") == []
    assert search_boolean("   ") == []


def test_and_search():
    results = search_boolean(
        "dharma AND karma",
        lang="en",
        zone="commentary"
    )

    assert "7.15" in results
    assert "9.2" in results


def test_or_search():
    results = search_boolean(
        "dharma OR karma",
        lang="en",
        zone="commentary"
    )

    assert "7.15" in results
    assert "1.42" in results


def test_and_not_search():
    results = search_boolean(
        "dharma AND NOT karma",
        lang="en",
        zone="commentary"
    )

    assert "1.42" in results
    assert "7.15" not in results


def test_parentheses():
    results = search_boolean(
        "(dharma OR karma) AND yoga",
        lang="en",
        zone="commentary"
    )

    assert "2.39" in results


def test_not_search():
    results = search_boolean(
        "NOT dharma",
        lang="en",
        zone="commentary"
    )

    assert "7.15" not in results
    assert "1.1" not in results


def test_invalid_language():
    with pytest.raises(ValueError):
        search_boolean("dharma", lang="fr")


def test_invalid_zone():
    with pytest.raises(ValueError):
        search_boolean("dharma", zone="invalid")


def test_missing_closing_parenthesis():
    with pytest.raises(ValueError):
        search_boolean("(dharma AND karma")


def test_unexpected_operator():
    with pytest.raises(ValueError):
        search_boolean("dharma AND AND karma")


def test_operator_precedence():
    first = search_boolean(
        "dharma OR karma AND yoga",
        lang="en",
        zone="commentary"
    )

    second = search_boolean(
        "dharma OR (karma AND yoga)",
        lang="en",
        zone="commentary"
    )

    assert first == second


def test_invalid_hindi_language_query():
    # Hindi preprocessing should handle a real Hindi query.
    results = search_boolean(
        "धर्म",
        lang="hi",
        zone="translation"
    )

    assert isinstance(results, list)
