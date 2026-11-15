
import math
import pytest

from retrieval.tfidf import (
    calculate_tf,
    calculate_idf,
    calculate_document_norms,
    search_tfidf,
)


def test_calculate_tf():
    assert calculate_tf(0) == 0
    assert calculate_tf(1) == 1
    assert calculate_tf(2) == pytest.approx(1 + math.log(2))


def test_calculate_idf():
    assert calculate_idf(10, 0) == 0
    assert calculate_idf(10, 10) == 0
    assert calculate_idf(10, 2) == pytest.approx(math.log(5))


def test_calculate_document_norms():
    from indexing.inverted_index import InvertedIndex

    index = InvertedIndex()

    verses = [
        {
            "id": "1.1",
            "translations": {"en_prabhupada": "dharma karma"},
            "commentary": "",
            "transliteration": "",
        },
        {
            "id": "1.2",
            "translations": {"en_prabhupada": "dharma yoga"},
            "commentary": "",
            "transliteration": "",
        },
    ]

    index.build_index(verses, "en")

    norms = calculate_document_norms(index, "translation")

    assert "1.1" in norms
    assert "1.2" in norms
    assert norms["1.1"] > 0
    assert norms["1.2"] > 0


def test_search_empty_query():
    assert search_tfidf([], k=5) == []


def test_search_unknown_term():
    assert search_tfidf(["nonexistentword"], k=5) == []


def test_search_invalid_language():
    with pytest.raises(ValueError):
        search_tfidf(["dharma"], lang="fr")


def test_search_invalid_zone():
    with pytest.raises(ValueError):
        search_tfidf(["dharma"], zone="invalid")
