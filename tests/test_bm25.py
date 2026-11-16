
import math
import pytest

from retrieval.bm25 import (
    calculate_idf,
    calculate_bm25_score,
    search_bm25,
)


def test_calculate_idf():
    assert calculate_idf(0, 0) == 0
    assert calculate_idf(100, 10) == pytest.approx(
        math.log(1 + 90.5 / 10.5)
    )


def test_calculate_bm25_score():
    score = calculate_bm25_score(3, 100, 120, 10, 100)

    assert score > 0


def test_bm25_zero_term_frequency():
    assert calculate_bm25_score(0, 100, 120, 10, 100) == 0


def test_bm25_invalid_parameters():
    with pytest.raises(ValueError):
        search_bm25(["dharma"], b=1.5)

    with pytest.raises(ValueError):
        search_bm25(["dharma"], k1=-1)


def test_bm25_empty_query():
    assert search_bm25([], k=5) == []


def test_bm25_unknown_term():
    assert search_bm25(["nonexistentword"], k=5) == []


def test_bm25_invalid_language():
    with pytest.raises(ValueError):
        search_bm25(["dharma"], lang="fr")


def test_bm25_invalid_zone():
    with pytest.raises(ValueError):
        search_bm25(["dharma"], zone="invalid")
