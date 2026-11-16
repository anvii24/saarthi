
import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    mean_average_precision,
    mrr,
    ndcg_at_k,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["a", "x", "c"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)


def test_f1_at_k():
    assert f1_at_k(["a", "x", "c"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)


def test_average_precision():
    assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(5 / 6)


def test_mean_average_precision():
    rankings = [["a", "x", "b"], ["x", "c"]]
    relevants = [{"a", "b"}, {"c"}]
    assert mean_average_precision(rankings, relevants) == pytest.approx(2 / 3)


def test_mrr():
    assert mrr(["x", "b", "a"], {"a", "b"}) == pytest.approx(1 / 2)


def test_ndcg_at_k():
    ranked = ["a", "b", "c"]
    relevant = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(1.0)


def test_empty_relevant():
    assert precision_at_k(["a"], set(), 1) == 0.0
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert average_precision(["a"], set()) == 0.0
    assert mrr(["a"], set()) == 0.0
    assert ndcg_at_k(["a"], {}, 1) == 0.0


def test_invalid_k():
    assert precision_at_k(["a"], {"a"}, 0) == 0.0
    assert recall_at_k(["a"], {"a"}, -1) == 0.0
    assert f1_at_k(["a"], {"a"}, 0) == 0.0
    assert ndcg_at_k(["a"], {"a": 2}, 0) == 0.0
