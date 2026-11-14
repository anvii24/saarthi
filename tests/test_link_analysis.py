import math

from indexing.link_analysis import build_graph, pagerank, hits

import pytest


def test_pagerank_invalid_damping():
    with pytest.raises(ValueError):
        pagerank({"A": set()}, damping=1.0)


def test_pagerank_invalid_iterations():
    with pytest.raises(ValueError):
        pagerank({"A": set()}, max_iterations=0)


def test_pagerank_invalid_tolerance():
    with pytest.raises(ValueError):
        pagerank({"A": set()}, tolerance=0)


def test_hits_invalid_iterations():
    with pytest.raises(ValueError):
        hits({"A": set()}, max_iterations=0)


def test_hits_invalid_tolerance():
    with pytest.raises(ValueError):
        hits({"A": set()}, tolerance=0)

def test_build_graph():
    verses = [{"id": "1.1"}, {"id": "1.2"}, {"id": "1.3"}]
    links = [
        {"from": "1.1", "to": "1.2", "source": "test"},
        {"from": "1.1", "to": "1.3", "source": "test"},
        {"from": "1.1", "to": "1.2", "source": "test"}
    ]

    graph = build_graph(verses, links)

    assert len(graph) == 3
    assert graph["1.1"] == {"1.2", "1.3"}
    assert graph["1.2"] == set()
    assert graph["1.3"] == set()


def test_pagerank():
    graph = {
        "A": {"B", "C"},
        "B": {"C"},
        "C": set()
    }

    scores = pagerank(graph)

    assert len(scores) == 3
    assert math.isclose(sum(scores.values()), 1.0)
    assert scores["C"] > scores["A"]


def test_pagerank_empty_graph():
    assert pagerank({}) == {}


def test_hits():
    graph = {
        "A": {"C"},
        "B": {"C"},
        "C": set()
    }

    hubs, authorities = hits(graph)

    assert len(hubs) == 3
    assert len(authorities) == 3
    assert math.isclose(sum(x ** 2 for x in hubs.values()), 1.0)
    assert math.isclose(sum(x ** 2 for x in authorities.values()), 1.0)
    assert authorities["C"] > authorities["A"]


def test_hits_empty_graph():
    hubs, authorities = hits({})

    assert hubs == {}
    assert authorities == {}