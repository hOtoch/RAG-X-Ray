"""Métricas de retrieval com casos calculados à mão."""

import math

import pytest

from eval.metrics.retrieval import doc_level, matches, mrr, ndcg_at_k, recall_at_k

RETRIEVED = ["a#chunk-0", "b#chunk-1", "c#chunk-2", "d#chunk-3", "e#chunk-4"]


def test_matches_chunk_and_doc_level():
    assert matches("a#chunk-0", "a#chunk-0")
    assert not matches("a#chunk-1", "a#chunk-0")
    assert matches("a#chunk-1", "a")
    assert not matches("ab#chunk-1", "a")


def test_recall():
    assert recall_at_k(RETRIEVED, ["b#chunk-1", "z#chunk-9"], 5) == 0.5
    assert recall_at_k(RETRIEVED, ["c#chunk-2"], 2) == 0.0
    assert recall_at_k(RETRIEVED, ["c#chunk-2"], 3) == 1.0


def test_recall_doc_level_counts_each_relevant_once():
    # dois chunks do mesmo documento não contam como dois acertos
    assert recall_at_k(["a#chunk-0", "a#chunk-1"], ["a", "b"], 5) == 0.5


def test_mrr():
    assert mrr(RETRIEVED, ["c#chunk-2"]) == pytest.approx(1 / 3)
    assert mrr(RETRIEVED, ["c#chunk-2", "a#chunk-0"]) == 1.0
    assert mrr(RETRIEVED, ["z#chunk-0"]) == 0.0


def test_ndcg():
    # um relevante na posição 3: DCG = 1/log2(4) = 0.5; ideal = 1
    assert ndcg_at_k(RETRIEVED, ["c#chunk-2"], 5) == pytest.approx(0.5)
    # relevantes nas posições 1 e 3; ideal nas posições 1 e 2
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RETRIEVED, ["a#chunk-0", "c#chunk-2"], 5) == pytest.approx(expected)
    assert ndcg_at_k(RETRIEVED, ["a#chunk-0"], 5) == 1.0


def test_undefined_without_relevant():
    for fn in (lambda: recall_at_k(RETRIEVED, [], 5), lambda: mrr(RETRIEVED, []), lambda: ndcg_at_k(RETRIEVED, [], 5)):
        with pytest.raises(ValueError):
            fn()


def test_doc_level():
    assert doc_level(["a#chunk-0", "a#chunk-3", "b#chunk-1"]) == ["a", "b"]
