"""Mapa do corpus: posição da pergunta e agrupamento por área da documentação."""

import pytest

from app.retrieval.embedding_map import map_payload, query_position
from ingest.corpus_map import group_of, normalize

COORDS = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (0.0, 1.0)}


def test_query_position_weights_by_inverse_rank():
    # pesos 1, 1/2: x = (0*1 + 1*0.5) / 1.5
    assert query_position([("a", 1), ("b", 2)], COORDS) == (pytest.approx(1 / 3, abs=1e-4), 0.0)


def test_query_position_ignores_chunks_outside_the_map():
    assert query_position([("zz", 1), ("c", 2)], COORDS) == (0.0, 1.0)
    assert query_position([("zz", 1)], COORDS) is None


def test_map_payload_uses_dense_ranks_and_lists_all_hits():
    hits = [{"id": "b", "dense_rank": None}, {"id": "a", "dense_rank": 1}, {"id": "c", "dense_rank": 2}]
    payload = map_payload(hits, COORDS)
    assert payload["hit_ids"] == ["b", "a", "c"]
    assert payload["query_xy"] == (0.0, pytest.approx(1 / 3, abs=1e-4))


def test_map_payload_without_map_is_none():
    assert map_payload([{"id": "a", "dense_rank": 1}], {}) is None


def test_group_of_uses_docs_area():
    assert group_of("https://docs.langchain.com/oss/python/langgraph/persistence#threads") == "langgraph"
    assert group_of("https://docs.langchain.com/oss/python/common-errors") == "geral"


def test_normalize_keeps_aspect_ratio():
    import numpy as np

    out = normalize(np.array([[2.0, 1.0], [6.0, 3.0]]), q=0)
    assert out.tolist() == [[0.0, 0.0], [1.0, 0.5]]


def test_normalize_clips_outliers():
    import numpy as np

    xy = np.array([[float(i), float(i)] for i in range(100)] + [[1000.0, 1000.0]])
    out = normalize(xy, q=0.01)
    assert out.max() == 1.0
    assert out[50, 0] == pytest.approx(0.5, abs=0.02)  # sem o corte, o ponto do meio ficaria em 0,05
