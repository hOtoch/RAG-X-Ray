"""Mapa 2D do corpus: projeta os embeddings dos chunks com UMAP.

Uso (a partir de backend/, depois da ingestão):
    uv sync --group map
    uv run python -m ingest.corpus_map

Grava as coordenadas na tabela `chunk_map` (usada pela API para posicionar a pergunta no mapa) e em
frontend/public/corpus_map.json (desenhado pela interface, inclusive no modo replay). Não chama nenhuma API:
os embeddings já estão no pgvector.

O UMAP fica num grupo opcional de dependências porque puxa numba e scikit-learn, que o servidor não precisa.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.config import ROOT, get_settings

OUT = ROOT / "frontend" / "public" / "corpus_map.json"
MAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk_map (
    id TEXT PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    x  REAL NOT NULL,
    y  REAL NOT NULL
);
"""
MAX_GROUPS = 9  # cores distintas na interface; o resto vira "outros"


def group_of(url: str) -> str:
    """Área da documentação a partir da URL: /oss/python/langgraph/... -> langgraph."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "oss":
        return parts[2] if len(parts) > 3 else "geral"
    return "geral"


def normalize(xy: np.ndarray, q: float = 0.005) -> np.ndarray:
    """Leva as coordenadas para [0, 1] mantendo a proporção entre os eixos. Os limites são os quantis
    q e 1-q, e não o mínimo e o máximo: meia dúzia de trechos isolados encolheria o resto do mapa num canto."""
    low, high = np.quantile(xy, q, axis=0), np.quantile(xy, 1 - q, axis=0)
    xy = np.clip(xy, low, high) - low
    return xy / (high - low).max()


def project(vectors: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap  # grupo opcional `map`

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    return normalize(reducer.fit_transform(vectors))


def main() -> None:
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        register_vector(conn)
        rows = conn.execute("SELECT id, doc_id, title, url, embedding FROM chunks ORDER BY id").fetchall()
        print(f"corpus_map: {len(rows)} chunks")
        start = time.perf_counter()
        xy = project(np.stack([np.asarray(r[4].to_numpy(), dtype=np.float32) for r in rows]))
        print(f"corpus_map: UMAP em {time.perf_counter() - start:.0f} s")

        conn.execute(MAP_SCHEMA)
        conn.execute("TRUNCATE chunk_map")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunk_map (id, x, y) VALUES (%s, %s, %s)",
                [(r[0], float(p[0]), float(p[1])) for r, p in zip(rows, xy, strict=True)],
            )
        conn.commit()

    groups_all = [group_of(r[3]) for r in rows]
    top = [g for g, _ in Counter(groups_all).most_common(MAX_GROUPS)]
    groups = [*top, "outros"]
    docs: dict[str, int] = {}
    titles: list[str] = []
    for r in rows:
        if r[1] not in docs:
            docs[r[1]] = len(titles)
            titles.append(r[2])
    # Formato em colunas, para o arquivo ficar pequeno: o i-ésimo ponto é (ids[i], x[i], y[i], ...).
    payload = {
        "groups": groups,
        "titles": titles,
        "ids": [r[0] for r in rows],
        "x": [round(float(p[0]), 4) for p in xy],
        "y": [round(float(p[1]), 4) for p in xy],
        "group": [groups.index(g) if g in top else len(top) for g in groups_all],
        "doc": [docs[r[1]] for r in rows],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"corpus_map: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), grupos: {', '.join(groups)}")


if __name__ == "__main__":
    main()
