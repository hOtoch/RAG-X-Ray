"""Pipeline completo de ingestão: uv run python -m ingest"""

from ingest.chunk import chunk_corpus
from ingest.fetch import fetch
from ingest.index import index
from ingest.load import load

if __name__ == "__main__":
    fetch()
    docs = load()
    chunks = chunk_corpus(docs)
    index(chunks)
