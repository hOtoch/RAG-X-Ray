"""Ambiente fixo para os testes: nenhum teste lê o .env real nem chama API externa."""

import os

TEST_ENV = {
    "LLM_PROVIDER": "openai",
    "LLM_MODEL_SMALL": "test-small",
    "LLM_MODEL_MEDIUM": "test-medium",
    "LLM_MODEL_LARGE": "test-large",
    "EMBEDDING_PROVIDER": "openai",
    "EMBEDDING_MODEL": "test-embedding",
    "EMBEDDING_DIM": "8",
    "OPENAI_API_KEY": "test-key",
    "RAGXRAY_ENV_FILE": "nao-existe.env",
}
os.environ.update(TEST_ENV)
