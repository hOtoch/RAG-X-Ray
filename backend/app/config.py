"""Configuração da aplicação. Tudo vem do ambiente (.env); nenhum nome de modelo fica no código."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# Os testes apontam RAGXRAY_ENV_FILE para um arquivo inexistente, para nunca lerem o .env real.
ENV_FILE = Path(os.environ.get("RAGXRAY_ENV_FILE", ROOT / ".env"))

Tier = Literal["small", "medium", "large"]


class ModelPrice(BaseModel):
    """Preço em USD por 1 milhão de tokens."""

    input: float = 0.0
    cached_input: float | None = None
    output: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # LLMs por nível
    llm_provider: str
    llm_model_small: str
    llm_model_medium: str
    llm_model_large: str

    # Embeddings
    embedding_provider: str
    embedding_model: str
    embedding_dim: int

    # Reranker (cross-encoder local, via fastembed)
    reranker_model: str = ""

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "ragxray"
    postgres_user: str = "ragxray"
    postgres_password: str = "ragxray"

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 7 * 24 * 3600
    cors_origins: str = "http://localhost:3001,http://localhost:3002"

    # Langfuse (tracing). Sem as chaves, o tracing fica desligado.
    langfuse_enabled: bool = True
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Retrieval e orçamento
    retrieval_top_k: int = 5
    hnsw_ef_search: int = 200
    max_retries: int = 2
    request_budget_usd: float = 0.05

    # Proteções do demo público (0 desliga)
    rate_limit_per_minute: int = 10
    daily_budget_usd: float = 2.0
    live_mode_enabled: bool = True

    prices_file: Path = ROOT / "configs" / "prices.yaml"
    pipelines_file: Path = ROOT / "configs" / "pipelines.yaml"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def model_for(self, tier: Tier) -> str:
        models = {
            "small": self.llm_model_small,
            "medium": self.llm_model_medium,
            "large": self.llm_model_large,
        }
        return models[tier]


@lru_cache
def get_settings() -> Settings:
    # As bibliotecas dos provedores leem as chaves de os.environ, não do Settings.
    load_dotenv(ENV_FILE, override=False)
    return Settings()  # type: ignore[call-arg]


@lru_cache
def get_prices() -> dict[str, ModelPrice]:
    path = get_settings().prices_file
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: ModelPrice(**values) for name, values in raw.items()}
