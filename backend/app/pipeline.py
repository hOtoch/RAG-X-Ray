"""Configs nomeadas de pipeline. Cada versão liga ou desliga features do grafo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.config import Tier, get_settings


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""

    # Retrieval
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    candidates_k: int = Field(default=20, ge=1, description="Candidatos por fonte antes da fusão/rerank")
    top_k: int = Field(default=5, ge=1, description="Chunks que chegam à geração")
    multi_hop_top_k: int = Field(default=8, ge=1, description="Chunks que chegam à geração na rota multi-hop")
    rerank: bool = False
    compress: bool = False
    context_token_budget: int | None = Field(default=None, description="Teto de tokens do contexto")

    # Agente
    router: bool = False
    groundedness: bool = False
    cache: bool = False

    cache_threshold: float = Field(default=0.95, ge=0, le=1, description="θ do cache semântico (similaridade)")

    generate_tier: Tier = "medium"
    # Com router: nível do modelo de geração por rota. Rotas ausentes usam generate_tier.
    route_tiers: dict[str, Tier] = Field(default_factory=dict)
    escalate: bool = Field(default=True, description="Se a groundedness falhar, tenta uma vez o nível acima")
    request_budget_usd: float | None = Field(default=None, description="Teto de custo por request; None usa o .env")


def load_pipelines(path: Path | None = None) -> dict[str, PipelineConfig]:
    path = path or get_settings().pipelines_file
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: PipelineConfig(name=name, **values) for name, values in raw.items()}


@lru_cache
def _pipelines() -> dict[str, PipelineConfig]:
    return load_pipelines()


def load_pipeline(name: str) -> PipelineConfig:
    pipelines = _pipelines()
    if name not in pipelines:
        raise KeyError(f"Pipeline '{name}' não existe. Disponíveis: {', '.join(pipelines)}")
    return pipelines[name]


def budget_usd(pipeline: PipelineConfig) -> float:
    """Teto de custo por request: o da pipeline, se definido; senão o do .env."""
    if pipeline.request_budget_usd is not None:
        return pipeline.request_budget_usd
    return get_settings().request_budget_usd
