"""Fábrica de modelos: o código pede um nível (small/medium/large) e o .env decide o modelo."""

from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Tier, get_settings


@lru_cache
def get_llm(tier: Tier) -> BaseChatModel:
    s = get_settings()
    return init_chat_model(s.model_for(tier), model_provider=s.llm_provider, temperature=0)


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    return init_embeddings(s.embedding_model, provider=s.embedding_provider)
