"""Montagem do grafo a partir de uma config de pipeline."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.instrument import instrumented
from app.graph.nodes.generate import make_generate
from app.graph.nodes.retrieve import make_retrieve
from app.graph.state import AgentState
from app.pipeline import PipelineConfig, load_pipeline


def build_graph(pipeline: PipelineConfig):
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", instrumented("retrieve", make_retrieve(pipeline)))
    builder.add_node("generate", instrumented("generate", make_generate(pipeline)))
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


@lru_cache
def get_graph(pipeline_name: str):
    return build_graph(load_pipeline(pipeline_name))
