"""Chamada com saída estruturada (JSON validado por um modelo Pydantic), usada pelos nós internos."""

from __future__ import annotations

from pydantic import BaseModel

from app.config import Tier, get_settings
from app.llm.factory import get_llm
from app.llm.usage import NodeUsage, usage_from_message


def call_structured[T: BaseModel](
    tier: Tier, schema: type[T], system: str, human: str, node: str
) -> tuple[T, NodeUsage]:
    """Retorna (objeto validado, uso). O prompt de sistema é fixo e vem primeiro, para favorecer o cache."""
    llm = get_llm(tier).with_structured_output(schema, include_raw=True)
    out = llm.invoke([("system", system), ("human", human)])
    usage = usage_from_message(node, get_settings().model_for(tier), out["raw"])
    if out["parsed"] is None:
        raise ValueError(f"Saída estruturada inválida no nó {node}: {out.get('parsing_error')}")
    return out["parsed"], usage
