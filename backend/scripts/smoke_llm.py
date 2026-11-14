"""Chamada real ao nível small, para conferir chave, modelo e cálculo de custo."""

import time

from app.config import get_settings
from app.llm.factory import get_llm
from app.llm.usage import usage_from_message

start = time.perf_counter()
message = get_llm("small").invoke("Responda apenas: ok")
elapsed_ms = (time.perf_counter() - start) * 1000
print(repr(message.content))
print(usage_from_message("smoke", get_settings().model_for("small"), message, elapsed_ms))
