"""Faz uma pergunta real e confere o trace no Langfuse (spans por nó, tokens e custo).

Uso: uv run python scripts/smoke_trace.py "sua pergunta"
"""

import base64
import json
import sys
import time
import urllib.request

from app import tracing
from app.config import get_settings
from app.service import QueryRun

question = sys.argv[1] if len(sys.argv) > 1 else "How do I visualize a LangGraph graph?"
s = get_settings()
run = QueryRun(question, "v0")
events = list(run.events())
tracing.flush()
print(f"{len(events)} eventos | custo medido US$ {run.result.total_cost_usd:.6f} | request_id {run.request_id}")

time.sleep(10)  # o Langfuse processa os eventos de forma assíncrona
token = base64.b64encode(f"{s.langfuse_public_key}:{s.langfuse_secret_key}".encode()).decode()
url = f"{s.langfuse_host}/api/public/traces/{tracing.trace_id_for(run.request_id)}"
request = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
trace = json.load(urllib.request.urlopen(request, timeout=30))
print("trace:", trace["name"], "| sessão = request_id:", trace["sessionId"] == run.request_id, "|", trace["tags"])
print("scores:", {score["name"]: round(score["value"], 6) for score in trace.get("scores", [])})
for obs in sorted(trace["observations"], key=lambda o: o["startTime"]):
    usage = (obs.get("metadata") or {}).get("usage")
    print(f"  {obs['type']:<10} {obs['name']:<16} modelo={obs.get('model')} uso={usage}")
