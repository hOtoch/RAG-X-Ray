"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AnswerWithCitations } from "@/components/AnswerWithCitations";
import { CompareMode } from "@/components/CompareMode";
import { EmbeddingMap } from "@/components/EmbeddingMap";
import { GraphView } from "@/components/GraphView";
import { RankingPanel } from "@/components/RankingPanel";
import { RouterExplainer } from "@/components/RouterExplainer";
import { TraceTimeline } from "@/components/TraceTimeline";
import { type Demo, fetchDemos, fetchDemoTrace, fetchGraph, fetchPipelines, replayEvents, streamQuery } from "@/lib/api";
import type { GraphTopology, PipelineInfo } from "@/lib/events";
import { useRun } from "@/lib/store";

const EXAMPLES = [
  "How do I pause a LangGraph graph to wait for human approval and then resume it?",
  "qual a diferença entre o padrão de subagents e o de handoffs no langchain?",
  "How do I set the max_parallel_sends parameter on StateGraph?",
  "Qual é a melhor receita de bolo de cenoura?",
];

export default function Home() {
  const [mode, setMode] = useState<"replay" | "live">("replay");
  const [view, setView] = useState<"single" | "compare">("single");
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [pipeline, setPipeline] = useState("v3");
  const [topology, setTopology] = useState<GraphTopology | null>(null);
  const [demos, setDemos] = useState<Demo[]>([]);
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [replayTopology, setReplayTopology] = useState<GraphTopology | null>(null);
  const abort = useRef<AbortController | null>(null);
  const { status, reset, apply, fail } = useRun();

  useEffect(() => {
    fetchPipelines()
      .then((list) => {
        setPipelines(list);
        if (!list.some((p) => p.name === "v3") && list.length) setPipeline(list.at(-1)!.name);
      })
      .catch((e) => setBackendError(`Backend indisponível (${e.message}). O modo replay funciona sem ele.`));
    fetchDemos()
      .then((list) => {
        setDemos(list);
        if (list[0]) fetchDemoTrace(list[0].request_id).then((t) => setReplayTopology(t.topology)).catch(() => {});
      })
      .catch(() => setDemos([]));
  }, []);

  useEffect(() => {
    setTopology(null);
    fetchGraph(pipeline).then(setTopology).catch(() => setTopology(null));
  }, [pipeline]);

  const runLive = useCallback(async () => {
    abort.current?.abort();
    abort.current = new AbortController();
    reset();
    try {
      await streamQuery(question, pipeline, apply, abort.current.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") fail((e as Error).message);
    }
  }, [question, pipeline, reset, apply, fail]);

  const runReplay = useCallback(
    async (demo: Demo) => {
      abort.current?.abort();
      abort.current = new AbortController();
      setQuestion(demo.question);
      reset();
      try {
        const trace = await fetchDemoTrace(demo.request_id);
        setReplayTopology(trace.topology);
        await replayEvents(trace.events, apply, { speed: 1, signal: abort.current.signal });
      } catch (e) {
        fail((e as Error).message);
      }
    },
    [reset, apply, fail],
  );

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>
            RAG <span>X-Ray</span>
          </h1>
          <p>Cada etapa do agente, ao vivo: cache, roteamento, retrieval, rerank, geração e verificação.</p>
        </div>
        <div className="toggles">
          <div className="mode-toggle" role="tablist" aria-label="Visualização">
            <button role="tab" aria-selected={view === "single"} onClick={() => setView("single")}>
              Uma execução
            </button>
            <button role="tab" aria-selected={view === "compare"} onClick={() => setView("compare")}>
              Comparar
            </button>
          </div>
          <div className="mode-toggle" role="tablist" aria-label="Origem">
            <button role="tab" aria-selected={mode === "replay"} onClick={() => setMode("replay")}>
              Replay
            </button>
            <button role="tab" aria-selected={mode === "live"} onClick={() => setMode("live")}>
              Ao vivo
            </button>
          </div>
        </div>
      </header>

      {backendError && mode === "live" && <p className="banner">{backendError}</p>}

      {view === "compare" ? (
        <CompareMode mode={mode} pipelines={pipelines} demos={demos} question={question} setQuestion={setQuestion} />
      ) : (
        <>
          <section className="ask">
            {mode === "live" ? (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (question.trim()) runLive();
                }}
              >
                <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Pergunte sobre LangGraph…" />
                <select value={pipeline} onChange={(e) => setPipeline(e.target.value)} aria-label="Versão do pipeline">
                  {pipelines.map((p) => (
                    <option key={p.name} value={p.name} title={p.description}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <button type="submit" disabled={status === "running"}>
                  {status === "running" ? "Rodando…" : "Perguntar"}
                </button>
              </form>
            ) : (
              <div className="demo-list">
                {demos.length === 0 && <span className="text-[var(--muted)]">Nenhum trace gravado ainda.</span>}
                {demos.map((d) => (
                  <button key={d.request_id} onClick={() => runReplay(d)} disabled={status === "running"} title={d.note}>
                    <span className="demo-tag">{d.label}</span>
                    {d.question}
                  </button>
                ))}
              </div>
            )}
            {mode === "live" && (
              <div className="examples">
                {EXAMPLES.map((q) => (
                  <button key={q} onClick={() => setQuestion(q)}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </section>

          <RouterExplainer />

          <section className="grid-main">
            <div className="panel panel-graph">
              <h2>
                Grafo <small>{mode === "replay" ? replayTopology?.pipeline ?? "" : pipeline}</small>
              </h2>
              <GraphView topology={mode === "replay" ? replayTopology : topology} />
            </div>
            <div className="panel panel-answer">
              <h2>Resposta</h2>
              <AnswerWithCitations live={mode === "live"} />
            </div>
            <div className="panel panel-timeline">
              <h2>Linha do tempo</h2>
              <TraceTimeline />
            </div>
            <div className="panel panel-ranking">
              <h2>Ranking dos trechos</h2>
              <RankingPanel />
            </div>
            <div className="panel panel-map">
              <h2>
                Mapa do corpus <small>UMAP dos embeddings</small>
              </h2>
              <EmbeddingMap />
            </div>
          </section>
        </>
      )}
    </main>
  );
}
