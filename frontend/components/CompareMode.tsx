"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useStore } from "zustand";

import { AnswerWithCitations } from "@/components/AnswerWithCitations";
import { GraphView } from "@/components/GraphView";
import { TraceTimeline } from "@/components/TraceTimeline";
import { type Demo, fetchDemoTrace, fetchGraph, replayEvents, streamQuery } from "@/lib/api";
import type { GraphTopology, PipelineInfo, RetrievedChunk, RouterData } from "@/lib/events";
import { RunProvider, type RunStore, useNewRunStore } from "@/lib/store";

type Side = "A" | "B";

interface Props {
  mode: "replay" | "live";
  pipelines: PipelineInfo[];
  demos: Demo[];
  question: string;
  setQuestion: (q: string) => void;
}

/** Duas execuções lado a lado: a mesma pergunta em dois pipelines (ao vivo) ou dois traces gravados (replay).
 * Cada lado tem o próprio store; os painéis são os mesmos da tela principal. */
export function CompareMode({ mode, pipelines, demos, question, setQuestion }: Props) {
  const stores: Record<Side, RunStore> = { A: useNewRunStore(), B: useNewRunStore() };
  const [pick, setPick] = useState<Record<Side, string>>({ A: "v0", B: "v3" });
  const [demoPick, setDemoPick] = useState<Record<Side, string>>({ A: "", B: "" });
  const [topology, setTopology] = useState<Record<Side, GraphTopology | null>>({ A: null, B: null });
  const abort = useRef<AbortController | null>(null);
  const running = [useStore(stores.A, (s) => s.status), useStore(stores.B, (s) => s.status)].includes("running");

  // padrões: os dois primeiros demos; no ao vivo, a primeira e a última versão disponível
  useEffect(() => {
    if (demos.length && !demoPick.A)
      setDemoPick({ A: demos[0].request_id, B: (demos[1] ?? demos[0]).request_id });
  }, [demos, demoPick.A]);
  useEffect(() => {
    if (pipelines.length) setPick({ A: pipelines[0].name, B: pipelines.at(-1)!.name });
  }, [pipelines]);

  useEffect(() => {
    if (mode !== "replay") return;
    (["A", "B"] as Side[]).forEach((side) => {
      if (demoPick[side])
        fetchDemoTrace(demoPick[side])
          .then((t) => setTopology((prev) => ({ ...prev, [side]: t.topology })))
          .catch(() => {});
    });
  }, [mode, demoPick]);

  useEffect(() => {
    if (mode !== "live") return;
    (["A", "B"] as Side[]).forEach((side) =>
      fetchGraph(pick[side])
        .then((t) => setTopology((prev) => ({ ...prev, [side]: t })))
        .catch(() => setTopology((prev) => ({ ...prev, [side]: null }))),
    );
  }, [mode, pick]);

  const runLive = useCallback(async () => {
    abort.current?.abort();
    abort.current = new AbortController();
    const signal = abort.current.signal;
    await Promise.all(
      (["A", "B"] as Side[]).map(async (side) => {
        const { reset, apply, fail } = stores[side].getState();
        reset();
        try {
          await streamQuery(question, pick[side], apply, signal);
        } catch (e) {
          if ((e as Error).name !== "AbortError") fail((e as Error).message);
        }
      }),
    );
    // stores são estáveis (useNewRunStore)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [question, pick]);

  const runReplay = useCallback(async () => {
    abort.current?.abort();
    abort.current = new AbortController();
    const signal = abort.current.signal;
    await Promise.all(
      (["A", "B"] as Side[]).map(async (side) => {
        const { reset, apply, fail } = stores[side].getState();
        reset();
        try {
          const trace = await fetchDemoTrace(demoPick[side]);
          setTopology((prev) => ({ ...prev, [side]: trace.topology }));
          await replayEvents(trace.events, apply, { speed: 1, signal });
        } catch (e) {
          fail((e as Error).message);
        }
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoPick]);

  const label = (side: Side) =>
    mode === "live" ? pick[side] : (demos.find((d) => d.request_id === demoPick[side])?.label ?? side);

  return (
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
            {(["A", "B"] as Side[]).map((side) => (
              <select
                key={side}
                value={pick[side]}
                onChange={(e) => setPick((p) => ({ ...p, [side]: e.target.value }))}
                aria-label={`Pipeline do lado ${side}`}
              >
                {pipelines.map((p) => (
                  <option key={p.name} value={p.name} title={p.description}>
                    {side}: {p.name}
                  </option>
                ))}
              </select>
            ))}
            <button type="submit" disabled={running}>
              {running ? "Rodando…" : "Comparar"}
            </button>
          </form>
        ) : (
          <form
            className="compare-replay"
            onSubmit={(e) => {
              e.preventDefault();
              runReplay();
            }}
          >
            {(["A", "B"] as Side[]).map((side) => (
              <select
                key={side}
                value={demoPick[side]}
                onChange={(e) => setDemoPick((p) => ({ ...p, [side]: e.target.value }))}
                aria-label={`Trace do lado ${side}`}
              >
                {demos.map((d) => (
                  <option key={d.request_id} value={d.request_id}>
                    {side}: [{d.label}] {d.question}
                  </option>
                ))}
              </select>
            ))}
            <button type="submit" disabled={running || !demos.length}>
              {running ? "Rodando…" : "Comparar"}
            </button>
          </form>
        )}
        {mode === "live" && (
          <p className="compare-note">A mesma pergunta roda nos dois pipelines ao mesmo tempo (conta como duas perguntas no limite).</p>
        )}
      </section>

      <CompareSummary a={stores.A} b={stores.B} labels={[label("A"), label("B")]} />

      <section className="compare-grid">
        {(["A", "B"] as Side[]).map((side) => (
          <RunProvider key={side} store={stores[side]}>
            <div className="compare-col">
              <div className="panel compare-graph">
                <h2>
                  {side} <small>{label(side)}</small>
                </h2>
                <GraphView topology={topology[side]} />
              </div>
              <div className="panel compare-timeline">
                <h2>Linha do tempo</h2>
                <TraceTimeline />
              </div>
              <div className="panel compare-answer">
                <h2>Resposta</h2>
                <AnswerWithCitations live={mode === "live"} />
              </div>
            </div>
          </RunProvider>
        ))}
      </section>
    </>
  );
}

function ratio(a: number, b: number) {
  if (!a || !b) return "—";
  const r = b / a;
  return r >= 1 ? `B ${r.toFixed(1)}× maior` : `B ${(1 / r).toFixed(1)}× menor`;
}

/** O que mudou de A para B: custo, latência, rota e quanto os trechos recuperados e citados coincidem. */
function CompareSummary({ a, b, labels }: { a: RunStore; b: RunStore; labels: [string, string] }) {
  const doneA = useStore(a, (s) => s.done);
  const doneB = useStore(b, (s) => s.done);
  const dataA = useStore(a, (s) => s.data);
  const dataB = useStore(b, (s) => s.data);
  if (!doneA || !doneB) return null;

  const ids = (data: Record<string, Record<string, unknown>[]>) =>
    new Set(((data.retrieve?.at(-1)?.chunks as RetrievedChunk[] | undefined) ?? []).slice(0, 5).map((c) => c.id));
  const cited = (citations: { chunk_id: string }[]) => new Set(citations.map((c) => c.chunk_id));
  const overlap = (x: Set<string>, y: Set<string>) => {
    if (!x.size && !y.size) return "—";
    const inter = [...x].filter((v) => y.has(v)).length;
    return `${inter} de ${new Set([...x, ...y]).size}`;
  };
  const route = (data: Record<string, Record<string, unknown>[]>) =>
    (data.router?.at(-1) as RouterData | undefined)?.route ?? "sem router";

  return (
    <section className="compare-summary" aria-label="Diferenças entre A e B">
      <dl>
        <div>
          <dt>Custo</dt>
          <dd>
            US$ {doneA.total_cost_usd.toFixed(5)} → US$ {doneB.total_cost_usd.toFixed(5)}
            <em>{ratio(doneA.total_cost_usd, doneB.total_cost_usd)}</em>
          </dd>
        </div>
        <div>
          <dt>Latência</dt>
          <dd>
            {(doneA.total_latency_ms / 1000).toFixed(1)} s → {(doneB.total_latency_ms / 1000).toFixed(1)} s
            <em>{ratio(doneA.total_latency_ms, doneB.total_latency_ms)}</em>
          </dd>
        </div>
        <div>
          <dt>Rota</dt>
          <dd>
            {route(dataA)} → {route(dataB)}
          </dd>
        </div>
        <div title="Trechos em comum entre os 5 primeiros do retrieve de cada lado">
          <dt>Top-5 em comum</dt>
          <dd>{overlap(ids(dataA), ids(dataB))}</dd>
        </div>
        <div title="Trechos citados nas duas respostas">
          <dt>Citações em comum</dt>
          <dd>{overlap(cited(doneA.citations), cited(doneB.citations))}</dd>
        </div>
      </dl>
      <p>
        {labels[0]} → {labels[1]}
      </p>
    </section>
  );
}
