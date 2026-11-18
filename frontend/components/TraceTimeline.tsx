"use client";

import { useRun } from "@/lib/store";

const usd = (v: number) => (v === 0 ? "US$ 0" : v < 0.0001 ? `US$ ${v.toExponential(1)}` : `US$ ${v.toFixed(5)}`);
const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);

export function TraceTimeline() {
  const spans = useRun((s) => s.spans);
  const done = useRun((s) => s.done);
  const status = useRun((s) => s.status);

  if (!spans.length) return <div className="panel-empty">A linha do tempo aparece quando a pergunta começa.</div>;

  const now = Math.max(...spans.map((s) => s.end ?? s.start), 1);
  const total = done?.total_latency_ms ?? now;
  const spent = spans.reduce((acc, s) => acc + (s.usage?.cost_usd ?? 0), 0);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1 text-xs">
        <span className="stat">
          <b>{ms(total)}</b> total
        </span>
        <span className="stat">
          <b>{usd(done?.total_cost_usd ?? spent)}</b> custo
        </span>
        {done?.counterfactual_cost_usd != null && done.counterfactual_cost_usd > 0 && (
          <span className="stat" title="Quanto custaria se toda chamada usasse o modelo large">
            <b>{usd(done.counterfactual_cost_usd)}</b> só com o modelo large
            {done.total_cost_usd < done.counterfactual_cost_usd && (
              <em className="ml-1 text-[var(--ok)]">
                −{Math.round((1 - done.total_cost_usd / done.counterfactual_cost_usd) * 100)}%
              </em>
            )}
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto pr-1">
        {spans.map((span, i) => {
          const end = span.end ?? (status === "running" ? now : span.start);
          const left = (span.start / total) * 100;
          const width = Math.max(((end - span.start) / total) * 100, 0.6);
          const u = span.usage;
          return (
            <div key={i} className="timeline-row" title={u?.model ?? ""}>
              <span className="timeline-label">{span.node}</span>
              <div className="timeline-track">
                <div
                  className={`timeline-bar ${span.end === null ? "timeline-open" : ""} ${u?.model ? "timeline-llm" : ""}`}
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
              </div>
              <span className="timeline-meta">
                {span.end === null ? "…" : ms(end - span.start)}
                {u && u.tokens_in + u.tokens_out > 0 ? ` · ${u.tokens_in}→${u.tokens_out} tok` : ""}
                {u && u.cost_usd > 0 ? ` · ${usd(u.cost_usd)}` : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
