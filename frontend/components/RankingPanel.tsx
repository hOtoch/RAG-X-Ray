"use client";

import { useMemo } from "react";

import type { RerankedChunk, RetrievedChunk } from "@/lib/events";
import { useRun } from "@/lib/store";

const STAGES = ["denso", "BM25", "RRF", "rerank"] as const;
const MAX_RANK = 20;

/** Bump chart: a posição de cada chunk em cada etapa do retrieval. Linhas que sobem mostram
 * trechos que uma etapa resgatou; pontos ausentes são trechos que a etapa não trouxe. */
export function RankingPanel() {
  const data = useRun((s) => s.data);
  const done = useRun((s) => s.done);

  const rows = useMemo(() => {
    const retrieved = (data.retrieve?.at(-1)?.chunks as RetrievedChunk[] | undefined) ?? [];
    const reranked = (data.rerank?.at(-1)?.chunks as RerankedChunk[] | undefined) ?? [];
    const rerankRank = new Map(reranked.map((c) => [c.id, c.rerank_rank]));
    const cited = new Set(done?.citations.map((c) => c.chunk_id) ?? []);
    return retrieved.slice(0, MAX_RANK).map((c) => ({
      id: c.id,
      label: c.section ? `${c.title} › ${c.section}` : c.title,
      ranks: [c.dense_rank, c.bm25_rank, c.rrf_rank, rerankRank.get(c.id) ?? null] as (number | null)[],
      cited: cited.has(c.id),
      final: rerankRank.has(c.id),
    }));
  }, [data, done]);

  const hasLexical = rows.some((r) => r.ranks[1] !== null);
  const hasRerank = rows.some((r) => r.ranks[3] !== null);
  const stages = STAGES.map((name, i) => ({ name, i })).filter(
    ({ i }) => (i !== 1 || hasLexical) && (i !== 2 || hasLexical) && (i !== 3 || hasRerank),
  );

  if (!rows.length) return <div className="panel-empty">O ranking aparece depois do retrieve.</div>;
  if (stages.length < 2)
    return (
      <div className="panel-empty">
        Esta versão só faz busca densa, então há uma única etapa de ranking. Nas versões com retrieval híbrido e
        rerank, o gráfico mostra como cada trecho sobe ou desce entre BM25, denso, RRF e rerank.
      </div>
    );

  const W = 360;
  const H = 240;
  const pad = { l: 8, r: 8, t: 22, b: 6 };
  const x = (k: number) => pad.l + (k * (W - pad.l - pad.r)) / Math.max(stages.length - 1, 1);
  const maxRank = Math.max(...rows.flatMap((r) => r.ranks.filter((v): v is number => v !== null)), 5);
  const y = (rank: number) => pad.t + ((rank - 1) * (H - pad.t - pad.b)) / Math.max(maxRank - 1, 1);

  return (
    <div className="flex h-full flex-col">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full flex-1" role="img" aria-label="Posição dos trechos por etapa">
        {stages.map((s, k) => (
          <g key={s.name}>
            <line x1={x(k)} x2={x(k)} y1={pad.t - 4} y2={H - pad.b} className="bump-axis" />
            <text x={x(k)} y={12} textAnchor={k === 0 ? "start" : k === stages.length - 1 ? "end" : "middle"} className="bump-stage">
              {s.name}
            </text>
          </g>
        ))}
        {rows.map((row) => {
          const points = stages
            .map((s, k) => (row.ranks[s.i] !== null ? { x: x(k), y: y(row.ranks[s.i] as number) } : null))
            .filter((p): p is { x: number; y: number } => p !== null);
          const cls = row.cited ? "bump-cited" : row.final ? "bump-final" : "bump-other";
          return (
            <g key={row.id} className={cls}>
              <title>{row.label}</title>
              <polyline points={points.map((p) => `${p.x},${p.y}`).join(" ")} fill="none" />
              {points.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={row.cited ? 3.4 : 2.4} />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex gap-4 text-[11px] text-[var(--muted)]">
        <span><i className="legend legend-cited" />citado na resposta</span>
        <span><i className="legend legend-final" />foi para a geração</span>
        <span><i className="legend legend-other" />descartado</span>
      </div>
    </div>
  );
}
