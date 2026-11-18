"use client";

import { Fragment, useState } from "react";

import { fetchChunk, sendFeedback } from "@/lib/api";
import type { ChunkDetail, Citation } from "@/lib/events";
import { useRun } from "@/lib/store";

/** Quebra o texto em blocos de código (```) e parágrafos, e troca [n] por botões de citação. */
function render(text: string, citations: Citation[], onCite: (c: Citation) => void) {
  const byN = new Map(citations.map((c) => [c.n, c]));
  const parts = text.split(/(```[\s\S]*?(?:```|$))/g);
  return parts.map((part, i) => {
    if (part.startsWith("```")) {
      const code = part.replace(/^```\w*\n?/, "").replace(/```$/, "");
      return <pre key={i} className="answer-code">{code}</pre>;
    }
    const pieces = part.split(/(\[\d+\])/g);
    return (
      <Fragment key={i}>
        {pieces.map((piece, j) => {
          const match = piece.match(/^\[(\d+)\]$/);
          const citation = match ? byN.get(Number(match[1])) : undefined;
          if (citation) {
            return (
              <button key={j} className="cite" onClick={() => onCite(citation)} title={citation.title}>
                {citation.n}
              </button>
            );
          }
          return <Fragment key={j}>{piece.replace(/`([^`]+)`/g, "$1")}</Fragment>;
        })}
      </Fragment>
    );
  });
}

export function AnswerWithCitations({ live }: { live: boolean }) {
  const answer = useRun((s) => s.answer);
  const done = useRun((s) => s.done);
  const status = useRun((s) => s.status);
  const error = useRun((s) => s.error);
  const requestId = useRun((s) => s.requestId);
  const [vote, setVote] = useState<{ id: string; value: 1 | -1 | "error" } | null>(null);
  const [open, setOpen] = useState<{ citation: Citation; chunk: ChunkDetail | null; error?: string } | null>(null);

  const cite = async (citation: Citation) => {
    setOpen({ citation, chunk: null });
    try {
      setOpen({ citation, chunk: await fetchChunk(citation.chunk_id) });
    } catch (e) {
      setOpen({ citation, chunk: null, error: String(e) });
    }
  };

  if (status === "idle") return <div className="panel-empty">A resposta aparece aqui, com as citações clicáveis.</div>;

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      <div className="answer flex-1 overflow-y-auto">
        {answer ? render(answer, done?.citations ?? [], cite) : status === "running" ? <span className="cursor">▍</span> : null}
        {status === "error" && <p className="text-[var(--bad)]">Erro: {error}</p>}
      </div>
      {live && status === "done" && requestId && (
        <div className="feedback">
          <span>A resposta ajudou?</span>
          {([1, -1] as const).map((value) => (
            <button
              key={value}
              aria-pressed={vote?.id === requestId && vote.value === value}
              disabled={vote?.id === requestId}
              onClick={() =>
                sendFeedback(requestId, value)
                  .then(() => setVote({ id: requestId, value }))
                  .catch(() => setVote({ id: requestId, value: "error" }))
              }
            >
              {value === 1 ? "👍" : "👎"}
            </button>
          ))}
          {vote?.id === requestId && <em>{vote.value === "error" ? "não foi possível enviar" : "obrigado, ligado ao trace"}</em>}
        </div>
      )}
      {open && (
        <aside className="source">
          <header>
            <span className="cite cite-static">{open.citation.n}</span>
            <a href={open.citation.url} target="_blank" rel="noreferrer">
              {open.citation.title}
              {open.citation.section ? ` › ${open.citation.section}` : ""} ↗
            </a>
            <button className="ml-auto text-[var(--muted)]" onClick={() => setOpen(null)} aria-label="Fechar">
              ✕
            </button>
          </header>
          <pre>{open.chunk?.text ?? open.error ?? "carregando…"}</pre>
        </aside>
      )}
    </div>
  );
}
