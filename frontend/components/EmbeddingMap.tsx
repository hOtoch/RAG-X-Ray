"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { fetchCorpusMap } from "@/lib/api";
import type { CorpusMap, EmbeddingMapData } from "@/lib/events";
import { useRun } from "@/lib/store";

// Uma cor por área da documentação; a última ("outros") fica neutra.
const PALETTE = ["#5b8def", "#c678dd", "#4fb6a8", "#e0a458", "#d16d9e", "#8fbf5a", "#d9746a", "#7fa7c9", "#b5a06b"];
const OTHER = "#4a5866";
const PAD = 14;

interface View {
  x0: number;
  y0: number;
  size: number; // lado do quadrado do mapa visível, em unidades do mapa (0..1)
}

const FULL: View = { x0: 0, y0: 0, size: 1 };

function colorOf(map: CorpusMap, group: number) {
  return group < map.groups.length - 1 ? PALETTE[group % PALETTE.length] : OTHER;
}

/** Nuvem de pontos do corpus (UMAP dos embeddings) com a pergunta projetada e os trechos recuperados.
 * Mostra em que região da documentação a pergunta caiu e se os trechos vieram de um só lugar ou de vários. */
export function EmbeddingMap() {
  const data = useRun((s) => s.data);
  const done = useRun((s) => s.done);
  const [map, setMap] = useState<CorpusMap | null | undefined>(undefined);
  const [zoomed, setZoomed] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);
  const [hover, setHover] = useState<{ i: number; left: number; top: number } | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchCorpusMap().then(setMap);
  }, []);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setSize({ w: entry.contentRect.width, h: entry.contentRect.height }),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [map]);

  const query = data.embedding_map?.at(-1) as EmbeddingMapData | undefined;
  const index = useMemo(() => new Map(map?.ids.map((id, i) => [id, i]) ?? []), [map]);
  const hits = useMemo(
    () => (query?.hit_ids ?? []).map((id) => index.get(id)).filter((i): i is number => i !== undefined),
    [query, index],
  );
  const cited = useMemo(
    () => new Set((done?.citations ?? []).map((c) => index.get(c.chunk_id)).filter((i) => i !== undefined)),
    [done, index],
  );

  useEffect(() => setZoomed(false), [query]);

  const view = useMemo<View>(() => {
    if (!zoomed || !map || !query) return FULL;
    const xs = [query.query_xy[0], ...hits.map((i) => map.x[i])];
    const ys = [query.query_xy[1], ...hits.map((i) => map.y[i])];
    const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.04) * 1.5;
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    return { x0: cx - side / 2, y0: cy - side / 2, size: side };
  }, [zoomed, map, query, hits]);

  // Do espaço do mapa (0..1, y para cima) para pixels do canvas, mantendo a proporção.
  const scale = Math.max(Math.min(size.w, size.h) - 2 * PAD, 1) / view.size;
  const offX = (size.w - view.size * scale) / 2;
  const offY = (size.h - view.size * scale) / 2;
  const toPx = (x: number, y: number): [number, number] => [
    offX + (x - view.x0) * scale,
    size.h - (offY + (y - view.y0) * scale),
  ];

  useEffect(() => {
    const el = canvas.current;
    if (!el || !map || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = size.w * dpr;
    el.height = size.h * dpr;
    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const base = query ? 0.22 : 0.5;
    const radius = zoomed ? 2.2 : 1.3;
    for (let i = 0; i < map.ids.length; i++) {
      const dim = focus !== null && map.group[i] !== focus;
      ctx.globalAlpha = dim ? 0.05 : base;
      ctx.fillStyle = colorOf(map, map.group[i]);
      const [px, py] = toPx(map.x[i], map.y[i]);
      ctx.fillRect(px - radius / 2, py - radius / 2, radius, radius);
    }
    ctx.globalAlpha = 1;

    if (query) {
      const [qx, qy] = toPx(...query.query_xy);
      // linhas da pergunta até cada trecho recuperado
      ctx.strokeStyle = "rgba(79, 209, 232, 0.28)";
      ctx.lineWidth = 1;
      for (const i of hits) {
        const [px, py] = toPx(map.x[i], map.y[i]);
        ctx.beginPath();
        ctx.moveTo(qx, qy);
        ctx.lineTo(px, py);
        ctx.stroke();
      }
      for (const i of hits) {
        const [px, py] = toPx(map.x[i], map.y[i]);
        ctx.beginPath();
        ctx.arc(px, py, cited.has(i) ? 4.5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = cited.has(i) ? "#6fd39a" : "#4fd1e8";
        ctx.fill();
      }
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(qx - 6, qy);
      ctx.lineTo(qx + 6, qy);
      ctx.moveTo(qx, qy - 6);
      ctx.lineTo(qx, qy + 6);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(qx, qy, 9, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    // toPx depende de view/size, já listados
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, size, query, hits, cited, focus, view, zoomed]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!map) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // pontos destacados primeiro, depois o resto
    let best = -1;
    let bestD = 64; // 8 px ao quadrado
    const check = (i: number, limit: number) => {
      const [px, py] = toPx(map.x[i], map.y[i]);
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < Math.min(bestD, limit)) {
        best = i;
        bestD = d;
      }
    };
    hits.forEach((i) => check(i, 100));
    if (best < 0) for (let i = 0; i < map.ids.length; i++) check(i, 25);
    setHover(best >= 0 ? { i: best, left: mx, top: my } : null);
  };

  if (map === undefined) return <div className="panel-empty">Carregando o mapa…</div>;
  if (map === null)
    return (
      <div className="panel-empty">
        Mapa não gerado. Rode <code>uv run python -m ingest.corpus_map</code> no backend.
      </div>
    );

  const hitRank = hover ? hits.indexOf(hover.i) : -1;

  return (
    <div className="emap">
      <div className="emap-canvas" ref={wrap}>
        <canvas
          ref={canvas}
          style={{ width: size.w, height: size.h }}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          aria-label="Mapa 2D dos trechos do corpus"
          role="img"
        />
        {hover && (
          <div className="emap-tip" style={{ left: hover.left + 12, top: hover.top + 12 }}>
            <strong>{map.titles[map.doc[hover.i]]}</strong>
            <span>
              {map.groups[map.group[hover.i]]}
              {hitRank >= 0 && ` · recuperado #${hitRank + 1}`}
              {cited.has(hover.i) && " · citado"}
            </span>
          </div>
        )}
        {query && (
          <button className="emap-zoom" onClick={() => setZoomed((z) => !z)}>
            {zoomed ? "Ver o corpus inteiro" : "Aproximar da pergunta"}
          </button>
        )}
        {!query && <p className="emap-hint">{map.ids.length.toLocaleString("pt-BR")} trechos. A pergunta aparece aqui depois do retrieve.</p>}
      </div>
      <ul className="emap-legend">
        {map.groups.map((g, i) => (
          <li key={g}>
            <button
              aria-pressed={focus === i}
              onClick={() => setFocus((f) => (f === i ? null : i))}
              title="Destacar só esta área"
            >
              <i style={{ background: colorOf(map, i) }} />
              {g}
            </button>
          </li>
        ))}
        <li className="emap-key">
          <i className="k-query" /> pergunta <i className="k-hit" /> recuperado <i className="k-cited" /> citado
        </li>
      </ul>
    </div>
  );
}
