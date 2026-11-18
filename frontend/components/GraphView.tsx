"use client";

import "@xyflow/react/dist/style.css";

import dagre from "@dagrejs/dagre";
import { Background, type Edge, type Node, ReactFlow } from "@xyflow/react";
import { useMemo } from "react";

import type { GraphTopology } from "@/lib/events";
import { useRun } from "@/lib/store";

const NODE_W = 150;
const NODE_H = 38;

const LABELS: Record<string, string> = {
  __start__: "início",
  __end__: "fim",
  cache_exact: "cache exato",
  cache_semantic: "cache semântico",
  direct_answer: "resposta direta",
  refuse: "recusa",
  retrieve_sub: "retrieve ×N",
  merge_results: "juntar",
  grade_docs: "avaliar docs",
  retry_retrieve: "nova busca",
  compress_context: "comprimir",
  groundedness_check: "groundedness",
  give_up: "não sei",
  cache_write: "gravar cache",
};

function layout(topology: GraphTopology) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", nodesep: 26, ranksep: 34 });
  g.setDefaultEdgeLabel(() => ({}));
  topology.nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
  topology.edges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);
  return Object.fromEntries(
    topology.nodes.map((n) => {
      const p = g.node(n.id);
      return [n.id, { x: p.x - NODE_W / 2, y: p.y - NODE_H / 2 }];
    }),
  );
}

export function GraphView({ topology }: { topology: GraphTopology | null }) {
  const nodeStatus = useRun((s) => s.nodeStatus);
  const visited = useRun((s) => s.visited);
  const runStatus = useRun((s) => s.status);

  const positions = useMemo(() => (topology ? layout(topology) : {}), [topology]);

  const { nodes, edges } = useMemo(() => {
    if (!topology) return { nodes: [] as Node[], edges: [] as Edge[] };
    const started = runStatus !== "idle";
    const finished = runStatus === "done";
    const path = ["__start__", ...visited, ...(finished ? ["__end__"] : [])];
    const walked = new Set(path.slice(1).map((target, i) => `${path[i]}->${target}`));
    // arestas de nós paralelos (retrieve_sub) e laços contam pelo par, independente da ordem exata
    const seen = new Set(path);

    const nodes: Node[] = topology.nodes.map((n) => {
      const status = n.kind === "terminal" ? (seen.has(n.id) && started ? "done" : "idle") : (nodeStatus[n.id] ?? "idle");
      return {
        id: n.id,
        position: positions[n.id],
        data: { label: LABELS[n.id] ?? n.id },
        draggable: false,
        className: `xr-node xr-${status} ${n.kind === "terminal" ? "xr-terminal" : ""} ${started && status === "idle" ? "xr-skipped" : ""}`,
        style: { width: NODE_W, height: NODE_H },
      };
    });
    const edges: Edge[] = topology.edges.map((e) => {
      const active = walked.has(`${e.source}->${e.target}`) || (seen.has(e.source) && seen.has(e.target) && started);
      return {
        id: `${e.source}->${e.target}`,
        source: e.source,
        target: e.target,
        animated: active && runStatus === "running",
        className: `${active ? "xr-edge-active" : ""} ${e.conditional ? "xr-edge-cond" : ""}`,
      };
    });
    return { nodes, edges };
  }, [topology, positions, nodeStatus, visited, runStatus]);

  if (!topology) return <div className="panel-empty">carregando o grafo…</div>;
  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.12 }}
        nodesConnectable={false}
        panOnScroll
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={18} size={1} color="var(--grid)" />
      </ReactFlow>
    </div>
  );
}
