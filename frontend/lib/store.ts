import { create } from "zustand";

import type { DonePayload, NodeEndPayload, XrayEvent } from "./events";

export type NodeStatus = "idle" | "running" | "done" | "error";

export interface NodeSpan {
  node: string;
  start: number; // ms desde o início do request
  end: number | null;
  usage: NodeEndPayload | null;
}

interface RunState {
  requestId: string | null;
  status: "idle" | "running" | "done" | "error";
  error: string | null;
  t0: number | null;
  events: XrayEvent[];
  nodeStatus: Record<string, NodeStatus>;
  spans: NodeSpan[];
  data: Record<string, Record<string, unknown>[]>; // eventos `data` por nó (retrieve_sub aparece várias vezes)
  visited: string[]; // ordem em que os nós rodaram
  answer: string;
  done: DonePayload | null;
  reset: () => void;
  apply: (event: XrayEvent) => void;
  fail: (message: string) => void;
}

const initial = {
  requestId: null,
  status: "idle" as const,
  error: null,
  t0: null,
  events: [],
  nodeStatus: {},
  spans: [],
  data: {},
  visited: [],
  answer: "",
  done: null,
};

export const useRun = create<RunState>((set) => ({
  ...initial,
  reset: () => set({ ...initial, status: "running" }),
  fail: (message) => set({ status: "error", error: message }),
  apply: (event) =>
    set((state) => {
      const ts = Date.parse(event.ts);
      const t0 = state.t0 ?? ts;
      const at = ts - t0;
      const next: Partial<RunState> = { t0, requestId: event.request_id, events: [...state.events, event] };
      const node = event.node ?? "";

      switch (event.type) {
        case "node_start":
          next.nodeStatus = { ...state.nodeStatus, [node]: "running" };
          next.spans = [...state.spans, { node, start: at, end: null, usage: null }];
          next.visited = [...state.visited, node];
          break;
        case "node_end": {
          next.nodeStatus = { ...state.nodeStatus, [node]: "done" };
          // fecha o último span aberto desse nó (nós paralelos têm o mesmo nome)
          const spans = [...state.spans];
          for (let i = spans.length - 1; i >= 0; i--) {
            if (spans[i].node === node && spans[i].end === null) {
              spans[i] = { ...spans[i], end: at, usage: event.payload as unknown as NodeEndPayload };
              break;
            }
          }
          next.spans = spans;
          break;
        }
        case "data":
          next.data = { ...state.data, [node]: [...(state.data[node] ?? []), event.payload] };
          break;
        case "token":
          next.answer = state.answer + String(event.payload.text ?? "");
          break;
        case "done": {
          const done = event.payload as unknown as DonePayload;
          next.done = done;
          next.answer = done.answer; // a resposta final pode vir de cache ou de "não sei", sem tokens
          next.status = "done";
          break;
        }
        case "error":
          next.status = "error";
          next.error = String(event.payload.message ?? "erro");
          next.nodeStatus = Object.fromEntries(
            Object.entries(state.nodeStatus).map(([k, v]) => [k, v === "running" ? "error" : v]),
          );
          break;
      }
      return next;
    }),
}));
