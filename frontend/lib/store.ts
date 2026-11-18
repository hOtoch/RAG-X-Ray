import { createContext, createElement, type ReactNode, useContext, useState } from "react";
import { createStore, type StoreApi, useStore } from "zustand";

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

const runState = (set: StoreApi<RunState>["setState"]): RunState => ({
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
});

export type RunStore = StoreApi<RunState>;

/** Um store por execução: a página principal usa o padrão; o modo de comparação cria um por lado. */
export const createRunStore = (): RunStore => createStore<RunState>()((set) => runState(set));

const defaultStore = createRunStore();
const RunContext = createContext<RunStore>(defaultStore);

/** Os componentes abaixo deste provider leem o store dado, em vez do padrão. */
export function RunProvider({ store, children }: { store: RunStore; children: ReactNode }) {
  return createElement(RunContext.Provider, { value: store }, children);
}

/** Hook usado pelos componentes: lê o store do provider mais próximo. */
export function useRun(): RunState;
export function useRun<T>(selector: (state: RunState) => T): T;
export function useRun<T>(selector?: (state: RunState) => T) {
  const store = useContext(RunContext);
  return useStore(store, selector ?? ((state: RunState) => state as unknown as T));
}

/** Cria um store estável para a vida do componente. */
export function useNewRunStore(): RunStore {
  const [store] = useState(createRunStore);
  return store;
}
