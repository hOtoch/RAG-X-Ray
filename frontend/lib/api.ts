import type { ChunkDetail, CorpusMap, GraphTopology, PipelineInfo, XrayEvent } from "./events";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export const fetchPipelines = () => getJson<PipelineInfo[]>("/pipelines");
export const fetchGraph = (pipeline: string) => getJson<GraphTopology>(`/graph/${encodeURIComponent(pipeline)}`);
// Trechos que vieram junto de um demo gravado: o replay abre citações sem chamar o backend.
const preloadedChunks = new Map<string, ChunkDetail>();

export const fetchChunk = async (id: string) =>
  preloadedChunks.get(id) ?? getJson<ChunkDetail>(`/chunks/${encodeURIComponent(id)}`);
export const fetchTrace = (requestId: string) => getJson<XrayEvent[]>(`/traces/${requestId}`);

/** Lê um corpo text/event-stream e chama onEvent para cada evento. */
async function readSse(body: ReadableStream<Uint8Array>, onEvent: (event: XrayEvent) => void) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (data) onEvent(JSON.parse(data) as XrayEvent);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

/** Pergunta ao vivo: POST /query/stream. */
export async function streamQuery(
  question: string,
  pipeline: string,
  onEvent: (event: XrayEvent) => void,
  signal?: AbortSignal,
) {
  const response = await fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, pipeline }),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`HTTP ${response.status} ${detail}`);
  }
  await readSse(response.body, onEvent);
}

/** Replay: reemite eventos gravados respeitando os intervalos originais, sem chamar o backend de novo. */
export async function replayEvents(
  events: XrayEvent[],
  onEvent: (event: XrayEvent) => void,
  { speed = 1, signal }: { speed?: number; signal?: AbortSignal } = {},
) {
  let previous: number | null = null;
  for (const event of events) {
    if (signal?.aborted) return;
    const ts = Date.parse(event.ts);
    if (previous !== null) {
      const wait = Math.min((ts - previous) / speed, 1500);
      if (wait > 4) await new Promise((resolve) => setTimeout(resolve, wait));
    }
    previous = ts;
    onEvent(event);
  }
}

export interface Demo {
  request_id: string;
  question: string;
  pipeline: string;
  label: string;
  note?: string;
}

export interface DemoTrace {
  demo: Demo;
  topology: GraphTopology;
  events: XrayEvent[];
  chunks?: ChunkDetail[];
}

/** Traces gravados para o modo replay. São arquivos estáticos em public/demos/, gerados por
 * backend/scripts/record_demos.py: o replay não depende do backend nem chama LLM. */
export async function fetchDemos(): Promise<Demo[]> {
  const response = await fetch("/demos/index.json");
  if (!response.ok) return [];
  return response.json() as Promise<Demo[]>;
}

export async function fetchDemoTrace(requestId: string): Promise<DemoTrace> {
  const response = await fetch(`/demos/${requestId}.json`);
  if (!response.ok) throw new Error(`trace ${requestId}: HTTP ${response.status}`);
  const trace = (await response.json()) as DemoTrace;
  trace.chunks?.forEach((chunk) => preloadedChunks.set(chunk.id, chunk));
  return trace;
}

export async function sendFeedback(requestId: string, value: 1 | -1) {
  const response = await fetch(`${API_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId, value }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
}

let corpusMap: Promise<CorpusMap | null> | null = null;

/** Mapa 2D do corpus (arquivo estático, ~1 MB). Baixado uma vez e compartilhado pelos painéis. */
export function fetchCorpusMap(): Promise<CorpusMap | null> {
  corpusMap ??= fetch("/corpus_map.json")
    .then((r) => (r.ok ? (r.json() as Promise<CorpusMap>) : null))
    .catch(() => null);
  return corpusMap;
}
