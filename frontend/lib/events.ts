// Tipos do contrato de eventos SSE. Espelham backend/app/events/schema.py (docs/events.schema.json).

export type EventType = "node_start" | "node_end" | "token" | "data" | "error" | "done";

export interface XrayEvent<P = Record<string, unknown>> {
  request_id: string;
  seq: number;
  ts: string;
  node: string | null;
  type: EventType;
  payload: P;
}

export interface NodeEndPayload {
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  tokens_cached: number;
  cost_usd: number;
  model: string | null;
}

export interface Citation {
  n: number;
  chunk_id: string;
  title: string;
  section: string;
  url: string;
}

export interface DonePayload {
  answer: string;
  citations: Citation[];
  total_cost_usd: number;
  total_latency_ms: number;
  counterfactual_cost_usd: number | null;
}

export interface RetrievedChunk {
  id: string;
  title: string;
  section: string;
  url: string;
  score: number;
  dense_rank: number | null;
  bm25_rank: number | null;
  rrf_rank: number | null;
}

export interface RerankedChunk {
  id: string;
  rerank_rank: number;
  score: number;
}

export interface RouterData {
  route: "trivial" | "out_of_scope" | "simple_rag" | "multi_hop";
  reason: string;
  confidence: number;
}

export interface CacheSemanticData {
  hit: boolean;
  similarity: number | null;
  threshold: number;
}

export interface GroundednessData {
  grounded: boolean;
  unsupported_claims: string[];
  next_action?: string;
}

export interface GraphTopology {
  pipeline: string;
  nodes: { id: string; kind: "terminal" | "node" }[];
  edges: { source: string; target: string; conditional: boolean; label: string | null }[];
}

export interface PipelineInfo {
  name: string;
  description: string;
  retrieval_mode: string;
  rerank: boolean;
  router: boolean;
  groundedness: boolean;
  cache: boolean;
}

export interface ChunkDetail {
  id: string;
  doc_id: string;
  title: string;
  section: string;
  url: string;
  text: string;
}

export interface EmbeddingMapData {
  query_xy: [number, number];
  hit_ids: string[];
}

/** frontend/public/corpus_map.json, gerado por backend/ingest/corpus_map.py. Em colunas: o i-ésimo
 * trecho é (ids[i], x[i], y[i], group[i], doc[i]); `group` indexa `groups` e `doc` indexa `titles`. */
export interface CorpusMap {
  groups: string[];
  titles: string[];
  ids: string[];
  x: number[];
  y: number[];
  group: number[];
  doc: number[];
}
