"use client";

import type { CacheSemanticData, GroundednessData, RouterData } from "@/lib/events";
import { useRun } from "@/lib/store";

const ROUTES: Record<string, string> = {
  trivial: "conversa",
  out_of_scope: "fora de escopo",
  simple_rag: "RAG simples",
  multi_hop: "multi-hop",
};

export function RouterExplainer() {
  const data = useRun((s) => s.data);
  const spans = useRun((s) => s.spans);
  const status = useRun((s) => s.status);

  const router = data.router?.at(-1) as RouterData | undefined;
  const exact = data.cache_exact?.at(-1) as { hit: boolean } | undefined;
  const semantic = data.cache_semantic?.at(-1) as CacheSemanticData | undefined;
  const grounded = data.groundedness_check as unknown as GroundednessData[] | undefined;
  const generations = spans.filter((s) => (s.node === "generate" || s.node === "direct_answer") && s.usage?.model);
  const model = generations.at(-1)?.usage?.model;

  if (status === "idle") return null;

  const cache = exact?.hit
    ? "acerto exato"
    : semantic?.hit
      ? `acerto semântico (${semantic.similarity?.toFixed(3)})`
      : semantic
        ? `miss${semantic.similarity != null ? ` · mais próximo ${semantic.similarity.toFixed(3)} ≤ θ ${semantic.threshold}` : ""}`
        : exact
          ? "miss"
          : null;

  const items: [string, string][] = [];
  if (router) items.push(["Rota", `${ROUTES[router.route] ?? router.route} · ${Math.round(router.confidence * 100)}%`]);
  if (model) items.push(["Modelo", generations.length > 1 ? `${generations.map((g) => g.usage?.model).join(" → ")}` : model]);
  if (router) items.push(["Motivo", router.reason]);
  if (cache) items.push(["Cache", cache]);
  if (grounded?.length) {
    const last = grounded.at(-1)!;
    items.push([
      "Groundedness",
      last.grounded
        ? grounded.length > 1 ? "ok após escalar o modelo" : "ok"
        : `falhou: ${last.unsupported_claims.slice(0, 2).join("; ")}`,
    ]);
  }
  if (!items.length) return null;

  return (
    <dl className="router-strip">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
