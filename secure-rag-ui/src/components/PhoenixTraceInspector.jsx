import React, { useState } from "react";
import { cn } from "../libs/utils";

export default function PhoenixTraceInspector({ inspectionData }) {
  const [selectedSpanIndex, setSelectedSpanIndex] = useState(0);
  const [detailTab, setDetailTab] = useState("attributes"); // "info" | "attributes" | "json"

  if (!inspectionData || !inspectionData.audit_trail || inspectionData.audit_trail.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-xs text-muted-foreground bg-[#090d16] p-6 text-center">
        <div className="w-8 h-8 rounded-full border border-dashed border-slate-700 flex items-center justify-center mb-2 font-mono">
          λ
        </div>
        No active trace selected. Execute a query to view spans and OpenInference attributes.
      </div>
    );
  }

  const spans = inspectionData.audit_trail.map((item, idx) => {
    let kind = "guard";
    if (item.layer.includes("generation") || item.layer.includes("output_filter")) kind = "llm";
    if (item.layer.includes("retrieval")) kind = "retriever";
    if (item.layer.includes("intent")) kind = "classifier";

    return {
      id: `span_${idx + 1}`,
      name: item.layer,
      kind: kind,
      passed: item.passed,
      latency_ms: item.latency_ms,
      detail: item.detail,
    };
  });

  const activeSpan = spans[selectedSpanIndex] || spans[0];
  const traceId = inspectionData.session_id || "42712b702f1b5901937d01c6794dd095";
  const totalLatency = spans.reduce((acc, s) => acc + (s.latency_ms || 0), 0);

  // Active span attribute key-values
  const spanAttributes = {
    "session_id": inspectionData.session_id,
    "query": inspectionData.rewritten_query || "N/A",
    "layer.name": activeSpan.name,
    "layer.status": activeSpan.passed ? "PASS" : "BLOCKED",
    "layer.detail": activeSpan.detail || "None",
    "llm.token_count.total": inspectionData.total_tokens,
    "cost_score.total": inspectionData.cost_score,
    "pipeline.confidence": inspectionData.overall_confidence,
  };

  return (
    <div className="h-full flex flex-col bg-[#090d16] text-slate-200 text-xs font-sans select-none overflow-hidden">
      {/* Top Bar / Breadcrumb */}
      <div className="h-10 border-b border-slate-800/80 px-3 flex items-center justify-between text-[11px] bg-[#0c1222]">
        <div className="flex items-center gap-1.5 font-mono text-slate-400">
          <span className="text-slate-500">Trace</span>
          <span className="text-slate-300 font-semibold truncate max-w-[150px]">ID {traceId.slice(0, 16)}...</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-[11px]">
            <span className="text-slate-500">Status</span>
            <span className={cn(
              "px-1.5 py-0.2 rounded font-mono font-medium text-[10px]",
              inspectionData.blocked ? "bg-rose-950/60 text-rose-400 border border-rose-800/50" : "bg-emerald-950/60 text-emerald-400 border border-emerald-800/50"
            )}>
              {inspectionData.blocked ? "Blocked" : "OK"}
            </span>
          </div>
          <div className="flex items-center gap-1 text-[11px]">
            <span className="text-slate-500">Total Latency</span>
            <span className="font-mono text-slate-300">{(totalLatency / 1000).toFixed(2)}s</span>
          </div>
        </div>
      </div>

      {/* Main Split Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column: Span Hierarchy Tree */}
        <div className="w-1/2 border-r border-slate-800/80 flex flex-col bg-[#0b101d]">
          <div className="p-2 border-b border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400 font-medium uppercase tracking-wider">
            <span>Spans ({spans.length})</span>
            <span>Latency</span>
          </div>

          <div className="flex-1 overflow-y-auto divide-y divide-slate-800/40">
            {spans.map((span, idx) => {
              const isSelected = selectedSpanIndex === idx;
              return (
                <div
                  key={span.id}
                  onClick={() => setSelectedSpanIndex(idx)}
                  className={cn(
                    "flex items-center justify-between px-3 py-2 cursor-pointer transition-colors text-xs",
                    isSelected ? "bg-slate-800/70 text-white font-medium border-l-2 border-cyan-500" : "hover:bg-slate-800/30 text-slate-400"
                  )}
                >
                  <div className="flex items-center gap-2 truncate">
                    {/* Status Hexagon/Circle */}
                    <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", span.passed ? "bg-emerald-400" : "bg-rose-500")} />

                    {/* Kind Pill */}
                    <span className={cn(
                      "px-1 py-0.2 rounded text-[9px] font-mono uppercase tracking-tight shrink-0",
                      span.kind === "llm" ? "bg-amber-950/60 text-amber-400 border border-amber-800/50" :
                      span.kind === "retriever" ? "bg-blue-950/60 text-blue-400 border border-blue-800/50" :
                      "bg-slate-800 text-slate-400 border border-slate-700"
                    )}>
                      {span.kind}
                    </span>

                    <span className="truncate font-mono text-[11px]">{span.name}</span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-500 shrink-0 ml-2">
                    {span.latency_ms.toFixed(1)}ms
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Span Inspector / Attributes */}
        <div className="w-1/2 flex flex-col bg-[#070b12]">
          {/* Sub Header */}
          <div className="p-2.5 border-b border-slate-800/80 bg-[#0c1222]">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-xs font-semibold text-slate-200">{activeSpan.name}</span>
              <span className="font-mono text-[10px] text-slate-400">{activeSpan.latency_ms.toFixed(2)}ms</span>
            </div>
            <div className="text-[10px] text-slate-500 font-mono">span_id: {activeSpan.id}</div>
          </div>

          {/* Sub Tab Switcher */}
          <div className="flex border-b border-slate-800/80 text-[11px] bg-[#090d16]">
            {["attributes", "json"].map((tab) => (
              <button
                key={tab}
                onClick={() => setDetailTab(tab)}
                className={cn(
                  "px-3 py-1.5 capitalize transition-colors border-b",
                  detailTab === tab ? "border-cyan-400 text-cyan-400 font-medium bg-slate-800/30" : "border-transparent text-slate-400 hover:text-slate-200"
                )}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab View */}
          <div className="flex-1 overflow-y-auto p-3 font-mono text-[11px]">
            {detailTab === "attributes" && (
              <div className="space-y-1">
                {Object.entries(spanAttributes).map(([k, v]) => (
                  <div key={k} className="flex flex-col border-b border-slate-800/40 py-1.5">
                    <span className="text-slate-500 text-[10px]">{k}</span>
                    <span className="text-slate-200 text-xs break-all mt-0.5 font-mono">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}

            {detailTab === "json" && (
              <pre className="p-2 rounded bg-slate-950 border border-slate-800 text-[10px] text-emerald-400 font-mono whitespace-pre-wrap">
                {JSON.stringify({ ...activeSpan, attributes: spanAttributes }, null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}