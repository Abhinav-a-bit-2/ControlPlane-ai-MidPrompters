import React, { useState } from "react";
import { cn } from "../libs/utils";
import { ActivityIcon, CheckCircleIcon, ExclamationCircleIcon } from "./Icons";

export default function PhoenixTraceInspector({ inspectionData }) {
  const [selectedSpanIndex, setSelectedSpanIndex] = useState(0);
  const [detailTab, setDetailTab] = useState("attributes"); // "attributes" | "json"

  if (!inspectionData || !inspectionData.audit_trail || inspectionData.audit_trail.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-xs text-muted-foreground bg-card/40 p-8 text-center">
        <div className="w-10 h-10 rounded-full border border-border flex items-center justify-center mb-3 text-muted-foreground bg-muted/30">
          <ActivityIcon className="w-5 h-5 opacity-60" />
        </div>
        <p className="font-medium text-foreground text-xs">No active telemetry trace</p>
        <p className="text-[11px] text-muted-foreground mt-1 max-w-[240px] font-sans">
          Execute a query in the chat console to record OpenInference span execution telemetry.
        </p>
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
    "session.id": inspectionData.session_id,
    "query.rewritten": inspectionData.rewritten_query || "N/A",
    "layer.name": activeSpan.name,
    "layer.verdict": activeSpan.passed ? "PASSED" : "BLOCKED",
    "layer.detail": activeSpan.detail || "None",
    "tokens.total": inspectionData.total_tokens || "N/A",
    "cost.score": inspectionData.cost_score !== undefined ? inspectionData.cost_score : "0.00",
    "grounding.confidence": inspectionData.overall_confidence ? `${(inspectionData.overall_confidence * 100).toFixed(1)}%` : "N/A",
  };

  return (
    <div className="h-full flex flex-col bg-card text-foreground text-xs font-sans select-none overflow-hidden border-t border-border">
      {/* Top Bar / Metadata */}
      <div className="h-11 border-b border-border px-4 flex items-center justify-between text-xs bg-muted/30">
        <div className="flex items-center gap-2 font-mono">
          <span className="text-muted-foreground text-[11px] uppercase tracking-wider">Trace:</span>
          <span className="text-foreground font-semibold truncate max-w-[140px]">{traceId.slice(0, 14)}...</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[11px]">
            <span className="text-muted-foreground font-mono">Status:</span>
            <span className={cn(
              "px-2 py-0.5 rounded text-[10px] font-mono font-medium inline-flex items-center gap-1",
              inspectionData.blocked 
                ? "bg-destructive/15 text-destructive border border-destructive/30" 
                : "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30"
            )}>
              {inspectionData.blocked ? <ExclamationCircleIcon className="w-3 h-3" /> : <CheckCircleIcon className="w-3 h-3" />}
              {inspectionData.blocked ? "BLOCKED" : "SANITIZED"}
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] font-mono">
            <span className="text-muted-foreground">Latency:</span>
            <span className="font-semibold text-primary">{(totalLatency / 1000).toFixed(2)}s</span>
          </div>
        </div>
      </div>

      {/* Split Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column: Span Hierarchy Tree */}
        <div className="w-1/2 border-r border-border flex flex-col bg-background/50">
          <div className="px-3 py-2 border-b border-border flex items-center justify-between text-[10px] text-muted-foreground font-mono uppercase tracking-wider bg-muted/20">
            <span>Execution Spans ({spans.length})</span>
            <span>Duration</span>
          </div>

          <div className="flex-1 overflow-y-auto divide-y divide-border/40">
            {spans.map((span, idx) => {
              const isSelected = selectedSpanIndex === idx;
              return (
                <div
                  key={span.id}
                  onClick={() => setSelectedSpanIndex(idx)}
                  className={cn(
                    "flex items-center justify-between px-3 py-2.5 cursor-pointer transition-colors text-xs",
                    isSelected 
                      ? "bg-primary/10 text-foreground font-medium border-l-2 border-primary" 
                      : "hover:bg-muted/40 text-muted-foreground hover:text-foreground"
                  )}
                >
                  <div className="flex items-center gap-2 truncate">
                    <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", span.passed ? "bg-emerald-500" : "bg-destructive")} />
                    
                    <span className={cn(
                      "px-1.5 py-0.2 rounded text-[9px] font-mono uppercase tracking-tight shrink-0 border",
                      span.kind === "llm" ? "bg-primary/10 text-primary border-primary/30" :
                      span.kind === "retriever" ? "bg-secondary/10 text-secondary border-secondary/30" :
                      "bg-muted text-muted-foreground border-border"
                    )}>
                      {span.kind}
                    </span>

                    <span className="truncate font-mono text-[11px]">{span.name}</span>
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground shrink-0 ml-2">
                    {span.latency_ms.toFixed(1)}ms
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Attributes Inspector */}
        <div className="w-1/2 flex flex-col bg-card">
          {/* Sub Header */}
          <div className="p-3 border-b border-border bg-muted/10">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-xs font-semibold text-foreground truncate">{activeSpan.name}</span>
              <span className="font-mono text-[10px] text-muted-foreground shrink-0">{activeSpan.latency_ms.toFixed(2)}ms</span>
            </div>
            <div className="text-[10px] text-muted-foreground font-mono">id: {activeSpan.id}</div>
          </div>

          {/* Tab Switcher */}
          <div className="flex border-b border-border text-xs bg-muted/20">
            {["attributes", "json"].map((tab) => (
              <button
                key={tab}
                onClick={() => setDetailTab(tab)}
                className={cn(
                  "px-3.5 py-1.5 capitalize font-mono text-[11px] transition-colors border-b-2",
                  detailTab === tab 
                    ? "border-primary text-primary font-semibold bg-card" 
                    : "border-transparent text-muted-foreground hover:text-foreground"
                )}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Tab View */}
          <div className="flex-1 overflow-y-auto p-3 font-mono text-xs">
            {detailTab === "attributes" && (
              <div className="space-y-2">
                {Object.entries(spanAttributes).map(([k, v]) => (
                  <div key={k} className="flex flex-col border-b border-border/50 pb-1.5">
                    <span className="text-muted-foreground text-[10px] uppercase font-sans tracking-wider">{k}</span>
                    <span className="text-foreground text-xs break-all mt-0.5 font-mono">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}

            {detailTab === "json" && (
              <pre className="p-3 rounded bg-muted/40 border border-border text-[11px] text-foreground font-mono whitespace-pre-wrap">
                {JSON.stringify({ ...activeSpan, attributes: spanAttributes }, null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}