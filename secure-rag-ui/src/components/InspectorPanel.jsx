import React, { useState } from "react";
import PhoenixTraceInspector from "./PhoenixTraceInspector";
import { cn } from "../libs/utils";

export default function InspectorPanel({ inspectionData, onSelectChunk }) {
    const [activeTab, setActiveTab] = useState("phoenix");

    return (
        <aside className="w-1/2 flex flex-col bg-card border-l border-border h-full">
            {/* Tab Navigation */}
            <div className="flex border-b border-border text-xs font-medium bg-muted/30">
                {[
                    { id: "phoenix", label: "Arize Phoenix Trace" },
                    { id: "grounding", label: "Grounding & Claims" },
                ].map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={cn(
                            "flex-1 py-3 border-b-2 text-center transition-colors",
                            activeTab === tab.id
                                ? "border-cyan-500 font-semibold text-foreground bg-background"
                                : "border-transparent text-muted-foreground hover:text-foreground"
                        )}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Main Container */}
            <div className="p-3 border-b border-border bg-slate-900/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">Classified Intent:</span>
                    <span className="px-2 py-0.5 rounded text-xs font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/60 font-semibold">
                        {inspectionData?.intent_label || "general"}
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">
                        ({((inspectionData?.intent_confidence || 1.0) * 100).toFixed(0)}%)
                    </span>
                </div>
            </div>
            <div className="flex-1 overflow-hidden">
                {activeTab === "phoenix" ? (
                    <PhoenixTraceInspector inspectionData={inspectionData} />
                ) : (
                    <div className="p-4 space-y-4 text-xs overflow-y-auto h-full">
                        {/* Grounding and safe chunk details */}
                        {inspectionData?.safe_chunks?.map((chunk, idx) => (
                            <button
                                key={idx}
                                onClick={() => onSelectChunk(chunk.chunk_id)}
                                className="w-full text-left p-2 border border-border bg-background hover:bg-muted/50 rounded flex items-center justify-between font-mono text-[11px]"
                            >
                                <span className="text-emerald-500 font-semibold">{chunk.chunk_id}</span>
                                <span className="text-[10px] text-muted-foreground">Inspect Chunk &rarr;</span>
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </aside>
    );
}