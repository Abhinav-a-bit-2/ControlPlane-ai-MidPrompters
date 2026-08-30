import React, { useState } from "react";
import PhoenixTraceInspector from "./PhoenixTraceInspector";
import { cn } from "../libs/utils";
import { LayersIcon, DocumentTextIcon } from "./Icons";

export default function InspectorPanel({ inspectionData, onSelectChunk }) {
    const [activeTab, setActiveTab] = useState("phoenix");

    return (
        <aside className="w-5/12 flex flex-col bg-card border-l border-border h-full">
            {/* Tab Navigation */}
            <div className="flex border-b border-border text-xs font-medium bg-muted/40 shrink-0">
                {[
                    { id: "phoenix", label: "OpenInference Trace", icon: LayersIcon },
                    { id: "grounding", label: "Grounding & Evidence", icon: DocumentTextIcon },
                ].map((tab) => {
                    const Icon = tab.icon;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={cn(
                                "flex-1 py-3 border-b-2 text-center transition-all text-xs font-mono flex items-center justify-center gap-1.5",
                                activeTab === tab.id
                                    ? "border-primary font-semibold text-primary bg-card"
                                    : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/30"
                            )}
                        >
                            <Icon className="w-3.5 h-3.5" />
                            <span>{tab.label}</span>
                        </button>
                    );
                })}
            </div>

            {/* Metadata Bar */}
            <div className="px-4 py-2.5 border-b border-border bg-muted/15 flex items-center justify-between text-xs font-mono shrink-0">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] uppercase text-muted-foreground">Classified Intent:</span>
                    <span className="px-2 py-0.5 rounded text-[11px] bg-secondary/15 text-secondary border border-secondary/30 font-medium">
                        {inspectionData?.intent_label || "general"}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                        ({((inspectionData?.intent_confidence || 1.0) * 100).toFixed(0)}%)
                    </span>
                </div>
                {inspectionData?.cost_score !== undefined && (
                    <div className="flex items-center gap-1.5 text-[11px]">
                        <span className="text-muted-foreground text-[10px] uppercase">L4 Metric:</span>
                        <span className="font-semibold text-foreground">{inspectionData.cost_score}</span>
                    </div>
                )}
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden">
                {activeTab === "phoenix" ? (
                    <PhoenixTraceInspector inspectionData={inspectionData} />
                ) : (
                    <div className="p-5 space-y-3 text-xs overflow-y-auto h-full bg-background/40">
                        <div className="flex items-center justify-between border-b border-border pb-2">
                            <span className="label-caps text-muted-foreground">
                                Retrieved Knowledge Passages ({inspectionData?.safe_chunks?.length || 0})
                            </span>
                            <span className="text-[10px] font-mono text-muted-foreground">Cos-Sim Rank</span>
                        </div>
                        {(!inspectionData?.safe_chunks || inspectionData.safe_chunks.length === 0) ? (
                            <div className="p-6 rounded border border-dashed border-border text-center text-muted-foreground text-xs font-mono">
                                No grounded passages cited in current pipeline step.
                            </div>
                        ) : (
                            inspectionData.safe_chunks.map((chunk, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => onSelectChunk(chunk.chunk_id)}
                                    className="w-full text-left p-3 border border-border bg-card hover:border-primary/50 hover:shadow-xs rounded transition-all group flex items-center justify-between font-mono text-xs"
                                >
                                    <div className="flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-primary" />
                                        <span className="text-foreground font-semibold group-hover:text-primary transition-colors">
                                            [{chunk.chunk_id}]
                                        </span>
                                    </div>
                                    <span className="text-[11px] text-muted-foreground group-hover:text-foreground font-sans">
                                        Inspect Vector &rarr;
                                    </span>
                                </button>
                            ))
                        )}
                    </div>
                )}
            </div>
        </aside>
    );
}