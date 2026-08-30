import React, { useState } from "react";

export function ChunkCitationBadge({ chunkId, chunks, onSelect }) {
  const [hovered, setHovered] = useState(false);
  const chunkData = chunks?.find((c) => c.chunk_id === chunkId);

  return (
    <span className="relative inline-block" onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <button
        type="button"
        onClick={() => onSelect(chunkId)}
        className="mx-0.5 px-1.5 py-0.2 text-[10px] font-mono rounded bg-primary/10 text-primary border border-primary/25 hover:bg-primary/20 hover:border-primary/40 font-medium transition-colors inline-flex items-center gap-0.5"
      >
        <span className="opacity-70 font-serif italic text-[11px]">§</span>
        <span>{chunkId.replace('chunk-', '')}</span>
      </button>

      {hovered && chunkData && (
        <div className="absolute bottom-full mb-1.5 left-1/2 -translate-x-1/2 w-72 p-3 rounded-md bg-card border border-border shadow-xl z-50 text-xs text-foreground pointer-events-none">
          <div className="flex justify-between items-center text-[10px] font-mono text-primary mb-1.5 border-b border-border pb-1">
            <span className="font-semibold">Passage ID: {chunkData.chunk_id}</span>
            <span className="text-[9px] uppercase tracking-wider text-muted-foreground">Grounding Context</span>
          </div>
          <p className="line-clamp-3 text-[11px] font-mono text-muted-foreground bg-muted/50 p-2 rounded border border-border/60 leading-relaxed">
            "{chunkData.content}"
          </p>
        </div>
      )}
    </span>
  );
}