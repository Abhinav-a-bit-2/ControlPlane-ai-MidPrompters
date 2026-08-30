import React, { useState } from "react";

export function ChunkCitationBadge({ chunkId, chunks, onSelect }) {
  const [hovered, setHovered] = useState(false);
  const chunkData = chunks?.find((c) => c.chunk_id === chunkId);

  return (
    <span className="relative inline-block" onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <button
        onClick={() => onSelect(chunkId)}
        className="mx-0.5 px-1.5 py-0.5 text-[10px] font-mono rounded bg-emerald-950/80 text-emerald-400 border border-emerald-700/60 hover:bg-emerald-800/40 font-bold transition-all"
      >
        [{chunkId}]
      </button>

      {hovered && chunkData && (
        <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 w-64 p-2.5 rounded-md bg-slate-900 border border-slate-700 shadow-2xl z-50 text-xs font-sans text-slate-200 pointer-events-none">
          <div className="flex justify-between items-center text-[10px] font-mono text-emerald-400 mb-1 border-b border-slate-800 pb-1">
            <span>ID: {chunkData.chunk_id}</span>
            <span>Grounding Match</span>
          </div>
          <p className="line-clamp-3 text-[11px] font-mono text-slate-300 bg-slate-950 p-1.5 rounded">
            "{chunkData.content}"
          </p>
        </div>
      )}
    </span>
  );
}