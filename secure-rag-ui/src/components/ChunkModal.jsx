import React, { useEffect, useState } from "react";
import { CloseIcon } from "./Icons";

export default function ChunkModal({ chunkId, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!chunkId) return;
    setLoading(true);
    fetch(`http://localhost:8080/api/chunks/${chunkId}`)
      .then((res) => res.json())
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [chunkId]);

  if (!chunkId) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-background border border-border w-full max-w-lg rounded-lg shadow-lg overflow-hidden flex flex-col max-h-[80vh]">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/40">
          <span className="font-mono text-xs font-semibold uppercase tracking-wider">{chunkId}</span>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="p-4 overflow-y-auto text-sm">
          {loading ? (
            <div className="text-xs text-muted-foreground">Fetching passage data...</div>
          ) : data?.content ? (
            <div className="space-y-3">
              <div className="text-xs text-muted-foreground font-mono">Source File: {data.metadata?.filename || "corpus"}</div>
              <div className="p-3 bg-muted/30 rounded border border-border text-foreground leading-relaxed whitespace-pre-wrap font-mono text-xs">
                {data.content}
              </div>
            </div>
          ) : (
            <div className="text-xs text-rose-500">Could not retrieve chunk content.</div>
          )}
        </div>
      </div>
    </div>
  );
}