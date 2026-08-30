import React, { useEffect, useState } from "react";
import { CloseIcon, DocumentTextIcon } from "./Icons";

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
    <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-card border border-border w-full max-w-xl rounded-lg shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border bg-muted/40">
          <div className="flex items-center gap-2.5">
            <DocumentTextIcon className="w-4 h-4 text-primary" />
            <span className="font-mono text-xs font-semibold text-foreground">
              Passage Excerpt: [{chunkId}]
            </span>
          </div>
          <button 
            onClick={onClose} 
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 overflow-y-auto text-sm space-y-3.5">
          {loading ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
              <span className="w-3.5 h-3.5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              Fetching index passage metadata...
            </div>
          ) : data?.content ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono border-b border-border/60 pb-2">
                <span>Source Corpus: <strong className="text-foreground">{data.metadata?.filename || "corpus.txt"}</strong></span>
                {data.metadata?.score && (
                  <span className="bg-muted px-2 py-0.5 rounded text-[10px] border border-border font-mono">
                    Score: {(data.metadata.score * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="p-4 bg-muted/30 rounded border border-border text-foreground leading-relaxed whitespace-pre-wrap font-mono text-xs selection:bg-primary/20">
                {data.content}
              </div>
            </div>
          ) : (
            <div className="text-xs text-destructive font-mono p-3 bg-destructive/10 border border-destructive/20 rounded">
              Could not retrieve passage excerpt from index.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}