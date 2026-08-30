"""
rag_client.py: Black-box client wrapper for interacting with the Secure RAG Pipeline during DeepEval testing.
Provides a clean interface that measures latency and extracts all relevant evaluation parameters.
"""
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

# We initialize telemetry if not already initialized
try:
    import telemetry
    telemetry.init_telemetry()
except Exception:
    pass

from base_rag import RAGEngine
from security.intent_pipeline import IntentRoutedPipeline

# Default knowledge base document
SOURCE_DOC = os.environ.get(
    "SOURCE_DOC",
    str(
        Path(__file__).parent.parent
        / "confluence"
        / "dsid_0a7f0607ab7a4043bc5a888e3fd7c7e7__tiered-degradation-and-incident-runbook-2026.txt"
    ),
)

@dataclass
class RAGResponse:
    query: str
    answer: str
    latency_ms: float
    blocked: bool
    blocked_at_layer: str
    hitl_ticket_id: Optional[str]
    total_tokens: int
    cost_score: float
    audit_trail: list = field(default_factory=list)
    safe_chunks: Any = None


class RAGClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, source_path: Optional[str] = None):
        self.source_path = source_path or SOURCE_DOC
        self.engine = RAGEngine(self.source_path)
        # Ensure indexed
        try:
            existing_cnt = len(self.engine.vector_store.get()["ids"])
            if existing_cnt == 0:
                self.engine.index()
        except Exception:
            pass
        self.pipeline = IntentRoutedPipeline(self.engine)

    def query(self, question: str, session_id: Optional[str] = None) -> RAGResponse:
        """
        Executes a black-box query against the RAG pipeline and returns the structured response.
        """
        sid = session_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        
        result = self.pipeline.ask(
            raw_question=question,
            original_question=question,
            session_id=sid
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RAGResponse(
            query=question,
            answer=result.answer,
            latency_ms=latency_ms,
            blocked=result.blocked,
            blocked_at_layer=result.blocked_at_layer,
            hitl_ticket_id=result.hitl_ticket_id,
            total_tokens=result.total_tokens,
            cost_score=result.cost_score,
            audit_trail=result.audit_trail,
            safe_chunks=result.safe_chunks
        )
