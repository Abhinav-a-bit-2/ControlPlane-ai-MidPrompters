"""
SecureRAGPipeline: composes Layers 1-3 around the plain RAGEngine.

Each stage is logged with which layer made the decision — this is what
feeds the "which layer caught what" demo dashboard.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from base_rag import RAGEngine, SYSTEM_PROMPT_TEMPLATE
from .layer1_sanitize import sanitize_input
from .layer2_heuristic import check_heuristics
from .layer3_ml_guard import MLGuard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security.pipeline")

REFUSAL_MESSAGE = "This request was blocked by the input security pipeline."


@dataclass
class AuditEntry:
    layer: str
    passed: bool
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class PipelineResult:
    answer: str
    blocked: bool = False
    blocked_at_layer: str = ""
    audit_trail: list = field(default_factory=list)
    quarantined_chunk_ids: list = field(default_factory=list)


class SecureRAGPipeline:
    def __init__(self, engine: RAGEngine, ml_guard: Optional[MLGuard] = None, max_tokens: int = 512):
        self.engine = engine
        self.ml_guard = ml_guard or MLGuard()
        self.max_tokens = max_tokens

    def ask(self, raw_question: str, k: int = 3) -> PipelineResult:
        audit = []

        # Layer 1: pre-processing & sanitization
        t0 = time.perf_counter()
        l1 = sanitize_input(raw_question, max_tokens=self.max_tokens)
        audit.append(AuditEntry("L1_sanitize", l1.passed, l1.reason, (time.perf_counter() - t0) * 1000))
        if not l1.passed:
            return PipelineResult(answer=REFUSAL_MESSAGE, blocked=True,
                                   blocked_at_layer="L1_sanitize", audit_trail=audit)
        question = l1.cleaned_text

        # Layer 2: fast heuristic firewall
        t0 = time.perf_counter()
        l2 = check_heuristics(question)
        audit.append(AuditEntry("L2_heuristic", l2.passed, l2.matched_pattern, (time.perf_counter() - t0) * 1000))
        if not l2.passed:
            return PipelineResult(answer=REFUSAL_MESSAGE, blocked=True,
                                   blocked_at_layer="L2_heuristic", audit_trail=audit)

        # Layer 3: ML/semantic detection
        t0 = time.perf_counter()
        l3 = self.ml_guard.check(question)
        audit.append(AuditEntry("L3_ml_guard", l3.is_safe, l3.category, (time.perf_counter() - t0) * 1000))
        if not l3.is_safe:
            return PipelineResult(answer=REFUSAL_MESSAGE, blocked=True,
                                   blocked_at_layer="L3_ml_guard", audit_trail=audit)

        # Retrieval (untrusted context, no longer isolated by L4)
        t0 = time.perf_counter()
        hits = self.engine.retrieve(question, k=k)
        audit.append(AuditEntry("retrieval", True, f"{len(hits)} chunks", (time.perf_counter() - t0) * 1000))

        if not hits:
            return PipelineResult(
                answer="I do not know based on the provided context.",
                blocked=False, audit_trail=audit,
                quarantined_chunk_ids=[],
            )

        context = self.engine.build_context(hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": f"Context : {context}\nQuestion: {question}"}
        ]

        t0 = time.perf_counter()
        completion = self.engine.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None,
        )
        result_text = ""
        for chunk in completion:
            result_text += chunk.choices[0].delta.content or ""
        audit.append(AuditEntry("generation", True, "", (time.perf_counter() - t0) * 1000))

        return PipelineResult(
            answer=result_text,
            blocked=False,
            audit_trail=audit,
            quarantined_chunk_ids=[],
        )
