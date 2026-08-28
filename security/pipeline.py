"""
SecureRAGPipeline: composes Layers 1-4 around the plain RAGEngine.
With OTEL Tracing, Caching, and HITL Queue functionality.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from base_rag import RAGEngine, SYSTEM_PROMPT_TEMPLATE
from .layer1_sanitize import sanitize_input
from .layer2_heuristic import check_heuristics
from .layer3_ml_guard import MLGuard
from session_manager import SessionManager
import telemetry

from opentelemetry import trace

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
    hitl_ticket_id: Optional[str] = None
    total_tokens: int = 0
    cost_score: float = 0.0


class SecureRAGPipeline:
    def __init__(self, engine: RAGEngine, ml_guard: Optional[MLGuard] = None, max_tokens: int = 512):
        self.engine = engine
        self.ml_guard = ml_guard or MLGuard()
        self.max_tokens = max_tokens
        self.session_mgr = SessionManager()

    def ask(self, raw_question: str, session_id: str = "default", initial_tokens: int = 0, k: int = 3) -> PipelineResult:
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span("Pipeline_Ask") as parent_span:
            parent_span.set_attribute("session_id", session_id)
            parent_span.set_attribute("query", raw_question)
            
            audit = []
            accumulated_tokens = initial_tokens
            
            # --- CACHE CHECK ---
            with telemetry.trace_span("Cache_Check") as cache_span:
                cached_answer = self.session_mgr.get_cache(raw_question)
                if cached_answer:
                    cache_span.set_attribute("cache.hit", True)
                    cache_span.add_event("Cache Hit: Returning saved answer")
                    parent_span.set_attribute("cache.hit", True)
                    parent_span.add_event("Skipping LLM due to Cache Hit")
                    return PipelineResult(answer=cached_answer, blocked=False, audit_trail=audit, total_tokens=accumulated_tokens)
                cache_span.set_attribute("cache.hit", False)
            
            # Layer 1: pre-processing & sanitization
            with telemetry.trace_span("L1_Sanitize") as l1_span:
                t0 = time.perf_counter()
                l1 = sanitize_input(raw_question, max_tokens=self.max_tokens)
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L1_sanitize", l1.passed, l1.reason, latency))
                
                if not l1.passed:
                    l1_span.set_attribute("security.blocked", True)
                    return PipelineResult(answer=REFUSAL_MESSAGE, blocked=True,
                                           blocked_at_layer="L1_sanitize", audit_trail=audit, total_tokens=accumulated_tokens)
                question = l1.cleaned_text
    
            # Layer 2: fast heuristic firewall
            with telemetry.trace_span("L2_Heuristic") as l2_span:
                t0 = time.perf_counter()
                l2 = check_heuristics(question)
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L2_heuristic", l2.passed, l2.matched_pattern, latency))
                
                if not l2.passed:
                    l2_span.set_attribute("security.blocked", True)
                    return PipelineResult(answer=REFUSAL_MESSAGE, blocked=True,
                                           blocked_at_layer="L2_heuristic", audit_trail=audit, total_tokens=accumulated_tokens)
    
            # Layer 3: ML/semantic detection
            with telemetry.trace_span("L3_ML_Guard") as l3_span:
                t0 = time.perf_counter()
                l3 = self.ml_guard.check(question)
                
                # Track tokens for L3 guard LLM
                l3_total = l3.prompt_tokens + l3.completion_tokens
                accumulated_tokens += l3_total
                telemetry.set_llm_attributes(l3_span, "openai/gpt-oss-120b", l3.prompt_tokens, l3.completion_tokens)
                
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L3_ml_guard", l3.is_safe, l3.category, latency))
                
                if not l3.is_safe:
                    l3_span.set_attribute("security.blocked", True)
                    return PipelineResult(answer=REFUSAL_MESSAGE, blocked=True,
                                           blocked_at_layer="L3_ml_guard", audit_trail=audit, total_tokens=accumulated_tokens)
    
            # Retrieval 
            with telemetry.trace_span("Chroma_Retrieval") as ret_span:
                t0 = time.perf_counter()
                hits = self.engine.retrieve(question, k=k)
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("retrieval", True, f"{len(hits)} chunks", latency))
                ret_span.set_attribute("retrieval.chunks", len(hits))
    
            if not hits:
                return PipelineResult(
                    answer="I do not know based on the provided context.",
                    blocked=False, audit_trail=audit,
                    quarantined_chunk_ids=[],
                    total_tokens=accumulated_tokens
                )
    
            context = self.engine.build_context(hits)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": f"Context : {context}\nQuestion: {question}"}
            ]
    
            # Generation
            with telemetry.trace_span("Groq_Generation") as gen_span:
                t0 = time.perf_counter()
                # Include usage to track exact tokens
                completion = self.engine.groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=2048,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=True,
                    stop=None
                )
                result_text = ""
                for chunk in completion:
                    if chunk.choices and len(chunk.choices) > 0:
                        result_text += chunk.choices[0].delta.content or ""
                    
                import tiktoken
                tokenizer = tiktoken.get_encoding("cl100k_base")
                prompt_text = context + question
                prompt_toks = len(tokenizer.encode(prompt_text))
                comp_toks = len(tokenizer.encode(result_text))
                
                accumulated_tokens += prompt_toks + comp_toks
                
                telemetry.set_llm_attributes(
                    gen_span, 
                    "openai/gpt-oss-120b", 
                    prompt_toks, 
                    comp_toks
                )
                
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("generation", True, "", latency))
                
            # Output SLM Filter
            with telemetry.trace_span("Output_Filter") as filter_span:
                t0 = time.perf_counter()
                filtered_text = self.engine.filter(result_text)
                
                # Filter is an LLM stream, estimate tokens
                prompt_toks_f = len(tokenizer.encode(result_text))
                comp_toks_f = len(tokenizer.encode(filtered_text))
                accumulated_tokens += prompt_toks_f + comp_toks_f
                
                telemetry.set_llm_attributes(
                    filter_span, 
                    "openai/gpt-oss-120b", 
                    prompt_toks_f, 
                    comp_toks_f
                )
                
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("output_filter", True, "", latency))
                
            # L4: Cost & Confidence Check
            with telemetry.trace_span("L4_Confidence_Check") as l4_span:
                t0 = time.perf_counter()
                
                total_latency = sum(e.latency_ms for e in audit)
                num_traces = len(audit)
                
                # Hardcoded weights to generate an overall cost parameter
                W_TOKEN = 0.01      # 1000 tokens = 10 cost
                W_LATENCY = 0.005   # 2000 ms = 10 cost
                W_TRACE = 2.0       # 5 traces = 10 cost
                
                cost_score = (accumulated_tokens * W_TOKEN) + (total_latency * W_LATENCY) + (num_traces * W_TRACE)
                l4_span.set_attribute("cost_score.total", cost_score)
                parent_span.set_attribute("cost_score.total", cost_score)
                
                # Trigger HITL if cost is too high
                COST_THRESHOLD = 50.0  # adjust as needed
                confidence_is_high = cost_score <= COST_THRESHOLD
                
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L4_confidence_check", confidence_is_high, f"Cost Score: {cost_score:.2f}", latency))
                
                if not confidence_is_high:
                    l4_span.set_attribute("confidence.high", False)
                    # Push to HITL
                    trace_id = format(parent_span.get_span_context().trace_id, '032x')
                    ticket_id = self.session_mgr.enqueue_hitl(session_id, raw_question, trace_id)
                    return PipelineResult(
                        answer=f"Query exceeded complexity cost threshold (Score: {cost_score:.2f}). Routed to a human agent (Ticket: {ticket_id}).",
                        blocked=True,
                        blocked_at_layer="L4_confidence_check",
                        audit_trail=audit,
                        hitl_ticket_id=ticket_id,
                        total_tokens=accumulated_tokens,
                        cost_score=cost_score
                    )
                
                l4_span.set_attribute("confidence.high", True)

            # If we reached here, cache the answer
            self.session_mgr.set_cache(raw_question, filtered_text)
    
            return PipelineResult(
                answer=filtered_text,
                blocked=False,
                audit_trail=audit,
                quarantined_chunk_ids=[],
                total_tokens=accumulated_tokens,
                cost_score=cost_score
            )
