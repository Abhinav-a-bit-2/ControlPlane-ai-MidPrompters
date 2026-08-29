"""
IntentRoutedPipeline: Dual-path pipeline that classifies user intent via
zero-shot NLI (bart-large-mnli) and routes queries to a cheap or expensive
path based on classifier confidence.

- Cheap  path:  fewer chunks, shorter generation, intent-focused prompt
- Expensive path: full retrieval, full generation, comprehensive prompt
- Fallback: cheap→expensive if answer quality is low; expensive→HITL if still low

Security layers (L1-L3) and final layers (Output Filter + L4) are common
to both paths.
"""
import logging
import re
import time
from typing import Optional

import tiktoken
from opentelemetry import trace

from base_rag import RAGEngine, SYSTEM_PROMPT_TEMPLATE
from .layer1_sanitize import sanitize_input
from .layer2_heuristic import check_heuristics
from .layer3_ml_guard import MLGuard
from .pipeline import AuditEntry, PipelineResult, REFUSAL_MESSAGE
from session_manager import SessionManager
from intent_classifier import (
    IntentClassifier,
    IntentResult,
    INTENT_PROMPT_ADDENDA,
    INTENT_RETRIEVAL_K,
    LABEL_SHORT_NAMES,
)
import telemetry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security.intent_pipeline")

# Confidence threshold: above this → cheap path, at or below → expensive path
INTENT_CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Answer quality gate (lightweight heuristic, <1ms)
# ---------------------------------------------------------------------------
def is_answer_adequate(answer: str) -> bool:
    """
    Fast heuristic to decide whether a generated answer is good enough.
    Returns False if the answer looks like a failure – triggers fallback.
    """
    if not answer or len(answer.strip()) < 20:
        logger.warning(f"Answer inadequate: too short ({len(answer) if answer else 0})")
        return False
    if "I do not know" in answer:
        # Retrieval miss (out-of-scope query). This is a successful, adequate refusal.
        return True
    if not re.search(r'chunk-\d+', answer, flags=re.IGNORECASE):
        # No citations -> likely hallucinating or unhelpful
        logger.warning(f"Answer inadequate: no citations. Answer: {repr(answer)}")
        return False
    return True


class IntentRoutedPipeline:
    """
    Dual-path secure RAG pipeline with intent-based routing.

    Shares the same security layers, caching, HITL queue, and telemetry
    as SecureRAGPipeline but adds:
      1. Zero-shot intent classification (before path selection)
      2. Cheap path (fewer chunks, shorter gen, intent-focused prompt)
      3. Expensive path (full retrieval, full gen, comprehensive prompt)
      4. Automatic fallback: cheap → expensive → HITL
    """

    def __init__(
        self,
        engine: RAGEngine,
        ml_guard: Optional[MLGuard] = None,
        max_tokens: int = 512,
        confidence_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
    ):
        self.engine = engine
        self.ml_guard = ml_guard or MLGuard()
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        self.session_mgr = SessionManager()
        self.classifier = IntentClassifier()
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def ask(
        self,
        raw_question: str,
        original_question: str = "",
        session_id: str = "default",
        initial_tokens: int = 0,
    ) -> PipelineResult:
        """
        Args:
            raw_question:      The contextualized/rewritten query (used for retrieval + generation).
            original_question: The raw user input before rewriting (used for intent classification).
                               Falls back to raw_question if not provided.
            session_id:        Redis session key.
            initial_tokens:    Token count already consumed by the contextualizer.
        """
        tracer = trace.get_tracer(__name__)
        classify_on = original_question or raw_question

        with tracer.start_as_current_span("Pipeline_Ask") as parent_span:
            parent_span.set_attribute("session_id", session_id)
            parent_span.set_attribute("query", raw_question)
            parent_span.set_attribute("original_query", classify_on)

            audit: list[AuditEntry] = []
            accumulated_tokens = initial_tokens

            # =============================================================
            # COMMON PRE-PROCESSING: Cache → L1 → L2 → L3
            # =============================================================

            # --- Semantic Cache Check ---
            with telemetry.trace_span("Cache_Check") as cache_span:
                cached_answer = self.session_mgr.get_cache(raw_question)
                if cached_answer:
                    cache_span.set_attribute("cache.hit", True)
                    cache_span.add_event("Cache Hit: Returning saved answer")
                    parent_span.set_attribute("cache.hit", True)
                    parent_span.add_event("Skipping LLM due to Cache Hit")
                    return PipelineResult(
                        safe_chunks=" ", answer=cached_answer,
                        blocked=False, audit_trail=audit,
                        total_tokens=accumulated_tokens,
                    )
                cache_span.set_attribute("cache.hit", False)

            # --- L1: Sanitize ---
            with telemetry.trace_span("L1_Sanitize") as l1_span:
                t0 = time.perf_counter()
                l1 = sanitize_input(raw_question, max_tokens=self.max_tokens)
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L1_sanitize", l1.passed, l1.reason, latency))
                if not l1.passed:
                    l1_span.set_attribute("security.blocked", True)
                    return PipelineResult(
                        safe_chunks=" ", answer=REFUSAL_MESSAGE, blocked=True,
                        blocked_at_layer="L1_sanitize", audit_trail=audit,
                        total_tokens=accumulated_tokens,
                    )
                question = l1.cleaned_text

            # --- L2: Heuristic Firewall ---
            with telemetry.trace_span("L2_Heuristic") as l2_span:
                t0 = time.perf_counter()
                l2 = check_heuristics(question)
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L2_heuristic", l2.passed, l2.matched_pattern, latency))
                if not l2.passed:
                    l2_span.set_attribute("security.blocked", True)
                    return PipelineResult(
                        safe_chunks=" ", answer=REFUSAL_MESSAGE, blocked=True,
                        blocked_at_layer="L2_heuristic", audit_trail=audit,
                        total_tokens=accumulated_tokens,
                    )

            # --- L3: ML Guard ---
            with telemetry.trace_span("L3_ML_Guard") as l3_span:
                t0 = time.perf_counter()
                l3 = self.ml_guard.check(question)
                l3_total = l3.prompt_tokens + l3.completion_tokens
                accumulated_tokens += l3_total
                telemetry.set_llm_attributes(
                    l3_span, "openai/gpt-oss-20b",
                    l3.prompt_tokens, l3.completion_tokens,
                )
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("L3_ml_guard", l3.is_safe, l3.category, latency))
                if not l3.is_safe:
                    l3_span.set_attribute("security.blocked", True)
                    return PipelineResult(
                        safe_chunks=" ", answer=REFUSAL_MESSAGE, blocked=True,
                        blocked_at_layer="L3_ml_guard", audit_trail=audit,
                        total_tokens=accumulated_tokens,
                    )

            # =============================================================
            # INTENT CLASSIFICATION
            # =============================================================
            with telemetry.trace_span("Intent_Classification") as ic_span:
                t0 = time.perf_counter()
                intent: IntentResult = self.classifier.classify(classify_on)
                latency = (time.perf_counter() - t0) * 1000
                ic_span.set_attribute("intent.label", intent.short_label)
                ic_span.set_attribute("intent.full_label", intent.label)
                ic_span.set_attribute("intent.confidence", intent.confidence)
                parent_span.set_attribute("intent.label", intent.short_label)
                parent_span.set_attribute("intent.confidence", intent.confidence)
                audit.append(AuditEntry(
                    "intent_classification", True,
                    f"{intent.short_label} ({intent.confidence:.3f})", latency,
                ))

            # =============================================================
            # PATH ROUTING
            # =============================================================
            if intent.confidence > self.confidence_threshold:
                path = "cheap"
            else:
                path = "expensive"

            parent_span.set_attribute("pipeline.path", path)
            logger.info(
                "Routing to %s path (intent=%s, conf=%.3f, threshold=%.2f)",
                path, intent.label, intent.confidence, self.confidence_threshold,
            )

            # --- CHEAP PATH ---
            if path == "cheap":
                with telemetry.trace_span("Cheap_Path") as cheap_span:
                    cheap_span.set_attribute("intent.label", intent.label)
                    result_text, hits, path_tokens, path_audit = self._generate_path(
                        question=question,
                        intent=intent,
                        mode="cheap",
                    )
                    accumulated_tokens += path_tokens
                    audit.extend(path_audit)

                # --- ANSWER QUALITY GATE ---
                with telemetry.trace_span("Answer_Quality_Gate") as qg_span:
                    adequate = is_answer_adequate(result_text)
                    qg_span.set_attribute("quality.adequate", adequate)
                    qg_span.set_attribute("quality.path", "cheap")
                    audit.append(AuditEntry(
                        "quality_gate_cheap", adequate,
                        f"adequate={adequate}", 0.0,
                    ))

                if not adequate:
                    # FALLBACK: re-run through expensive path
                    logger.info("Cheap path answer inadequate — falling back to expensive path")
                    parent_span.set_attribute("pipeline.path", "fallback")
                    parent_span.add_event("Cheap path fallback to expensive path")

                    with telemetry.trace_span("Expensive_Path_Fallback") as exp_span:
                        exp_span.set_attribute("intent.label", intent.label)
                        exp_span.set_attribute("fallback", True)
                        result_text, hits, path_tokens, path_audit = self._generate_path(
                            question=question,
                            intent=intent,
                            mode="expensive",
                        )
                        accumulated_tokens += path_tokens
                        audit.extend(path_audit)

                    # Quality gate on expensive fallback
                    with telemetry.trace_span("Answer_Quality_Gate") as qg2_span:
                        adequate = is_answer_adequate(result_text)
                        qg2_span.set_attribute("quality.adequate", adequate)
                        qg2_span.set_attribute("quality.path", "expensive_fallback")
                        audit.append(AuditEntry(
                            "quality_gate_expensive", adequate,
                            f"adequate={adequate}", 0.0,
                        ))

                    if not adequate:
                        # Double failure → HITL
                        return self._escalate_to_hitl(
                            parent_span, session_id, raw_question,
                            audit, accumulated_tokens, hits,
                            reason="Both cheap and expensive paths produced inadequate answers",
                        )

            # --- EXPENSIVE PATH (direct) ---
            else:
                with telemetry.trace_span("Expensive_Path") as exp_span:
                    exp_span.set_attribute("intent.label", intent.label)
                    result_text, hits, path_tokens, path_audit = self._generate_path(
                        question=question,
                        intent=intent,
                        mode="expensive",
                    )
                    accumulated_tokens += path_tokens
                    audit.extend(path_audit)

                # Quality gate on expensive path
                with telemetry.trace_span("Answer_Quality_Gate") as qg_span:
                    adequate = is_answer_adequate(result_text)
                    qg_span.set_attribute("quality.adequate", adequate)
                    qg_span.set_attribute("quality.path", "expensive")
                    audit.append(AuditEntry(
                        "quality_gate_expensive", adequate,
                        f"adequate={adequate}", 0.0,
                    ))

                if not adequate:
                    return self._escalate_to_hitl(
                        parent_span, session_id, raw_question,
                        audit, accumulated_tokens, hits,
                        reason="Expensive path produced inadequate answer",
                    )

            # =============================================================
            # COMMON FINAL LAYERS: Output Filter → L4
            # =============================================================

            # --- Output SLM Filter (PII Masking) ---
            with telemetry.trace_span("Output_Filter") as filter_span:
                t0 = time.perf_counter()
                filtered_text = self.engine.filter(result_text)
                prompt_toks_f = len(self._tokenizer.encode(result_text))
                comp_toks_f = len(self._tokenizer.encode(filtered_text))
                accumulated_tokens += prompt_toks_f + comp_toks_f
                telemetry.set_llm_attributes(
                    filter_span, "openai/gpt-oss-20b",
                    prompt_toks_f, comp_toks_f,
                )
                latency = (time.perf_counter() - t0) * 1000
                audit.append(AuditEntry("output_filter", True, "", latency))

            # --- L4: Cost & Confidence Check ---
            with telemetry.trace_span("L4_Confidence_Check") as l4_span:
                total_latency = sum(e.latency_ms for e in audit)
                num_traces = len(audit)

                W_TOKEN = 0.04
                W_LATENCY = 0.0
                W_TRACE = 1.0
                cost_score = (
                    (accumulated_tokens * W_TOKEN)
                    + (total_latency * W_LATENCY)
                    + (num_traces * W_TRACE)
                )
                l4_span.set_attribute("cost_score.total", cost_score)
                parent_span.set_attribute("cost_score.total", cost_score)

                COST_THRESHOLD = 50
                confidence_is_high = cost_score <= COST_THRESHOLD

                audit.append(AuditEntry(
                    "L4_confidence_check", confidence_is_high,
                    f"Cost Score: {cost_score:.2f}", 0.0,
                ))

                if not confidence_is_high:
                    return self._escalate_to_hitl(
                        parent_span, session_id, raw_question,
                        audit, accumulated_tokens, hits,
                        reason=f"Cost score {cost_score:.2f} exceeded threshold",
                        cost_score=cost_score,
                    )

                l4_span.set_attribute("confidence.high", True)

            # --- Cache & Return ---
            self.session_mgr.set_cache(raw_question, filtered_text)

            return PipelineResult(
                safe_chunks=hits,
                answer=filtered_text,
                blocked=False,
                audit_trail=audit,
                quarantined_chunk_ids=[],
                total_tokens=accumulated_tokens,
                cost_score=cost_score,
            )

    # ------------------------------------------------------------------
    # Internal: Generate answer via cheap or expensive path
    # ------------------------------------------------------------------
    def _generate_path(
        self,
        question: str,
        intent: IntentResult,
        mode: str,  # "cheap" or "expensive"
    ) -> tuple:
        """
        Runs retrieval + generation for a given path mode.

        Returns:
            (result_text, hits, tokens_consumed, audit_entries)
        """
        path_audit: list[AuditEntry] = []
        tokens = 0

        # --- Retrieval ---
        if mode == "cheap":
            k = INTENT_RETRIEVAL_K.get(intent.short_label, 2)
        else:
            k = max(INTENT_RETRIEVAL_K.get(intent.short_label, 3), 4)

        with telemetry.trace_span(f"Chroma_Retrieval_{mode}") as ret_span:
            t0 = time.perf_counter()
            hits = self.engine.retrieve(question, k=k)
            latency = (time.perf_counter() - t0) * 1000
            ret_span.set_attribute("retrieval.chunks", len(hits))
            ret_span.set_attribute("retrieval.mode", mode)
            path_audit.append(AuditEntry(
                f"retrieval_{mode}", True,
                f"{len(hits)} chunks (k={k})", latency,
            ))

        if not hits:
            return (
                "I do not know based on the provided context.",
                [], 0, path_audit,
            )

        # --- Build intent-aware prompt ---
        context = self.engine.build_context(hits)
        intent_addendum = INTENT_PROMPT_ADDENDA.get(intent.label, "")
        system_prompt = SYSTEM_PROMPT_TEMPLATE + intent_addendum

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context : {context}\nQuestion: {question}"},
        ]

        # --- Generation ---
        if mode == "cheap":
            max_completion_tokens = 1024
            reasoning_effort = "low"
        else:
            max_completion_tokens = 2048
            reasoning_effort = "medium"

        with telemetry.trace_span(f"Groq_Generation_{mode}") as gen_span:
            t0 = time.perf_counter()
            completion = self.engine.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=1,
                max_completion_tokens=max_completion_tokens,
                top_p=1,
                reasoning_effort=reasoning_effort,
                stream=True,
                stop=None,
            )
            result_text = ""
            for chunk in completion:
                if chunk.choices and len(chunk.choices) > 0:
                    result_text += chunk.choices[0].delta.content or ""

            prompt_text = context + question
            prompt_toks = len(self._tokenizer.encode(prompt_text))
            comp_toks = len(self._tokenizer.encode(result_text))
            tokens += prompt_toks + comp_toks

            gen_span.set_attribute("generation.mode", mode)
            gen_span.set_attribute("generation.reasoning_effort", reasoning_effort)
            gen_span.set_attribute("generation.max_tokens", max_completion_tokens)
            telemetry.set_llm_attributes(
                gen_span, "openai/gpt-oss-20b",
                prompt_toks, comp_toks,
            )
            latency = (time.perf_counter() - t0) * 1000
            path_audit.append(AuditEntry(
                f"generation_{mode}", True,
                f"reasoning={reasoning_effort}, max_tok={max_completion_tokens}",
                latency,
            ))

        return result_text, hits, tokens, path_audit

    # ------------------------------------------------------------------
    # Internal: Escalate to HITL queue
    # ------------------------------------------------------------------
    def _escalate_to_hitl(
        self,
        parent_span,
        session_id: str,
        raw_question: str,
        audit: list[AuditEntry],
        accumulated_tokens: int,
        hits,
        reason: str,
        cost_score: float = 0.0,
    ) -> PipelineResult:
        """Push query to the human-in-the-loop queue and return a blocked result."""
        parent_span.set_attribute("confidence.high", False)
        trace_id = format(parent_span.get_span_context().trace_id, "032x")
        ticket_id = self.session_mgr.enqueue_hitl(session_id, raw_question, trace_id)
        logger.info("Escalated to HITL: ticket=%s reason=%s", ticket_id, reason)

        return PipelineResult(
            answer=(
                f"Your query has been escalated to a human agent. "
                f"Reason: {reason}. (Ticket: {ticket_id})"
            ),
            blocked=True,
            blocked_at_layer="hitl_escalation",
            audit_trail=audit,
            hitl_ticket_id=ticket_id,
            total_tokens=accumulated_tokens,
            cost_score=cost_score,
            safe_chunks=hits if hits else " ",
        )
