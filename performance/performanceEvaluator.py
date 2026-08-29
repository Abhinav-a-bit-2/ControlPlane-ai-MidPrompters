import logging
from typing import List, Dict, Any, Optional
from .models import extract_claims, evaluate_claims, build_report, Chunk, ClaimEvaluation, EvaluationReport
from .grounding import GroundingChecker
from .semantic_entropy import SemanticEntropyChecker

logger = logging.getLogger(__name__)
class PerformanceEvaluator:
    def __init__(self, groq_client=None, grounding_checker: Optional[GroundingChecker] = None, risk_threshold: float = 0.50, session_manager=None, sampling_period: int = 2):
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.entropy_checker = SemanticEntropyChecker(groq_client, self.grounding_checker) if groq_client else None
        self.eval_count = 0
        self.risk_threshold = risk_threshold
        self.session_manager = session_manager
        self.sampling_period = sampling_period
    @staticmethod    
    def _extract_chunk_data(doc: Any) -> tuple[Any, str]:
        """Helper to safely extract (chunk_id, content) from dicts, Document objects, or tuples."""
        # If doc is a (chunk, score) tuple/list
        if isinstance(doc, (tuple, list)):
            doc = doc[0]

        # LangChain / LlamaIndex Document object
        if hasattr(doc, "metadata"):
            chunk_id = doc.metadata.get("chunk_id")
            content = getattr(doc, "page_content", "")
            return chunk_id, content

        # Dictionary representation
        if isinstance(doc, dict):
            chunk_id = doc.get("chunk_id")
            content = doc.get("content", doc.get("page_content", ""))
            return chunk_id, content

        # Raw string or fallback object
        chunk_id = getattr(doc, "chunk_id", str(doc))
        content = getattr(doc, "content", str(doc))
        return chunk_id, content

    def evaluate(
        self,
        answer: str,
        retrieved_chunks: List[Any],
        messages: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> EvaluationReport:
        self.eval_count += 1
        periodic_sample = (self.eval_count % self.sampling_period == 0)

        try:
            # Build Chunk objects for models.evaluate_claims
            chunks = []
            for doc in retrieved_chunks:
                chunk_id, content = self._extract_chunk_data(doc)
                if chunk_id and content:
                    chunks.append(Chunk(chunk_id=str(chunk_id), text=content))

            claims = extract_claims(answer)
            if not claims:
                return EvaluationReport(overall_confidence="high", risk_score=0.0)

            # Grounding-based evaluation via models.py
            evaluations = evaluate_claims(claims, chunks)

            has_ambiguity = any(e.status in ("neutral", "uncited", "contradiction") for e in evaluations)

            # Entropy sampling: only on periodic tick OR ambiguity
            entropy_val = 0.0
            should_sample = has_ambiguity or periodic_sample

            if should_sample and self.entropy_checker:
                try:
                    # Use tokenBudgetedChats for episodic memory if available
                    sampled_messages = messages
                    if session_id and self.session_manager:
                        sampled_messages = self.session_manager.tokenBudgetedChats(session_id, token_budget=2048)
                    if sampled_messages:
                        entropy_val = self.entropy_checker.compute_entropy(sampled_messages)
                except Exception as exc:
                    logger.warning("Sampling-based entropy check skipped: %s", exc)
                    entropy_val = 0.5

            # Apply entropy multiplier to flagged/ambiguous claims
            if entropy_val > 0:
                adjusted = []
                for e in evaluations:
                    if e.status in ("neutral", "uncited", "contradiction"):
                        new_risk = min(1.0, e.risk_score * (1.0 + entropy_val))
                        adjusted.append(ClaimEvaluation(
                            text=e.text, status=e.status,
                            entailment_score=e.entailment_score,
                            risk_score=new_risk,
                            is_flagged=e.is_flagged or new_risk > self.risk_threshold,
                            cited_chunks=e.cited_chunks,
                            best_matching_chunk=e.best_matching_chunk,
                        ))
                    else:
                        adjusted.append(e)
                evaluations = adjusted

            report = build_report(evaluations)
            report.semantic_entropy = entropy_val
            return report

        except Exception as exc:
            logger.error("Performance evaluation failed-open: %s", exc, exc_info=True)
            return EvaluationReport(overall_confidence="unknown")
