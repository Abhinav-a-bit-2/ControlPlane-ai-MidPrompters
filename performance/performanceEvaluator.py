import logging
from typing import List, Dict, Any, Optional
from .models import extract_claims, ClaimEvaluation, EvaluationReport
from .grounding import GroundingChecker
from .semantic_entropy import SemanticEntropyChecker

logger = logging.getLogger(__name__)
class PerformanceEvaluator:
    def __init__(self, groq_client=None, grounding_checker: Optional[GroundingChecker] = None, risk_threshold: float = 0.50):
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.entropy_checker = SemanticEntropyChecker(groq_client, self.grounding_checker) if groq_client else None
        self.eval_count = 0
        self.risk_threshold = risk_threshold
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
    ) -> EvaluationReport:
        self.eval_count += 1
        periodic_sample = (self.eval_count % 1 == 0)

        try:
            chunk_lookup = dict(
                self._extract_chunk_data(doc) for doc in retrieved_chunks
            )
            claims = extract_claims(answer)
            if not claims:
                return EvaluationReport(overall_confidence="high", risk_score=0.0)

            preliminary = []
            has_ambiguity = False

            for claim in claims:
                if not claim.cited_chunk_ids:
                    preliminary.append({"claim": claim, "status": "uncited", "entailment_score": 0.0, "needs_escalation": True})
                    has_ambiguity = True
                    continue

                chunk_text = " ".join(chunk_lookup.get(cid, "") for cid in claim.cited_chunk_ids)
                res = self.grounding_checker.check_claim(claim.text, chunk_text)
                is_ambiguous = (res["label"] == "neutral" or res["entailment_score"] < 0.55)
                if is_ambiguous:
                    has_ambiguity = True

                preliminary.append({
                    "claim": claim,
                    "status": res["label"],
                    "entailment_score": res["entailment_score"],
                    "needs_escalation": is_ambiguous
                })

            entropy_val = 0.0
            should_sample = (has_ambiguity or periodic_sample)

            if should_sample and self.entropy_checker and messages:
                try:
                    entropy_val = self.entropy_checker.compute_entropy(messages)
                except Exception as exc:
                    logger.warning("Sampling-based entropy check skipped: %s", exc)
                    entropy_val = 0.5

            all_evals = []
            flagged = []
            total_risk = 0.0

            for item in preliminary:
                c = item["claim"]
                entailment = item["entailment_score"]
                status = item["status"]

                unfaithfulness = 1.0 - entailment
                uncertainty_mult = 1.0 + (entropy_val if item["needs_escalation"] else 0.0)
                risk = min(1.0, unfaithfulness * uncertainty_mult)
                is_flagged = (status == "uncited") or (risk > self.risk_threshold)

                evaluation = ClaimEvaluation(
                    text=c.text,
                    status=status,
                    entailment_score=entailment,
                    risk_score=risk,
                    is_flagged=is_flagged,
                    cited_chunks=c.cited_chunk_ids,
                )
                all_evals.append(evaluation)
                total_risk += risk
                if is_flagged:
                    flagged.append(evaluation)

            avg_risk = total_risk / len(all_evals)
            if not flagged and avg_risk < 0.30:
                conf = "high"
            elif avg_risk > 0.60 or (len(flagged) / len(all_evals)) > 0.50:
                conf = "low"
            else:
                conf = "medium"

            return EvaluationReport(
                overall_confidence=conf,
                risk_score=avg_risk,
                semantic_entropy=entropy_val,
                flagged_claims=flagged,
                all_claims=all_evals,
            )
        except Exception as exc:
            logger.error("Performance evaluation failed-open: %s", exc, exc_info=True)
            return EvaluationReport(overall_confidence="unknown")
