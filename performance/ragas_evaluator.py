"""
ragas_evaluator.py: Ragas-based Vector Retrieval Quality Evaluator.

Evaluates vector database retrieval quality BEFORE the LLM sees the documents,
calculating strict metrics including Context Precision and Context Recall
without requiring human-annotated reference answers.
"""

from __future__ import annotations

import os
import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# Try importing Ragas components
try:
    from ragas.llms import llm_factory
    from ragas.prompt import PydanticPrompt
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
    from ragas.metrics._context_recall import LLMContextRecall
    RAGAS_AVAILABLE = True
except Exception as e:
    logger.warning("Failed to import some ragas modules: %s", e)
    RAGAS_AVAILABLE = False
    from typing import Generic, TypeVar
    T = TypeVar("T")
    U = TypeVar("U")
    class PydanticPrompt(Generic[T, U]):
        pass
    class LLMContextPrecisionWithoutReference:
        pass
    class LLMContextRecall:
        pass
    def llm_factory(*args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Pydantic Schemas for Reference-Free Retrieval Evaluation Prompts
# ---------------------------------------------------------------------------

class QueryContextInput(BaseModel):
    question: str = Field(..., description="The user query being investigated")
    context: str = Field(..., description="The concatenated retrieved context chunks")


class RequirementCoverageItem(BaseModel):
    requirement: str = Field(..., description="Specific informational requirement or question element needed to answer the query")
    covered: int = Field(..., description="1 if the requirement can be completely answered from the retrieved context, 0 if missing")
    reason: str = Field(..., description="Brief factual justification based strictly on the context")


class RequirementCoverageOutput(BaseModel):
    requirements: List[RequirementCoverageItem] = Field(
        ...,
        description="Decomposed requirements and whether the context satisfies them"
    )


class ReferenceFreeRecallPrompt(PydanticPrompt[QueryContextInput, RequirementCoverageOutput]):
    """
    Ragas prompt that decomposes a user query into atomic informational requirements,
    then evaluates which proportion of those requirements are covered by the retrieved contexts.
    Enables strict Context Recall calculation without human-annotated ground truths.
    """
    name: str = "reference_free_context_recall"
    instruction: str = (
        "Identify the essential factual points, constraints, or sub-questions needed to completely answer the question. "
        "For each point, evaluate whether the provided context contains sufficient information to satisfy it (verdict 1) "
        "or if the information is missing or incomplete (verdict 0). Output JSON adhering to the schema."
    )
    input_model = QueryContextInput
    output_model = RequirementCoverageOutput
    examples = [
        (
            QueryContextInput(
                question="What was created on the first day and what was upon the face of the deep?",
                context="In the beginning God created heaven, and earth. And the earth was void and empty, and darkness was upon the face of the deep. And God said: Be light made. And light was made.",
            ),
            RequirementCoverageOutput(
                requirements=[
                    RequirementCoverageItem(
                        requirement="What was created on the first day (creation of light)",
                        covered=1,
                        reason="Context explicitly states God said 'Be light made' and light was created.",
                    ),
                    RequirementCoverageItem(
                        requirement="What was upon the face of the deep",
                        covered=1,
                        reason="Context explicitly mentions 'darkness was upon the face of the deep'.",
                    ),
                ]
            ),
        ),
        (
            QueryContextInput(
                question="What are the dimensions of Noah's ark and what were the names of his three sons?",
                context="Make thee an ark of timber planks. The length of the ark shall be three hundred cubits: the breadth of it fifty cubits, and the height of it thirty cubits.",
            ),
            RequirementCoverageOutput(
                requirements=[
                    RequirementCoverageItem(
                        requirement="Dimensions of Noah's ark (length, breadth, height)",
                        covered=1,
                        reason="Context specifies 300 cubits length, 50 cubits breadth, and 30 cubits height.",
                    ),
                    RequirementCoverageItem(
                        requirement="Names of Noah's three sons",
                        covered=0,
                        reason="Context does not mention Noah's sons or their names.",
                    ),
                ]
            ),
        ),
    ]


class ChunkRelevanceItem(BaseModel):
    chunk_index: int = Field(..., description="1-based index of the retrieved chunk")
    is_useful: int = Field(..., description="1 if chunk contains relevant signal for the query, 0 if noise/irrelevant")
    reason: str = Field(..., description="Why this chunk is or is not relevant to the query")


class ChunkRelevanceOutput(BaseModel):
    verdicts: List[ChunkRelevanceItem] = Field(..., description="Relevance classification per chunk")


class PreGenerationPrecisionPrompt(PydanticPrompt[QueryContextInput, ChunkRelevanceOutput]):
    """
    Ragas prompt to assess chunk-by-chunk relevance before generation to calculate
    Rank-Aware Context Precision (Mean Average Precision) directly at the vector DB level.
    """
    name: str = "pre_generation_context_precision"
    instruction: str = (
        "Given the user question and the numbered retrieved context chunks, evaluate each chunk independently. "
        "Classify each chunk as useful (1) if it directly addresses the question or provides necessary background context, "
        "or not useful / noise (0) if it is irrelevant. Output JSON."
    )
    input_model = QueryContextInput
    output_model = ChunkRelevanceOutput


# ---------------------------------------------------------------------------
# Output Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkDiagnostic:
    chunk_id: str
    rank: int
    content_snippet: str
    is_relevant: bool
    relevance_score: float
    reason: str


@dataclass
class RecallDiagnostic:
    requirement: str
    is_covered: bool
    reason: str


@dataclass
class RetrievalEvaluationResult:
    query: str
    k_retrieved: int
    context_precision: float
    context_recall: float
    signal_to_noise_ratio: float
    is_sufficient: bool
    verdict: str
    chunk_diagnostics: List[ChunkDiagnostic] = field(default_factory=list)
    recall_diagnostics: List[RecallDiagnostic] = field(default_factory=list)
    latency_ms: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "k_retrieved": self.k_retrieved,
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "signal_to_noise_ratio": round(self.signal_to_noise_ratio, 4),
            "is_sufficient": self.is_sufficient,
            "verdict": self.verdict,
            "chunk_diagnostics": [
                {
                    "chunk_id": c.chunk_id,
                    "rank": c.rank,
                    "is_relevant": c.is_relevant,
                    "relevance_score": c.relevance_score,
                    "reason": c.reason,
                    "content_snippet": c.content_snippet,
                }
                for c in self.chunk_diagnostics
            ],
            "recall_diagnostics": [
                {
                    "requirement": r.requirement,
                    "is_covered": r.is_covered,
                    "reason": r.reason,
                }
                for r in self.recall_diagnostics
            ],
            "latency_ms": round(self.latency_ms, 2),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class RagasRetrievalEvaluator:
    """
    Ragas-based evaluation engine specialized in vector retrieval quality.
    
    Verifies that your vector database (e.g. ChromaDB) fetches the right documents
    before passing them downstream to the LLM generation layer.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        precision_threshold: float = 0.60,
        recall_threshold: float = 0.60,
    ):
        self.model_name = (
            model_name
            or os.getenv("RAGAS_EVAL_MODEL")
            or "openai/gpt-oss-20b"
        )
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold

        # Initialize OpenAI client pointed at Groq
        self.openai_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Initialize Ragas LLM wrapper
        self.ragas_llm = llm_factory(
            self.model_name,
            client=self.openai_client,
            temperature=0.01,
        )

        # Initialize prompts & metrics
        self.recall_prompt = ReferenceFreeRecallPrompt()
        self.precision_prompt = PreGenerationPrecisionPrompt()

        try:
            self.native_precision_metric = LLMContextPrecisionWithoutReference(llm=self.ragas_llm)
        except Exception:
            self.native_precision_metric = None

        logger.info(
            "RagasRetrievalEvaluator initialized with model '%s' (base_url=%s)",
            self.model_name, self.base_url
        )

    @staticmethod
    def _extract_chunk_text_and_id(chunk: Any, idx: int) -> Tuple[str, str]:
        """Safely extract (chunk_id, text) from LangChain Document, dict, or tuple."""
        if isinstance(chunk, (tuple, list)):
            chunk = chunk[0]

        if hasattr(chunk, "page_content"):
            cid = getattr(chunk, "metadata", {}).get("chunk_id", f"chunk-{idx}")
            return str(cid), str(chunk.page_content)

        if isinstance(chunk, dict):
            cid = chunk.get("chunk_id", chunk.get("metadata", {}).get("chunk_id", f"chunk-{idx}"))
            text = chunk.get("content", chunk.get("page_content", str(chunk)))
            return str(cid), str(text)

        cid = getattr(chunk, "chunk_id", f"chunk-{idx}")
        text = getattr(chunk, "content", getattr(chunk, "text", str(chunk)))
        return str(cid), str(text)

    async def aevaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Any],
        response: Optional[str] = None,
    ) -> RetrievalEvaluationResult:
        """
        Asynchronously evaluate retrieval quality for a single query using Ragas.
        """
        t0 = time.perf_counter()

        if not retrieved_chunks:
            return RetrievalEvaluationResult(
                query=query,
                k_retrieved=0,
                context_precision=0.0,
                context_recall=0.0,
                signal_to_noise_ratio=0.0,
                is_sufficient=False,
                verdict="RETRIEVAL_MISS",
                latency_ms=(time.perf_counter() - t0) * 1000,
                summary="No chunks were retrieved by the vector database.",
            )

        # 1. Parse chunks into clean strings and IDs
        parsed_chunks: List[Tuple[str, str]] = []
        for i, doc in enumerate(retrieved_chunks, start=1):
            cid, text = self._extract_chunk_text_and_id(doc, i)
            parsed_chunks.append((cid, text))

        formatted_context_str = "\n\n".join(
            f"[Chunk {i}] (ID: {cid}):\n{text}"
            for i, (cid, text) in enumerate(parsed_chunks, start=1)
        )

        # 2. Compute Reference-Free Context Recall
        recall_task = self.recall_prompt.generate(
            data=QueryContextInput(question=query, context=formatted_context_str),
            llm=self.ragas_llm,
        )

        # 3. Compute Rank-Aware Context Precision
        precision_task = self.precision_prompt.generate(
            data=QueryContextInput(question=query, context=formatted_context_str),
            llm=self.ragas_llm,
        )

        recall_res, precision_res = await asyncio.gather(recall_task, precision_task)

        # --- Calculate Context Recall Score ---
        recall_items = recall_res.requirements if recall_res and recall_res.requirements else []
        if recall_items:
            num_covered = sum(item.covered for item in recall_items)
            recall_score = num_covered / len(recall_items)
        else:
            recall_score = 0.0

        recall_diagnostics = [
            RecallDiagnostic(
                requirement=item.requirement,
                is_covered=bool(item.covered),
                reason=item.reason,
            )
            for item in recall_items
        ]

        # --- Calculate Context Precision Score (MAP) ---
        precision_items = precision_res.verdicts if precision_res and precision_res.verdicts else []
        verdict_map = {item.chunk_index: item for item in precision_items}

        chunk_diagnostics = []
        verdict_binary_list = []

        for rank, (cid, text) in enumerate(parsed_chunks, start=1):
            verdict_item = verdict_map.get(rank)
            is_rel = bool(verdict_item.is_useful) if verdict_item else False
            reason = verdict_item.reason if verdict_item else "No evaluation verdict produced"
            verdict_binary_list.append(1 if is_rel else 0)

            chunk_diagnostics.append(
                ChunkDiagnostic(
                    chunk_id=cid,
                    rank=rank,
                    content_snippet=(text[:120] + "...") if len(text) > 120 else text,
                    is_relevant=is_rel,
                    relevance_score=1.0 if is_rel else 0.0,
                    reason=reason,
                )
            )

        # Rank-aware Mean Average Precision (MAP) as defined in Ragas
        total_relevant = sum(verdict_binary_list)
        if total_relevant > 0:
            numerator = sum(
                (sum(verdict_binary_list[:i + 1]) / (i + 1)) * verdict_binary_list[i]
                for i in range(len(verdict_binary_list))
            )
            precision_score = numerator / total_relevant
        else:
            precision_score = 0.0

        signal_to_noise = total_relevant / len(parsed_chunks) if parsed_chunks else 0.0

        # --- Overall Verdict ---
        is_sufficient = (
            precision_score >= self.precision_threshold
            and recall_score >= self.recall_threshold
        )

        if precision_score >= 0.8 and recall_score >= 0.8:
            verdict = "EXCELLENT"
        elif is_sufficient:
            verdict = "ADEQUATE"
        elif total_relevant == 0 or recall_score == 0:
            verdict = "RETRIEVAL_MISS"
        else:
            verdict = "POOR"

        latency = (time.perf_counter() - t0) * 1000

        covered_count = sum(1 for r in recall_diagnostics if r.is_covered)
        summary = (
            f"Retrieval {verdict}: Precision={precision_score:.2f}, Recall={recall_score:.2f}. "
            f"{total_relevant}/{len(parsed_chunks)} chunks relevant. "
            f"{covered_count}/{len(recall_diagnostics)} requirements satisfied."
        )

        return RetrievalEvaluationResult(
            query=query,
            k_retrieved=len(parsed_chunks),
            context_precision=precision_score,
            context_recall=recall_score,
            signal_to_noise_ratio=signal_to_noise,
            is_sufficient=is_sufficient,
            verdict=verdict,
            chunk_diagnostics=chunk_diagnostics,
            recall_diagnostics=recall_diagnostics,
            latency_ms=latency,
            summary=summary,
        )

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Any],
        response: Optional[str] = None,
    ) -> RetrievalEvaluationResult:
        """Synchronous wrapper for aevaluate_retrieval."""
        return asyncio.run(self.aevaluate_retrieval(query, retrieved_chunks, response))

    def evaluate_engine_queries(
        self,
        engine: Any,
        queries: List[str],
        k: int = 3,
    ) -> List[RetrievalEvaluationResult]:
        """
        Directly test the RAGEngine / vector database on a list of queries.
        """
        results = []
        for q in queries:
            logger.info("Evaluating retrieval for query: %s", q[:60])
            hits = engine.retrieve(q, k=k)
            res = self.evaluate_retrieval(query=q, retrieved_chunks=hits)
            results.append(res)
        return results

    def format_markdown_report(self, results: List[RetrievalEvaluationResult]) -> str:
        """Generate a formatted markdown evaluation report for the retrieval results."""
        if not results:
            return "# Ragas Retrieval Quality Report\n\nNo evaluation results to display."

        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_snr = sum(r.signal_to_noise_ratio for r in results) / len(results)
        sufficiency_rate = (sum(1 for r in results if r.is_sufficient) / len(results)) * 100.0

        lines = [
            "# 📊 Ragas Vector Retrieval Quality Report",
            "",
            "## Executive Summary",
            f"- **Evaluated Queries:** {len(results)}",
            f"- **Mean Context Precision (MAP):** `{avg_precision:.3f}` (Target: `≥{self.precision_threshold:.2f}`)",
            f"- **Mean Context Recall (Reference-Free):** `{avg_recall:.3f}` (Target: `≥{self.recall_threshold:.2f}`)",
            f"- **Average Signal-to-Noise Ratio:** `{avg_snr:.1%}`",
            f"- **Retrieval Sufficiency Rate:** `{sufficiency_rate:.1f}%`",
            "",
            "## Metric Definitions",
            "- **Context Precision:** Measures whether retrieved chunks are relevant to the query and penalizes ranking irrelevant chunks above relevant ones (Mean Average Precision).",
            "- **Context Recall (Reference-Free):** Decomposes the user query into key informational requirements and calculates the percentage covered by the retrieved contexts, without requiring human reference answers.",
            "",
            "## Evaluation Matrix",
            "",
            "| # | Query | Precision | Recall | S/N Ratio | Sufficiency | Verdict |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|",
        ]

        for i, r in enumerate(results, start=1):
            q_clean = r.query.replace("|", "\\|").replace("\n", " ")
            if len(q_clean) > 55:
                q_clean = q_clean[:52] + "..."
            suff_str = "✅ PASS" if r.is_sufficient else "❌ FAIL"
            lines.append(
                f"| {i} | {q_clean} | `{r.context_precision:.2f}` | `{r.context_recall:.2f}` | `{r.signal_to_noise_ratio:.0%}` | {suff_str} | **{r.verdict}** |"
            )

        lines.append("")
        lines.append("## Detailed Diagnostics per Query")
        lines.append("")

        for i, r in enumerate(results, start=1):
            lines.append(f"### Query {i}: {r.query}")
            lines.append(f"**Verdict:** `{r.verdict}` | **Precision:** `{r.context_precision:.2f}` | **Recall:** `{r.context_recall:.2f}` | **Latency:** `{r.latency_ms:.1f}ms`")
            lines.append("")
            lines.append("**Retrieved Chunks Evaluation:**")
            lines.append("")
            lines.append("| Rank | Chunk ID | Useful? | Justification | Snippet |")
            lines.append("|:---:|:---:|:---:|---|---|")
            for c in r.chunk_diagnostics:
                useful_icon = "✅ Yes" if c.is_relevant else "❌ No"
                snip = c.content_snippet.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {c.rank} | `{c.chunk_id}` | {useful_icon} | {c.reason} | *\"{snip}\"* |")

            lines.append("")
            lines.append("**Query Requirement Coverage (Recall Breakdown):**")
            lines.append("")
            for req in r.recall_diagnostics:
                icon = "✅ [Covered]" if req.is_covered else "❌ [Missing]"
                lines.append(f"- **{icon}:** *{req.requirement}* — {req.reason}")
            lines.append("")
            lines.append("---")

        return "\n".join(lines)
