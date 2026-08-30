"""
run_ragas_eval.py: CLI and Test Runner for Ragas Vector Retrieval Evaluation.

Tests whether ChromaDB vector search retrieves the right documents before the LLM
generation phase, calculating Context Precision (MAP) and Context Recall (Reference-Free).

Usage:
    python run_ragas_eval.py
    python run_ragas_eval.py --query "What did God create on the first day?"
    python run_ragas_eval.py --k 4 --output-report ragas_retrieval_report.md
"""

import os
import sys
import argparse
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

from base_rag import RAGEngine
from performance.ragas_evaluator import RagasRetrievalEvaluator, RetrievalEvaluationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RagasRunner")

DEFAULT_BENCHMARK_QUERIES = [
    # 1. High-precision In-Scope Factual Query
    "In the beginning, what did God create?",
    # 2. Multi-part informational requirement (testing Context Recall coverage)
    "What was upon the face of the deep before light was made, and what did God name the light?",
    # 3. Specific entity query (testing chunk ranking & precision)
    "Who was the husbandman who killed his shepherd brother Abel?",
    # 4. Partial / Edge-case query (only partially covered in early chapters)
    "What were the exact dimensions of Noah's ark, and how many years did Abraham live?",
    # 5. Completely Out-of-Scope query (testing retrieval miss detection)
    "What is the capital city of Australia and what are its geographic coordinates?",
]


def run_evaluation(
    queries: List[str],
    source_path: str = "confluence/genesis.txt",
    k: int = 3,
    output_report: str = "ragas_retrieval_report.md",
    eval_model: str = "openai/gpt-oss-20b",
):
    print("=" * 80)
    print("  RAGAS VECTOR RETRIEVAL QUALITY EVALUATION SUITE")
    print("=" * 80)
    print(f"Vector Database Context: {source_path}")
    print(f"Retrieval Depth (k):     {k}")
    print(f"Evaluator LLM Model:     {eval_model}")
    print(f"Total Test Queries:      {len(queries)}")
    print("=" * 80)

    # Initialize RAG engine and Ragas evaluator
    engine = RAGEngine(source_path)
    evaluator = RagasRetrievalEvaluator(model_name=eval_model)

    results: List[RetrievalEvaluationResult] = []

    for i, q in enumerate(queries, start=1):
        print(f"\n[{i}/{len(queries)}] Testing Query: '{q}'")
        hits = engine.retrieve(q, k=k)
        print(f"  Retrieved {len(hits)} chunks from ChromaDB.")

        eval_result = evaluator.evaluate_retrieval(query=q, retrieved_chunks=hits)
        results.append(eval_result)

        print(f"  -> Context Precision: {eval_result.context_precision:.2f}")
        print(f"  -> Context Recall:    {eval_result.context_recall:.2f}")
        print(f"  -> S/N Ratio:         {eval_result.signal_to_noise_ratio:.0%}")
        print(f"  -> Verdict:           {eval_result.verdict} ({'SUFFICIENT' if eval_result.is_sufficient else 'INSUFFICIENT'})")
        print(f"  -> Summary:           {eval_result.summary}")

    # Generate Markdown Report
    report_md = evaluator.format_markdown_report(results)

    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETE - SUMMARY REPORT")
    print("=" * 80)
    print(report_md)
    print(f"\n[Saved] Detailed report saved to: {output_report}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vector DB Retrieval with Ragas")
    parser.add_argument("--query", type=str, default=None, help="Run evaluation on a single custom query")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve (k)")
    parser.add_argument("--source", type=str, default="confluence/genesis.txt", help="Knowledge base source file")
    parser.add_argument("--output-report", type=str, default="ragas_retrieval_report.md", help="Output markdown report path")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Groq model to use for Ragas evaluation")

    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_BENCHMARK_QUERIES
    run_evaluation(
        queries=queries,
        source_path=args.source,
        k=args.k,
        output_report=args.output_report,
        eval_model=args.model,
    )


if __name__ == "__main__":
    main()
