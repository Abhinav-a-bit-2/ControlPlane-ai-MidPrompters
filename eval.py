#!/usr/bin/env python3
"""
EnterpriseRAG-Bench Evaluation Script

This script:
1. Loads the benchmark questions from questions.jsonl
2. Runs your RAG pipeline (embedding + retrieval + generation) on each question
3. Writes answers in the required JSONL format
4. (Optional) Runs the official metrics-based evaluation

Usage:
    python evaluate_rag.py --questions questions.jsonl --output answers.jsonl
    python evaluate_rag.py --evaluate --answers answers.jsonl

Environment variables required:
    OPENAI_BASE_URL      # Your OpenAI-compatible endpoint
    LLM_API_KEY          # API key for the LLM (if using OpenAI/Anthropic)
    LLM_PROVIDER         # "openai" or "anthropic" (default: "openai")
    LLM_MODEL_NAME       # Optional: override default model
"""

import json
import os
import argparse
from typing import List, Dict, Any
from pathlib import Path

# ============================================================
# 1. Import your RAG pipeline components
# ============================================================
# Adapt these imports to match your actual RAG implementation.
# This example assumes you have a function `answer_with_rag(query: str) -> dict`
# that returns {"answer": str, "document_ids": List[str]}

from base_rag import final_answer as answer_with_rag  # <-- REPLACE with your actual import


# ============================================================
# 2. Load benchmark questions
# ============================================================
def load_questions(questions_path: str) -> List[Dict[str, Any]]:
    with open(questions_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []

        # If the file starts with '[', parse it as a standard JSON array
        if content.startswith("["):
            return json.loads(content)

        # Fallback: parse line-by-line as JSONL / single JSON object
        questions = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                questions.append(json.loads(line))
        return questions


# ============================================================
# 3. Run RAG pipeline on each question
# ============================================================
def run_rag_on_questions(
    questions: List[Dict[str, Any]],
    output_path: str,
    resume: bool = False,
) -> None:
    """
    Run the RAG pipeline on each question and write answers to a JSONL file.

    Args:
        questions: List of question objects from the benchmark.
        output_path: Path to write the answers JSONL file.
        resume: If True, skip questions already in the output file.
    """
    # Load existing answers if resuming
    existing_ids = set()
    if resume and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_ids.add(data.get("question_id"))

    mode = "a" if resume and Path(output_path).exists() else "w"
    with open(output_path, mode, encoding="utf-8") as out_f:
        for q in questions:
            q_id = q.get("question_id")
            if q_id in existing_ids:
                print(f"Skipping {q_id} (already processed)")
                continue

            query = q.get("question", "")
            print(f"Processing {q_id}: {query[:60]}...")

            try:
                # Call your RAG pipeline
                result = answer_with_rag(query)

                # Build the required output format
                # The benchmark expects:
                #   {"question_id": "...", "answer": "...", "document_ids": ["dsid_..."]}
                output_entry = {
                    "question_id": q_id,
                    "answer": result.get("answer", ""),
                    "document_ids": result.get("document_ids", []),
                }

                # Write as JSONL
                out_f.write(json.dumps(output_entry) + "\n")
                out_f.flush()

            except Exception as e:
                print(f"ERROR on {q_id}: {e}")
                # Write a fallback entry so evaluation can still run
                fallback = {
                    "question_id": q_id,
                    "answer": "[ERROR] Could not generate answer.",
                    "document_ids": [],
                }
                out_f.write(json.dumps(fallback) + "\n")
                out_f.flush()


# ============================================================
# 4. Run official metrics-based evaluation
# ============================================================
def run_metrics_evaluation(answers_path: str, parallelism: int = 4) -> None:
    """
    Run the official EnterpriseRAG-Bench metrics evaluation.

    This requires the benchmark repository to be available and the
    src.scripts.answer_evaluation.metrics_based_eval module to be importable.

    Args:
        answers_path: Path to the answers JSONL file.
        parallelism: Number of parallel evaluation workers.
    """
    try:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "src.scripts.answer_evaluation.metrics_based_eval",
            "--answers-file",
            answers_path,
            "--parallelism",
            str(parallelism),
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except ImportError:
        print(
            "WARNING: Could not import the official evaluation module.\n"
            "Make sure you have the EnterpriseRAG-Bench repository in your PYTHONPATH.\n"
            "You can run the evaluation manually with:\n"
            "  python -m src.scripts.answer_evaluation.metrics_based_eval "
            f"--answers-file {answers_path}"
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")


# ============================================================
# 5. Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG pipeline on EnterpriseRAG-Bench"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="manual_question.json",
        help="Path to the benchmark questions.jsonl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Path to write the answers JSONL file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed questions)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the official metrics-based evaluation after generating answers",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of parallel workers for evaluation (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to process (for quick testing)",
    )

    args = parser.parse_args()

    # Check that the questions file exists
    if not Path(args.questions).exists():
        print(f"ERROR: Questions file not found: {args.questions}")
        print("Download it from:")
        print("  https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/latest")
        return 1

    # Load questions
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]

    print(f"Loaded {len(questions)} questions from {args.questions}")

    # Run RAG pipeline
    run_rag_on_questions(questions, args.output, resume=args.resume)

    print(f"\n✅ Answers written to {args.output}")

    # Optionally run evaluation
    if args.evaluate:
        run_metrics_evaluation(args.output, parallelism=args.parallelism)

    return 0


if __name__ == "__main__":
    exit(main())