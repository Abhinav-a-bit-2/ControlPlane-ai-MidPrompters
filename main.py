"""
Entry point. Replace SOURCE with your document path.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import uuid
from base_rag import RAGEngine
from session_manager import SessionManager
from security.pipeline import SecureRAGPipeline
from performance.performanceEvaluator import PerformanceEvaluator

SOURCE = str(
    Path(__file__).parent
    / "confluence"
    / "dsid_0a3c5810b26347739f3e1a3b0a774d7c__slo-driven-fidelity-onboarding-checklist-2029-06-14.txt"
)



groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def contextualize_query(question: str, chat_history: list[dict]) -> str:
    if not chat_history:
        return question

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone question containing all necessary context for document search. Do not answer, only rewrite.

Chat History:
{history_text}

Follow-up Question: {question}
Standalone Question:"""

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        reasoning_effort="low"
    )
    rewritten = completion.choices[0].message.content.strip()
    print("="*70)
    print("this is what is passed into rewritter: \n"+history_text+"\n\n"+"*"*100)
    print("\nthis is what is rewritten: \n"+rewritten+"\n\n"+"*"*100)
    print("="*70)

    return rewritten if rewritten else question

def main():
    engine = RAGEngine(SOURCE)
    existing_cnt = len(engine.vector_store.get()["ids"])
    if existing_cnt == 0:
        print(f"Collection '{engine.uuid}' is empty. Indexing document...")
        n_chunks = engine.index()
        print(f"{n_chunks} chunks indexed into collection '{engine.uuid}'.")
    else:
        print(f"Collection '{engine.uuid}' already contains {existing_cnt} chunks. Skipping indexing.")
    session_mgr = SessionManager()
    session_id = str(uuid.uuid4())
    pipeline = SecureRAGPipeline(engine)
    evaluator = PerformanceEvaluator(groq_client=groq_client)

    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() == "quit":
            break

        recent_turns = session_mgr.recentKChats(session_id, k=10)
        search_query = contextualize_query(question, recent_turns)
        if search_query != question:
            print(f"  [Rewritten Query for Search]: {search_query}")

        result = pipeline.ask(search_query)

        print(f"\n--- Audit trail ---")
        for entry in result.audit_trail:
            status = "PASS" if entry.passed else "BLOCK"
            print(f"  [{status}] {entry.layer}: {entry.detail} ({entry.latency_ms:.1f}ms)")

        if result.quarantined_chunk_ids:
            print(f"  Quarantined chunks: {result.quarantined_chunk_ids}")

        print(f"\nAnswer: {result.answer}")

        passed_security = not result.audit_trail or all(e.passed for e in result.audit_trail)
        if passed_security and getattr(result, "safe_chunks", None):
            safe_hits = [(chunk, 1.0) for chunk in result.safe_chunks]
            eval_report = evaluator.evaluate(
                answer=result.answer,
                retrieved_chunks=safe_hits,
                messages=getattr(result, "generation_messages", None)
            )

            print(f"\n--- Performance & Grounding Evaluation ---")
            print(f"  Confidence Level : {eval_report.overall_confidence.upper()}")
            print(f"  Risk Score       : {eval_report.risk_score:.2f}")
            if eval_report.semantic_entropy > 0:
                print(f"  Semantic Entropy : {eval_report.semantic_entropy:.2f}")

            if eval_report.flagged_claims:
                print("  Flagged Claims:")
                for fc in eval_report.flagged_claims:
                    print(f"    • [{fc.status.upper()}] (Risk: {fc.risk_score:.2f}) -> {fc.text[:80]}...")

        if not result.audit_trail or all(e.passed for e in result.audit_trail):
            session_mgr.addChats(session_id, role="user", content=question)
            session_mgr.addChats(session_id, role="assistant", content=result.answer)


if __name__ == "__main__":
    main()
