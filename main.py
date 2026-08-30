"""
Entry point. Replace SOURCE with your document path.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import uuid
from base_rag import RAGEngine
from session_manager import SessionManager
from security.pipeline import SecureRAGPipeline
from security.intent_pipeline import IntentRoutedPipeline

# Toggle between pipelines for A/B testing
USE_INTENT_PIPELINE = True
import telemetry

telemetry.init_telemetry()
from performance.performanceEvaluator import PerformanceEvaluator

SOURCE = str(
    Path(__file__).parent
    / "confluence"
    # / "dsid_0a2cd37d53ff47d4aced289cd9a76fe8__evidence-driven-offer-evaluation-and-onboarding-trigger-playbook-2028.txt"
    /"genesis.txt"
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# def contextualize_query(question: str, chat_history: list[dict]) -> tuple[str, int]:
#     if not chat_history:
#         return question, 0

#     history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

#     prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone question containing all necessary context for document search. Do not answer. If the current query,introduces 
#     a new topic which is independent of conversation history then return the original query as it was given to you.

# Chat History:
# {history_text}

# Follow-up Question: {question}
# Standalone Question:"""

#     with telemetry.trace_span("Query_Contextualize") as span:
#         completion = groq_client.chat.completions.create(
#             model="openai/gpt-oss-120b",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=150,
#             reasoning_effort="low"
#         )
#         rewritten = completion.choices[0].message.content.strip()
        
#         ctx_tokens = 0
#         # Explicitly set OpenInference attributes so Phoenix calculates Cost
#         if completion.usage:
#             telemetry.set_llm_attributes(
#                 span, 
#                 "openai/gpt-oss-120b", 
#                 completion.usage.prompt_tokens, 
#                 completion.usage.completion_tokens
#             )
#             ctx_tokens = completion.usage.prompt_tokens + completion.usage.completion_tokens
            
#         print("="*70)
#         print("this is what is passed into rewritter: \n"+history_text+"\n\n"+"*"*100)
#         print("\nthis is what is rewritten: \n"+rewritten+"\n\n"+"*"*100)
#         print("="*70)
    
#         return (rewritten if rewritten else question), ctx_tokens


def _clean_history_message(role: str, content: str) -> str:
    """Strip chunk citations and compress multi-line assistant outputs."""

    cleaned = re.sub(r"[\[【]chunk-\d+[\]】]", "", content)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if role == "assistant":
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        cleaned = " ".join(sentences[:2])
    return f"{role.capitalize()}: {cleaned}"

def contextualize_query(question: str, chat_history: list[dict]) -> tuple[str, int]:
    trimmed_history = chat_history[-4:] if chat_history else []
    
    if not trimmed_history:
        return question, 0

    history_lines = [_clean_history_message(m["role"], m["content"]) for m in trimmed_history]
    history_text = "\n".join(history_lines)

    system_prompt = (
        "You rewrite conversational search queries into standalone search queries. "
        "If the question is self-contained or switches topics, return it unchanged. "
        "Output ONLY the final rewritten question without explanation."
    )
    user_prompt = f"Chat History:\n{history_text}\n\nFollow-up Question: {question}\nStandalone Question:"

    with telemetry.trace_span("Query_Contextualize") as span:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = completion.choices[0].message.content.strip()
        rewritten = (rewritten if rewritten else question)
        ctx_tokens = 0
        if completion.usage:
            telemetry.set_llm_attributes(
                span, 
                "openai/gpt-oss-20b", 
                completion.usage.prompt_tokens, 
                completion.usage.completion_tokens
            )
            ctx_tokens = completion.usage.prompt_tokens + completion.usage.completion_tokens
            print("\n\n ---  Rewritten  --- \n\n")
            print(f"Context given : {history_text}\n\n")
            print(f"Rewritten text : {rewritten}\n\n")
            print(f"Tokens used : {ctx_tokens}\n\n")

        return rewritten, ctx_tokens

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
    if USE_INTENT_PIPELINE:
        pipeline = IntentRoutedPipeline(engine)
        print("[Mode] Intent-Routed Dual-Path Pipeline")
    else:
        pipeline = SecureRAGPipeline(engine)
        print("[Mode] Standard Secure Pipeline")
    evaluator = PerformanceEvaluator(groq_client=groq_client, session_manager=session_mgr)

    while True:
        question = input("\nQuestion (or 'quit'): ").strip()
        if question.lower() == "quit":
            break

        with telemetry.trace_span("User_Turn", {"session_id": session_id, "query": question}) as span:
            recent_turns = session_mgr.recentKChats(session_id, k=4)
            search_query, ctx_tokens = contextualize_query(question, recent_turns)
            if search_query != question:
                print(f"  [Rewritten Query for Search]: {search_query}")
    
            if USE_INTENT_PIPELINE:
                result = pipeline.ask(
                    search_query,
                    original_question=question,
                    session_id=session_id,
                    initial_tokens=ctx_tokens,
                )
            else:
                result = pipeline.ask(search_query, session_id=session_id, initial_tokens=ctx_tokens)
            
            # result.total_tokens now includes ctx_tokens (because we passed it as initial_tokens)
            total_turn_tokens = result.total_tokens
            span.set_attribute("llm.token_count.total", total_turn_tokens)
    
            print(f"\n--- Audit trail ---")
            for entry in result.audit_trail:
                status = "PASS" if entry.passed else "BLOCK"
                print(f"  [{status}] {entry.layer}: {entry.detail} ({entry.latency_ms:.1f}ms)")
    
            if result.quarantined_chunk_ids:
                print(f"  Quarantined chunks: {result.quarantined_chunk_ids}")
            
            if result.hitl_ticket_id:
                print(f"  [HITL Escalation] Ticket ID: {result.hitl_ticket_id}")
    
            print(f"\nAnswer: {result.answer}")
            print(f"\n[Telemetry] Cost Score: {result.cost_score:.2f} | Tokens: {total_turn_tokens}")

            # Performance evaluation (risk scoring)
            report = evaluator.evaluate(
                answer=result.answer,
                retrieved_chunks=result.safe_chunks,
                session_id=session_id,
            )
            print(f"[Performance] Confidence: {report.overall_confidence} | Risk: {report.risk_score:.3f} | Entropy: {report.semantic_entropy:.3f}")
            if report.flagged_claims:
                for fc in report.flagged_claims:
                    print(f"  ⚠ [{fc.status}] {fc.text[:80]}")
    
            if not result.audit_trail or all(e.passed for e in result.audit_trail):
                session_mgr.addChats(session_id, role="user", content=question)
                session_mgr.addChats(session_id, role="assistant", content=result.answer)


if __name__ == "__main__":
    main()
