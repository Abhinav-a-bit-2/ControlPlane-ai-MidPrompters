import os
import uuid
import json
from pathlib import Path
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import json

import telemetry
from base_rag import RAGEngine
from session_manager import SessionManager
from security.pipeline import SecureRAGPipeline
from security.intent_pipeline import IntentRoutedPipeline
from performance.performanceEvaluator import PerformanceEvaluator
from main import contextualize_query, groq_client, SOURCE

load_dotenv()
telemetry.init_telemetry()

app = FastAPI(title="Secure RAG Control Plane API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Pipeline Initialization
# ---------------------------------------------------------
engine = RAGEngine(SOURCE)
existing_cnt = len(engine.vector_store.get()["ids"])
if existing_cnt == 0:
    engine.index()

session_mgr = SessionManager()
intent_pipeline = IntentRoutedPipeline(engine)
standard_pipeline = SecureRAGPipeline(engine)
evaluator = PerformanceEvaluator(groq_client=groq_client, session_manager=session_mgr)


# ---------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_intent_pipeline: bool = True


class AuditTrailItem(BaseModel):
    layer: str
    passed: bool
    detail: str
    latency_ms: float


class FlaggedClaim(BaseModel):
    text: str
    status: str
    risk_score: float
    cited_chunks: List[str]


class SafeChunk(BaseModel):
    chunk_id: str
    content: str


class SpanDetail(BaseModel):
    id: str
    name: str
    kind: str  # "llm" | "retriever" | "guard" | "classifier"
    status: str  # "ok" | "blocked"
    latency_ms: float
    attributes: dict


class ChatResponse(BaseModel):
    session_id: str
    rewritten_query: str
    answer: str
    blocked: bool
    blocked_at_layer: str
    audit_trail: List[AuditTrailItem]
    quarantined_chunk_ids: List[str]
    hitl_ticket_id: Optional[str] = None
    total_tokens: int
    cost_score: float
    overall_confidence: str
    risk_score: float
    semantic_entropy: float
    flagged_claims: List[FlaggedClaim]
    safe_chunks: List[SafeChunk]
    spans: List[SpanDetail]
    intent_label: Optional[str] = "general"
    intent_confidence: Optional[float] = 1.0
    prompt_delta: Optional[str] = "Standard Security Ruleset"


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def serialize_chunks(raw_chunks: Any) -> List[SafeChunk]:
    chunks_out = []
    if not raw_chunks or raw_chunks == " ":
        return chunks_out

    for item in raw_chunks:
        chunk_id, content = PerformanceEvaluator._extract_chunk_data(item)
        if chunk_id and content:
            chunks_out.append(SafeChunk(chunk_id=str(chunk_id), content=str(content)))
    return chunks_out


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    question = payload.message.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    recent_turns = session_mgr.recentKChats(session_id, k=4)
    search_query, ctx_tokens = contextualize_query(question, recent_turns)

    if payload.use_intent_pipeline:
        result = intent_pipeline.ask(
            search_query,
            original_question=question,
            session_id=session_id,
            initial_tokens=ctx_tokens,
        )
    else:
        result = standard_pipeline.ask(
            search_query,
            session_id=session_id,
            initial_tokens=ctx_tokens,
        )

    # Performance Evaluation
    report = evaluator.evaluate(
        answer=result.answer,
        retrieved_chunks=result.safe_chunks if result.safe_chunks != " " else [],
        session_id=session_id,
    )

    # Save to history on pass
    if not result.audit_trail or all(e.passed for e in result.audit_trail):
        session_mgr.addChats(session_id, role="user", content=question)
        session_mgr.addChats(session_id, role="assistant", content=result.answer)

    audit_items = [
        AuditTrailItem(
            layer=e.layer,
            passed=e.passed,
            detail=e.detail or "",
            latency_ms=round(e.latency_ms, 2),
        )
        for e in result.audit_trail
    ]
    intent_info = getattr(result, "intent_metadata", {"label": "retrieval_augmented", "confidence": 0.94, "delta": "Enforced strict context-grounding system prompt."})
    trace_spans = [
        SpanDetail(
            id=f"span_{i+1}",
            name=entry.layer,
            kind=(
                "guard"
                if "L" in entry.layer or "heuristic" in entry.layer or "sanitize" in entry.layer
                else (
                    "llm"
                    if "generation" in entry.layer or "filter" in entry.layer
                    else ("classifier" if "intent" in entry.layer else "retriever")
                )
            ),
            status="ok" if entry.passed else "blocked",
            latency_ms=round(entry.latency_ms, 2),
            attributes={
                "detail": entry.detail or "",
                "passed": entry.passed,
                "session_id": session_id,
                "query": question,
                "rewritten_query": search_query,
                "total_tokens": result.total_tokens,
                "cost_score": round(result.cost_score, 2),
            },
        )
        for i, entry in enumerate(result.audit_trail)
    ]

    flagged = [
        FlaggedClaim(
            text=fc.text,
            status=fc.status,
            risk_score=round(fc.risk_score, 3),
            cited_chunks=fc.cited_chunks,
        )
        for fc in report.flagged_claims
    ]

    safe_chunks_serialized = serialize_chunks(result.safe_chunks)

    return ChatResponse(
        session_id=session_id,
        rewritten_query=search_query,
        answer=result.answer,
        blocked=result.blocked,
        blocked_at_layer=result.blocked_at_layer,
        audit_trail=audit_items,
        quarantined_chunk_ids=result.quarantined_chunk_ids,
        hitl_ticket_id=result.hitl_ticket_id,
        total_tokens=result.total_tokens,
        cost_score=round(result.cost_score, 2),
        overall_confidence=report.overall_confidence,
        risk_score=round(report.risk_score, 3),
        semantic_entropy=round(report.semantic_entropy, 3),
        flagged_claims=flagged,
        safe_chunks=safe_chunks_serialized,
        spans=trace_spans,

    )

@app.post("/api/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    question = payload.message.strip()

    async def event_generator():
        recent_turns = session_mgr.recentKChats(session_id, k=4)
        search_query, ctx_tokens = contextualize_query(question, recent_turns)

        # 1. Run Guard Rails / Intent Pipeline up to generation
        if payload.use_intent_pipeline:
            result = intent_pipeline.ask(search_query, original_question=question, session_id=session_id, initial_tokens=ctx_tokens)
        else:
            result = standard_pipeline.ask(search_query, session_id=session_id, initial_tokens=ctx_tokens)

        # 2. Check if query blocked before generation
        if result.blocked:
            yield {
                "event": "blocked",
                "data": json.dumps({"reason": result.blocked_at_layer, "audit_trail": [e.__dict__ for e in result.audit_trail]})
            }
            return

        # 3. Stream Answer Tokens (Simulated or Groq Stream)
        tokens = result.answer.split(" ")
        for token in tokens:
            yield {
                "event": "token",
                "data": json.dumps({"token": token + " "})
            }

        # 4. Asynchronous / Post-Gen Telemetry Evaluation
        report = evaluator.evaluate(answer=result.answer, retrieved_chunks=result.safe_chunks if result.safe_chunks != " " else [], session_id=session_id)
        
        telemetry_payload = {
            "session_id": session_id,
            "rewritten_query": search_query,
            "total_tokens": result.total_tokens,
            "cost_score": round(result.cost_score, 2),
            "overall_confidence": report.overall_confidence,
            "risk_score": round(report.risk_score, 3),
            "semantic_entropy": round(report.semantic_entropy, 3),
            "flagged_claims": [fc.__dict__ for fc in report.flagged_claims],
            "safe_chunks": serialize_chunks(result.safe_chunks),
            "audit_trail": [e.__dict__ for e in result.audit_trail]
        }

        yield {
            "event": "telemetry",
            "data": json.dumps(telemetry_payload)
        }

    return EventSourceResponse(event_generator())

@app.get("/api/session/history")
async def get_history(session_id: str):
    return {"history": session_mgr.tokenBudgetedChats(session_id, token_budget=4096)}


@app.post("/api/session/reset")
async def reset_session(session_id: str):
    success = session_mgr.deleteChat(session_id)
    return {"status": "cleared" if success else "not_found", "session_id": session_id}


@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    raw_data = engine.vector_store.get(where={"chunk_id": chunk_id})
    if not raw_data["documents"]:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {
        "chunk_id": chunk_id,
        "content": raw_data["documents"][0],
        "metadata": raw_data["metadatas"][0] if raw_data["metadatas"] else {},
    }


static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)