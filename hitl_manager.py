import argparse
import time
from session_manager import SessionManager
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Re-use tracer configuration if we are executing this standalone
import telemetry

tracer = trace.get_tracer(__name__)

def resolve_ticket(ticket_id: str, answer: str, time_spent_sec: int):
    session_mgr = SessionManager()
    ticket = session_mgr.get_hitl_ticket(ticket_id)
    
    if not ticket:
        print(f"Ticket {ticket_id} not found.")
        return
        
    print(f"Resolving Ticket: {ticket_id}")
    print(f"Original Query: {ticket['query']}")
    print(f"Trace ID to link: {ticket['trace_id']}")
    
    # Calculate a mock human cost (e.g. $0.50 per minute)
    cost_per_sec = 0.50 / 60.0
    total_cost = time_spent_sec * cost_per_sec
    
    # We create a new trace for the human interaction, but we can link it
    # to the original trace ID so observability platforms can join them.
    # For simplicity, we'll just emit a new span that includes the original trace ID as an attribute.
    
    with telemetry.trace_span("Human_Resolution_Action") as span:
        span.set_attribute("original_trace_id", ticket['trace_id'])
        span.set_attribute("session_id", ticket['session_id'])
        span.set_attribute("human.time_spent_sec", time_spent_sec)
        span.set_attribute("human.cost_usd", total_cost)
        span.set_status(Status(StatusCode.OK))
        
        # Save to chat history as assistant
        session_mgr.addChats(ticket['session_id'], role="assistant", content=answer)
        
        # Resolve ticket in queue
        session_mgr.resolve_hitl_ticket(ticket_id)
        print("Ticket resolved and cost logged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HITL Ticket Resolver")
    parser.add_argument("--ticket", type=str, required=True, help="Ticket ID")
    parser.add_argument("--answer", type=str, required=True, help="Resolution answer")
    parser.add_argument("--time_sec", type=int, default=60, help="Time spent by human")
    
    args = parser.parse_args()
    resolve_ticket(args.ticket, args.answer, args.time_sec)
