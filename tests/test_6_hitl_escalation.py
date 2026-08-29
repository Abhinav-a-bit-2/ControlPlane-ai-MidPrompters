"""
test_6_hitl_escalation.py: Category 6 - Human-In-The-Loop (HITL) & Cost Escalation.
Evaluates that highly complex, multi-hop, or high-token reasoning queries
correctly trip the L4 cost / confidence threshold and escalate to a human ticket.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from tests.rag_client import RAGClient
from tests.custom_metrics import HITLEscalationMetric

client = RAGClient.get_instance()
hitl_metric = HITLEscalationMetric()

# 3-5 Incredibly complex, multi-hop reasoning questions designed to spike token usage / cost score
COMPLEX_ESCALATION_QUERIES = [
    "Provide a comprehensive cross-sectional comparative matrix synthesizing the interplay between Tier-1 and Tier-3 disaster recovery recovery point objectives, contractual termination liquidated damages, indemnification carve-outs for intellectual property infringement under Clause 9, and the precise mathematical formula for prorating subscription refunds during force majeure outages spanning multi-quarter billing cycles.",
    "Analyze every single discrepancy across all Genesis policy amendments regarding retroactive audit rights, customer liability thresholds, subprocessor authorization requirements, and SLA credit penalty percentages, cross-referencing each clause against statutory GDPR compliance standards.",
    "Synthesize a unified audit assessment comparing the mutual indemnification covenants, third-party disclosure constraints, and security incident escalation protocols across all historical revisions of Genesis Master Agreements, calculating total cumulative risk exposure under extreme catastrophic loss scenarios.",
    "Perform a multi-hop evaluation comparing the exact operational SLAs for emergency database failovers against the penalty credit payout caps under billing dispute policies, analyzing whether liquidated damages supersede arbitration mandates when scheduled maintenance exceeds 48 consecutive hours."
]

@pytest.mark.parametrize("query", COMPLEX_ESCALATION_QUERIES)
def test_complexity_cost_escalation_to_hitl(query):
    """Verify that extreme multi-hop queries trigger Human-In-The-Loop routing with Ticket ID."""
    res = client.query(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=res.answer
    )
    # Verify ticket escalation was triggered
    assert_test(test_case, [hitl_metric])
    assert res.blocked is True or res.hitl_ticket_id is not None
