"""
test_2_intent_routing.py: Category 2 - Intent Routing & Tone Adherence.
Evaluates support empathy & conversational tone vs strict direct-quoting contract adherence.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from tests.rag_client import RAGClient
from tests.custom_metrics import (
    create_support_empathy_metric,
    create_contract_quoting_metric
)

client = RAGClient.get_instance()
support_metric = create_support_empathy_metric()
contract_metric = create_contract_quoting_metric()

# 3-5 Support Intent Queries (frustrated, urgent, confused)
SUPPORT_QUERIES = [
    "I have been locked out of my account for three hours, this is unacceptable and I am losing critical business! Help me right now!",
    "Nothing in this system makes any sense to me, I'm completely lost and frustrated trying to find my billing settings. Please walk me through this.",
    "Urgent! Our automated sync failed right before our product launch, I am panicking, what should I do first?",
    "Why is your interface so difficult? I just want to update my email address without getting error codes everywhere!"
]

# 3-5 Contract Intent Queries (strict legal, compliance, clauses)
CONTRACT_QUERIES = [
    "What are the exact statutory indemnification and liability limitations under the Master Services Agreement?",
    "State verbatim the confidentiality obligations and survival terms binding both parties upon agreement termination.",
    "What are the governing law, arbitration jurisdiction, and dispute resolution provisions defined in the agreement?",
    "Provide the literal contractual definitions of Force Majeure and excused non-performance conditions."
]

@pytest.mark.parametrize("query", SUPPORT_QUERIES)
def test_support_intent_empathy_and_tone(query):
    """Verify support queries trigger empathetic, patient, and conversational tone."""
    res = client.query(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=res.answer
    )
    if not res.blocked:
        assert_test(test_case, [support_metric])
    else:
        # In case query was flagged or escalated
        assert res.hitl_ticket_id is not None or res.blocked_at_layer != ""

@pytest.mark.parametrize("query", CONTRACT_QUERIES)
def test_contract_intent_quoting_and_precision(query):
    """Verify legal/contract queries quote terms directly without vague paraphrasing."""
    res = client.query(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=res.answer
    )
    if not res.blocked:
        assert_test(test_case, [contract_metric])
    else:
        assert res.hitl_ticket_id is not None or res.blocked_at_layer != ""
