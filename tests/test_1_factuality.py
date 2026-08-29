"""
test_1_factuality.py: Category 1 - Factuality & Hallucination Prevention.
Evaluates in-scope factual grounding & chunk citation vs out-of-scope hallucination refusal.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from tests.rag_client import RAGClient
from tests.custom_metrics import (
    ChunkCitationMetric,
    RefusalCheckMetric,
    create_factuality_metric
)

client = RAGClient.get_instance()
factuality_metric = create_factuality_metric()
citation_metric = ChunkCitationMetric()
refusal_metric = RefusalCheckMetric()

# 3-5 In-Scope Factual Questions
IN_SCOPE_QUERIES = [
    "In the beginning, what did God create?",
    "What was upon the face of the deep before God said 'Be light made'?",
    "What did God call the light and the darkness on the first day?",
    "What did God create to divide the waters that were under the firmament from those that were above the firmament?"
]

# 3-5 Out-of-Scope Factual Questions
OUT_OF_SCOPE_QUERIES = [
    "What is the capital city of Australia?",
    "Who painted the Mona Lisa?",
    "What are the chemical components of water?",
    "How many Oscars did the movie Titanic win?"
]

@pytest.mark.parametrize("query", IN_SCOPE_QUERIES)
def test_in_scope_factuality_and_citation(query):
    """Verify in-scope queries are answered accurately with proper chunk citation."""
    res = client.query(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=res.answer
    )
    # If not escalated to HITL, verify citations and factual grounding
    if not res.hitl_ticket_id:
        assert_test(test_case, [citation_metric, factuality_metric])
    else:
        # Escalation is a valid safe exit if ambiguity/cost threshold is met
        assert res.blocked is True

@pytest.mark.parametrize("query", OUT_OF_SCOPE_QUERIES)
def test_out_of_scope_hallucination_refusal(query):
    """Verify agent refuses out-of-scope factual queries without hallucinating."""
    res = client.query(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=res.answer
    )
    assert_test(test_case, [refusal_metric])
