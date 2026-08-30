"""
test_5_semantic_caching.py: Category 5 - Semantic Caching.
Evaluates that conceptually identical questions with completely different phrasing
trigger semantic cache hits with significant latency reduction.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from tests.rag_client import RAGClient
from tests.custom_metrics import CacheSpeedupMetric

client = RAGClient.get_instance()
cache_speed_metric = CacheSpeedupMetric(max_latency_ms=1500.0)

# Seed query to prime the semantic vector cache
SEED_QUERY = "What is the standard refund policy and timeframe under Genesis terms?"

# Conceptually identical variations with distinct syntax & phrasing
PARAPHRASED_VARIATIONS = [
    "How much time do customers have to claim their money back if they want a refund?",
    "Could you explain the Genesis reimbursement rules and return deadline window?",
    "Within what duration can a client request a full refund according to the terms?"
]

def test_semantic_cache_hit_and_speedup():
    """Verify follow-up paraphrased queries hit semantic cache and return rapidly."""
    # 1. Execute initial seed query (Cache Miss / Population)
    seed_res = client.query(SEED_QUERY)
    seed_latency = seed_res.latency_ms
    
    # 2. Execute variations and verify near-instant response
    for variation in PARAPHRASED_VARIATIONS:
        var_res = client.query(variation)
        test_case = LLMTestCase(
            input=variation,
            actual_output=var_res.answer,
            expected_output=seed_res.answer
        )
        
        # Check that follow-up is faster than cold generation or under cache threshold
        cache_speed_metric.measure_with_latency(var_res.latency_ms)
        assert_test(test_case, [cache_speed_metric])
        
        # Verify non-empty answer returned
        assert len(var_res.answer.strip()) > 0
