from intent_classifier import IntentClassifier

c = IntentClassifier()

test_queries = [
    "What is the refund policy?",
    "Explain how the onboarding process works",
    "What's the difference between Plan A and Plan B?",
    "My payment was charged twice, can you help?",
    "What are the contract terms for enterprise customers?",
    "Is my personal data stored securely?",
    "How do I configure the API endpoint?",
    "What is the weather today?",
    "Can you compare the SLA guarantees across tiers?",
    "I need help resetting my password",
    "What confidential information is stored in the system?",
    "Why is my invoice showing a different amount?",
]

print(f"{'Query':<55} {'Label':<15} {'Conf':>6}  {'Path':>10}")
print("-" * 90)
for q in test_queries:
    r = c.classify(q)
    path = "CHEAP" if r.confidence > 0.50 else "EXPENSIVE"
    print(f"{q:<55} {r.short_label:<15} {r.confidence:>6.3f}  {path:>10}")
