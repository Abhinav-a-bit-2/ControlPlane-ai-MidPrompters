"""
Intent Classifier: Zero-Shot classification using facebook/bart-large-mnli.

Classifies user queries into domain-specific intents and provides
confidence scores for routing decisions in the dual-path pipeline.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("intent_classifier")


# ---------------------------------------------------------------------------
# Intent-aware system prompt fragments
# ---------------------------------------------------------------------------
# Each intent maps to an addendum appended to the base SYSTEM_PROMPT_TEMPLATE.
# This lets the LLM tailor its tone, depth, and citation style per intent.

INTENT_PROMPT_ADDENDA = {
    "a customer support request for help": (
        "\nBe empathetic and conversational. "
        "Provide step-by-step guidance where applicable. "
        "If the user seems frustrated, acknowledge their concern first. "
        "Always cite your sources using chunk IDs (e.g., [chunk-12])."
    ),
    "a legal contract, organizational policy, or SLA": (
        "\nCite every clause explicitly using chunk IDs (e.g., [chunk-12]). "
        "Do not paraphrase policy language — quote it directly. "
        "If the policy text is ambiguous, state the ambiguity clearly."
    ),
    "technical troubleshooting and system configuration": (
        "\nBe precise and technical. "
        "Include any relevant configuration values, parameters, or procedures from the context. "
        "Use bullet points for multi-step instructions. "
        "Always cite your sources using chunk IDs (e.g., [chunk-12])."
    ),
    "billing, payments, pricing, and invoices": (
        "\nInclude specific amounts, dates, and plan names. "
        "Be clear about what is and isn't included in the pricing. "
        "Cite the source chunk for every monetary figure using chunk IDs (e.g., [chunk-12])."
    ),
    "confidential data and privacy": (
        "\nThis query may involve sensitive or restricted information. "
        "Answer only from the provided context. "
        "Do NOT speculate or infer beyond what is explicitly stated. "
        "If the context does not contain sufficient detail, say so clearly. "
        "Always cite your sources using chunk IDs (e.g., [chunk-12])."
    ),
    "an explanation of a general concept": (
        "\nThe user is looking for a conceptual explanation. "
        "Structure your answer clearly with definitions first, then details. "
        "Use simple language and explain jargon where it appears in the context. "
        "Always cite your sources using chunk IDs (e.g., [chunk-12])."
    ),
    "a comparison between multiple options": (
        "\nThe user wants to understand differences or similarities. "
        "Structure the answer as a clear comparison — use a side-by-side format if there are multiple attributes. "
        "Cite each compared item's source chunk explicitly (e.g., [chunk-12])."
    ),
    "a direct factual question or inquiry about events, entities, or history": (
        "\nThe user is looking for a direct factual answer. "
        "Keep your response concise and directly address the specific fact or event requested. "
        "Explicitly cite the source chunk for the fact using chunk IDs (e.g., [chunk-12])."
    ),
    "a topic completely unrelated to this system": (
        # No special addendum — use base prompt and let retrieval decide
        "\nAlways cite your sources using chunk IDs (e.g., [chunk-12])."
    ),
}

# Short aliases for display, telemetry, and retrieval-k lookup
LABEL_SHORT_NAMES = {
    "a customer support request for help": "support",
    "a legal contract, organizational policy, or SLA": "contract",
    "technical troubleshooting and system configuration": "technical",
    "billing, payments, pricing, and invoices": "billing",
    "confidential data and privacy": "confidential",
    "an explanation of a general concept": "explanation",
    "a comparison between multiple options": "comparison",
    "a direct factual question or inquiry about events, entities, or history": "factual",
    "a topic completely unrelated to this system": "out_of_scope",
}

# Candidate labels for zero-shot classification
CANDIDATE_LABELS = list(INTENT_PROMPT_ADDENDA.keys())

# Retrieval depth per intent
INTENT_RETRIEVAL_K = {
    "support": 2,
    "contract": 4,
    "technical": 3,
    "billing": 2,
    "confidential": 3,
    "explanation": 3,
    "comparison": 4,
    "factual": 3,
    "out_of_scope": 2,
}


@dataclass
class IntentResult:
    """Result of zero-shot intent classification."""
    label: str                          # full descriptive label (used for prompt addenda)
    short_label: str = ""               # short name for display/telemetry/retrieval-k
    confidence: float = 0.0             # confidence of top label (0.0–1.0)
    all_scores: dict = field(default_factory=dict)  # {label: score, ...}


class IntentClassifier:
    """
    Singleton-style zero-shot intent classifier backed by
    facebook/bart-large-mnli.  The model is loaded once at init and
    kept in memory for fast inference (~50-100ms per query on CPU).
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: -1 for CPU, 0+ for GPU index.
        """
        from transformers import pipeline as hf_pipeline
        logger.info("Loading zero-shot classifier: %s (device=%d)", model_name, device)
        self._classifier = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
        self.candidate_labels = CANDIDATE_LABELS
        logger.info("Intent classifier ready. Labels: %s", self.candidate_labels)

    def classify(self, query: str, candidate_labels: list[str] | None = None) -> IntentResult:
        """
        Classify a raw user query into one of the candidate intent labels.

        Args:
            query: The original (non-rewritten) user question.
            candidate_labels: Override default labels if needed.

        Returns:
            IntentResult with top label, confidence, and all scores.
        """
        labels = candidate_labels or self.candidate_labels
        result = self._classifier(query, labels)

        all_scores = dict(zip(result["labels"], result["scores"]))
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        short = LABEL_SHORT_NAMES.get(top_label, top_label)

        logger.info(
            "Intent classified: '%s' -> %s [%s] (%.3f)",
            query[:60], short, top_label, top_score,
        )

        return IntentResult(
            label=top_label,
            short_label=short,
            confidence=top_score,
            all_scores=all_scores,
        )
