"""
Layer 1: Pre-Processing & Sanitization

Cheapest, fastest checks — run first so expensive layers (ML detection,
retrieval) never see input that should have been rejected outright.
"""
import re
import unicodedata
from dataclasses import dataclass

import tiktoken
from pydantic import BaseModel, Field, field_validator


# Zero-width / invisible chars used to hide instructions inside text
_ZERO_WIDTH = re.compile(
    "[\u200b\u200c\u200d\u2060\ufeff\u180e]"
)

# Bidirectional control characters (RTL/LTR override tricks)
_BIDI_CONTROLS = re.compile(
    "[\u202a-\u202e\u2066-\u2069]"
)

try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Falls back to a whitespace-based approximation if the tiktoken
    # vocab file can't be fetched (e.g. restricted network egress).
    # This under/over-counts slightly vs. true token count but still
    # enforces a meaningful upper bound - swap back to the real
    # tokenizer once network access is confirmed in your deployment env.
    _TOKENIZER = None


class SanitizationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class SanitizeResult:
    passed: bool
    cleaned_text: str = ""
    reason: str = ""
    token_count: int = 0


class QueryRequest(BaseModel):
    """Structured validation for API-facing input. Reject malformed
    requests before they ever reach the text sanitizer."""

    question: str = Field(..., min_length=1, max_length=4000)
    k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def not_just_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty/whitespace")
        return v


def normalize_unicode(text: str) -> str:
    """NFKC-normalize, then strip zero-width and bidi-control characters
    that can be used to hide or reorder malicious instructions."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _BIDI_CONTROLS.sub("", text)
    return text


def count_tokens(text: str) -> int:
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    return len(text.split())


def sanitize_input(raw_text: str, max_tokens: int = 512) -> SanitizeResult:
    """Orchestrates Layer 1. Order matters: validate structure first
    (cheap), then normalize, then check length on the *normalized* text
    (an attacker could pad with characters that vanish after normalization
    to sneak past a naive length check done before normalizing)."""

    try:
        validated = QueryRequest(question=raw_text)
    except Exception as e:
        return SanitizeResult(passed=False, reason=f"structural_validation_failed: {e}")

    cleaned = normalize_unicode(validated.question)

    n_tokens = count_tokens(cleaned)
    if n_tokens > max_tokens:
        return SanitizeResult(
            passed=False,
            reason=f"token_limit_exceeded: {n_tokens} > {max_tokens}",
            token_count=n_tokens,
        )

    return SanitizeResult(passed=True, cleaned_text=cleaned, token_count=n_tokens)
