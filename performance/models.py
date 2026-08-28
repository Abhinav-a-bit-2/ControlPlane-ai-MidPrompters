import re
from dataclasses import dataclass, field
from typing import List

_CITATION_PATTERN = re.compile(r"\[chunk-(\d+)\]")

@dataclass
class Claim:
    text: str
    cited_chunk_ids: List[str] = field(default_factory=list)

@dataclass
class ClaimEvaluation:
    text: str
    status: str  # "entailment" | "neutral" | "contradiction" | "uncited"
    entailment_score: float
    risk_score: float
    is_flagged: bool
    cited_chunks: List[str] = field(default_factory=list)

@dataclass
class EvaluationReport:
    overall_confidence: str  # "high" | "medium" | "low" | "unknown"
    risk_score: float = 0.0
    semantic_entropy: float = 0.0
    flagged_claims: List[ClaimEvaluation] = field(default_factory=list)
    all_claims: List[ClaimEvaluation] = field(default_factory=list)

def extract_claims(answer: str) -> List[Claim]:
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    claims = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        chunk_ids = [f"chunk-{n}" for n in _CITATION_PATTERN.findall(sentence)]
        clean_text = _CITATION_PATTERN.sub("", sentence).strip()
        if clean_text:
            claims.append(Claim(text=clean_text, cited_chunk_ids=chunk_ids))
    return claims