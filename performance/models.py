"""
Claim grounding via direct similarity to retrieved chunks.

Design change from the old pipeline:
  OLD: split answer into sentences -> check if a "[chunk-N]" tag is glued to
       that exact sentence -> UNCITED if not.
       Problem: a citation tag placed after a bullet list (or at the end of
       a paragraph) only attaches to the *last* sentence. Every earlier
       bullet/sentence in that block gets flagged UNCITED even though it's
       part of the same cited claim.

  NEW: two independent things are computed per claim:
    1. Attribution: which chunk IDs the claim is *associated with*. If the
       model tagged citations, an untagged sentence inherits the citation
       of the next tagged sentence in the same block (bullets before a
       trailing "[chunk-2]" all count as citing chunk-2). This removes the
       false UNCITED alarms without requiring the LLM to tag every line.
    2. Grounding: regardless of what got tagged, we directly measure how
       similar the claim text is to the *content* of the retrieved chunks
       (all of them, not just the tagged one) using embedding/TF-IDF
       cosine similarity. This is what actually answers "is this claim
       supported by evidence in the corpus" and lets you catch:
         - claims that cite a chunk but aren't actually supported by it
           (citation present, similarity low -> contradiction/neutral)
         - claims that are well supported by a chunk the model forgot to
           tag (no citation, similarity high -> should not be flagged as
           harshly as a truly ungrounded claim)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .grounding import GroundingChecker

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

_CITATION_PATTERN = re.compile(r"[\[【]chunk-(\d+)[\]】]")

# Tune these once against a labeled sample of your own answers.
ENTAILMENT_THRESHOLD = 0.55   # similarity >= this -> entailment
NEUTRAL_THRESHOLD = 0.30      # similarity >= this (but < entailment) -> neutral
# below NEUTRAL_THRESHOLD with no supporting chunk -> uncited/unsupported


@dataclass
class Chunk:
    chunk_id: str          # e.g. "chunk-2"
    text: str


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
    best_matching_chunk: Optional[str] = None


@dataclass
class EvaluationReport:
    overall_confidence: str  # "high" | "medium" | "low" | "unknown"
    risk_score: float = 0.0
    semantic_entropy: float = 0.0
    flagged_claims: List[ClaimEvaluation] = field(default_factory=list)
    all_claims: List[ClaimEvaluation] = field(default_factory=list)
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Step 1: extract claims, but group by "citation block" not raw sentence.
# ---------------------------------------------------------------------------

def extract_claims(answer: str) -> List[Claim]:
    """
    Split into sentence/line-level units, but propagate a trailing citation
    backward over any preceding uncited units (bullets, list items, etc.)
    until the previous citation or the start of the text. This fixes the
    "bullets before a single trailing [chunk-N]" false-negative case.
    """
    # Split on sentence boundaries AND newlines, so bullet points that don't
    # end in punctuation are still separated. Negative lookahead prevents
    # splitting between a sentence and a citation tag that trails it
    # (e.g. "overview. [chunk-2]" must stay one unit, not split into
    # "overview." + "[chunk-2]" which would silently drop the citation).
    raw_units = re.split(r'(?<=[.!?])\s+(?!\[chunk-\d+\])|\n+', answer.strip())
    raw_units = [u for u in raw_units if u.strip()]

    # First pass: pull out per-unit citations and clean text.
    parsed = []
    for unit in raw_units:
        chunk_ids = [f"chunk-{n}" for n in _CITATION_PATTERN.findall(unit)]
        clean = _CITATION_PATTERN.sub("", unit).strip()
        clean = re.sub(r'^[\*\-\u2022]\s*', '', clean)  # strip bullet markers
        if clean:
            parsed.append((clean, chunk_ids))

    # Second pass: backfill citations from the next tagged unit onto any
    # run of untagged units that precede it (same block).
    claims: List[Claim] = [Claim(text=t, cited_chunk_ids=list(c)) for t, c in parsed]
    pending: List[int] = []
    for i, claim in enumerate(claims):
        if claim.cited_chunk_ids:
            for j in pending:
                claims[j].cited_chunk_ids = list(claim.cited_chunk_ids)
            pending = []
        else:
            pending.append(i)
    # trailing uncited units with nothing after them stay uncited -- correct,
    # there's no citation anywhere to inherit.

    return claims


# ---------------------------------------------------------------------------
# Step 2: ground each claim against chunk *content*, not just its tag.
# ---------------------------------------------------------------------------

# def _similarity_matrix(claim_texts: List[str], chunk_texts: List[str]):
#     if not claim_texts or not chunk_texts:
#         return None
#     vectorizer = TfidfVectorizer(stop_words="english")
#     all_texts = claim_texts + chunk_texts
#     tfidf = vectorizer.fit_transform(all_texts)
#     claim_vecs = tfidf[: len(claim_texts)]
#     chunk_vecs = tfidf[len(claim_texts):]
#     return cosine_similarity(claim_vecs, chunk_vecs)


# def evaluate_claims(
#     claims: List[Claim],
#     chunks: List[Chunk],
#     entailment_threshold: float = ENTAILMENT_THRESHOLD,
#     neutral_threshold: float = NEUTRAL_THRESHOLD,
# ) -> List[ClaimEvaluation]:
#     chunk_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}
#     claim_texts = [c.text for c in claims]
#     chunk_texts = [c.text for c in chunks]

#     sims = _similarity_matrix(claim_texts, chunk_texts)  # [n_claims, n_chunks] or None

#     evaluations: List[ClaimEvaluation] = []
#     for i, claim in enumerate(claims):
#         if sims is None:
#             best_score = 0.0
#             best_chunk_id = None
#         else:
#             row = sims[i]
#             best_idx = row.argmax()
#             best_score = float(row[best_idx])
#             best_chunk_id = chunks[best_idx].chunk_id if chunks else None

#         # Similarity specifically to the chunk(s) the claim cited, if any.
#         cited_score = 0.0
#         if claim.cited_chunk_ids and sims is not None:
#             for cid in claim.cited_chunk_ids:
#                 if cid in chunk_by_id:
#                     idx = [c.chunk_id for c in chunks].index(cid)
#                     cited_score = max(cited_score, float(sims[i][idx]))

#         # Grounding score = best evidence we can find anywhere, cited or not.
#         grounding_score = max(best_score, cited_score)

#         if grounding_score >= entailment_threshold:
#             status = "entailment"
#             risk = 0.0
#         elif grounding_score >= neutral_threshold:
#             status = "neutral"
#             risk = 0.4
#         elif claim.cited_chunk_ids:
#             # Tagged a chunk, but content doesn't actually support it.
#             status = "contradiction"
#             risk = 0.9
#         else:
#             status = "uncited"
#             risk = 1.0

#         is_flagged = status in ("contradiction", "uncited")

#         evaluations.append(
#             ClaimEvaluation(
#                 text=claim.text,
#                 status=status,
#                 entailment_score=round(grounding_score, 3),
#                 risk_score=risk,
#                 is_flagged=is_flagged,
#                 cited_chunks=claim.cited_chunk_ids,
#                 best_matching_chunk=best_chunk_id,
#             )
#         )
#     return evaluations

def evaluate_claims(claims: List[Claim],chunks: List[Chunk], checker: GroundingChecker)->List[ClaimEvaluation]:
    evals = []
    chunks_map = {chunk.chunk_id:chunk.text for chunk in chunks}
    chunks_comb = [c.text for c in chunks]
    for claim in claims:
        cited_text = ([chunks_map[chunk_id] for chunk_id in claim.cited_chunk_ids if chunk_id in chunks_map])

        if cited_text:
            check_text = " ".join(cited_text)
            ground_res = checker.checkClaim(output=claim.text, chunk=check_text)
        elif chunks_comb:
            candidate_results = [checker.checkClaim(output=claim.text, chunk=text) for text in chunks_comb]
            ground_res = max(candidate_results, key=lambda r: r["entailment_score"])
        else:
            ground_res = {
                "label": "uncited",
                "entailment_score": 0.0,
                "neutral_score": 0.0,
                "contradiction_score": 1.0,
                "entropy": 0.0
            }
        
        status = ground_res["label"]
        p_e = ground_res["entailment_score"]
        p_n = ground_res["neutral_score"]
        p_c = ground_res["contradiction_score"]
        risk = 0.0*p_e + 0.4 * p_n + 1.0*p_c

        evals.append(
            ClaimEvaluation(
                text=claim.text,
                status=status,
                entailment_score=round(p_e),
                risk_score= risk,
                is_flagged= status in ("contradiction", "uncited"),
                cited_chunks=claim.cited_chunk_ids,
            )
        )
    return evals


def build_report(evaluations: List[ClaimEvaluation]) -> EvaluationReport:
    if not evaluations:
        return EvaluationReport(overall_confidence="unknown")

    avg_risk = sum(e.risk_score for e in evaluations) / len(evaluations)
    flagged = [e for e in evaluations if e.is_flagged]

    if avg_risk < 0.2:
        confidence = "high"
    elif avg_risk < 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    return EvaluationReport(
        overall_confidence=confidence,
        risk_score=round(avg_risk, 3),
        flagged_claims=flagged,
        all_claims=evaluations,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    answer = (
        "The Pragmatic Playbook's core principles are to\n"
        "* Map customer SLOs — latency, availability, and correctness — to the appropriate fidelity tier.\n"
        "* Define the required telemetry for each tier.\n"
        "* Specify the operational checks that must be performed.\n"
        "* Provide a step-by-step checklist for both commercial and technical leads.\n"
        "These elements are described in the playbook's overview. [chunk-2]"
    )

    chunks = [
        Chunk(
            chunk_id="chunk-2",
            text=(
                "Overview: The Pragmatic Playbook maps customer SLOs (latency, "
                "availability, correctness) to fidelity tiers, defines required "
                "telemetry per tier, lists operational checks, and gives a "
                "step-by-step checklist for commercial and technical leads."
            ),
        )
    ]

    claims = extract_claims(answer)
    evals = evaluate_claims(claims, chunks)
    report = build_report(evals)

    print(f"Confidence: {report.overall_confidence}  Risk: {report.risk_score}")
    for e in evals:
        flag = "FLAGGED" if e.is_flagged else "ok"
        print(f"[{e.status:12s} {flag:8s} score={e.entailment_score}] {e.text[:70]}")
