"""
Trust scoring.

A match score says how good a fit looks.  It says nothing about how much the
system should be believed about it.  Those are different questions, and
collapsing them into one number is how recommenders end up confidently wrong.

Trust is composed of three independently meaningful signals:

* **Model confidence** — how decisive the match score is.  A score sitting on
  the qualification threshold is a coin flip and is reported as such.
* **Evidence density** — how much of the job's requirement set was actually
  reachable in the graph.  A high score built on two of eleven requirements
  deserves less confidence than the same score built on nine.
* **Faithfulness** — the measured grounding rate of the explanation.

Keeping them separate means a low trust score can be *diagnosed*: the interface
shows which of the three is dragging it down.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..config import SETTINGS, TrustWeights
from ..types import EvidencePath, JobPosting, TrustScore, ValidationResult


def model_confidence(score: float, threshold: float) -> float:
    """
    How far the score sits from the decision boundary, normalised to [0, 1].

    Distance is scaled by the larger side of the boundary so a score at either
    extreme reaches 1.0 while a score exactly on the threshold reaches 0.0.
    """
    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
    if score >= threshold:
        span = 1.0 - threshold
        return min(1.0, (score - threshold) / span) if span > 0 else 1.0
    return min(1.0, (threshold - score) / threshold)


def evidence_density(evidence: Sequence[EvidencePath], job: JobPosting) -> float:
    """
    Share of the job's requirements that have any supporting evidence.

    Weighted so essential requirements count double: evidence covering
    optional extras while leaving the essentials unaddressed should not read as
    dense support.
    """
    essential_total = len(job.essential_skills)
    optional_total = len(job.optional_skills)
    denominator = 2.0 * essential_total + optional_total
    if denominator == 0:
        return 0.0

    covered_essential = sum(1 for e in evidence if e.requirement_type == "essential")
    covered_optional = sum(1 for e in evidence if e.requirement_type == "optional")
    numerator = 2.0 * covered_essential + covered_optional
    return min(1.0, numerator / denominator)


def compute_trust(
    score: float,
    evidence: Sequence[EvidencePath],
    job: JobPosting,
    validation: Optional[ValidationResult] = None,
    threshold: Optional[float] = None,
    weights: Optional[TrustWeights] = None,
) -> TrustScore:
    """Combine the three signals into an overall trust score."""
    w = weights or SETTINGS.trust_weights
    thr = threshold if threshold is not None else SETTINGS.qualification_threshold

    confidence = model_confidence(score, thr)
    density = evidence_density(evidence, job)
    faithfulness = validation.grounding_rate if validation else 0.0

    overall = (
        w.model_confidence * confidence
        + w.evidence_density * density
        + w.faithfulness * faithfulness
    )
    total_weight = w.model_confidence + w.evidence_density + w.faithfulness
    overall = overall / total_weight if total_weight else 0.0

    return TrustScore(
        model_confidence=round(confidence, 4),
        evidence_density=round(density, 4),
        faithfulness=round(faithfulness, 4),
        overall=round(max(0.0, min(1.0, overall)), 4),
    )


def trust_band(overall: float) -> str:
    """Coarse label for the interface. Thresholds are presentational only."""
    if overall >= 0.75:
        return "high"
    if overall >= 0.5:
        return "moderate"
    return "low"
