"""
Match scoring.

Turns evidence paths into a single defensible number, keeping every component
visible in a `ScoreBreakdown` so any score can be taken apart and questioned.

Two design decisions worth stating, because both are places where a naive
implementation would be quietly wrong:

1. **Weight renormalisation.**  If a job lists no optional skills, the optional
   term is not scored as zero — that would punish the job for being concisely
   written.  Its weight is removed from both numerator and denominator so the
   remaining components still sum to one.

2. **Asymmetric seniority fit.**  Being under-qualified is penalised harder
   than being over-qualified, because for a job seeker a stretch role is a
   worse recommendation than a comfortable one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..config import SETTINGS, ScoreWeights
from ..parsing.document import SENIORITY_ORDER
from ..types import EvidencePath, JobPosting, ScoreBreakdown


def coverage(
    evidence: Sequence[EvidencePath],
    requirement_uris: Sequence[str],
    requirement_type: str,
) -> float:
    """
    Fraction of requirements covered, weighted by evidence strength.

    An exact match contributes 1.0, a specialisation 0.8, and so on, so a
    candidate who holds every requirement exactly scores 1.0 while one who
    holds only adjacent skills scores well below it.
    """
    if not requirement_uris:
        return 0.0
    by_uri = {
        e.required_skill_uri: e
        for e in evidence
        if e.requirement_type == requirement_type
    }
    total = sum(by_uri[u].strength for u in requirement_uris if u in by_uri)
    return total / len(requirement_uris)


def seniority_fit(candidate_seniority: str, job_seniority: str) -> float:
    """
    How well the candidate's level matches the job's, in [0, 1].

    Returns a neutral 0.6 when either side is unspecified: absence of a stated
    level is not evidence of a mismatch, and scoring it as zero would bury
    every job posting that omits a seniority label.
    """
    cand = SENIORITY_ORDER.get(candidate_seniority)
    job = SENIORITY_ORDER.get(job_seniority)
    if cand is None or job is None:
        return 0.6

    diff = job - cand
    if diff <= 0:
        fit = 1.0 - 0.12 * abs(diff)   # over-qualified: mild penalty
    else:
        fit = 1.0 - 0.28 * diff        # under-qualified: steeper penalty
    return max(0.0, min(1.0, fit))


def score_match(
    evidence: Sequence[EvidencePath],
    job: JobPosting,
    semantic_similarity: float = 0.0,
    candidate_seniority: str = "unspecified",
    weights: Optional[ScoreWeights] = None,
) -> ScoreBreakdown:
    """Combine all signals into a final score with a full breakdown."""
    w = weights or SETTINGS.score_weights

    ess_cov = coverage(evidence, job.essential_skills, "essential")
    opt_cov = coverage(evidence, job.optional_skills, "optional")
    sem = max(0.0, min(1.0, float(semantic_similarity)))
    sen = seniority_fit(candidate_seniority, job.seniority)

    # Only include components that are meaningful for this job, then
    # renormalise so the weights of the components actually used sum to 1.
    terms: List[Tuple[float, float]] = [(w.essential_coverage, ess_cov)]
    if job.optional_skills:
        terms.append((w.optional_coverage, opt_cov))
    terms.append((w.semantic_similarity, sem))
    terms.append((w.seniority_fit, sen))

    weight_sum = sum(weight for weight, _ in terms)
    final = sum(weight * value for weight, value in terms) / weight_sum if weight_sum else 0.0

    return ScoreBreakdown(
        essential_coverage=round(ess_cov, 4),
        optional_coverage=round(opt_cov, 4),
        semantic_similarity=round(sem, 4),
        seniority_fit=round(sen, 4),
        final_score=round(max(0.0, min(1.0, final)), 4),
    )


def score_with_skills(
    kg,
    held_skills: Sequence[str],
    job: JobPosting,
    semantic_similarity: float = 0.0,
    candidate_seniority: str = "unspecified",
    weights: Optional[ScoreWeights] = None,
) -> Tuple[float, ScoreBreakdown, List[EvidencePath]]:
    """
    Score a hypothetical skill set against a job.

    The counterfactual engine calls this repeatedly with modified skill sets to
    measure what acquiring a skill would actually be worth, so it recomputes
    evidence from scratch rather than trusting a cached score.
    """
    evidence, _uncovered = kg.evidence_paths(held_skills, job)
    breakdown = score_match(
        evidence, job, semantic_similarity, candidate_seniority, weights
    )
    return breakdown.final_score, breakdown, evidence


def explain_score_arithmetic(
    breakdown: ScoreBreakdown,
    job: JobPosting,
    weights: Optional[ScoreWeights] = None,
) -> List[Dict[str, object]]:
    """
    The score as a list of (component, weight, value, contribution) rows.

    Surfaced in the interface so a user can see precisely how the number was
    assembled rather than being asked to trust it.
    """
    w = weights or SETTINGS.score_weights
    rows: List[Tuple[str, float, float]] = [
        ("Essential skill coverage", w.essential_coverage, breakdown.essential_coverage)
    ]
    if job.optional_skills:
        rows.append(
            ("Optional skill coverage", w.optional_coverage, breakdown.optional_coverage)
        )
    rows.append(("Semantic similarity", w.semantic_similarity, breakdown.semantic_similarity))
    rows.append(("Seniority fit", w.seniority_fit, breakdown.seniority_fit))

    weight_sum = sum(weight for _, weight, _ in rows) or 1.0
    return [
        {
            "component": name,
            "weight": round(weight / weight_sum, 4),
            "value": round(value, 4),
            "contribution": round((weight / weight_sum) * value, 4),
        }
        for name, weight, value in rows
    ]
