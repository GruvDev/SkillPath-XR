"""
Evaluation metrics.

Ranking metrics follow their standard definitions.  The explanation metrics are
the part specific to this work: grounding rate, counterfactual validity and
counterfactual minimality, which measure the properties the system claims to
provide rather than merely the quality of its ranking.

All functions take plain Python sequences so they can be reused on any
system's output, including baselines, and are written to be read against the
definitions in the report.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def _relevance_vector(ranked: Sequence[str], relevant: Set[str], k: int) -> List[int]:
    return [1 if item in relevant else 0 for item in ranked[:k]]


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    hits = sum(_relevance_vector(ranked, relevant, k))
    return hits / min(k, len(ranked))


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(_relevance_vector(ranked, relevant, k))
    return hits / len(relevant)


def f1_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def hit_rate_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """1.0 if at least one relevant item appears in the top k."""
    return 1.0 if any(_relevance_vector(ranked, relevant, k)) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant item, 0 if none."""
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Set[str]) -> float:
    """Mean of precision values taken at each relevant hit."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the standard log2(rank + 1) discount."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(list(gains)[:k]))


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Normalised DCG at k.

    Uses binary relevance.  The ideal ranking places every relevant item first,
    capped at k, so a query with fewer relevant items than k can still reach
    1.0 rather than being unfairly penalised.
    """
    if not relevant:
        return 0.0
    gains = _relevance_vector(ranked, relevant, k)
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    ranked: Sequence[str], relevant: Set[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> Dict[str, float]:
    """All ranking metrics for one query."""
    results: Dict[str, float] = {
        "MRR": reciprocal_rank(ranked, relevant),
        "MAP": average_precision(ranked, relevant),
    }
    for k in ks:
        results[f"P@{k}"] = precision_at_k(ranked, relevant, k)
        results[f"R@{k}"] = recall_at_k(ranked, relevant, k)
        results[f"F1@{k}"] = f1_at_k(ranked, relevant, k)
        results[f"HR@{k}"] = hit_rate_at_k(ranked, relevant, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
    return results


def aggregate(per_query: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across queries."""
    if not per_query:
        return {}
    keys = sorted({k for row in per_query for k in row})
    return {
        k: round(sum(row.get(k, 0.0) for row in per_query) / len(per_query), 4)
        for k in keys
    }


# ---------------------------------------------------------------------------
# Explanation quality — the metrics specific to this work
# ---------------------------------------------------------------------------


def grounding_rate(validations: Sequence[object]) -> float:
    """Mean sentence-level grounding rate across explanations."""
    rates = [
        float(getattr(v, "grounding_rate", 0.0)) for v in validations if v is not None
    ]
    return round(sum(rates) / len(rates), 4) if rates else 0.0


def explanation_pass_rate(validations: Sequence[object]) -> float:
    """Share of explanations clearing the minimum grounding threshold."""
    flags = [bool(getattr(v, "passed", False)) for v in validations if v is not None]
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def evidence_coverage(matches: Sequence[object]) -> float:
    """
    Mean share of each job's requirements that carry supporting evidence.

    Distinguishes a high score built on broad support from one resting on a
    couple of lucky matches.
    """
    shares: List[float] = []
    for match in matches:
        job = getattr(match, "job", None)
        evidence = getattr(match, "evidence", [])
        if job is None:
            continue
        total = len(job.essential_skills) + len(job.optional_skills)
        if total:
            shares.append(len(evidence) / total)
    return round(sum(shares) / len(shares), 4) if shares else 0.0


def counterfactual_validity(results: Sequence[bool]) -> float:
    """
    Share of counterfactuals that verifiably do what they claim.

    Computed by re-running the scorer with the suggested skills added, so this
    is a measured property rather than an assertion.
    """
    flags = list(results)
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def counterfactual_minimality(counterfactuals: Sequence[object]) -> float:
    """Share of achieved counterfactuals that are locally minimal."""
    achieved = [c for c in counterfactuals if getattr(c, "achieves_threshold", False)]
    if not achieved:
        return 0.0
    minimal = [c for c in achieved if getattr(c, "is_minimal", False)]
    return round(len(minimal) / len(achieved), 4)


def mean_counterfactual_size(counterfactuals: Sequence[object]) -> float:
    """Average number of skills suggested, over counterfactuals that achieved."""
    sizes = [
        len(getattr(c, "skills", []))
        for c in counterfactuals
        if getattr(c, "achieves_threshold", False) and getattr(c, "skills", [])
    ]
    return round(sum(sizes) / len(sizes), 4) if sizes else 0.0
