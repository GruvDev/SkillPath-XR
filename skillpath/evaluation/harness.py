"""
Evaluation harness.

Runs the system and its ablations over a labelled set and prints a results
table.  This is what produces the numbers for the report.

Ground truth
------------
Relevance labels are derived from the occupation layer of the taxonomy rather
than hand-written per candidate: a job is relevant to a candidate if the
candidate evidences at least `min_essential_ratio` of that job's essential
requirements.  Deriving labels from the same taxonomy the system uses is a real
limitation and is stated plainly here and in the report — it measures internal
consistency and ranking behaviour, not agreement with human recruiters.  A
human-labelled set is the correct next step and is listed as such in the
documentation.

Baselines and ablations
-----------------------
* `lexical`  — TF-IDF-style token overlap, no taxonomy, no graph. The classical
  keyword baseline.
* `dense`    — semantic retrieval only (ablates the graph).
* `graph`    — structural retrieval only (ablates dense retrieval).
* `hybrid`   — the full system.

All four run through the same scorer, so differences are attributable to
retrieval rather than to incidental scoring differences.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..pipeline import SkillPathPipeline, load_sample_resumes
from ..types import JobPosting
from .metrics import (
    aggregate,
    counterfactual_minimality,
    counterfactual_validity,
    evaluate_ranking,
    evidence_coverage,
    explanation_pass_rate,
    grounding_rate,
    mean_counterfactual_size,
)

_TOKEN = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class LexicalBaseline:
    """
    TF-IDF cosine over raw tokens.

    Represents the pre-semantic generation of job matchers: no taxonomy, no
    graph, no notion that "PySpark" and "distributed data processing" are
    related.  Included so the gain from the taxonomy layer is measurable rather
    than assumed.
    """

    def __init__(self, jobs: Sequence[JobPosting], taxonomy) -> None:
        self.jobs = list(jobs)
        self.taxonomy = taxonomy
        self.doc_tokens: Dict[str, Counter] = {}
        df: Counter = Counter()

        for job in self.jobs:
            skills = [taxonomy.label(u) for u in job.essential_skills + job.optional_skills]
            blob = " ".join([job.title, job.description, " ".join(skills)])
            counts = Counter(_tokens(blob))
            self.doc_tokens[job.job_id] = counts
            df.update(counts.keys())

        n = max(1, len(self.jobs))
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _vector(self, counts: Counter) -> Dict[str, float]:
        return {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in counts.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def rank(self, resume_text: str, top_k: int = 10) -> List[str]:
        query = self._vector(Counter(_tokens(resume_text)))
        scored = [
            (jid, self._cosine(query, self._vector(counts)))
            for jid, counts in self.doc_tokens.items()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [jid for jid, _ in scored[:top_k]]


def build_ground_truth(
    pipeline: SkillPathPipeline,
    resumes: Sequence[Dict[str, str]],
    min_essential_ratio: float = 0.6,
) -> Dict[str, Set[str]]:
    """
    Derive relevance labels from essential-requirement coverage.

    A job counts as relevant to a candidate when the candidate evidences at
    least `min_essential_ratio` of its essential skills through *exact* matches
    only.  Exact-only is deliberate: using graded evidence here would define
    relevance in terms of the very mechanism under test.
    """
    truth: Dict[str, Set[str]] = {}

    for resume in resumes:
        profile = pipeline.build_profile(resume["text"], resume["candidate_id"])
        held = set(profile.skill_uris)
        relevant: Set[str] = set()

        for job in pipeline.jobs:
            essential = list(job.essential_skills)
            if not essential:
                continue
            covered = sum(1 for u in essential if u in held)
            if covered / len(essential) >= min_essential_ratio:
                relevant.add(job.job_id)

        truth[resume["candidate_id"]] = relevant

    return truth


def run_evaluation(
    pipeline: Optional[SkillPathPipeline] = None,
    resumes: Optional[Sequence[Dict[str, str]]] = None,
    ks: Sequence[int] = (1, 3, 5, 10),
    top_k: int = 10,
) -> Dict[str, object]:
    """Run every configuration and return the full results structure."""
    pipeline = pipeline or SkillPathPipeline()
    resumes = list(resumes if resumes is not None else load_sample_resumes())
    truth = build_ground_truth(pipeline, resumes)

    lexical = LexicalBaseline(pipeline.jobs, pipeline.taxonomy)
    configurations = ["lexical", "dense", "graph", "hybrid"]
    ranking_results: Dict[str, List[Dict[str, float]]] = {c: [] for c in configurations}

    explanation_validations: List[object] = []
    all_matches: List[object] = []
    cf_objects: List[object] = []
    cf_verified: List[bool] = []

    for resume in resumes:
        cid = resume["candidate_id"]
        relevant = truth.get(cid, set())
        if not relevant:
            continue  # a candidate with no relevant job cannot discriminate systems

        ranking_results["lexical"].append(
            evaluate_ranking(lexical.rank(resume["text"], top_k), relevant, ks)
        )

        for mode in ("dense", "graph", "hybrid"):
            result = pipeline.recommend(
                resume["text"], cid, top_k=top_k, mode=mode, explain=(mode == "hybrid")
            )
            ranked = [m.job.job_id for m in result.matches]
            ranking_results[mode].append(evaluate_ranking(ranked, relevant, ks))

            if mode == "hybrid":
                held = result.candidate.skill_uris
                seniority = result.candidate.document.seniority
                for match in result.matches:
                    all_matches.append(match)
                    if match.explanation and match.explanation.validation:
                        explanation_validations.append(match.explanation.validation)
                    if match.counterfactual:
                        cf_objects.append(match.counterfactual)
                        cf_verified.append(
                            pipeline.counterfactual_engine.verify(
                                match.counterfactual,
                                match.job,
                                held,
                                match.breakdown.semantic_similarity,
                                seniority,
                            )
                        )

    return {
        "n_candidates": len([r for r in resumes if truth.get(r["candidate_id"])]),
        "n_jobs": len(pipeline.jobs),
        "ranking": {c: aggregate(rows) for c, rows in ranking_results.items()},
        "explanation": {
            "mean_grounding_rate": grounding_rate(explanation_validations),
            "explanation_pass_rate": explanation_pass_rate(explanation_validations),
            "evidence_coverage": evidence_coverage(all_matches),
            "n_explanations": len(explanation_validations),
        },
        "counterfactual": {
            "validity": counterfactual_validity(cf_verified),
            "minimality": counterfactual_minimality(cf_objects),
            "mean_size": mean_counterfactual_size(cf_objects),
            "n_counterfactuals": len(cf_objects),
        },
        "providers": pipeline.stats()["providers"],
    }


def format_results(results: Dict[str, object]) -> str:
    """Render the results as fixed-width tables for the console and the report."""
    lines: List[str] = []
    ranking: Dict[str, Dict[str, float]] = results["ranking"]  # type: ignore[assignment]

    headline = ["NDCG@5", "NDCG@10", "MRR", "MAP", "P@5", "R@10", "HR@5"]
    lines.append("")
    lines.append("RANKING QUALITY (mean over candidates)")
    lines.append("-" * 78)
    lines.append(f"{'Configuration':<14}" + "".join(f"{h:>9}" for h in headline))
    lines.append("-" * 78)

    labels = {
        "lexical": "Lexical",
        "dense": "Dense only",
        "graph": "Graph only",
        "hybrid": "Hybrid (ours)",
    }
    for key in ("lexical", "dense", "graph", "hybrid"):
        row = ranking.get(key, {})
        cells = "".join(f"{row.get(h, 0.0):>9.4f}" for h in headline)
        lines.append(f"{labels[key]:<14}" + cells)
    lines.append("-" * 78)

    expl: Dict[str, float] = results["explanation"]  # type: ignore[assignment]
    lines.append("")
    lines.append("EXPLANATION QUALITY")
    lines.append("-" * 78)
    lines.append(f"{'Mean grounding rate':<34}{expl['mean_grounding_rate']:>10.4f}")
    lines.append(f"{'Explanation pass rate':<34}{expl['explanation_pass_rate']:>10.4f}")
    lines.append(f"{'Evidence coverage':<34}{expl['evidence_coverage']:>10.4f}")
    lines.append(f"{'Explanations evaluated':<34}{int(expl['n_explanations']):>10d}")

    cf: Dict[str, float] = results["counterfactual"]  # type: ignore[assignment]
    lines.append("")
    lines.append("COUNTERFACTUAL QUALITY")
    lines.append("-" * 78)
    lines.append(f"{'Validity (verified by re-scoring)':<34}{cf['validity']:>10.4f}")
    lines.append(f"{'Local minimality':<34}{cf['minimality']:>10.4f}")
    lines.append(f"{'Mean skills suggested':<34}{cf['mean_size']:>10.4f}")
    lines.append(f"{'Counterfactuals evaluated':<34}{int(cf['n_counterfactuals']):>10d}")
    lines.append("-" * 78)
    lines.append(f"Providers: {results['providers']}")
    lines.append("")
    lines.append(
        "Note: relevance labels are derived from taxonomy coverage, so these "
        "figures measure ranking behaviour and internal consistency, not "
        "agreement with human recruiters. See docs/05-evaluation.md."
    )
    return "\n".join(lines)


def main() -> None:
    results = run_evaluation()
    print(format_results(results))


if __name__ == "__main__":
    main()
