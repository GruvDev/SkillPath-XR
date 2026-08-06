"""
Counterfactual skill-gap analysis.

Answers the question existing explainable recommenders do not: *what would have
to change for this to become a match?*

Given a candidate below the qualification threshold for a job, this finds a
small set of skills whose acquisition would lift them over it.

Algorithm
---------
Finding the truly minimum such set is a set-cover style problem and is not
tractable in general, so this uses the standard greedy construction followed by
a pruning pass:

1. **Greedy.**  Repeatedly add the missing skill with the largest marginal
   score gain, recomputing the score against the graph each time rather than
   assuming gains add up.  They do not: adding a parent skill can partially
   cover several requirements at once, so marginal value changes after every
   addition.
2. **Prune.**  Walk the selected set and drop any skill the set still clears
   the threshold without.  This removes members that were worth adding early
   but became redundant once later skills were included.

The result is *locally minimal*: no single member can be removed. That is a
weaker guarantee than global minimality, and the code says so rather than
overclaiming.

Every result is **verified** — the projected score is recomputed from the
modified skill set through the same scorer the recommendation used, so a
counterfactual that does not actually work cannot be reported as if it does.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..config import SETTINGS, ScoreWeights
from ..kg.graph import SkillKnowledgeGraph
from ..ranking.scorer import score_with_skills
from ..types import Counterfactual, JobPosting, SkillGap


class CounterfactualEngine:
    """Finds minimal skill sets that flip a near-miss into a match."""

    def __init__(
        self,
        kg: SkillKnowledgeGraph,
        threshold: Optional[float] = None,
        max_skills: Optional[int] = None,
        weights: Optional[ScoreWeights] = None,
    ) -> None:
        self.kg = kg
        self.threshold = threshold if threshold is not None else SETTINGS.qualification_threshold
        self.max_skills = (
            max_skills if max_skills is not None else SETTINGS.max_counterfactual_skills
        )
        self.weights = weights or SETTINGS.score_weights

    def _score(
        self,
        skills: Sequence[str],
        job: JobPosting,
        semantic: float,
        seniority: str,
    ) -> float:
        score, _breakdown, _evidence = score_with_skills(
            self.kg, skills, job, semantic, seniority, self.weights
        )
        return score

    def candidate_skills_to_add(self, job: JobPosting, held: Sequence[str]) -> List[str]:
        """
        Requirements not already held outright.

        Only the job's own requirements are considered: recommending a skill
        the employer never asked for would be noise, however useful it might be
        in general.
        """
        held_set = set(held)
        return [
            uri
            for uri in list(job.essential_skills) + list(job.optional_skills)
            if uri not in held_set and uri in self.kg.taxonomy
        ]

    def compute(
        self,
        job: JobPosting,
        held: Sequence[str],
        semantic_similarity: float = 0.0,
        candidate_seniority: str = "unspecified",
    ) -> Counterfactual:
        """Compute the minimal skill set that reaches the threshold."""
        held = list(held)
        baseline = self._score(held, job, semantic_similarity, candidate_seniority)

        if baseline >= self.threshold:
            return Counterfactual(
                target_threshold=self.threshold,
                baseline_score=round(baseline, 4),
                projected_score=round(baseline, 4),
                skills=[],
                achieves_threshold=True,
                is_minimal=True,
                note="Already at or above the qualification threshold.",
            )

        pool = self.candidate_skills_to_add(job, held)
        if not pool:
            return Counterfactual(
                target_threshold=self.threshold,
                baseline_score=round(baseline, 4),
                projected_score=round(baseline, 4),
                skills=[],
                achieves_threshold=False,
                is_minimal=True,
                note=(
                    "Every listed requirement is already evidenced; the shortfall is "
                    "in seniority or semantic fit rather than skills."
                ),
            )

        selected, current_score = self._greedy(
            pool, held, job, semantic_similarity, candidate_seniority
        )

        if current_score >= self.threshold:
            selected = self._prune(
                selected, held, job, semantic_similarity, candidate_seniority
            )
            current_score = self._score(
                held + selected, job, semantic_similarity, candidate_seniority
            )

        gaps = self._build_gap_records(
            selected, held, job, semantic_similarity, candidate_seniority, baseline
        )
        achieved = current_score >= self.threshold

        return Counterfactual(
            target_threshold=self.threshold,
            baseline_score=round(baseline, 4),
            projected_score=round(current_score, 4),
            skills=gaps,
            achieves_threshold=achieved,
            is_minimal=achieved,
            search_method="greedy+prune",
            note=(
                ""
                if achieved
                else (
                    f"Acquiring these {len(selected)} skills raises the match to "
                    f"{current_score:.2f}, short of the {self.threshold:.2f} threshold. "
                    f"The remaining shortfall is in seniority or semantic fit."
                )
            ),
        )

    # -- search phases -------------------------------------------------------

    def _greedy(
        self,
        pool: Sequence[str],
        held: Sequence[str],
        job: JobPosting,
        semantic: float,
        seniority: str,
    ) -> Tuple[List[str], float]:
        """Add the highest-marginal-gain skill until the threshold or the cap."""
        selected: List[str] = []
        current = self._score(list(held), job, semantic, seniority)
        remaining = list(pool)

        while remaining and len(selected) < self.max_skills and current < self.threshold:
            best_uri, best_score = None, current

            for uri in remaining:
                trial = self._score(
                    list(held) + selected + [uri], job, semantic, seniority
                )
                # Tie-break on label so runs are reproducible.
                if trial > best_score or (
                    best_uri is not None
                    and abs(trial - best_score) < 1e-12
                    and self.kg.taxonomy.label(uri) < self.kg.taxonomy.label(best_uri)
                ):
                    best_uri, best_score = uri, trial

            if best_uri is None or best_score <= current + 1e-12:
                break  # no remaining skill improves the score

            selected.append(best_uri)
            remaining.remove(best_uri)
            current = best_score

        return selected, current

    def _prune(
        self,
        selected: Sequence[str],
        held: Sequence[str],
        job: JobPosting,
        semantic: float,
        seniority: str,
    ) -> List[str]:
        """Drop any member the set still clears the threshold without."""
        kept = list(selected)
        for uri in list(kept):
            trial = [u for u in kept if u != uri]
            if self._score(list(held) + trial, job, semantic, seniority) >= self.threshold:
                kept = trial
        return kept

    def _build_gap_records(
        self,
        selected: Sequence[str],
        held: Sequence[str],
        job: JobPosting,
        semantic: float,
        seniority: str,
        baseline: float,
    ) -> List[SkillGap]:
        """
        Attach each selected skill's individual marginal gain.

        Gains are measured against the baseline one skill at a time, so the
        interface can show what each skill is worth on its own.  These do not
        sum to the total, and the docs say so explicitly.
        """
        records: List[SkillGap] = []
        for uri in selected:
            individual = self._score(list(held) + [uri], job, semantic, seniority)
            requirement = "essential" if uri in job.essential_skills else "optional"
            records.append(
                SkillGap(
                    skill_uri=uri,
                    skill_label=self.kg.taxonomy.label(uri),
                    requirement_type=requirement,
                    best_relation="none",
                    best_strength=0.0,
                    nearest_held_skill=self.kg._nearest_held(held, uri),
                    score_gain=round(max(0.0, individual - baseline), 4),
                )
            )
        # Highest-value skill first.
        records.sort(key=lambda g: (-g.score_gain, g.skill_label))
        return records

    def verify(
        self,
        counterfactual: Counterfactual,
        job: JobPosting,
        held: Sequence[str],
        semantic_similarity: float = 0.0,
        candidate_seniority: str = "unspecified",
    ) -> bool:
        """
        Independently confirm the counterfactual does what it claims.

        Recomputes the score with the suggested skills added and checks it
        matches the reported projection.  Used by the evaluation harness to
        measure counterfactual validity as a first-class metric.
        """
        if not counterfactual.skills:
            return counterfactual.achieves_threshold
        augmented = list(held) + [g.skill_uri for g in counterfactual.skills]
        actual = self._score(augmented, job, semantic_similarity, candidate_seniority)
        return abs(actual - counterfactual.projected_score) < 1e-6
