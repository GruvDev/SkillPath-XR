"""
The SkillPath-XR pipeline.

Wires every stage into one object:

    resume text
      -> parse -> extract skills -> normalise to taxonomy
      -> hybrid retrieval (graph + dense, RRF-fused)
      -> graded scoring over evidence paths
      -> explanation generation constrained to the evidence subgraph
      -> faithfulness validation
      -> counterfactual skill-gap analysis
      -> trust scoring

Construct it once and call `recommend()` per request: building the graph and
the vector index is the expensive part, and neither depends on the query.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import SETTINGS, Settings
from .embeddings.providers import BaseEmbedder, get_embedder
from .explain.counterfactual import CounterfactualEngine
from .explain.generator import (
    BaseExplanationGenerator,
    GroundedTemplateGenerator,
    get_generator,
)
from .explain.validator import FaithfulnessValidator
from .extraction.skills import SkillExtractor
from .kg.graph import SkillKnowledgeGraph
from .ontology.esco import Taxonomy, load_taxonomy
from .parsing.document import parse_document
from .ranking.scorer import explain_score_arithmetic, score_match
from .retrieval.hybrid import DenseRetriever, GraphRetriever, HybridRetriever
from .retrieval.vector_index import VectorIndex
from .scoring.trust import compute_trust, trust_band
from .types import (
    CandidateProfile,
    Explanation,
    JobMatch,
    JobPosting,
    RecommendationResult,
    SkillGap,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_jobs(path: Optional[str] = None) -> List[JobPosting]:
    """Load job postings from JSON."""
    path = path or os.path.join(DATA_DIR, "jobs.json")
    with open(path, "r", encoding="utf-8") as f:
        return [JobPosting(**row) for row in json.load(f)]


def load_sample_resumes(path: Optional[str] = None) -> List[Dict[str, str]]:
    path = path or os.path.join(DATA_DIR, "resumes.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SkillPathPipeline:
    """End-to-end explainable job recommendation."""

    def __init__(
        self,
        taxonomy: Optional[Taxonomy] = None,
        jobs: Optional[Sequence[JobPosting]] = None,
        settings: Optional[Settings] = None,
        embedder: Optional[BaseEmbedder] = None,
        generator: Optional[BaseExplanationGenerator] = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.taxonomy = taxonomy or load_taxonomy()
        self.jobs: List[JobPosting] = list(jobs) if jobs is not None else load_jobs()
        self.jobs_by_id: Dict[str, JobPosting] = {j.job_id: j for j in self.jobs}

        self.extractor = SkillExtractor(self.taxonomy)
        self.kg = SkillKnowledgeGraph(self.taxonomy, self.settings.match_weights)
        self.kg.add_jobs(self.jobs)

        self.embedder = embedder or get_embedder(self.settings)
        self.index = VectorIndex(self.embedder.dim)
        self._index_jobs()

        self.graph_retriever = GraphRetriever(self.kg, self.taxonomy)
        self.dense_retriever = DenseRetriever(self.index)
        self.retriever = HybridRetriever(
            self.graph_retriever, self.dense_retriever, self.settings.rrf_k
        )

        self.generator = generator or get_generator(self.settings)
        self.fallback_generator = GroundedTemplateGenerator()
        self.validator = FaithfulnessValidator(self.extractor)
        self.counterfactual_engine = CounterfactualEngine(
            self.kg,
            self.settings.qualification_threshold,
            self.settings.max_counterfactual_skills,
            self.settings.score_weights,
        )

    # -- indexing ------------------------------------------------------------

    def _job_text(self, job: JobPosting) -> str:
        """
        Text used to embed a job.

        Includes the *labels* of required skills, not just the prose: a posting
        may name a skill only in a structured field, and embedding the prose
        alone would make the dense retriever blind to it.
        """
        skills = [self.taxonomy.label(u) for u in job.essential_skills]
        skills += [self.taxonomy.label(u) for u in job.optional_skills]
        return " . ".join(
            [job.title, job.seniority, job.description, " ; ".join(skills)]
        )

    def _index_jobs(self) -> None:
        if not self.jobs:
            return
        texts = [self._job_text(j) for j in self.jobs]
        vectors = self.embedder.encode(texts)
        self.index.add([j.job_id for j in self.jobs], vectors)

    def _candidate_text(self, profile: CandidateProfile) -> str:
        skills = [self.taxonomy.label(u) for u in profile.skill_uris]
        return " . ".join(
            [
                profile.document.title,
                profile.document.seniority,
                profile.document.raw_text,
                " ; ".join(skills),
            ]
        )

    # -- main entry point ----------------------------------------------------

    def build_profile(
        self, resume_text: str, candidate_id: str = "C000", name: str = ""
    ) -> CandidateProfile:
        """Parse a résumé and link its skills to the taxonomy."""
        document = parse_document(resume_text, candidate_id, "resume")
        spans = self.extractor.extract(resume_text, document)
        return CandidateProfile(
            candidate_id=candidate_id, document=document, spans=spans, name=name
        )

    def recommend(
        self,
        resume_text: str,
        candidate_id: str = "C000",
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        explain: bool = True,
        name: str = "",
    ) -> RecommendationResult:
        """Produce ranked, explained job recommendations for a résumé."""
        timings: Dict[str, float] = {}
        top_k = top_k or self.settings.top_k

        t0 = time.perf_counter()
        profile = self.build_profile(resume_text, candidate_id, name)
        held = profile.skill_uris
        timings["parse_extract_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        query_vector = self.embedder.encode_one(self._candidate_text(profile))
        retrieved = self.retriever.retrieve(
            held, query_vector, top_k=self.settings.retrieval_candidates, mode=mode
        )
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        matches: List[JobMatch] = []
        for job_id, _fused, sources in retrieved:
            job = self.jobs_by_id.get(job_id)
            if job is None:
                continue
            matches.append(
                self._build_match(job, profile, held, query_vector, sources, explain=False)
            )
        matches.sort(key=lambda m: (-m.score, m.job.job_id))
        matches = matches[:top_k]
        timings["scoring_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if explain:
            t0 = time.perf_counter()
            for match in matches:
                self._attach_explanation(match, held, profile.document.seniority)
            timings["explanation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return RecommendationResult(
            candidate=profile,
            matches=matches,
            timings_ms=timings,
            providers={
                **self.settings.describe_providers(),
                "embedder": self.embedder.name,
                "generator": self.generator.name,
            },
        )

    # -- per-match assembly --------------------------------------------------

    def _build_match(
        self,
        job: JobPosting,
        profile: CandidateProfile,
        held: Sequence[str],
        query_vector: np.ndarray,
        sources: Sequence[str],
        explain: bool = True,
    ) -> JobMatch:
        evidence, uncovered = self.kg.evidence_paths(held, job)
        semantic = self.index.score_for(job.job_id, query_vector)
        breakdown = score_match(
            evidence,
            job,
            semantic_similarity=semantic,
            candidate_seniority=profile.document.seniority,
            weights=self.settings.score_weights,
        )

        gaps = [
            SkillGap(
                skill_uri=uri,
                skill_label=label,
                requirement_type=req_type,
                best_relation="none",
                best_strength=strength,
                nearest_held_skill=nearest,
            )
            for uri, label, req_type, strength, nearest in uncovered
        ]

        match = JobMatch(
            job=job,
            score=breakdown.final_score,
            breakdown=breakdown,
            evidence=evidence,
            gaps=gaps,
            qualified=breakdown.final_score >= self.settings.qualification_threshold,
            retrieval_sources=list(sources),
        )
        if explain:
            self._attach_explanation(match, held, profile.document.seniority)
        return match

    def _attach_explanation(
        self, match: JobMatch, held: Sequence[str], candidate_seniority: str
    ) -> None:
        """
        Generate, validate and attach an explanation, plus its counterfactual.

        If the configured generator fails or its output does not clear the
        grounding threshold, the deterministic template output is used instead
        and the substitution is recorded.  The user is never shown an
        unvalidated explanation, and never shown nothing.
        """
        allowed_extra = [g.skill_uri for g in match.gaps]

        text, provider = self._generate_text(match)
        validation = self.validator.validate(
            text,
            match.evidence,
            allowed_extra_uris=allowed_extra,
            min_grounding_rate=self.settings.min_grounding_rate,
        )

        withheld, reason = False, ""
        if not validation.passed and provider != self.fallback_generator.name:
            logger.warning(
                "explanation for %s failed grounding (%.2f); using template",
                match.job.job_id,
                validation.grounding_rate,
            )
            text = self.fallback_generator.generate(
                match.job, match.evidence, match.gaps, match.score
            )
            provider = f"{self.fallback_generator.name} (fallback)"
            validation = self.validator.validate(
                text,
                match.evidence,
                allowed_extra_uris=allowed_extra,
                min_grounding_rate=self.settings.min_grounding_rate,
            )
            reason = "generated explanation failed the grounding check"
        elif not validation.passed:
            withheld = True
            reason = "explanation did not meet the minimum grounding rate"

        sentences = self.validator.annotate_sentences(
            text, match.evidence, allowed_extra_uris=allowed_extra
        )

        match.explanation = Explanation(
            text=text,
            sentences=sentences,
            evidence=list(match.evidence),
            validation=validation,
            provider=provider,
            withheld=withheld,
            withheld_reason=reason,
        )

        match.counterfactual = self.counterfactual_engine.compute(
            match.job,
            held,
            semantic_similarity=match.breakdown.semantic_similarity,
            candidate_seniority=candidate_seniority,
        )

        match.trust = compute_trust(
            match.score,
            match.evidence,
            match.job,
            validation,
            self.settings.qualification_threshold,
            self.settings.trust_weights,
        )

    def _generate_text(self, match: JobMatch) -> tuple:
        try:
            text = self.generator.generate(
                match.job, match.evidence, match.gaps, match.score
            )
            if text and text.strip():
                return text, self.generator.name
            logger.warning("generator returned empty text for %s", match.job.job_id)
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the request
            logger.warning("generator failed for %s (%s)", match.job.job_id, exc)
        return (
            self.fallback_generator.generate(
                match.job, match.evidence, match.gaps, match.score
            ),
            self.fallback_generator.name,
        )

    # -- interactive simulation ----------------------------------------------

    def simulate(
        self,
        resume_text: str,
        job_id: str,
        added_skill_uris: Sequence[str],
        candidate_id: str = "C000",
    ) -> Dict[str, object]:
        """
        Recompute a single match with extra skills added.

        This backs the interface's counterfactual control: a user toggles a
        missing skill and sees the score, the evidence and the gaps update.
        It runs the real scorer over the real graph, so what the user sees is
        the system's actual behaviour, not an animation of it.
        """
        job = self.jobs_by_id.get(job_id)
        if job is None:
            raise KeyError(f"unknown job_id: {job_id}")

        profile = self.build_profile(resume_text, candidate_id)
        base_held = profile.skill_uris
        valid_added = [u for u in added_skill_uris if u in self.taxonomy]
        augmented = list(dict.fromkeys(list(base_held) + valid_added))

        query_vector = self.embedder.encode_one(self._candidate_text(profile))
        semantic = self.index.score_for(job_id, query_vector)

        def build(skills: Sequence[str]) -> Dict[str, object]:
            evidence, uncovered = self.kg.evidence_paths(skills, job)
            breakdown = score_match(
                evidence,
                job,
                semantic_similarity=semantic,
                candidate_seniority=profile.document.seniority,
                weights=self.settings.score_weights,
            )
            return {
                "score": breakdown.final_score,
                "qualified": breakdown.final_score >= self.settings.qualification_threshold,
                "breakdown": breakdown.to_dict(),
                "evidence": [e.to_dict() for e in evidence],
                "gaps": [
                    {
                        "skill_uri": uri,
                        "skill_label": label,
                        "requirement_type": req_type,
                        "nearest_held_skill": nearest,
                    }
                    for uri, label, req_type, _strength, nearest in uncovered
                ],
                "subgraph": self.kg.subgraph_for_evidence(evidence),
            }

        before = build(base_held)
        after = build(augmented)

        return {
            "job_id": job_id,
            "job_title": job.title,
            "threshold": self.settings.qualification_threshold,
            "added_skills": [
                {"uri": u, "label": self.taxonomy.label(u)} for u in valid_added
            ],
            "before": before,
            "after": after,
            "delta": round(after["score"] - before["score"], 4),  # type: ignore[operator]
            "crossed_threshold": bool(after["qualified"]) and not bool(before["qualified"]),
        }

    # -- presentation helpers ------------------------------------------------

    def match_detail(self, match: JobMatch) -> Dict[str, object]:
        """Full payload for one match, including everything the UI renders."""
        return {
            **match.to_dict(),
            "subgraph": self.kg.subgraph_for_evidence(match.evidence),
            "score_arithmetic": explain_score_arithmetic(
                match.breakdown, match.job, self.settings.score_weights
            ),
            "trust_band": trust_band(match.trust.overall) if match.trust else "low",
        }

    def stats(self) -> Dict[str, object]:
        return {
            "taxonomy_skills": len(self.taxonomy),
            "occupations": len(self.taxonomy.occupations),
            "jobs": len(self.jobs),
            "graph": self.kg.stats(),
            "embedding_dim": self.embedder.dim,
            "providers": {
                **self.settings.describe_providers(),
                "embedder": self.embedder.name,
                "generator": self.generator.name,
            },
            "thresholds": {
                "qualification": self.settings.qualification_threshold,
                "min_grounding_rate": self.settings.min_grounding_rate,
            },
        }
