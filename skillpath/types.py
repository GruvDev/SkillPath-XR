"""
Core data types for SkillPath-XR.

These dataclasses are the contract between modules.  Everything that crosses a
module boundary is one of these, and every one of them is JSON-serialisable so
the same object can go straight to the API without a translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillConcept:
    """A single normalised skill from the taxonomy (ESCO-shaped)."""

    uri: str
    label: str
    skill_type: str = "skill"          # "skill" | "knowledge"
    alt_labels: Tuple[str, ...] = ()
    broader: Tuple[str, ...] = ()      # parent concept URIs
    related: Tuple[str, ...] = ()      # laterally related concept URIs
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alt_labels"] = list(self.alt_labels)
        d["broader"] = list(self.broader)
        d["related"] = list(self.related)
        return d


@dataclass(frozen=True)
class Occupation:
    """An occupation with its essential and optional skill requirements."""

    uri: str
    label: str
    essential_skills: Tuple[str, ...] = ()
    optional_skills: Tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["essential_skills"] = list(self.essential_skills)
        d["optional_skills"] = list(self.optional_skills)
        return d


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@dataclass
class SkillSpan:
    """A skill mention located in source text and linked to the taxonomy."""

    surface: str                 # the exact text that matched
    start: int                   # character offset in the source document
    end: int
    uri: Optional[str] = None    # taxonomy URI, None if unlinked
    label: Optional[str] = None  # canonical taxonomy label
    confidence: float = 0.0
    method: str = "alias"        # how it was linked: alias | fuzzy | vector
    section: str = "body"        # which document section it came from
    negated: bool = False        # "no experience with X" -> True, excluded

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    """Structured view of a resume or job description."""

    doc_id: str
    raw_text: str
    doc_type: str = "resume"                     # "resume" | "job"
    sections: Dict[str, str] = field(default_factory=dict)
    title: str = ""
    years_experience: Optional[float] = None
    seniority: str = "unspecified"               # junior|mid|senior|lead|unspecified
    location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProfile:
    """A parsed resume plus its normalised skill set."""

    candidate_id: str
    document: ParsedDocument
    spans: List[SkillSpan] = field(default_factory=list)
    name: str = ""

    @property
    def skill_uris(self) -> List[str]:
        """Unique, order-preserving list of linked, non-negated skill URIs."""
        seen, out = set(), []
        for s in self.spans:
            if s.uri and not s.negated and s.uri not in seen:
                seen.add(s.uri)
                out.append(s.uri)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "document": self.document.to_dict(),
            "spans": [s.to_dict() for s in self.spans],
            "skill_uris": self.skill_uris,
        }


@dataclass
class JobPosting:
    """A vacancy with graded skill requirements."""

    job_id: str
    title: str
    company: str = ""
    location: str = ""
    seniority: str = "unspecified"
    description: str = ""
    essential_skills: List[str] = field(default_factory=list)
    optional_skills: List[str] = field(default_factory=list)
    occupation_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evidence and explanation
# ---------------------------------------------------------------------------


@dataclass
class EvidencePath:
    """
    One traceable link between a skill the candidate holds and a skill the job
    requires.  This is the atomic unit of grounding: an explanation sentence is
    faithful only if every claim in it maps onto one of these.
    """

    evidence_id: str                 # "E1", "E2", ... stable within a match
    candidate_skill_uri: str
    candidate_skill_label: str
    required_skill_uri: str
    required_skill_label: str
    relation: str                    # exact | narrower | broader | related
    strength: float
    requirement_type: str            # "essential" | "optional"
    hops: int = 1
    path_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillGap:
    """A required skill the candidate does not adequately cover."""

    skill_uri: str
    skill_label: str
    requirement_type: str
    best_relation: str = "none"
    best_strength: float = 0.0
    nearest_held_skill: Optional[str] = None      # label of closest held skill
    score_gain: float = 0.0                       # marginal gain if acquired

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Counterfactual:
    """
    The minimal set of skills whose acquisition moves the candidate over the
    qualification threshold, together with proof that it works.
    """

    target_threshold: float
    baseline_score: float
    projected_score: float
    skills: List[SkillGap] = field(default_factory=list)
    achieves_threshold: bool = False
    is_minimal: bool = True
    search_method: str = "greedy+prune"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["skills"] = [s.to_dict() for s in self.skills]
        return d


@dataclass
class ValidationResult:
    """Output of the faithfulness validator."""

    grounding_rate: float
    total_sentences: int
    grounded_sentences: int
    unsupported_claims: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Explanation:
    """A generated explanation with per-sentence evidence citations."""

    text: str
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[EvidencePath] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    provider: str = "grounded_template"
    withheld: bool = False
    withheld_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sentences": self.sentences,
            "evidence": [e.to_dict() for e in self.evidence],
            "validation": self.validation.to_dict() if self.validation else None,
            "provider": self.provider,
            "withheld": self.withheld,
            "withheld_reason": self.withheld_reason,
        }


@dataclass
class ScoreBreakdown:
    """Every component that fed the final score, for auditability."""

    essential_coverage: float = 0.0
    optional_coverage: float = 0.0
    semantic_similarity: float = 0.0
    seniority_fit: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrustScore:
    model_confidence: float = 0.0
    evidence_density: float = 0.0
    faithfulness: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobMatch:
    """A single recommended job with its full audit trail."""

    job: JobPosting
    score: float
    breakdown: ScoreBreakdown
    evidence: List[EvidencePath] = field(default_factory=list)
    gaps: List[SkillGap] = field(default_factory=list)
    explanation: Optional[Explanation] = None
    counterfactual: Optional[Counterfactual] = None
    trust: Optional[TrustScore] = None
    qualified: bool = False
    retrieval_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "score": self.score,
            "breakdown": self.breakdown.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "gaps": [g.to_dict() for g in self.gaps],
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "counterfactual": self.counterfactual.to_dict() if self.counterfactual else None,
            "trust": self.trust.to_dict() if self.trust else None,
            "qualified": self.qualified,
            "retrieval_sources": self.retrieval_sources,
        }


@dataclass
class RecommendationResult:
    """The complete response for one candidate."""

    candidate: CandidateProfile
    matches: List[JobMatch] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    providers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "matches": [m.to_dict() for m in self.matches],
            "timings_ms": self.timings_ms,
            "providers": self.providers,
        }
