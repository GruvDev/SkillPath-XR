"""
Central configuration for SkillPath-XR.

Every external dependency (embedding model, graph database, LLM) is selected
here by name.  Each has an offline fallback so the system runs with zero
network access, and a production adapter you enable by changing one value.

Environment variables override defaults, so deployment needs no code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SKILLPATH_{name}", default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(f"SKILLPATH_{name}", default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(f"SKILLPATH_{name}", default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MatchWeights:
    """
    Strength assigned to each kind of evidence path between a skill the
    candidate holds and a skill the job requires.

    These are deliberately ordered: holding the exact skill is worth more than
    holding a specialisation of it, which is worth more than holding a merely
    adjacent skill.  Values are documented in docs/04-explainability.md and are
    the single place to tune graded coverage.
    """

    exact: float = 1.00        # candidate holds the required skill itself
    narrower: float = 0.80     # candidate holds a specialisation of it
    broader: float = 0.60      # candidate holds the general parent of it
    related: float = 0.45      # candidate holds an adjacent/transferable skill
    none: float = 0.00

    def for_relation(self, relation: str) -> float:
        return {
            "exact": self.exact,
            "narrower": self.narrower,
            "broader": self.broader,
            "related": self.related,
        }.get(relation, self.none)


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for combining signals into the final match score."""

    essential_coverage: float = 0.55
    optional_coverage: float = 0.15
    semantic_similarity: float = 0.20
    seniority_fit: float = 0.10

    def total(self) -> float:
        return (
            self.essential_coverage
            + self.optional_coverage
            + self.semantic_similarity
            + self.seniority_fit
        )


@dataclass(frozen=True)
class TrustWeights:
    """Weights for the composite trust score shown next to every explanation."""

    model_confidence: float = 0.30
    evidence_density: float = 0.30
    faithfulness: float = 0.40


@dataclass
class Settings:
    # ---- provider selection -------------------------------------------------
    # "hashing"  : deterministic offline embedder, no downloads (default)
    # "sentence_transformers" : real dense embeddings (needs the package)
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDINGS", "hashing"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 256))

    # "networkx" : in-process graph, no server (default)
    # "neo4j"    : production graph database
    graph_backend: str = field(default_factory=lambda: _env("GRAPH_BACKEND", "networkx"))
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", ""))

    # "grounded_template" : deterministic generator, provably faithful (default)
    # "ollama" / "openai" : real LLM, output checked by the faithfulness validator
    explanation_provider: str = field(default_factory=lambda: _env("EXPLAINER", "grounded_template"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3.1:8b"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 60.0))

    # ---- retrieval ----------------------------------------------------------
    retrieval_candidates: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATES", 50))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 10))

    # ---- decision thresholds ------------------------------------------------
    # A job is called a "match" at or above this score; the counterfactual
    # engine targets this number when computing the minimal skill gap.
    qualification_threshold: float = field(
        default_factory=lambda: _env_float("QUALIFY_THRESHOLD", 0.62)
    )
    # Explanations below this grounding rate are withheld from the user.
    min_grounding_rate: float = field(default_factory=lambda: _env_float("MIN_GROUNDING", 0.80))
    max_counterfactual_skills: int = field(default_factory=lambda: _env_int("MAX_CF_SKILLS", 5))

    # ---- weights ------------------------------------------------------------
    match_weights: MatchWeights = field(default_factory=MatchWeights)
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    trust_weights: TrustWeights = field(default_factory=TrustWeights)

    # ---- server -------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe_providers(self) -> Dict[str, str]:
        """Human-readable provider summary, surfaced in the UI status bar."""
        return {
            "embeddings": self.embedding_provider,
            "graph": self.graph_backend,
            "explainer": self.explanation_provider,
        }


SETTINGS = Settings()
