"""
Hybrid retrieval — the GraphRAG retrieval stage.

Two retrievers run over the same corpus and their rankings are fused:

* **Graph retriever** walks `requires` edges backwards from the skills the
  candidate holds (including specialisations and adjacent skills) to reach jobs
  that need them.  Precise and explainable, but blind to anything the skill
  extractor missed.

* **Dense retriever** compares the candidate's profile embedding against job
  embeddings.  Catches semantic overlap the taxonomy does not encode, but
  cannot say *why* it matched.

Fusion uses Reciprocal Rank Fusion (RRF), which combines rankings rather than
scores.  This matters: graph overlap counts and cosine similarities are not on
the same scale, and normalising them against each other would make the blend
depend on the score distribution of whatever corpus happens to be loaded. RRF
only needs rank order, so it is stable across corpora.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SETTINGS
from ..kg.graph import SkillKnowledgeGraph
from ..ontology.esco import Taxonomy
from .vector_index import VectorIndex


class GraphRetriever:
    """Structural candidate generation over the skill knowledge graph."""

    def __init__(self, kg: SkillKnowledgeGraph, taxonomy: Taxonomy) -> None:
        self.kg = kg
        self.taxonomy = taxonomy

    def expand_skills(self, held: Sequence[str]) -> Dict[str, float]:
        """
        Grow the held skill set along the taxonomy, with decayed weights.

        A candidate who knows PyTorch should surface deep-learning jobs even
        without the literal phrase on their résumé.  Expansion weights decay by
        relation type so expanded skills never outrank directly held ones.
        """
        weights: Dict[str, float] = {}

        def bump(uri: str, value: float) -> None:
            if uri in self.taxonomy and value > weights.get(uri, 0.0):
                weights[uri] = value

        for uri in held:
            bump(uri, 1.0)
            for parent in self.taxonomy.ancestors(uri, max_depth=2):
                bump(parent, 0.6)
            for child in self.taxonomy.descendants(uri, max_depth=1):
                bump(child, 0.5)
            for other in self.taxonomy.related_of(uri):
                bump(other, 0.45)

        return weights

    def retrieve(self, held: Sequence[str], top_k: int = 50) -> List[Tuple[str, float]]:
        """Return (job_id, structural_score) ranked by weighted skill overlap."""
        if not held:
            return []

        expanded = self.expand_skills(held)
        scores: Dict[str, float] = {}

        for uri, weight in expanded.items():
            for job_id, _count in self.kg.jobs_requiring([uri]).items():
                scores[job_id] = scores.get(job_id, 0.0) + weight

        if not scores:
            return []

        # Normalise into [0, 1] so the raw value is comparable across queries.
        peak = max(scores.values())
        ranked = sorted(
            ((jid, s / peak) for jid, s in scores.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:top_k]


class DenseRetriever:
    """Semantic candidate generation over job embeddings."""

    def __init__(self, index: VectorIndex) -> None:
        self.index = index

    def retrieve(self, query_vector: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        return self.index.search(query_vector, top_k=top_k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    k: int = 60,
    labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float, List[str]]]:
    """
    Fuse ranked lists with Reciprocal Rank Fusion.

    Each list contributes 1 / (k + rank) for every item, with rank starting at
    1.  The constant k damps the influence of the very top positions so a
    single retriever cannot dominate the fused order.

    Returns (item_id, fused_score, contributing_sources) sorted best first.
    The source list is carried through so the interface can show which
    retriever surfaced a job — part of making the pipeline auditable end to
    end rather than only at the explanation step.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive")

    fused: Dict[str, float] = {}
    sources: Dict[str, List[str]] = {}
    names = list(labels) if labels else [f"retriever_{i}" for i in range(len(rankings))]

    for list_idx, ranking in enumerate(rankings):
        name = names[list_idx] if list_idx < len(names) else f"retriever_{list_idx}"
        for position, (item_id, _score) in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + position)
            sources.setdefault(item_id, []).append(name)

    return sorted(
        ((item_id, score, sources[item_id]) for item_id, score in fused.items()),
        key=lambda triple: (-triple[1], triple[0]),
    )


class HybridRetriever:
    """Graph and dense retrieval fused into one candidate list."""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        dense_retriever: DenseRetriever,
        rrf_k: Optional[int] = None,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k if rrf_k is not None else SETTINGS.rrf_k

    def retrieve(
        self,
        held_skills: Sequence[str],
        query_vector: np.ndarray,
        top_k: int = 50,
        mode: str = "hybrid",
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Retrieve candidate jobs.

        `mode` selects the ablation: "hybrid" (both), "graph" (structural only)
        or "dense" (semantic only).  Exposing the ablation here rather than in
        the experiment script means the evaluation measures the same code path
        the application runs.
        """
        pool = max(top_k * 2, top_k)
        graph_hits = self.graph_retriever.retrieve(held_skills, top_k=pool)
        dense_hits = self.dense_retriever.retrieve(query_vector, top_k=pool)

        if mode == "graph":
            return [(jid, s, ["graph"]) for jid, s in graph_hits[:top_k]]
        if mode == "dense":
            return [(jid, s, ["dense"]) for jid, s in dense_hits[:top_k]]
        if mode != "hybrid":
            raise ValueError(f"unknown retrieval mode: {mode!r}")

        fused = reciprocal_rank_fusion(
            [graph_hits, dense_hits], k=self.rrf_k, labels=["graph", "dense"]
        )
        return fused[:top_k]
