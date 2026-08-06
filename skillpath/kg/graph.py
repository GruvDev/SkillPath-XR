"""
The Skill Knowledge Graph.

This module holds the structure that makes explanations auditable.  Nodes are
skills, occupations, jobs and candidates; edges are the typed relations between
them.  The central operation is `evidence_paths`: given a skill a candidate
holds and a skill a job requires, find the traceable connection between them
and grade its strength.

The grading is what turns binary matching into graded coverage:

    exact     candidate holds the required skill itself
    narrower  candidate holds a specialisation (holds PyTorch, job wants
              deep learning) — strong transferable evidence
    broader   candidate holds the parent (holds machine learning, job wants
              PyTorch) — partial, the specific tool is not evidenced
    related   candidate holds an adjacent skill (holds Docker, job wants
              Kubernetes) — weak but real transferability

Every returned path is a first-class object with an id, so an explanation
sentence can cite it and the interface can highlight the exact edge.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from ..config import SETTINGS, MatchWeights
from ..ontology.esco import Taxonomy
from ..types import EvidencePath, JobPosting

# Node type prefixes keep the namespaces separate inside one graph.
JOB_PREFIX = "job::"
CAND_PREFIX = "cand::"


class SkillKnowledgeGraph:
    """A typed multi-relational graph over skills, jobs and candidates."""

    def __init__(self, taxonomy: Taxonomy, weights: Optional[MatchWeights] = None) -> None:
        self.taxonomy = taxonomy
        self.weights = weights or SETTINGS.match_weights
        self.graph = nx.MultiDiGraph()
        self._build_taxonomy_layer()

    # -- construction --------------------------------------------------------

    def _build_taxonomy_layer(self) -> None:
        """Load skills and their relations as the base layer of the graph."""
        for uri, concept in self.taxonomy.skills.items():
            self.graph.add_node(
                uri,
                node_type="skill",
                label=concept.label,
                skill_type=concept.skill_type,
            )

        for uri, concept in self.taxonomy.skills.items():
            for parent in concept.broader:
                if parent in self.taxonomy:
                    # child --broader--> parent, and the inverse for traversal
                    self.graph.add_edge(uri, parent, key="broader", relation="broader")
                    self.graph.add_edge(parent, uri, key="narrower", relation="narrower")
            for other in concept.related:
                if other in self.taxonomy:
                    self.graph.add_edge(uri, other, key="related", relation="related")

    def add_job(self, job: JobPosting) -> str:
        """Add a job node with `requires` edges to its skills."""
        node = JOB_PREFIX + job.job_id
        self.graph.add_node(
            node,
            node_type="job",
            label=job.title,
            job_id=job.job_id,
            company=job.company,
            seniority=job.seniority,
        )
        for uri in job.essential_skills:
            if uri in self.taxonomy:
                self.graph.add_edge(
                    node, uri, key="requires_essential",
                    relation="requires", requirement="essential",
                )
        for uri in job.optional_skills:
            if uri in self.taxonomy:
                self.graph.add_edge(
                    node, uri, key="requires_optional",
                    relation="requires", requirement="optional",
                )
        return node

    def add_jobs(self, jobs: Iterable[JobPosting]) -> None:
        for job in jobs:
            self.add_job(job)

    def add_candidate(self, candidate_id: str, skill_uris: Sequence[str]) -> str:
        """Add a candidate node with `has_skill` edges."""
        node = CAND_PREFIX + candidate_id
        self.graph.add_node(node, node_type="candidate", label=candidate_id)
        for uri in skill_uris:
            if uri in self.taxonomy:
                self.graph.add_edge(node, uri, key="has_skill", relation="has_skill")
        return node

    # -- queries -------------------------------------------------------------

    def jobs_requiring(self, skill_uris: Iterable[str]) -> Dict[str, int]:
        """
        Job ids that require any of these skills, with how many they match.

        Used by the graph retriever as a fast structural candidate generator:
        it walks the `requires` edges backwards from the candidate's skills.
        """
        counts: Dict[str, int] = {}
        for uri in set(skill_uris):
            if uri not in self.graph:
                continue
            for source, _target, data in self.graph.in_edges(uri, data=True):
                if data.get("relation") == "requires" and source.startswith(JOB_PREFIX):
                    job_id = self.graph.nodes[source].get("job_id")
                    if job_id:
                        counts[job_id] = counts.get(job_id, 0) + 1
        return counts

    def relation_between(self, held: str, required: str) -> Tuple[str, int, List[str]]:
        """
        Classify how a held skill connects to a required skill.

        Returns (relation, hops, path_labels).  Checked strongest first and
        returns on the first hit, so a skill that is both a specialisation and
        laterally related is reported as the stronger of the two.

        Depth is capped at two hops for hierarchy: beyond that the semantic
        link is too weak to defend to a user as evidence, which matters more
        here than recall.
        """
        if held == required:
            return "exact", 0, [self.taxonomy.label(held)]

        label_h = self.taxonomy.label(held)
        label_r = self.taxonomy.label(required)

        # Held skill is a specialisation of the requirement (strongest transfer).
        if held in self.taxonomy.descendants(required, max_depth=2):
            return "narrower", 1, [label_h, "is a kind of", label_r]

        # Held skill is the general parent of the requirement (partial).
        if held in self.taxonomy.ancestors(required, max_depth=2):
            return "broader", 1, [label_h, "is the broader field of", label_r]

        # Laterally related concepts.
        if required in self.taxonomy.related_of(held) or held in self.taxonomy.related_of(required):
            return "related", 1, [label_h, "is related to", label_r]

        # Siblings: share a direct parent. Treated as `related` strength.
        parents_h = set(self.taxonomy.broader_of(held))
        parents_r = set(self.taxonomy.broader_of(required))
        shared = parents_h & parents_r
        if shared:
            parent_label = self.taxonomy.label(sorted(shared)[0])
            return "related", 2, [label_h, f"shares the field {parent_label} with", label_r]

        return "none", -1, []

    def best_evidence_for_requirement(
        self,
        held_skills: Sequence[str],
        required: str,
        requirement_type: str,
        evidence_id: str,
    ) -> Optional[EvidencePath]:
        """
        Strongest evidence path from any held skill to one requirement.

        Ties are broken deterministically by skill label so repeated runs
        produce identical output — important for reproducible evaluation.
        """
        best: Optional[Tuple[float, str, int, List[str], str]] = None

        for held in held_skills:
            relation, hops, labels = self.relation_between(held, required)
            if relation == "none":
                continue
            strength = self.weights.for_relation(relation)
            if strength <= 0:
                continue
            candidate = (strength, held, hops, labels, relation)
            if best is None:
                best = candidate
            else:
                # higher strength wins; then fewer hops; then stable label order
                if (strength, -hops, self.taxonomy.label(held)) > (
                    best[0],
                    -best[2],
                    self.taxonomy.label(best[1]),
                ):
                    best = candidate

        if best is None:
            return None

        strength, held, hops, labels, relation = best
        return EvidencePath(
            evidence_id=evidence_id,
            candidate_skill_uri=held,
            candidate_skill_label=self.taxonomy.label(held),
            required_skill_uri=required,
            required_skill_label=self.taxonomy.label(required),
            relation=relation,
            strength=strength,
            requirement_type=requirement_type,
            hops=hops,
            path_labels=labels,
        )

    def evidence_paths(
        self,
        held_skills: Sequence[str],
        job: JobPosting,
        start_index: int = 1,
    ) -> Tuple[List[EvidencePath], List[Tuple[str, str, str, float, Optional[str]]]]:
        """
        Full evidence set for one candidate/job pair.

        Returns (evidence, uncovered) where `uncovered` lists requirements with
        no acceptable evidence, as
        (skill_uri, label, requirement_type, best_strength, nearest_held_label).
        Together these two lists cover every requirement exactly once, which is
        what lets the interface prove nothing was quietly dropped.
        """
        evidence: List[EvidencePath] = []
        uncovered: List[Tuple[str, str, str, float, Optional[str]]] = []
        counter = start_index

        requirements = [(u, "essential") for u in job.essential_skills]
        requirements += [(u, "optional") for u in job.optional_skills]

        for required, req_type in requirements:
            if required not in self.taxonomy:
                continue
            path = self.best_evidence_for_requirement(
                held_skills, required, req_type, f"E{counter}"
            )
            if path is not None:
                evidence.append(path)
                counter += 1
            else:
                nearest = self._nearest_held(held_skills, required)
                uncovered.append(
                    (required, self.taxonomy.label(required), req_type, 0.0, nearest)
                )

        return evidence, uncovered

    def _nearest_held(self, held_skills: Sequence[str], required: str) -> Optional[str]:
        """
        Closest held skill by shared ancestry, for gap messaging.

        Only used to make a gap easier to act on ("closest thing you have is
        X"), never as evidence of a match.
        """
        req_ancestors = set(self.taxonomy.ancestors(required, max_depth=3))
        if not req_ancestors:
            return None
        best_label, best_overlap = None, 0
        for held in held_skills:
            overlap = len(req_ancestors & set(self.taxonomy.ancestors(held, max_depth=3)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = self.taxonomy.label(held)
        return best_label

    # -- introspection -------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        types: Dict[str, int] = {}
        for _node, data in self.graph.nodes(data=True):
            key = data.get("node_type", "unknown")
            types[key] = types.get(key, 0) + 1
        relations: Dict[str, int] = {}
        for _u, _v, data in self.graph.edges(data=True):
            key = data.get("relation", "unknown")
            relations[key] = relations.get(key, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            **{f"nodes_{k}": v for k, v in sorted(types.items())},
            **{f"edges_{k}": v for k, v in sorted(relations.items())},
        }

    def subgraph_for_evidence(self, evidence: Sequence[EvidencePath]) -> Dict[str, object]:
        """
        A minimal node/edge payload describing just these evidence paths.

        This is the retrieval context handed to the explanation generator and
        the exact structure the interface draws.  Keeping generation and
        rendering on the same payload is what guarantees the picture a user
        sees is the evidence the text was written from.
        """
        nodes: Dict[str, Dict[str, object]] = {}
        edges: List[Dict[str, object]] = []

        for path in evidence:
            nodes[path.candidate_skill_uri] = {
                "id": path.candidate_skill_uri,
                "label": path.candidate_skill_label,
                "role": "held",
            }
            nodes[path.required_skill_uri] = {
                "id": path.required_skill_uri,
                "label": path.required_skill_label,
                "role": "required",
            }
            edges.append(
                {
                    "evidence_id": path.evidence_id,
                    "source": path.candidate_skill_uri,
                    "target": path.required_skill_uri,
                    "relation": path.relation,
                    "strength": path.strength,
                    "requirement_type": path.requirement_type,
                }
            )

        return {"nodes": list(nodes.values()), "edges": edges}
