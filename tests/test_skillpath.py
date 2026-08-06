"""
SkillPath-XR test suite (standard library `unittest`, no pytest required).

Run:  python -m unittest discover -s tests -v
      python tests/test_skillpath.py

The suite is organised by module.  The tests that matter most for the research
claims are:

* `TestFaithfulnessValidator` — proves the validator *rejects* hallucinated
  explanations.  A validator that always passes would report a perfect
  grounding rate and mean nothing, so these are the tests that give the metric
  its value.
* `TestCounterfactual` — proves the suggested skills actually move the score,
  verified by re-running the scorer rather than by assertion.
* `TestInvariants` — properties that must hold for every candidate/job pair.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from skillpath.config import MatchWeights, ScoreWeights, Settings  # noqa: E402
from skillpath.embeddings.providers import HashingEmbedder  # noqa: E402
from skillpath.evaluation.metrics import (  # noqa: E402
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from skillpath.explain.counterfactual import CounterfactualEngine  # noqa: E402
from skillpath.explain.generator import GroundedTemplateGenerator  # noqa: E402
from skillpath.explain.validator import FaithfulnessValidator, split_sentences  # noqa: E402
from skillpath.extraction.skills import SkillExtractor  # noqa: E402
from skillpath.kg.graph import SkillKnowledgeGraph  # noqa: E402
from skillpath.ontology.esco import Taxonomy, load_seed_taxonomy, normalise  # noqa: E402
from skillpath.parsing.document import (  # noqa: E402
    detect_seniority,
    detect_years_experience,
    parse_document,
    seniority_from_years,
)
from skillpath.pipeline import SkillPathPipeline, load_jobs, load_sample_resumes  # noqa: E402
from skillpath.ranking.scorer import coverage, score_match, seniority_fit  # noqa: E402
from skillpath.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from skillpath.retrieval.vector_index import VectorIndex  # noqa: E402
from skillpath.scoring.trust import compute_trust, evidence_density, model_confidence  # noqa: E402
from skillpath.types import EvidencePath, JobPosting, SkillConcept  # noqa: E402

# Built once: constructing the pipeline is the expensive part and nothing in
# the suite mutates it.
TAXONOMY = load_seed_taxonomy()
PIPELINE = SkillPathPipeline(taxonomy=TAXONOMY)
RESUMES = load_sample_resumes()


class TestTaxonomy(unittest.TestCase):
    def test_loads_and_indexes(self):
        self.assertGreater(len(TAXONOMY), 50)
        self.assertGreater(len(TAXONOMY.occupations), 5)
        self.assertGreater(len(TAXONOMY.alias_index), len(TAXONOMY))

    def test_normalise_preserves_meaningful_punctuation(self):
        self.assertEqual(normalise("C++"), "c++")
        self.assertEqual(normalise("Node.js"), "node.js")
        self.assertEqual(normalise("  Machine   LEARNING  "), "machine learning")
        self.assertEqual(normalise("REST/API!"), "rest api")

    def test_alias_lookup_matches_alt_labels(self):
        self.assertTrue(TAXONOMY.lookup_alias("ML"))
        self.assertEqual(TAXONOMY.lookup_alias("ML"), TAXONOMY.lookup_alias("machine learning"))

    def test_hierarchy_navigation_is_consistent(self):
        for uri in TAXONOMY.skills:
            for parent in TAXONOMY.broader_of(uri):
                self.assertIn(uri, TAXONOMY.narrower_of(parent))

    def test_ancestors_terminate_on_cycles(self):
        # A deliberately cyclic taxonomy must not hang or recurse forever.
        cyclic = Taxonomy([
            SkillConcept(uri="a", label="A", broader=("b",)),
            SkillConcept(uri="b", label="B", broader=("a",)),
        ])
        self.assertEqual(set(cyclic.ancestors("a", max_depth=5)), {"b", "a"} - {"a"} | {"b"})
        self.assertLessEqual(len(cyclic.ancestors("a", max_depth=5)), 2)

    def test_no_dangling_references(self):
        for uri, concept in TAXONOMY.skills.items():
            for ref in list(concept.broader) + list(concept.related):
                self.assertIn(ref, TAXONOMY.skills, f"{uri} references missing {ref}")


class TestParsing(unittest.TestCase):
    def test_sections_are_segmented(self):
        doc = parse_document(RESUMES[0]["text"], "C001")
        self.assertIn("skills", doc.sections)
        self.assertIn("experience", doc.sections)

    def test_years_experience(self):
        self.assertEqual(detect_years_experience("4 years of experience"), 4.0)
        self.assertEqual(detect_years_experience("Experience: 7 years"), 7.0)
        self.assertIsNone(detect_years_experience("no numbers here"))
        # Months are converted, not ignored.
        self.assertAlmostEqual(detect_years_experience("6 months internship"), 0.5, places=2)

    def test_seniority_precedence(self):
        # "lead" must win over "senior" when both appear.
        self.assertEqual(detect_seniority("Senior Lead Engineer"), "lead")
        self.assertEqual(detect_seniority("Junior Developer"), "junior")
        self.assertEqual(detect_seniority("Developer"), "unspecified")

    def test_seniority_inferred_from_years(self):
        self.assertEqual(seniority_from_years(1.0), "junior")
        self.assertEqual(seniority_from_years(3.0), "mid")
        self.assertEqual(seniority_from_years(6.0), "senior")
        self.assertEqual(seniority_from_years(12.0), "lead")
        self.assertEqual(seniority_from_years(None), "unspecified")

    def test_seniority_falls_back_to_years(self):
        doc = parse_document("Backend Engineer\n\nSUMMARY\n4 years of experience.", "X")
        self.assertEqual(doc.seniority, "mid")

    def test_empty_document_does_not_crash(self):
        doc = parse_document("", "empty")
        self.assertEqual(doc.raw_text, "")
        self.assertEqual(doc.seniority, "unspecified")

    def test_rejects_non_string(self):
        with self.assertRaises(TypeError):
            parse_document(None, "x")  # type: ignore[arg-type]


class TestSkillExtraction(unittest.TestCase):
    def setUp(self):
        self.ex = SkillExtractor(TAXONOMY)

    def test_extracts_basic_skills(self):
        labels = {TAXONOMY.label(u) for u in self.ex.extract_uris("I know Python and SQL")}
        self.assertIn("Python", labels)
        self.assertIn("SQL", labels)

    def test_longest_match_wins(self):
        spans = self.ex.extract("experience with machine learning")
        surfaces = {s.surface.lower() for s in spans}
        self.assertIn("machine learning", surfaces)
        self.assertNotIn("learning", surfaces)

    def test_negation_excludes_skill(self):
        uris = self.ex.extract_uris("No experience with Kubernetes.")
        self.assertFalse(any("kubernetes" in u for u in uris))
        # The span is still recorded so the interface can show it was considered.
        spans = self.ex.extract("No experience with Kubernetes.")
        self.assertTrue(any(s.negated for s in spans))

    def test_negation_does_not_cross_sentence_boundary(self):
        uris = self.ex.extract_uris("No experience with Kubernetes. I use Docker daily.")
        self.assertTrue(any("docker" in u for u in uris))

    def test_sentence_final_skill_is_extracted(self):
        # Regression: the token pattern admits '.', so "Java." must still match.
        for text in ["SQL is related to Java.", "I know Python.", "We use Docker."]:
            with self.subTest(text=text):
                self.assertTrue(self.ex.extract_uris(text), f"nothing extracted from {text!r}")

    def test_internal_dot_preserved(self):
        labels = {TAXONOMY.label(u) for u in self.ex.extract_uris("I use Node.js daily.")}
        self.assertIn("Node.js", labels)

    def test_offsets_point_at_source_text(self):
        text = "Skilled in Python and Docker."
        for span in self.ex.extract(text):
            self.assertEqual(text[span.start : span.end], span.surface)

    def test_proficiency_affects_confidence(self):
        expert = self.ex.extract("expert in Python")[0]
        learning = self.ex.extract("currently learning Python")[0]
        self.assertGreater(expert.confidence, learning.confidence)

    def test_empty_text(self):
        self.assertEqual(self.ex.extract(""), [])
        self.assertEqual(self.ex.extract_uris(""), [])

    def test_deduplicates_repeat_mentions(self):
        uris = self.ex.extract_uris("Python Python Python and more Python")
        self.assertEqual(len(uris), len(set(uris)))


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.emb = HashingEmbedder(dim=128)

    def test_deterministic(self):
        a = self.emb.encode_one("machine learning engineer")
        b = self.emb.encode_one("machine learning engineer")
        np.testing.assert_allclose(a, b)

    def test_vectors_are_normalised(self):
        vecs = self.emb.encode(["python developer", "data scientist"])
        for v in vecs:
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_empty_string_is_zero_not_nan(self):
        v = self.emb.encode_one("")
        self.assertFalse(np.isnan(v).any())
        self.assertAlmostEqual(float(np.linalg.norm(v)), 0.0, places=6)

    def test_similar_text_scores_higher(self):
        q = self.emb.encode_one("python data analysis")
        near = self.emb.encode_one("python data analyst role")
        far = self.emb.encode_one("graphic design illustration")
        self.assertGreater(float(q @ near), float(q @ far))

    def test_rejects_bad_dim(self):
        with self.assertRaises(ValueError):
            HashingEmbedder(dim=0)


class TestVectorIndex(unittest.TestCase):
    def test_search_orders_by_similarity(self):
        emb = HashingEmbedder(dim=128)
        idx = VectorIndex(128)
        texts = ["python machine learning", "java spring backend", "figma ux design"]
        idx.add(["a", "b", "c"], emb.encode(texts))
        hits = idx.search(emb.encode_one("python ml engineer"), top_k=3)
        self.assertEqual(hits[0][0], "a")
        self.assertEqual(len(hits), 3)
        # Scores must be descending.
        self.assertTrue(all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1)))

    def test_empty_index_returns_empty(self):
        self.assertEqual(VectorIndex(16).search(np.zeros(16), top_k=5), [])

    def test_dimension_mismatch_raises(self):
        idx = VectorIndex(8)
        with self.assertRaises(ValueError):
            idx.add(["a"], np.zeros((1, 4), dtype=np.float32))
        idx.add(["a"], np.zeros((1, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            idx.search(np.zeros(4), top_k=1)

    def test_id_vector_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            VectorIndex(4).add(["a", "b"], np.zeros((1, 4), dtype=np.float32))

    def test_scores_never_negative(self):
        idx = VectorIndex(4)
        idx.add(["a"], np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.assertGreaterEqual(idx.search(np.array([-1, 0, 0, 0], dtype=np.float32))[0][1], 0.0)


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.kg = PIPELINE.kg

    def test_graph_is_populated(self):
        stats = self.kg.stats()
        self.assertGreater(stats["nodes"], 100)
        self.assertGreater(stats["edges_requires"], 0)

    def test_exact_relation(self):
        uri = TAXONOMY.lookup_alias("Python")[0]
        rel, hops, _ = self.kg.relation_between(uri, uri)
        self.assertEqual(rel, "exact")
        self.assertEqual(hops, 0)

    def test_broader_and_narrower_are_symmetric_opposites(self):
        ml = TAXONOMY.lookup_alias("machine learning")[0]
        dl = TAXONOMY.lookup_alias("deep learning")[0]
        self.assertEqual(self.kg.relation_between(ml, dl)[0], "broader")
        self.assertEqual(self.kg.relation_between(dl, ml)[0], "narrower")

    def test_unrelated_skills_return_none(self):
        a = TAXONOMY.lookup_alias("Figma")[0]
        b = TAXONOMY.lookup_alias("Apache Kafka")[0]
        self.assertEqual(self.kg.relation_between(a, b)[0], "none")

    def test_relation_strength_ordering(self):
        w = MatchWeights()
        self.assertGreater(w.exact, w.narrower)
        self.assertGreater(w.narrower, w.broader)
        self.assertGreater(w.broader, w.related)
        self.assertEqual(w.for_relation("nonsense"), 0.0)

    def test_evidence_and_gaps_partition_requirements(self):
        """Every requirement is either evidenced or reported as a gap — never dropped."""
        held = PIPELINE.extractor.extract_uris(RESUMES[0]["text"])
        for job in PIPELINE.jobs:
            with self.subTest(job=job.job_id):
                evidence, uncovered = self.kg.evidence_paths(held, job)
                total = len(job.essential_skills) + len(job.optional_skills)
                self.assertEqual(len(evidence) + len(uncovered), total)

    def test_evidence_ids_are_unique_and_sequential(self):
        held = PIPELINE.extractor.extract_uris(RESUMES[3]["text"])
        evidence, _ = self.kg.evidence_paths(held, PIPELINE.jobs[0])
        ids = [e.evidence_id for e in evidence]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [f"E{i}" for i in range(1, len(ids) + 1)])

    def test_no_skills_yields_no_evidence(self):
        evidence, uncovered = self.kg.evidence_paths([], PIPELINE.jobs[0])
        self.assertEqual(evidence, [])
        self.assertGreater(len(uncovered), 0)

    def test_subgraph_matches_evidence(self):
        held = PIPELINE.extractor.extract_uris(RESUMES[0]["text"])
        evidence, _ = self.kg.evidence_paths(held, PIPELINE.jobs[0])
        sub = self.kg.subgraph_for_evidence(evidence)
        self.assertEqual(len(sub["edges"]), len(evidence))


class TestScorer(unittest.TestCase):
    def _evidence(self, uri, relation, strength, req_type="essential"):
        return EvidencePath(
            evidence_id="E1", candidate_skill_uri=uri, candidate_skill_label="x",
            required_skill_uri=uri, required_skill_label="x",
            relation=relation, strength=strength, requirement_type=req_type,
        )

    def test_coverage_is_strength_weighted(self):
        ev = [self._evidence("a", "exact", 1.0), self._evidence("b", "related", 0.45)]
        ev[1].required_skill_uri = "b"
        self.assertAlmostEqual(coverage(ev, ["a", "b"], "essential"), 0.725)

    def test_coverage_of_no_requirements_is_zero(self):
        self.assertEqual(coverage([], [], "essential"), 0.0)

    def test_seniority_fit_penalises_under_qualification_harder(self):
        under = seniority_fit("junior", "senior")   # candidate below the role
        over = seniority_fit("senior", "junior")    # candidate above the role
        self.assertGreater(over, under)
        self.assertEqual(seniority_fit("mid", "mid"), 1.0)
        self.assertEqual(seniority_fit("unspecified", "mid"), 0.6)

    def test_seniority_fit_bounded(self):
        for a in ("junior", "mid", "senior", "lead", "unspecified"):
            for b in ("junior", "mid", "senior", "lead", "unspecified"):
                self.assertTrue(0.0 <= seniority_fit(a, b) <= 1.0)

    def test_weights_renormalise_when_no_optional_skills(self):
        """A job without optional skills must not be penalised for it."""
        job_with = JobPosting(job_id="A", title="t", essential_skills=["a"], optional_skills=["b"])
        job_without = JobPosting(job_id="B", title="t", essential_skills=["a"], optional_skills=[])
        ev = [self._evidence("a", "exact", 1.0)]
        with_opt = score_match(ev, job_with, 0.5, "mid")
        without_opt = score_match(ev, job_without, 0.5, "mid")
        self.assertGreater(without_opt.final_score, with_opt.final_score)

    def test_score_bounded(self):
        job = JobPosting(job_id="A", title="t", essential_skills=["a"], seniority="mid")
        ev = [self._evidence("a", "exact", 1.0)]
        self.assertTrue(0.0 <= score_match(ev, job, 1.0, "mid").final_score <= 1.0)
        self.assertTrue(0.0 <= score_match([], job, 0.0, "junior").final_score <= 1.0)

    def test_score_weights_sum_to_one(self):
        self.assertAlmostEqual(ScoreWeights().total(), 1.0, places=6)


class TestRRF(unittest.TestCase):
    def test_fusion_rewards_agreement(self):
        a = [("x", 1.0), ("y", 0.9), ("z", 0.8)]
        b = [("y", 1.0), ("x", 0.9), ("w", 0.7)]
        fused = reciprocal_rank_fusion([a, b], k=60, labels=["a", "b"])
        top = [item for item, _score, _src in fused[:2]]
        self.assertIn("x", top)
        self.assertIn("y", top)

    def test_sources_are_tracked(self):
        fused = reciprocal_rank_fusion([[("x", 1.0)], [("x", 1.0)]], k=60, labels=["g", "d"])
        self.assertEqual(sorted(fused[0][2]), ["d", "g"])

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([], k=60), [])

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[("x", 1.0)]], k=0)


class TestFaithfulnessValidator(unittest.TestCase):
    """
    The tests that give the grounding metric its meaning.

    Each asserts the validator *rejects* something it should reject.  Without
    these, a validator that returned 1.0 unconditionally would pass every other
    test in this file.
    """

    def setUp(self):
        self.validator = FaithfulnessValidator(PIPELINE.extractor)
        held = PIPELINE.extractor.extract_uris(RESUMES[0]["text"])
        job = next(j for j in PIPELINE.jobs if j.job_id == "J001")
        self.evidence, uncovered = PIPELINE.kg.evidence_paths(held, job)
        self.gap_uris = [u for u, _l, _t, _s, _n in uncovered]

    def test_sentence_splitting(self):
        self.assertEqual(len(split_sentences("One. Two! Three?")), 3)
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   "), [])

    def test_template_output_is_fully_grounded(self):
        text = GroundedTemplateGenerator().generate(
            PIPELINE.jobs[0], self.evidence, [], 0.8
        )
        result = self.validator.validate(text, self.evidence, self.gap_uris)
        self.assertEqual(result.grounding_rate, 1.0)
        self.assertTrue(result.passed)

    def test_rejects_hallucinated_entity(self):
        text = "The candidate has extensive Kubernetes and Terraform experience."
        result = self.validator.validate(text, self.evidence, self.gap_uris)
        self.assertFalse(result.passed)
        self.assertLess(result.grounding_rate, 1.0)
        flagged = result.unsupported_claims[0]["ungrounded_entities"]
        self.assertIn("Kubernetes", flagged)

    def test_rejects_fabricated_relation(self):
        text = "Python is a kind of SQL, so the requirement is met."
        result = self.validator.validate(text, self.evidence, self.gap_uris)
        self.assertFalse(result.passed)
        self.assertIn("narrower", result.unsupported_claims[0]["unsupported_relations"])

    def test_flags_only_the_bad_sentence(self):
        """A half-true explanation must be partially, not wholly, rejected."""
        text = "The resume evidences Python directly. The candidate also brings deep Docker expertise."
        result = self.validator.validate(text, self.evidence, self.gap_uris)
        self.assertEqual(result.total_sentences, 2)
        self.assertEqual(result.grounded_sentences, 1)
        self.assertAlmostEqual(result.grounding_rate, 0.5)

    def test_empty_explanation_fails(self):
        result = self.validator.validate("", self.evidence, self.gap_uris)
        self.assertFalse(result.passed)
        self.assertEqual(result.grounding_rate, 0.0)

    def test_gap_skills_may_be_mentioned(self):
        """Naming a missing skill is honest reporting, not a hallucination."""
        # Pick a pairing that genuinely has gaps rather than skipping.
        held = PIPELINE.extractor.extract_uris(RESUMES[0]["text"])
        for job in PIPELINE.jobs:
            evidence, uncovered = PIPELINE.kg.evidence_paths(held, job)
            if uncovered:
                gap_uris = [u for u, _l, _t, _s, _n in uncovered]
                label = TAXONOMY.label(gap_uris[0])
                result = self.validator.validate(
                    f"No evidence was found for {label}.", evidence, gap_uris
                )
                self.assertTrue(
                    result.passed,
                    f"mentioning gap {label!r} was wrongly flagged as unfaithful",
                )
                return
        self.fail("no candidate/job pairing in the corpus produced a gap")

    def test_annotations_cite_evidence(self):
        text = GroundedTemplateGenerator().generate(PIPELINE.jobs[0], self.evidence, [], 0.8)
        records = self.validator.annotate_sentences(text, self.evidence, self.gap_uris)
        self.assertTrue(records)
        self.assertTrue(any(r["evidence_ids"] for r in records))


class TestCounterfactual(unittest.TestCase):
    def setUp(self):
        self.engine = CounterfactualEngine(PIPELINE.kg, threshold=0.62, max_skills=5)

    def test_already_qualified_suggests_nothing(self):
        held = PIPELINE.extractor.extract_uris(RESUMES[0]["text"])
        job = next(j for j in PIPELINE.jobs if j.job_id == "J001")
        cf = self.engine.compute(job, held, 0.7, "junior")
        self.assertTrue(cf.achieves_threshold)
        self.assertEqual(cf.skills, [])

    def test_near_miss_produces_verified_counterfactual(self):
        """The suggested skills must actually move the score when applied."""
        checked = 0
        for resume in RESUMES:
            profile = PIPELINE.build_profile(resume["text"], resume["candidate_id"])
            held = profile.skill_uris
            for job in PIPELINE.jobs:
                cf = self.engine.compute(job, held, 0.5, profile.document.seniority)
                if not cf.skills:
                    continue
                checked += 1
                self.assertTrue(
                    self.engine.verify(cf, job, held, 0.5, profile.document.seniority),
                    f"counterfactual for {job.job_id} did not verify",
                )
                self.assertGreater(cf.projected_score, cf.baseline_score)
        self.assertGreater(checked, 0, "no counterfactuals were exercised")

    def test_respects_max_skills_cap(self):
        engine = CounterfactualEngine(PIPELINE.kg, threshold=0.99, max_skills=2)
        cf = engine.compute(PIPELINE.jobs[0], [], 0.0, "junior")
        self.assertLessEqual(len(cf.skills), 2)

    def test_unreachable_threshold_is_reported_honestly(self):
        engine = CounterfactualEngine(PIPELINE.kg, threshold=0.999, max_skills=3)
        cf = engine.compute(PIPELINE.jobs[0], [], 0.0, "junior")
        self.assertFalse(cf.achieves_threshold)
        self.assertTrue(cf.note)

    def test_only_suggests_skills_the_job_asks_for(self):
        job = PIPELINE.jobs[0]
        cf = self.engine.compute(job, [], 0.0, "mid")
        allowed = set(job.essential_skills) | set(job.optional_skills)
        for gap in cf.skills:
            self.assertIn(gap.skill_uri, allowed)

    def test_selected_set_is_locally_minimal(self):
        """No single suggested skill can be dropped while still clearing the bar."""
        for resume in RESUMES[:2]:
            profile = PIPELINE.build_profile(resume["text"], resume["candidate_id"])
            held = profile.skill_uris
            for job in PIPELINE.jobs:
                cf = self.engine.compute(job, held, 0.5, profile.document.seniority)
                if not (cf.achieves_threshold and len(cf.skills) > 1):
                    continue
                uris = [g.skill_uri for g in cf.skills]
                for drop in uris:
                    subset = [u for u in uris if u != drop]
                    score = self.engine._score(
                        held + subset, job, 0.5, profile.document.seniority
                    )
                    self.assertLess(
                        score, self.engine.threshold,
                        f"{job.job_id}: dropping {drop} still clears the threshold",
                    )


class TestTrust(unittest.TestCase):
    def test_confidence_is_zero_at_threshold(self):
        self.assertAlmostEqual(model_confidence(0.62, 0.62), 0.0, places=6)

    def test_confidence_rises_away_from_threshold(self):
        self.assertGreater(model_confidence(0.95, 0.62), model_confidence(0.70, 0.62))
        self.assertGreater(model_confidence(0.10, 0.62), model_confidence(0.55, 0.62))

    def test_evidence_density_weights_essentials_double(self):
        job = JobPosting(job_id="A", title="t", essential_skills=["a", "b"], optional_skills=["c"])
        ev = [EvidencePath("E1", "a", "a", "a", "a", "exact", 1.0, "essential")]
        only_optional = [EvidencePath("E1", "c", "c", "c", "c", "exact", 1.0, "optional")]
        self.assertGreater(evidence_density(ev, job), evidence_density(only_optional, job))

    def test_trust_is_bounded(self):
        job = JobPosting(job_id="A", title="t", essential_skills=["a"])
        trust = compute_trust(0.9, [], job, None, 0.62)
        self.assertTrue(0.0 <= trust.overall <= 1.0)


class TestMetrics(unittest.TestCase):
    def test_precision_recall(self):
        ranked, relevant = ["a", "b", "c", "d"], {"a", "c"}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 4), 1.0)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "y", "a"], {"a"}), 1 / 3)
        self.assertEqual(reciprocal_rank(["x"], {"a"}), 0.0)

    def test_ndcg_perfect_and_empty(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3), 1.0)
        self.assertEqual(ndcg_at_k(["a"], set(), 3), 0.0)

    def test_ndcg_rewards_better_ordering(self):
        good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        bad = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        self.assertGreater(good, bad)

    def test_average_precision(self):
        self.assertAlmostEqual(average_precision(["a", "b"], {"a", "b"}), 1.0)
        self.assertEqual(average_precision(["a"], set()), 0.0)


class TestPipeline(unittest.TestCase):
    def test_recommend_returns_ranked_matches(self):
        result = PIPELINE.recommend(RESUMES[0]["text"], top_k=5)
        self.assertTrue(result.matches)
        self.assertLessEqual(len(result.matches), 5)
        scores = [m.score for m in result.matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_match_carries_a_full_audit_trail(self):
        result = PIPELINE.recommend(RESUMES[3]["text"], top_k=3)
        for match in result.matches:
            self.assertIsNotNone(match.explanation)
            self.assertIsNotNone(match.explanation.validation)
            self.assertIsNotNone(match.counterfactual)
            self.assertIsNotNone(match.trust)
            self.assertTrue(0.0 <= match.score <= 1.0)

    def test_all_explanations_meet_the_grounding_bar(self):
        for resume in RESUMES:
            result = PIPELINE.recommend(resume["text"], top_k=5)
            for match in result.matches:
                validation = match.explanation.validation
                self.assertGreaterEqual(
                    validation.grounding_rate, PIPELINE.settings.min_grounding_rate,
                    f"{resume['candidate_id']}/{match.job.job_id}: {validation.grounding_rate}",
                )

    def test_retrieval_modes_all_work(self):
        for mode in ("hybrid", "graph", "dense"):
            with self.subTest(mode=mode):
                result = PIPELINE.recommend(RESUMES[1]["text"], top_k=3, mode=mode)
                self.assertTrue(result.matches)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            PIPELINE.recommend(RESUMES[0]["text"], mode="nonsense")

    def test_empty_resume_yields_no_crash(self):
        result = PIPELINE.recommend("", top_k=3)
        self.assertIsInstance(result.matches, list)

    def test_nonsense_resume_yields_no_matches_or_low_scores(self):
        result = PIPELINE.recommend("qqq zzz wibble flurb", top_k=3)
        for match in result.matches:
            self.assertLess(match.score, 0.6)

    def test_simulate_reflects_added_skills(self):
        resume = RESUMES[0]["text"]
        result = PIPELINE.recommend(resume, top_k=12)
        near = [m for m in result.matches if not m.qualified and m.counterfactual.skills]
        if not near:
            self.skipTest("no near-miss match available")
        match = near[0]
        added = [g.skill_uri for g in match.counterfactual.skills]
        sim = PIPELINE.simulate(resume, match.job.job_id, added)
        self.assertGreater(sim["after"]["score"], sim["before"]["score"])
        self.assertAlmostEqual(
            sim["delta"], sim["after"]["score"] - sim["before"]["score"], places=6
        )

    def test_simulate_unknown_job_raises(self):
        with self.assertRaises(KeyError):
            PIPELINE.simulate(RESUMES[0]["text"], "NOPE", [])

    def test_simulate_ignores_unknown_skill_uris(self):
        sim = PIPELINE.simulate(RESUMES[0]["text"], PIPELINE.jobs[0].job_id, ["urn:not:a:skill"])
        self.assertEqual(sim["added_skills"], [])

    def test_output_is_json_serialisable(self):
        import json

        result = PIPELINE.recommend(RESUMES[0]["text"], top_k=2)
        payload = result.to_dict()
        payload["matches"] = [PIPELINE.match_detail(m) for m in result.matches]
        self.assertIsInstance(json.dumps(payload), str)

    def test_results_are_deterministic(self):
        a = PIPELINE.recommend(RESUMES[2]["text"], top_k=5)
        b = PIPELINE.recommend(RESUMES[2]["text"], top_k=5)
        self.assertEqual(
            [(m.job.job_id, m.score) for m in a.matches],
            [(m.job.job_id, m.score) for m in b.matches],
        )


class TestInvariants(unittest.TestCase):
    """Properties that must hold for every candidate/job pair in the corpus."""

    def test_global_invariants(self):
        for resume in RESUMES:
            profile = PIPELINE.build_profile(resume["text"], resume["candidate_id"])
            held = profile.skill_uris
            for job in PIPELINE.jobs:
                with self.subTest(candidate=resume["candidate_id"], job=job.job_id):
                    evidence, uncovered = PIPELINE.kg.evidence_paths(held, job)

                    # 1. Requirements are partitioned: evidenced or gap, never both or neither.
                    total = len(job.essential_skills) + len(job.optional_skills)
                    self.assertEqual(len(evidence) + len(uncovered), total)

                    # 2. Evidence never references a skill the candidate lacks.
                    for path in evidence:
                        self.assertIn(path.candidate_skill_uri, held)

                    # 3. Strength always matches the declared relation.
                    weights = MatchWeights()
                    for path in evidence:
                        self.assertAlmostEqual(
                            path.strength, weights.for_relation(path.relation)
                        )

                    # 4. Score stays in range.
                    breakdown = score_match(evidence, job, 0.5, profile.document.seniority)
                    self.assertTrue(0.0 <= breakdown.final_score <= 1.0)

    def test_more_skills_never_lowers_structural_coverage(self):
        """Monotonicity: acquiring a required skill cannot reduce coverage."""
        job = PIPELINE.jobs[0]
        base = PIPELINE.extractor.extract_uris(RESUMES[1]["text"])
        base_ev, _ = PIPELINE.kg.evidence_paths(base, job)
        base_cov = coverage(base_ev, job.essential_skills, "essential")

        for extra in job.essential_skills:
            with self.subTest(extra=extra):
                more_ev, _ = PIPELINE.kg.evidence_paths(base + [extra], job)
                more_cov = coverage(more_ev, job.essential_skills, "essential")
                self.assertGreaterEqual(more_cov + 1e-9, base_cov)


class TestConfig(unittest.TestCase):
    def test_settings_serialise(self):
        self.assertIn("embedding_provider", Settings().to_dict())

    def test_provider_summary(self):
        self.assertEqual(
            set(Settings().describe_providers()), {"embeddings", "graph", "explainer"}
        )


class TestData(unittest.TestCase):
    def test_jobs_reference_known_skills(self):
        for job in load_jobs():
            for uri in list(job.essential_skills) + list(job.optional_skills):
                self.assertIn(uri, TAXONOMY, f"{job.job_id} references unknown {uri}")

    def test_no_essential_optional_overlap(self):
        for job in load_jobs():
            self.assertFalse(set(job.essential_skills) & set(job.optional_skills))

    def test_job_ids_unique(self):
        ids = [j.job_id for j in load_jobs()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_samples_are_parseable(self):
        for resume in RESUMES:
            self.assertTrue(PIPELINE.extractor.extract_uris(resume["text"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
