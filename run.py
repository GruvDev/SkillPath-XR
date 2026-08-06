#!/usr/bin/env python3
"""
SkillPath-XR entry point.

    python run.py                 start the server on http://127.0.0.1:8000
    python run.py --port 9000     start on another port
    python run.py --eval          run the evaluation harness and print results
    python run.py --test          run the test suite
    python run.py --demo          run one recommendation in the terminal
    python run.py --build-data    regenerate the bundled seed dataset
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="skillpath-xr", description="Explainable job recommendation over a skill knowledge graph."
    )
    parser.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (default 8000)")
    parser.add_argument("--eval", action="store_true", help="run the evaluation harness")
    parser.add_argument("--test", action="store_true", help="run the test suite")
    parser.add_argument("--demo", action="store_true", help="print one recommendation")
    parser.add_argument("--build-data", action="store_true", help="regenerate seed data")
    args = parser.parse_args()

    if args.build_data:
        from skillpath.data.build_seed_data import main as build
        build()
        return 0

    if args.test:
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover("tests")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.eval:
        from skillpath.evaluation.harness import format_results, run_evaluation
        print(format_results(run_evaluation()))
        return 0

    if args.demo:
        from skillpath.pipeline import SkillPathPipeline, load_sample_resumes
        pipeline = SkillPathPipeline()
        sample = load_sample_resumes()[0]
        print(f"Candidate: {sample['name']}\n" + "=" * 72)
        result = pipeline.recommend(sample["text"], sample["candidate_id"], top_k=3)
        for match in result.matches:
            flag = "QUALIFIED" if match.qualified else "NEAR MISS"
            print(f"\n[{flag}] {match.job.title} — {match.job.company}   score {match.score:.3f}")
            print(f"  {match.explanation.text}")
            validation = match.explanation.validation
            print(f"  grounding {validation.grounding_rate:.0%} "
                  f"({validation.grounded_sentences}/{validation.total_sentences} sentences verified)")
            cf = match.counterfactual
            if cf and cf.skills:
                names = ", ".join(g.skill_label for g in cf.skills)
                print(f"  to qualify: {names}  ->  {cf.projected_score:.3f}")
        return 0

    from server.stdlib_server import serve
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
