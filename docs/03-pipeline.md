# 03 — Pipeline and Scoring

## Skill extraction

Three behaviours are load-bearing for correctness.

**Longest-match resolution.** Candidate spans are sorted by length descending
and accepted greedily if they do not overlap an already-accepted span. Without
this, "machine learning engineer" would report both "machine learning" and a
spurious shorter match.

**Negation handling.** A cue list (`no experience with`, `never used`, `yet to
learn`, …) is scanned in a 60-character window before each mention. A sentence
boundary between the cue and the mention cancels the negation, so "No
experience with Kubernetes. I use Docker daily." correctly credits Docker.
Negated spans are *kept* and flagged rather than dropped, so the interface can
show what was considered and rejected.

**Offset fidelity.** Every span carries exact character offsets satisfying
`text[span.start:span.end] == span.surface`. This is what lets the interface
highlight the literal résumé text behind a piece of evidence.

> **Fixed defect worth knowing about.** The token pattern admits `.` so that
> "Node.js" survives tokenisation. That meant a sentence-final "Java." became
> the token `java.` and never matched the alias `java` — silently dropping any
> skill that ended a sentence. Trailing dots are now trimmed (with the end
> offset moved with them) while internal dots are preserved. Regression test:
> `test_sentence_final_skill_is_extracted`.

## Evidence path grading

For each required skill, the strongest connection from any held skill is found:

| Relation | Meaning | Strength |
|---|---|---|
| `exact` | Candidate holds the required skill itself | 1.00 |
| `narrower` | Candidate holds a specialisation (PyTorch → deep learning) | 0.80 |
| `broader` | Candidate holds the general parent (machine learning → PyTorch) | 0.60 |
| `related` | Candidate holds an adjacent skill (Docker → Kubernetes) | 0.45 |
| `none` | No defensible connection | 0.00 |

Hierarchy traversal is capped at two hops. Beyond that the semantic link is too
weak to defend to a user as evidence, and precision matters more than recall
when the output is a claim about a person.

**Partition invariant.** Every requirement ends up either in `evidence` or in
`uncovered` — never both, never neither. `TestKnowledgeGraph.test_evidence_and_gaps_partition_requirements`
checks this for every candidate/job pair in the corpus, which is what proves
nothing is quietly dropped.

## Retrieval

Two retrievers, fused by Reciprocal Rank Fusion with k = 60:

- **Graph retriever** expands held skills along the taxonomy with decaying
  weights (self 1.0, ancestors 0.6, descendants 0.5, related 0.45) and walks
  `requires` edges backwards to reach jobs.
- **Dense retriever** compares the candidate's profile embedding against job
  embeddings. Job text includes the *labels* of required skills, not just the
  prose, because a posting may name a skill only in a structured field.

`mode` selects the ablation — `hybrid`, `graph`, or `dense` — in the same code
path the application runs, so the evaluation measures the real system.

## Scoring

```
score = ( w_ess·essential_coverage
        + w_opt·optional_coverage      ← dropped if the job lists none
        + w_sem·semantic_similarity
        + w_sen·seniority_fit ) / Σ(applicable weights)
```

Defaults: 0.55 / 0.15 / 0.20 / 0.10.

**Weight renormalisation.** A job with no optional skills does not score zero on
that term — the weight is removed from both numerator and denominator. Scoring
it as zero would punish a posting for being concisely written. Test:
`test_weights_renormalise_when_no_optional_skills`.

**Asymmetric seniority fit.** Over-qualification is penalised at 0.12 per band,
under-qualification at 0.28 per band, because for a job seeker a stretch role
is a worse recommendation than a comfortable one. An unspecified level on
either side returns a neutral 0.6: absence of a stated level is not evidence of
a mismatch.

## Counterfactual search

Finding the truly minimum skill set is set-cover-like and intractable in
general, so:

1. **Greedy** — repeatedly add the missing skill with the largest marginal
   gain, **recomputing against the graph each time**. Gains are not additive:
   adding a parent skill can partially cover several requirements at once, so
   marginal value changes after every addition.
2. **Prune** — walk the selected set and drop any member the set still clears
   the threshold without.

The result is **locally minimal** (no single member can be removed), which is
weaker than global minimality. The code and the interface both say so rather
than overclaiming. Test: `test_selected_set_is_locally_minimal`.

Only skills the job actually asks for are ever suggested. Every result is
verified by re-running the scorer with the skills applied, so a counterfactual
that does not work cannot be reported as if it does.
