# 04 — Explainability and Faithfulness

This is the module the project is built around.

## Generation is constrained, not trusted

Two providers implement one interface.

**`GroundedTemplateGenerator` (default)** composes sentences only from fields of
`EvidencePath` objects. It cannot mention a skill outside the evidence because
it never has access to any other text — faithful *by construction*. That gives
a guaranteed-correct baseline and, importantly, a known-good input for testing
the validator.

**`LLMGenerator`** sends the evidence subgraph as JSON to an Ollama or
OpenAI-compatible endpoint under a strict evidence-only instruction. Fluency
improves; faithfulness stops being guaranteed — which is exactly why the
validator exists.

## The faithfulness validator

1. Split the explanation into sentences.
2. Run the **same skill extractor used on résumés** over each sentence, so a
   claim about a skill is detected identically wherever it appears.
3. A sentence is **grounded** when every skill it names is in the evidence
   subgraph (or in the whitelisted gap set), and any relation it asserts
   between two skills exists as an edge.
4. Report `grounding_rate = grounded_sentences / total_sentences` and list
   every unsupported claim.

**Gap whitelisting.** Skills identified as *missing* are permitted to be named.
Without this the validator would reject honest statements about what the
candidate lacks — the content users most need.

### Why this is a real check and not a rubber stamp

A validator that never rejects would report a perfect grounding rate and mean
nothing. These tests assert that it **does** reject:

| Test | Input | Expected |
|---|---|---|
| `test_rejects_hallucinated_entity` | "extensive Kubernetes and Terraform experience" | flagged, rate < 1.0 |
| `test_rejects_fabricated_relation` | "Python is a kind of SQL" | `narrower` flagged as unsupported |
| `test_flags_only_the_bad_sentence` | one true sentence + one false | exactly 0.5, not 0.0 |
| `test_empty_explanation_fails` | "" | fails |
| `test_gap_skills_may_be_mentioned` | "No evidence was found for X" | passes |

Measured behaviour on the bundled corpus:

```
HONEST (template)        rate=1.00 passed=True
HALLUCINATED entity      rate=0.50 passed=False  → Kubernetes, Terraform
FABRICATED relation      rate=0.00 passed=False  → narrower
SUBTLE half-true         rate=0.50 passed=False  → Docker
EMPTY                    rate=0.00 passed=False
```

## Degradation policy

If the configured generator fails, returns empty, or produces output below
`min_grounding_rate` (default 0.80), the pipeline substitutes the template
output and records the substitution in `explanation.provider`. **The user is
never shown an unvalidated explanation, and never shown nothing.**

## The explainability stack

Mapped to the three-layer taxonomy of Tang et al. (2025):

| Layer | Mechanism here |
|---|---|
| Data | Skill spans with exact source offsets; negation flags |
| Model | KG evidence paths with typed relations and graded strength |
| Output | Natural-language explanation, per-sentence citations, counterfactual |

Also provided: rule-based coverage (`score_arithmetic`), confidence, evidence
density, and a composite trust score.

## Trust scoring

A match score says how good a fit looks; it says nothing about how much the
system should be believed. Trust keeps three signals separate so a low score
can be *diagnosed*:

- **model confidence** — distance from the decision boundary (a score sitting
  on the threshold is a coin flip and reports as such)
- **evidence density** — share of requirements with support, essentials
  weighted double
- **faithfulness** — the measured grounding rate

Weights: 0.30 / 0.30 / 0.40.
