# SkillPath-XR Documentation

| Document | Contents |
|---|---|
| [01 — Architecture](01-architecture.md) | Problem, stage pipeline, design rationale, module map |
| [02 — Data and Ontology](02-data-and-ontology.md) | Seed dataset, loading real ESCO, adding jobs, datasets for the report |
| [03 — Pipeline and Scoring](03-pipeline.md) | Extraction, evidence grading, retrieval, scoring, counterfactual search |
| [04 — Explainability](04-explainability.md) | Generation, the faithfulness validator, degradation policy, trust |
| [05 — Evaluation](05-evaluation.md) | Results, metrics, baselines, **what the numbers do and do not show** |
| [06 — API Reference](06-api.md) | Every route, payload and limit |
| [07 — Deployment](07-deployment.md) | Local, environment variables, FastAPI, Docker, security notes |
| [08 — Decisions and Defects](08-decisions-log.md) | Bugs found and fixed, design rationale, known limitations |

Start with 01, then 04 — the faithfulness validator is the part of this system
that constitutes the research contribution.
