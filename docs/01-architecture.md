# 01 — Architecture

## The problem this system addresses

Existing job recommenders fall into two families, and neither does what a job
seeker actually needs.

**Knowledge-graph recommenders** produce structurally interpretable output —
you can point at a path in a graph — but they cannot articulate it in language,
and the closest published system (XAIRec, Tiwary et al. 2025) is evaluated on
e-commerce rather than jobs.

**LLM-based recruiters** produce fluent language but have no grounding
structure, so nothing prevents the model asserting a qualification the
candidate does not have. In hiring, an unverified explanation is a liability.

SkillPath-XR occupies the gap between them: an LLM *and* an ontology-grounded
skill knowledge graph *and* a mechanical faithfulness check, producing
explanations that are both readable and verifiable — plus the thing neither
family provides, an **actionable counterfactual**: what to acquire to qualify.

## Stage pipeline

```
uploaded file (PDF/DOCX/ODT/RTF/TXT) ──▶ parsing/documents_io.py ──┐
                                                                   │
resume text ◀──────────────────────────────────────────────────────┘
   │
   ▼ parsing/document.py        sections, seniority, years, location
   ▼ extraction/skills.py       skill spans → taxonomy URIs (longest match,
   │                            negation-aware, exact source offsets)
   ▼ ontology/esco.py           normalisation against ESCO/O*NET-shaped taxonomy
   │
   ├─▶ kg/graph.py              skill KG: skills, jobs, candidates, typed edges
   │      └── evidence_paths()  graded connections (exact/narrower/broader/related)
   │
   ├─▶ embeddings/providers.py  candidate & job vectors
   │      └── retrieval/vector_index.py
   │
   ▼ retrieval/hybrid.py        graph retrieval ⊕ dense retrieval, fused by RRF
   ▼ ranking/scorer.py          graded coverage + semantic + seniority → score
   ▼ explain/generator.py       explanation built ONLY from the evidence subgraph
   ▼ explain/validator.py       ← the faithfulness check; can and does reject
   ▼ explain/counterfactual.py  minimal skill set that flips the decision
   ▼ scoring/trust.py           confidence × evidence density × faithfulness
   │
   ▼ ranked matches, each with a full audit trail
```

## Why each choice was made

| Decision | Reason |
|---|---|
| **RRF for fusion** | Graph overlap counts and cosine similarities are not on the same scale. Normalising them against each other makes the blend depend on the corpus's score distribution. RRF needs only rank order, so it is stable across corpora. |
| **Graded relations, not binary** | Holding PyTorch is real evidence for a deep-learning role. Binary matching throws that away; graded coverage keeps it while ranking it below an exact match. |
| **Evidence as first-class objects** | Each `EvidencePath` has a stable id (`E1`, `E2`…), so an explanation sentence can cite it and the interface can highlight that exact edge. This is what makes the explanation auditable rather than merely plausible. |
| **Template generator as default** | It is faithful by construction — it never has access to any text but the evidence. That gives a guaranteed-correct baseline and a known-good input for testing the validator. |
| **Validator runs on all providers** | Template and LLM output are measured on identical terms, so the comparison between them is meaningful. |
| **Adapters with offline fallbacks** | The system runs with zero installs and degrades in a documented way rather than refusing to start when a model download fails. |

## Module map

| Path | Responsibility |
|---|---|
| `skillpath/config.py` | All tunable weights and provider selection, env-overridable |
| `skillpath/types.py` | Dataclass contract between modules; all JSON-serialisable |
| `skillpath/ontology/esco.py` | Taxonomy, alias index, hierarchy navigation, real-ESCO loader |
| `skillpath/parsing/document.py` | Section segmentation, seniority, experience |
| `skillpath/parsing/documents_io.py` | PDF/DOCX/ODT/RTF/TXT → plain text |
| `skillpath/extraction/skills.py` | Span extraction and URI linking |
| `skillpath/kg/graph.py` | Skill knowledge graph and evidence-path discovery |
| `skillpath/embeddings/providers.py` | Hashing (offline) and sentence-transformers embedders |
| `skillpath/retrieval/` | Vector index, graph retriever, RRF hybrid |
| `skillpath/ranking/scorer.py` | Graded coverage scoring with breakdown |
| `skillpath/explain/` | Generation, **validation**, counterfactual search |
| `skillpath/scoring/trust.py` | Composite trust score |
| `skillpath/evaluation/` | Metrics and the baseline/ablation harness |
| `skillpath/pipeline.py` | Orchestration |
| `server/multipart.py` | RFC 7578 subset for file uploads (`cgi` is removed in 3.13) |
| `server/` | stdlib HTTP server (default) and FastAPI app (production) |
| `ui/index.html` | Single-file interface, no CDN, no build step |
