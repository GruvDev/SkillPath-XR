# SkillPath-XR

**Explainable job recommendation over an ontology-grounded skill knowledge graph, where every sentence of every explanation is traceable to a graph edge.**

Final-year engineering project implementing the research gap identified in
*"Explainable Job Recommendation System Using Large Language Models and Skill
Knowledge Graphs."*

---

## Quickstart

```bash
python run.py
```

Open <http://127.0.0.1:8000>. No install step, no API key, no model download,
no database. Requires Python 3.9+ with `numpy` and `networkx`.

**Upload a resume** (drag and drop, or browse), paste text, or pick a sample.
DOCX, ODT, RTF and TXT are read with the standard library — nothing to install.
PDF needs one of `pdfplumber`, `pypdf`, or the `pdftotext` binary; the interface
checks and tells you if it is unavailable rather than failing after the
upload.

```bash
python run.py --demo     # terminal demo, no browser
python run.py --test     # 96 tests
python run.py --eval     # evaluation harness with baselines and ablations
```

---

## What this does that existing systems do not

| | KG recommenders<br>(KGIMCS, JobEdKG, XAIRec) | LLM recruiters<br>(LLM-GAN, Zhu et al.) | GraphRAG XAI<br>(G-Refer, XRec) | **SkillPath-XR** |
|---|---|---|---|---|
| Job/skill domain | yes | yes | no (e-commerce) | **yes** |
| LLM generation | no | yes | yes | **yes (optional)** |
| Skill knowledge graph | yes | no | partial | **yes** |
| Ontology grounding (ESCO) | no | no | no | **yes** |
| Hybrid graph + dense retrieval | no | no | yes | **yes** |
| **Faithfulness validator** | n/a | absent | decoding constraint | **explicit check + metric** |
| **Actionable skill-gap counterfactual** | no | no | no | **yes, verified** |

The two bold rows are the contribution. Everything else exists somewhere in the
literature; this is the first system to put them together for job
recommendation.

---

## How it works

```
resume → parse → extract skills → normalise to ESCO-shaped taxonomy
       → hybrid retrieval (graph ⊕ dense, fused by RRF)
       → graded scoring over evidence paths
       → explanation generated ONLY from the evidence subgraph
       → faithfulness validation        ← rejects ungrounded claims
       → counterfactual skill-gap search ← verified by re-scoring
       → trust score
```

**Graded evidence.** A required skill is matched not just exactly but through
the taxonomy: holding PyTorch is real evidence for a deep-learning role
(`narrower`, 0.80), holding Docker is weak evidence for Kubernetes (`related`,
0.45). Binary matching throws this away.

**Faithfulness is measured, not assumed.** The validator runs the same skill
extractor over each generated sentence and checks every named skill and
asserted relation against the retrieved subgraph. It rejects hallucinated
entities, fabricated relations, and flags only the bad sentence in a half-true
explanation. Adversarial tests prove it fails when it should — a validator that
never rejects would report a perfect score and mean nothing.

**Counterfactuals are verified.** Greedy search plus a pruning pass finds a
locally minimal skill set that crosses the qualification threshold, then
re-runs the scorer with those skills applied to confirm it actually works.

---

## Resume input

Three ways in, all producing identical results (`test_every_format_yields_identical_recommendations`):

| Format | Reader | Installs needed |
|---|---|---|
| `.txt` `.md` | direct decode, encoding-sniffed | none |
| `.docx` | stdlib `zipfile` + XML — a DOCX is a ZIP of XML | none |
| `.odt` | stdlib `zipfile` + XML | none |
| `.rtf` | control-word stripper, metadata groups discarded | none |
| `.pdf` | pdfplumber → pypdf → `pdftotext`, best first | one of these |
| `.doc` | not supported; message says to save as .docx | — |

Format is detected from **magic bytes, not the file extension** — a résumé
saved as `cv.txt` that is really a PDF still parses correctly, and a
mislabelled file cannot silently feed binary noise to the skill extractor.

## The interface

Two elements carry the research claim:

**Citation-linked explanations.** Every sentence carries `[E1] [E2]` chips.
Hover a sentence and the exact graph edges backing it light up; hover a
citation and only that edge lights. A grounding meter shows what fraction of
sentences verified.

**Live counterfactual simulator.** Tick a missing skill and the match score,
evidence graph and gap list recompute against the real scorer — not an
animation. A threshold marker on the track shows the moment a near miss becomes
a qualified match.

Colour encodes truth-status rather than decoration: teal is evidenced, amber is
a gap. Data identifiers are set in monospace throughout, because in this domain
they are citations.

---

## Configuration

Runs offline by default; every production provider is one environment variable
away, with automatic fallback if unavailable.

```bash
export SKILLPATH_EMBEDDINGS=sentence_transformers   # real dense embeddings
export SKILLPATH_EXPLAINER=ollama                   # real LLM explanations
export SKILLPATH_ESCO_SKILLS_CSV=/path/skills_en.csv # real ESCO taxonomy
```

Full list in [docs/07-deployment.md](docs/07-deployment.md).

---

## Layout

```
skillpath/          core package
  ontology/         taxonomy + real-ESCO loader
  parsing/          resume and job parsing
  extraction/       skill spans → taxonomy URIs
  kg/               skill knowledge graph, evidence paths
  embeddings/       hashing (offline) + sentence-transformers
  retrieval/        vector index, graph retriever, RRF fusion
  ranking/          graded coverage scoring
  explain/          generation, VALIDATION, counterfactual
  scoring/          trust
  evaluation/       metrics + baseline/ablation harness
  data/             seed taxonomy, jobs, resumes (+ generator)
server/             stdlib server (default), FastAPI app (production)
ui/index.html       single-file interface, no CDN, no build
tests/              96 tests
docs/               8 documents
```

---

## Status and limitations

**Verified working:** all 96 tests pass; the evaluation harness runs; every API
endpoint returns correct payloads and status codes; the counterfactual
simulator performs real recomputation.

**Honest limitations — read before writing these results up:**

- **The ranking comparison is not discriminating.** With 24 jobs, lexical,
  dense, graph and hybrid all score alike because retrieving 10 captures
  nearly everything relevant. This is a corpus-size artefact, not evidence the
  retrievers are equivalent. **Scale to thousands of postings before quoting
  any retrieval figure.**
- **Relevance labels are derived from the taxonomy**, so they measure internal
  consistency rather than agreement with recruiters. Human annotation is the
  correct next step.
- The explanation and counterfactual metrics *are* meaningful now — they do not
  depend on corpus size.
- `server/app_fastapi.py` was never executed in the authoring environment
  (FastAPI was not installable there). Verify locally before relying on it.
- Extraction is dictionary-based, so skills outside the taxonomy are missed.
- English only. No authentication — not internet-safe as-is.

Full list in [docs/08-decisions-log.md](docs/08-decisions-log.md), along with
the four defects found and fixed during development.

---

## Documentation

Start with [docs/00-index.md](docs/00-index.md). Read
[04 — Explainability](docs/04-explainability.md) first if you only read one:
the faithfulness validator is where the research contribution lives.

---

## Base paper

Extends and is compared against:

> N. Tiwary, S. A. M. Noah, and F. Fauzi, "Prioritising explainable AI-driven
> recommendations with knowledge graphs and reinforcement learning,"
> *J. King Saud Univ. – Comput. Inf. Sci.*, vol. 37, art. 156, 2025.
> doi: 10.1007/s44443-025-00173-5

Framing anchor: Tang et al., *Front. Artif. Intell.* 8:1660548, 2025.
Method anchor: Pan et al., *IEEE TKDE* 36(7), 2024.
