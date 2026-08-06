# 05 — Evaluation

## Running it

```bash
python run.py --eval
```

## Current results on the bundled corpus

```
RANKING QUALITY (mean over candidates)
------------------------------------------------------------------------------
Configuration    NDCG@5  NDCG@10      MRR      MAP      P@5     R@10     HR@5
------------------------------------------------------------------------------
Lexical          0.9443   0.9452   1.0000   0.8854   0.6500   0.9583   1.0000
Dense only       0.9215   0.9814   1.0000   0.9475   0.6000   1.0000   1.0000
Graph only       0.9215   0.9814   1.0000   0.9475   0.6000   1.0000   1.0000
Hybrid (ours)    0.9215   0.9814   1.0000   0.9475   0.6000   1.0000   1.0000
------------------------------------------------------------------------------

EXPLANATION QUALITY
------------------------------------------------------------------------------
Mean grounding rate                   1.0000
Explanation pass rate                 1.0000
Evidence coverage                     0.8475
Explanations evaluated                    40

COUNTERFACTUAL QUALITY
------------------------------------------------------------------------------
Validity (verified by re-scoring)     1.0000
Local minimality                      1.0000
Mean skills suggested                 1.2941
Counterfactuals evaluated                 40
------------------------------------------------------------------------------
Providers: {'embeddings': 'hashing', 'graph': 'networkx', 'explainer': 'grounded_template', 'embedder': 'hashing', 'generator': 'grounded_template'}

Note: relevance labels are derived from taxonomy coverage, so these figures measure ranking behaviour and internal consistency, not agreement with human recruiters. See docs/05-evaluation.md.
```

## What these numbers do and do not show

**Read this before quoting any figure.**

**The ranking numbers are not discriminating.** Lexical, dense, graph and hybrid
score almost identically. That is not evidence that the retrievers are
equivalent — it is a corpus-size artefact. With 24 jobs and roughly 5 relevant
per candidate, retrieving 10 captures nearly everything relevant regardless of
method, so there is no headroom in which the methods can differ. **To produce a
meaningful retrieval comparison for the report you need a corpus in the
thousands.** This is the single most important extension before writing up.

**Relevance labels are derived, not human.** A job counts as relevant when the
candidate evidences at least 60% of its essential requirements via *exact*
matches. Exact-only is deliberate — using graded evidence would define
relevance in terms of the mechanism under test. But labels derived from the
same taxonomy the system uses measure internal consistency, not agreement with
recruiters. A human-labelled set is the correct next step.

**The explanation and counterfactual numbers are meaningful now.** They do not
depend on corpus size:

- **Grounding rate 1.00** — every sentence of every explanation is traceable to
  a graph edge. Expected for the template generator (faithful by construction);
  the number becomes an interesting measurement when you enable an LLM.
- **Counterfactual validity 1.00** — verified by re-running the scorer with the
  suggested skills applied, not asserted.
- **Local minimality 1.00** — no single suggested skill can be dropped while
  still clearing the threshold.
- **Mean size 1.25 skills** — the suggestions are small enough to act on.

## Metrics implemented

`skillpath/evaluation/metrics.py`

**Ranking:** Precision@k, Recall@k, F1@k, HitRate@k, MRR, MAP, NDCG@k
(binary relevance, log2(rank+1) discount, ideal ranking capped at k).

**Explanation:** mean grounding rate, explanation pass rate, evidence coverage.

**Counterfactual:** validity (re-scored), local minimality, mean size.

## Baselines and ablations

| Configuration | What it isolates |
|---|---|
| `lexical` | TF-IDF cosine, no taxonomy, no graph — the pre-semantic baseline |
| `dense` | Ablates the graph |
| `graph` | Ablates dense retrieval |
| `hybrid` | The full system |

All four run through the same scorer, so differences are attributable to
retrieval rather than incidental scoring differences.

## What to add before submission

Ordered by impact on the write-up:

1. **Scale the corpus** to thousands of postings so retrieval ablations
   discriminate. Without this the ranking table says nothing.
2. **Human-labelled relevance** for a subset — even 50 pairs from 2–3
   annotators with an agreement statistic.
3. **Enable the LLM generator** and report grounding rate against the template
   baseline. This is where the validator earns its place in the paper.
4. **Small human study** (n ≈ 20) on usefulness, trust and actionability,
   comparing explanation-with-counterfactual against path-only and
   explanation-only conditions.
5. **Latency and cost** per explanation under each provider.
6. **Fairness audit** — do explanations differ systematically across
   demographic proxies?
