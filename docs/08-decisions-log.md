# 08 — Decisions and Defect Log

A record of the non-obvious choices and the bugs found during development.
Kept because a project like this is judged partly on whether its author can
account for its behaviour.

## Defects found and fixed

### D1 — Sentence-final skills silently dropped *(severity: high)*

The token pattern admits `.` so "Node.js" and ".NET" survive tokenisation. That
meant a sentence-final "Java." tokenised as `java.`, which never matched the
alias `java`. **Any skill ending a sentence was invisible to the extractor.**

Surfaced indirectly: the template generator is faithful by construction, yet
the measured grounding rate was 0.933 rather than 1.0. Tracing that gap led to
the extractor rather than the validator.

Fix: trim trailing dots in `_tokenise_with_offsets`, moving the end offset with
them, preserving internal and leading dots. Grounding went to 1.0 and evidence
coverage rose 0.827 → 0.848 because more skills now extract.
Regression: `test_sentence_final_skill_is_extracted`.

**Worth noting:** this was found by a metric disagreeing with a proof, not by a
test. A component that should be perfect by construction is a useful canary.

### D2 — Counterfactual used the wrong seniority *(severity: medium)*

`_seniority_of()` read `self._current_seniority`, an attribute never set
anywhere, so it always returned `"unspecified"`. The counterfactual engine
therefore scored against a neutral seniority fit while the recommendation used
the real one — the two could disagree.

Fix: deleted the helper and threaded `candidate_seniority` explicitly through
`_attach_explanation`.

### D3 — RTF metadata leaked into résumé text *(severity: medium)*

The first RTF stripper removed control *words* but kept the *contents* of
destination groups. Font tables, colour tables and `\info` blocks therefore
appeared as body text — so "Calibri" showed up in the extracted résumé.

This is worse than cosmetic. Those groups hold author names, company names and
generator strings; a résumé produced at a company called "Kubernetes
Consulting" would have credited the candidate with Kubernetes, and the
faithfulness validator would have passed it, because the evidence graph really
would contain that skill. **A bad extractor produces confidently wrong
explanations, not obviously wrong ones.**

Fix: `_strip_rtf_destinations()` removes known destination groups and their
contents with brace-depth tracking, plus any `\*`-prefixed group.
Regression: `test_rtf_metadata_never_becomes_a_skill`.

*(Found while inspecting output that already scored 100% skill recall — the
recall metric could not see it, because the leak added a skill rather than
removing one.)*

### D4 — Test fixture, not code *(worth recording)*

RTF extraction measured 86% recall against the other formats' 100%. The cause
was the fixture generator running `.replace('\\','')` *after* inserting `\par`,
stripping the backslashes it had just added. The extractor was correct.

Recorded because the instinct was to patch the extractor. Verifying the fixture
first saved a wrong "fix" to working code.

### D5 — Plural agreement in generated text *(severity: low)*

"The optional requirement Docker and microservices architecture **is** also
unevidenced." Fixed with count-aware noun/verb selection.

### D6 — Repeated sentence frames *(severity: low)*

The generator emitted one "Partial evidence comes from…" sentence per relation
type, so an explanation could open three sentences the same way. Merged into a
single clause list.

## Design decisions

### Why RRF rather than weighted score blending
Graph overlap counts and cosine similarities are not on the same scale.
Normalising them against each other makes the blend depend on the score
distribution of whatever corpus is loaded. RRF uses only rank order.

### Why the template generator is the default
It is faithful by construction — it never has access to text outside the
evidence. That provides a guaranteed-correct baseline and a known-good input
against which to test the validator. The LLM is opt-in.

### Why the validator whitelists gap skills
An explanation must be able to say "no evidence was found for X". Without the
whitelist the validator would reject exactly the honest statements users most
need.

### Why local rather than global minimality
Minimum-set-cover is intractable in general. Greedy + pruning gives a set where
no single member can be removed. The code, the docs and the interface all say
"locally minimal" rather than "minimal".

### Why offline fallbacks everywhere
The system was authored in an environment with no network access, which forced
every external dependency behind an adapter. That constraint turned out to be
good engineering: the project runs on a clean Python install, the test suite is
reproducible, and a failed model download degrades the system in a documented
way instead of preventing start-up.

### Why marginal gains do not sum
`SkillGap.score_gain` is measured against the baseline **one skill at a time**.
Gains are not additive — adding a parent skill can partially cover several
requirements at once — so the individual figures will not sum to the total
projected gain. The interface shows both and does not imply otherwise.

## Known limitations

| Limitation | Consequence | Fix |
|---|---|---|
| 24-job corpus | Retrieval ablations do not discriminate | Scale to thousands |
| Derived relevance labels | Measures internal consistency, not recruiter agreement | Human annotation |
| Dictionary-based extraction | Misses skills absent from the taxonomy | Add a neural span tagger (SkillSpan) as a second extractor |
| Seed taxonomy, not real ESCO | Limited vocabulary | Set `SKILLPATH_ESCO_SKILLS_CSV` |
| No authentication | Not internet-safe | See docs/07 |
| FastAPI app unexecuted | Unverified in authoring environment | Run it locally before relying on it |
| English only | No multilingual support | ESCOXLM-R for cross-lingual linking |
| No OCR | Scanned PDFs are detected but unreadable | Tesseract, if scans matter |
| `.doc` unsupported | Legacy Word files rejected with a message | Convert to .docx |
