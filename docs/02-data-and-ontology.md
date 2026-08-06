# 02 — Data and Ontology

## What ships in the repository

`python -m skillpath.data.build_seed_data` generates three files into
`skillpath/data/`:

| File | Contents |
|---|---|
| `taxonomy.json` | 108 skill concepts, 15 occupations |
| `jobs.json` | 24 job postings across 12 companies and 4 seniority bands |
| `resumes.json` | 4 sample résumés spanning distinct career profiles |

The generator runs a `validate()` pass that fails loudly on dangling URIs,
self-referencing concepts, duplicate ids, and any skill listed as both
essential and optional for the same job. Regenerating is therefore also a
consistency check.

## Why a seed taxonomy rather than real ESCO

The real ESCO release contains roughly 13,900 skills, is distributed by the
European Commission under its own terms, and is too large to bundle. The seed
taxonomy is deliberately ESCO-*shaped*: every concept has a URI, a preferred
label, alternative labels, a `broader` hierarchy and `related` lateral links;
occupations declare `essential` and `optional` skills.

Because the shape matches, **swapping in the real taxonomy is a loader change
and nothing else**.

## Loading the real ESCO taxonomy

1. Download the ESCO CSV distribution from
   `https://esco.ec.europa.eu/en/use-esco/download` (registration required;
   check the licence terms for your use).
2. Point the environment at the files:

```bash
export SKILLPATH_ESCO_SKILLS_CSV=/path/to/skills_en.csv
export SKILLPATH_ESCO_RELATIONS_CSV=/path/to/broaderRelationsSkillPillar_en.csv
export SKILLPATH_ESCO_OCCUPATIONS_CSV=/path/to/occupationSkillRelations_en.csv
python run.py
```

`load_taxonomy()` picks these up automatically. The CSV reader resolves column
names defensively (`conceptUri`/`conceptURI`/`uri`, and so on) so minor
release-to-release changes do not break it.

## Adding your own jobs

`skillpath/data/jobs.json` is a list of objects:

```json
{
  "job_id": "J025",
  "title": "Data Scientist",
  "company": "Example Ltd",
  "location": "Nagpur, IN",
  "seniority": "mid",
  "description": "Free text.",
  "occupation_uri": "urn:skillpath:occupation:data-scientist",
  "essential_skills": ["urn:skillpath:skill:python"],
  "optional_skills":  ["urn:skillpath:skill:pytorch"]
}
```

Every URI must exist in the taxonomy — `TestData.test_jobs_reference_known_skills`
enforces this.

## Datasets for the report

For the thesis evaluation, these are the public resources to cite and use:

| Purpose | Resource |
|---|---|
| Skill taxonomy | **ESCO** (EU), **O*NET** (US) |
| Labelled skill spans | **SkillSpan** (NAACL 2022), **Chinese-SkillSpan** |
| Synthetic postings | **JOBSKAPE** |
| Multilingual taxonomy encoder | **ESCOXLM-R** |
| Shared-task benchmark | **TalentCLEF 2026** |
| Résumés | Kaggle résumé datasets (check licence before redistributing) |

Do not scrape live job boards without checking their terms. ESCO + O*NET +
Kaggle + synthetic data is a clean, defensible combination for a student
project.


## Résumé file ingestion

`skillpath/parsing/documents_io.py` turns an uploaded document into plain text.

| Format | Reader | Installs |
|---|---|---|
| `.txt` `.md` | decode, with `chardet` if present | none |
| `.docx` | stdlib `zipfile` + `xml.etree` | none |
| `.odt` | stdlib `zipfile` + `xml.etree` | none |
| `.rtf` | control-word stripper | none |
| `.pdf` | pdfplumber → pypdf → `pdftotext` | one of these |
| `.doc` | unsupported, reported clearly | — |

DOCX and ODT are ZIP archives of XML, so the standard library reads them
directly — no `python-docx` dependency, which keeps the zero-install promise
for every format except PDF.

### Two things worth knowing

**Detection is by magic bytes, not extension.** People rename files and mail
clients relabel them. A PDF named `cv.txt` still parses; binary noise cannot be
silently fed to the skill extractor as if it were text.

**RTF metadata is stripped, not just its control words.** Font tables, colour
tables and `\info` groups contain text — author names, company names, generator
strings — that is *not* résumé content. Leaving it in creates false skills: the
test fixture embeds a company called "Kubernetes Consulting" and asserts the
candidate is never credited with Kubernetes.

### Enabling PDF

```bash
pip install pdfplumber      # best layout fidelity
# or: pip install pypdf
# or: apt-get install poppler-utils   (provides pdftotext)
```

`GET /api/upload/formats` reports what is actually available, and the interface
greys out PDF and says so when no backend is present.

Scanned PDFs with no text layer are detected and reported rather than returning
an empty résumé. OCR is out of scope; the message tells the user to paste the
text instead.
