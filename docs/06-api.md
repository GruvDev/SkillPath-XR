# 06 — API Reference

Base URL `http://127.0.0.1:8000`. The stdlib server and the FastAPI app expose
identical routes and payloads.

Errors use one envelope: `{"error": "message", "status": 400}`.

---

### `GET /api/health`
```json
{ "status": "ok",
  "providers": {"graph":"networkx","embedder":"hashing","generator":"grounded_template"},
  "thresholds": {"qualification": 0.62, "min_grounding_rate": 0.8} }
```

### `GET /api/stats`
Corpus and graph statistics: taxonomy size, job count, node/edge counts by type.

### `GET /api/samples`
The bundled sample résumés.

### `GET /api/skills?q=<term>&limit=<n>`
Taxonomy search backing the skill picker. Matches labels and alt labels;
prefix matches sort first.

---

### `POST /api/recommend`

```json
{ "resume_text": "...", "top_k": 6, "mode": "hybrid", "explain": true }
```

`mode` ∈ `hybrid` | `graph` | `dense`.

Returns `candidate` (with spans and linked URIs), `matches`, `timings_ms`,
`providers`, `threshold`. Each match contains:

| Field | Meaning |
|---|---|
| `score`, `qualified` | Final score and whether it clears the threshold |
| `breakdown` | The four scoring components |
| `score_arithmetic` | Component × weight → contribution rows |
| `evidence[]` | Evidence paths, each with `evidence_id`, relation, strength |
| `gaps[]` | Requirements with no supporting evidence |
| `subgraph` | `{nodes, edges}` — exactly what the interface draws |
| `explanation` | `text`, per-sentence `sentences[]` with `evidence_ids`, `validation` |
| `counterfactual` | Minimal skill set, baseline/projected score, `achieves_threshold` |
| `trust`, `trust_band` | Composite trust and its three components |

### `POST /api/match`
```json
{ "resume_text": "...", "job_id": "J005" }
```
One job in full detail, bypassing retrieval. 404 on unknown `job_id`.

### `POST /api/simulate`
```json
{ "resume_text": "...", "job_id": "J021",
  "added_skills": ["urn:skillpath:skill:rest-api"] }
```

Recomputes the match with skills added. Backs the interface's counterfactual
control — it runs the **real scorer over the real graph**, so what the user
sees is the system's actual behaviour, not an animation of it.

```json
{ "before": {"score":0.591,"evidence":[…],"gaps":[…]},
  "after":  {"score":0.741,"evidence":[…],"gaps":[…]},
  "delta": 0.151, "crossed_threshold": true }
```

Unknown skill URIs are ignored rather than erroring, so a stale client cannot
break the call.

---

### `POST /api/upload`

`multipart/form-data` with one file part. Returns the extracted text, ready to
pass to `/api/recommend`.

```
POST /api/upload
Content-Type: multipart/form-data; boundary=...

<file part named "file">
```

```json
{ "filename": "resume.pdf",
  "resume_text": "PRIYA SHARMA\nFinal-year B.E. ...",
  "format": "pdf", "method": "pdfplumber",
  "pages": 1, "chars": 891, "truncated": false, "warnings": [] }
```

| Status | Meaning |
|---|---|
| 200 | Text extracted |
| 400 | No file, bad multipart body, or wrong Content-Type |
| 413 | File over 10 MB |
| 422 | File received but unreadable — `error` explains why, verbatim-safe for display |

422 rather than 500 is deliberate: an unreadable résumé is a client-side
problem with an actionable fix, not a server fault.

### `GET /api/upload/formats`

Which formats this server can read **right now**. PDF support depends on an
optional backend, so the interface asks rather than assuming — better to grey
out PDF up front than to accept a file and fail after the user has waited.

```json
{ "extensions": [".txt",".md",".text",".pdf",".docx",".odt",".rtf"],
  "max_bytes": 10485760,
  "pdf_supported": true, "pdf_backend": "pdfplumber",
  "always_available": [".txt",".md",".docx",".odt",".rtf"] }
```

---

## Limits

| Limit | Value |
|---|---|
| JSON request body | 2 MB |
| Uploaded file | 10 MB |
| Extracted text kept | 60,000 characters |
| Résumé length | 60,000 characters |
| `top_k` | 1–50 |
| `/api/skills` limit | 1–200 |
