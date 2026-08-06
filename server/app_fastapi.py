"""
SkillPath-XR — FastAPI application (production deployment).

Exposes exactly the same routes and payloads as server/stdlib_server.py, so the
interface works against either without modification.  Use this one when you
want ASGI, multiple workers, or the auto-generated OpenAPI schema at /docs.

    pip install fastapi "uvicorn[standard]" pydantic
    uvicorn server.app_fastapi:app --host 0.0.0.0 --port 8000 --workers 4

The stdlib server is the default because it needs no installs; this module is
imported only when you run it, so the package has no hard FastAPI dependency.

NOTE: this file could not be executed in the authoring environment (FastAPI was
not installable there). The handler logic is shared with the stdlib server via
skillpath.pipeline, which is fully covered by the test suite; verify this
module in your own environment before relying on it in production.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillpath.config import SETTINGS  # noqa: E402
from skillpath.parsing.documents_io import (  # noqa: E402
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    extract_text,
)
from skillpath.pipeline import SkillPathPipeline, load_sample_resumes  # noqa: E402

UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
MAX_RESUME_CHARS = 60_000

app = FastAPI(
    title="SkillPath-XR",
    version="1.0.0",
    description="Explainable job recommendation over an ontology-grounded skill knowledge graph.",
)

_pipeline: Optional[SkillPathPipeline] = None


def get_pipeline() -> SkillPathPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SkillPathPipeline()
    return _pipeline


@app.on_event("startup")
def _warm() -> None:
    """Build the pipeline at start-up so the first request is not slow."""
    get_pipeline()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    resume_text: str = Field(..., description="Plain-text résumé")
    top_k: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|graph|dense)$")
    candidate_id: str = "C000"
    explain: bool = True


class MatchRequest(BaseModel):
    resume_text: str
    job_id: str


class SimulateRequest(BaseModel):
    resume_text: str
    job_id: str
    added_skills: List[str] = Field(default_factory=list)


def _clean_resume(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Resume text is empty. Paste a resume or pick a sample.")
    if len(text) > MAX_RESUME_CHARS:
        raise HTTPException(413, f"Resume exceeds {MAX_RESUME_CHARS:,} characters.")
    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "providers": get_pipeline().stats()["providers"],
        "thresholds": {
            "qualification": SETTINGS.qualification_threshold,
            "min_grounding_rate": SETTINGS.min_grounding_rate,
        },
    }


@app.get("/api/stats")
def stats() -> Dict[str, object]:
    return get_pipeline().stats()


@app.get("/api/samples")
def samples() -> Dict[str, object]:
    return {"samples": load_sample_resumes()}


@app.get("/api/skills")
def skills(q: str = "", limit: int = 50) -> Dict[str, object]:
    pipeline = get_pipeline()
    term = q.strip().lower()
    limit = max(1, min(200, limit))
    results = [
        {"uri": uri, "label": c.label, "skill_type": c.skill_type}
        for uri, c in pipeline.taxonomy.skills.items()
        if not term or term in " ".join([c.label] + list(c.alt_labels)).lower()
    ]
    results.sort(key=lambda r: (not r["label"].lower().startswith(term), r["label"]))
    return {"query": term, "count": len(results), "skills": results[:limit]}


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> Dict[str, object]:
    pipeline = get_pipeline()
    result = pipeline.recommend(
        _clean_resume(req.resume_text),
        candidate_id=req.candidate_id,
        top_k=req.top_k,
        mode=req.mode,
        explain=req.explain,
    )
    payload = result.to_dict()
    payload["matches"] = [pipeline.match_detail(m) for m in result.matches]
    payload["threshold"] = SETTINGS.qualification_threshold
    return payload


@app.post("/api/match")
def match(req: MatchRequest) -> Dict[str, object]:
    pipeline = get_pipeline()
    job = pipeline.jobs_by_id.get(req.job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id: {req.job_id}")

    profile = pipeline.build_profile(_clean_resume(req.resume_text))
    query_vector = pipeline.embedder.encode_one(pipeline._candidate_text(profile))
    built = pipeline._build_match(
        job, profile, profile.skill_uris, query_vector, ["direct"], explain=True
    )
    return {
        "candidate": profile.to_dict(),
        "match": pipeline.match_detail(built),
        "threshold": SETTINGS.qualification_threshold,
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest) -> Dict[str, object]:
    pipeline = get_pipeline()
    try:
        return pipeline.simulate(
            _clean_resume(req.resume_text), req.job_id, req.added_skills
        )
    except KeyError:
        raise HTTPException(404, f"Unknown job_id: {req.job_id}")


@app.get("/api/upload/formats")
def upload_formats() -> Dict[str, object]:
    """Report which formats this server can actually read (PDF is optional)."""
    import shutil

    backend = None
    for module in ("pdfplumber", "pypdf"):
        try:
            __import__(module)
            backend = module
            break
        except ImportError:
            continue
    if backend is None and shutil.which("pdftotext"):
        backend = "pdftotext"

    return {
        "extensions": list(SUPPORTED_EXTENSIONS),
        "max_bytes": MAX_FILE_BYTES,
        "pdf_supported": backend is not None,
        "pdf_backend": backend,
        "always_available": [".txt", ".md", ".docx", ".odt", ".rtf"],
        "note": (
            "PDF needs one of pdfplumber, pypdf or the pdftotext binary. "
            "Everything else is read with the standard library."
        ),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, object]:
    """
    Accept a résumé file and return its extracted text.

    An unreadable file is a client problem, so it returns 422 with an
    explanation the interface can display verbatim.
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB.")

    result = extract_text(data, file.filename or "")
    if not result.ok:
        raise HTTPException(422, result.error)

    return {
        "filename": file.filename,
        "resume_text": result.text,
        "format": result.fmt,
        "method": result.method,
        "pages": result.pages,
        "chars": result.chars,
        "truncated": result.truncated,
        "warnings": result.warnings,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(UI_DIR, "index.html"))


if os.path.isdir(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@app.exception_handler(HTTPException)
def http_error(_request, exc: HTTPException) -> JSONResponse:
    """Match the stdlib server's error envelope so the UI handles both."""
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.detail, "status": exc.status_code}
    )
