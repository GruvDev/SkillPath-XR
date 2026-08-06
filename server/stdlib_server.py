"""
SkillPath-XR HTTP server (standard library only).

Runs with no third-party packages so the project starts on a clean Python
install.  A FastAPI equivalent is provided in server/app_fastapi.py for
production deployments that want ASGI, workers and auto-generated schema docs;
both expose exactly the same routes and payloads.

Routes
------
GET  /                      the interface
GET  /api/health            liveness and provider status
GET  /api/stats             corpus and graph statistics
GET  /api/samples           bundled sample résumés
GET  /api/skills?q=         taxonomy search, for the skill picker
POST /api/recommend         {resume_text, top_k?, mode?} -> ranked matches
POST /api/match             {resume_text, job_id} -> one match in full detail
POST /api/simulate          {resume_text, job_id, added_skills[]} -> before/after
POST /api/upload            multipart file upload -> extracted resume text
GET  /api/upload/formats    which formats this server can read right now
"""

from __future__ import annotations

import json
import logging
import os
import sys
import shutil
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.multipart import MultipartError, parse_multipart  # noqa: E402
from skillpath.config import SETTINGS  # noqa: E402
from skillpath.parsing.documents_io import (  # noqa: E402
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    extract_text,
)
from skillpath.pipeline import SkillPathPipeline, load_sample_resumes  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("skillpath.server")

UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
MAX_BODY_BYTES = 2 * 1024 * 1024        # JSON bodies
MAX_UPLOAD_BYTES = MAX_FILE_BYTES + (1024 * 1024)  # file + multipart overhead
MAX_RESUME_CHARS = 60_000

# Built once at start-up and shared across request threads.  Every pipeline
# method used by a request is read-only with respect to pipeline state, so a
# lock is only needed around lazy mutation, of which there is none.
_pipeline: Optional[SkillPathPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> SkillPathPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                logger.info("building pipeline ...")
                _pipeline = SkillPathPipeline()
                stats = _pipeline.stats()
                logger.info(
                    "ready: %s skills, %s jobs, providers=%s",
                    stats["taxonomy_skills"],
                    stats["jobs"],
                    stats["providers"],
                )
    return _pipeline


class ApiError(Exception):
    """Raised for client errors; carries the HTTP status to return."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _require(body: Dict[str, Any], field: str, kind: type) -> Any:
    if field not in body or body[field] is None:
        raise ApiError(f"'{field}' is required")
    value = body[field]
    if not isinstance(value, kind):
        raise ApiError(f"'{field}' must be {kind.__name__}, got {type(value).__name__}")
    return value


def _clean_resume(text: str) -> str:
    text = text.strip()
    if not text:
        raise ApiError("Resume text is empty. Paste a resume or pick a sample.")
    if len(text) > MAX_RESUME_CHARS:
        raise ApiError(f"Resume exceeds {MAX_RESUME_CHARS:,} characters.", status=413)
    return text


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_health(_body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "providers": pipeline.stats()["providers"],
        "thresholds": {
            "qualification": SETTINGS.qualification_threshold,
            "min_grounding_rate": SETTINGS.min_grounding_rate,
        },
    }


def handle_stats(_body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    return get_pipeline().stats()


def handle_samples(_body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    return {"samples": load_sample_resumes()}


def handle_skills(_body: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    """Taxonomy search backing the skill picker in the interface."""
    pipeline = get_pipeline()
    term = (query.get("q") or "").strip().lower()
    try:
        limit = max(1, min(200, int(query.get("limit", 50))))
    except (TypeError, ValueError):
        limit = 50

    results = []
    for uri, concept in pipeline.taxonomy.skills.items():
        haystack = " ".join([concept.label] + list(concept.alt_labels)).lower()
        if not term or term in haystack:
            results.append(
                {"uri": uri, "label": concept.label, "skill_type": concept.skill_type}
            )
    results.sort(key=lambda r: (not r["label"].lower().startswith(term), r["label"]))
    return {"query": term, "count": len(results), "skills": results[:limit]}


def handle_recommend(body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_pipeline()
    resume_text = _clean_resume(_require(body, "resume_text", str))

    top_k = body.get("top_k", SETTINGS.top_k)
    try:
        top_k = max(1, min(50, int(top_k)))
    except (TypeError, ValueError):
        raise ApiError("'top_k' must be an integer")

    mode = body.get("mode", "hybrid")
    if mode not in ("hybrid", "graph", "dense"):
        raise ApiError("'mode' must be one of: hybrid, graph, dense")

    result = pipeline.recommend(
        resume_text,
        candidate_id=str(body.get("candidate_id", "C000")),
        top_k=top_k,
        mode=mode,
        explain=bool(body.get("explain", True)),
    )

    payload = result.to_dict()
    payload["matches"] = [pipeline.match_detail(m) for m in result.matches]
    payload["threshold"] = SETTINGS.qualification_threshold
    return payload


def handle_match(body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_pipeline()
    resume_text = _clean_resume(_require(body, "resume_text", str))
    job_id = _require(body, "job_id", str)

    job = pipeline.jobs_by_id.get(job_id)
    if job is None:
        raise ApiError(f"Unknown job_id: {job_id}", status=404)

    profile = pipeline.build_profile(resume_text)
    held = profile.skill_uris
    query_vector = pipeline.embedder.encode_one(pipeline._candidate_text(profile))
    match = pipeline._build_match(
        job, profile, held, query_vector, ["direct"], explain=True
    )
    return {
        "candidate": profile.to_dict(),
        "match": pipeline.match_detail(match),
        "threshold": SETTINGS.qualification_threshold,
    }


def handle_simulate(body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_pipeline()
    resume_text = _clean_resume(_require(body, "resume_text", str))
    job_id = _require(body, "job_id", str)
    added = body.get("added_skills", [])
    if not isinstance(added, list):
        raise ApiError("'added_skills' must be a list of skill URIs")
    added = [str(u) for u in added]

    try:
        return pipeline.simulate(resume_text, job_id, added)
    except KeyError:
        raise ApiError(f"Unknown job_id: {job_id}", status=404)


def handle_upload_formats(_body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Report which formats this server can actually read.

    PDF support depends on an optional backend, so the interface asks rather
    than assuming — better to grey out PDF up front than to accept a file and
    fail after the user has waited.
    """
    pdf_backend = None
    for module in ("pdfplumber", "pypdf"):
        try:
            __import__(module)
            pdf_backend = module
            break
        except ImportError:
            continue
    if pdf_backend is None and shutil.which("pdftotext"):
        pdf_backend = "pdftotext"

    return {
        "extensions": list(SUPPORTED_EXTENSIONS),
        "max_bytes": MAX_FILE_BYTES,
        "pdf_supported": pdf_backend is not None,
        "pdf_backend": pdf_backend,
        "always_available": [".txt", ".md", ".docx", ".odt", ".rtf"],
        "note": (
            "PDF needs one of pdfplumber, pypdf or the pdftotext binary. "
            "Everything else is read with the standard library."
        ),
    }


def handle_upload(_body: Dict[str, Any], _query: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder — upload is handled directly in the request handler."""
    raise ApiError("Upload must be sent as multipart/form-data.", status=415)


ROUTES: Dict[Tuple[str, str], Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    ("GET", "/api/health"): handle_health,
    ("GET", "/api/stats"): handle_stats,
    ("GET", "/api/samples"): handle_samples,
    ("GET", "/api/skills"): handle_skills,
    ("GET", "/api/upload/formats"): handle_upload_formats,
    ("POST", "/api/recommend"): handle_recommend,
    ("POST", "/api/match"): handle_match,
    ("POST", "/api/simulate"): handle_simulate,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SkillPathXR/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        if isinstance(payload, (dict, list)):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        elif isinstance(payload, bytes):
            data = payload
        else:
            data = str(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected before response completed")

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message, "status": status})

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError("Invalid Content-Length header")
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ApiError("Request body too large", status=413)

        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(f"Invalid JSON body: {exc}")
        if not isinstance(parsed, dict):
            raise ApiError("Request body must be a JSON object")
        return parsed

    # -- static --------------------------------------------------------------

    def _serve_ui(self, path: str) -> None:
        filename = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        # Contain path traversal: resolve and confirm the result is inside UI_DIR.
        full = os.path.realpath(os.path.join(UI_DIR, filename))
        if not full.startswith(os.path.realpath(UI_DIR)) or not os.path.isfile(full):
            self._error(404, "Not found")
            return

        types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".json": "application/json",
        }
        ext = os.path.splitext(full)[1]
        with open(full, "rb") as f:
            self._send(200, f.read(), types.get(ext, "application/octet-stream"))

    # -- verbs ---------------------------------------------------------------

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        if method == "GET" and not path.startswith("/api"):
            self._serve_ui(path)
            return

        if method == "POST" and path == "/api/upload":
            self._handle_upload_request()
            return

        handler = ROUTES.get((method, path))
        if handler is None:
            self._error(404, f"No route for {method} {path}")
            return

        try:
            body = self._read_json() if method == "POST" else {}
            self._send(200, handler(body, query))
        except ApiError as exc:
            self._error(exc.status, exc.message)
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to a client
            logger.error("unhandled error on %s %s: %s", method, path, exc)
            logger.debug(traceback.format_exc())
            self._error(500, "Internal server error. Check the server log for details.")

    def _handle_upload_request(self) -> None:
        """
        Accept a résumé file and return its extracted text.

        Returns 200 with the text on success. A file that cannot be read is a
        client problem, not a server fault, so it returns 422 with an
        explanation the interface can show verbatim — never a stack trace.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "Invalid Content-Length header.")
            return

        if length <= 0:
            self._error(400, "No file was uploaded.")
            return
        if length > MAX_UPLOAD_BYTES:
            self._error(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            return

        raw = self.rfile.read(length)

        try:
            form = parse_multipart(raw, self.headers.get("Content-Type", ""))
        except MultipartError as exc:
            self._error(400, str(exc))
            return

        part = form.first_file()
        if part is None or not part.data:
            self._error(400, "The request contained no file. Choose a resume to upload.")
            return

        result = extract_text(part.data, part.filename or "")

        if not result.ok:
            self._send(422, {
                "error": result.error,
                "status": 422,
                "filename": part.filename,
                "format": result.fmt,
                "warnings": result.warnings,
            })
            return

        logger.info(
            "extracted %s chars from %s (%s via %s)",
            result.chars, part.filename, result.fmt, result.method,
        )
        self._send(200, {
            "filename": part.filename,
            "resume_text": result.text,
            "format": result.fmt,
            "method": result.method,
            "pages": result.pages,
            "chars": result.chars,
            "truncated": result.truncated,
            "warnings": result.warnings,
        })

    def do_GET(self) -> None:      # noqa: N802 - required name
        self._dispatch("GET")

    def do_POST(self) -> None:     # noqa: N802
        self._dispatch("POST")


def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    host = host or SETTINGS.host
    port = port if port is not None else SETTINGS.port

    get_pipeline()  # build before binding so start-up errors surface immediately

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    logger.info("SkillPath-XR listening on http://%s:%s", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
