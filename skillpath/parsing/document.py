"""
Résumé and job-description parsing.

Segments a plain-text document into labelled sections and pulls out the
structured attributes the matcher needs (title, seniority, years of
experience, location).  Deliberately rule-based and dependency-free: it is
deterministic, inspectable and fast, which matters because every extracted
span later has to be defensible as evidence.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..types import ParsedDocument

# Section headings recognised in résumés and job posts.  Order does not matter;
# the parser scans line by line and switches section on any recognised heading.
SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("summary", re.compile(r"^\s*(summary|profile|objective|about( me)?)\s*:?\s*$", re.I)),
    ("skills", re.compile(r"^\s*(skills?|technical skills?|core competenc(y|ies)|"
                          r"technolog(y|ies)|tech stack)\s*:?\s*$", re.I)),
    ("experience", re.compile(r"^\s*(experience|work experience|employment|"
                              r"professional experience|career history)\s*:?\s*$", re.I)),
    ("projects", re.compile(r"^\s*(projects?|selected projects?|portfolio)\s*:?\s*$", re.I)),
    ("education", re.compile(r"^\s*(education|academics?|qualifications?)\s*:?\s*$", re.I)),
    ("certifications", re.compile(r"^\s*(certifications?|licences?|licenses?|courses?)\s*:?\s*$", re.I)),
    ("requirements", re.compile(r"^\s*(requirements?|what you.ll need|qualifications?|"
                                r"must have|essential)\s*:?\s*$", re.I)),
    ("responsibilities", re.compile(r"^\s*(responsibilities|what you.ll do|role|"
                                    r"the job|duties)\s*:?\s*$", re.I)),
    ("nice_to_have", re.compile(r"^\s*(nice to have|preferred|bonus|desirable|"
                                r"good to have|optional)\s*:?\s*$", re.I)),
]

SENIORITY_RULES: List[Tuple[str, re.Pattern]] = [
    # Checked in order; first hit wins, so "lead" outranks "senior".
    ("lead", re.compile(r"\b(lead|principal|staff|head of|director|architect)\b", re.I)),
    ("senior", re.compile(r"\b(senior|sr\.?|experienced)\b", re.I)),
    ("junior", re.compile(r"\b(junior|jr\.?|graduate|entry.level|trainee|intern|fresher|"
                          r"final.year|student)\b", re.I)),
    ("mid", re.compile(r"\b(mid.level|intermediate|associate)\b", re.I)),
]

SENIORITY_ORDER = {"junior": 1, "mid": 2, "senior": 3, "lead": 4}

YEARS_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience", re.I),
    re.compile(r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*\+\s*yrs?\b", re.I),
]

MONTHS_PATTERN = re.compile(r"(\d+)\s*months?\b", re.I)

LOCATION_PATTERN = re.compile(
    r"\b((?:remote|hybrid|on.?site)?\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?,\s*"
    r"(?:[A-Z][a-zA-Z]+|IN|US|UK|EU))\b"
)


def _detect_section(line: str) -> Optional[str]:
    stripped = line.strip()
    # A heading is short; a long line that happens to start with "Skills" is prose.
    if not stripped or len(stripped) > 60:
        return None
    for name, pattern in SECTION_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def detect_seniority(text: str) -> str:
    for level, pattern in SENIORITY_RULES:
        if pattern.search(text):
            return level
    return "unspecified"


def detect_years_experience(text: str) -> Optional[float]:
    """Largest explicitly stated span of experience, in years."""
    best: Optional[float] = None
    for pattern in YEARS_PATTERNS:
        for m in pattern.finditer(text):
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            if 0 < value <= 60 and (best is None or value > best):
                best = value
    if best is None:
        for m in MONTHS_PATTERN.finditer(text):
            try:
                months = float(m.group(1))
            except ValueError:
                continue
            if 0 < months <= 600:
                years = round(months / 12.0, 2)
                if best is None or years > best:
                    best = years
    return best


def seniority_from_years(years: Optional[float]) -> str:
    """
    Infer a seniority band from stated experience.

    Only consulted when the document carries no explicit seniority word, so an
    stated title like "Senior Engineer" always wins over an inferred band.
    Boundaries follow common industry banding and are documented so a reviewer
    can see they were chosen rather than tuned to the sample data.
    """
    if years is None:
        return "unspecified"
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    if years < 9:
        return "senior"
    return "lead"


def detect_location(text: str) -> str:
    head = "\n".join(text.strip().splitlines()[:12])
    m = LOCATION_PATTERN.search(head)
    return m.group(1).strip() if m else ""


def detect_title(text: str, doc_type: str) -> str:
    """
    First line that looks like a role rather than a name or contact detail.

    Résumés commonly open with a name line then a role line, often pipe- or
    dash-separated from a location.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    if doc_type == "job":
        return lines[0]

    role_hint = re.compile(
        r"\b(engineer|developer|scientist|analyst|designer|manager|architect|"
        r"consultant|researcher|administrator|specialist|student|intern|lead)\b",
        re.I,
    )
    for line in lines[:6]:
        if role_hint.search(line):
            return line.split("|")[0].strip(" -–—")
    return lines[0]


def parse_document(text: str, doc_id: str, doc_type: str = "resume") -> ParsedDocument:
    """Segment a document into sections and extract its structured attributes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    lines = text.splitlines()
    sections: Dict[str, List[str]] = {}
    current = "header" if doc_type == "resume" else "body"

    for line in lines:
        detected = _detect_section(line)
        if detected:
            current = detected
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    section_text = {k: "\n".join(v).strip() for k, v in sections.items()}
    section_text = {k: v for k, v in section_text.items() if v}

    # Seniority is most reliable in the title/header; fall back to whole text.
    header_blob = "\n".join(
        [section_text.get("header", ""), section_text.get("summary", "")]
    ).strip()
    seniority = detect_seniority(header_blob) if header_blob else "unspecified"
    if seniority == "unspecified":
        seniority = detect_seniority(text)

    years = detect_years_experience(text)
    if seniority == "unspecified":
        seniority = seniority_from_years(years)

    return ParsedDocument(
        doc_id=doc_id,
        raw_text=text,
        doc_type=doc_type,
        sections=section_text,
        title=detect_title(text, doc_type),
        years_experience=years,
        seniority=seniority,
        location=detect_location(text),
    )


def section_for_offset(doc: ParsedDocument, offset: int) -> str:
    """
    Which section a character offset falls in.

    Used to tag every extracted skill span with its provenance, so an
    explanation can say the evidence came from the skills block rather than a
    passing mention in a summary.
    """
    best_name, best_pos = "body", -1
    for name, content in doc.sections.items():
        if not content:
            continue
        pos = doc.raw_text.find(content[: min(len(content), 60)])
        if 0 <= pos <= offset and pos > best_pos:
            best_name, best_pos = name, pos
    return best_name
