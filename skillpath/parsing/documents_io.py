"""
Résumé file ingestion — turning an uploaded document into plain text.

Format support and what it costs you
------------------------------------
| Format      | How it is read                          | Needs installs? |
|-------------|-----------------------------------------|-----------------|
| .txt .md    | decoded directly                        | no              |
| .docx       | stdlib `zipfile` + XML (DOCX is a ZIP)  | no              |
| .odt        | stdlib `zipfile` + XML (same structure) | no              |
| .rtf        | control-word stripper                   | no              |
| .pdf        | pdfplumber → pypdf → pdftotext          | **yes, one of** |
| .doc        | not supported — reported clearly        | —               |

Everything except PDF works on a clean Python install, matching the rest of
this project. PDF genuinely needs a parser; rather than shipping a fragile
hand-rolled one, the extractor tries each available backend in order of
fidelity and, if none is present, returns an error naming exactly what to
install.

Design notes
------------
* Format is detected from **magic bytes first**, extension second. A résumé
  saved as "cv.txt" that is really a PDF still parses, and a mislabelled file
  cannot silently produce garbage.
* Extraction never raises for a merely unreadable document. It returns an
  `ExtractionResult` carrying `ok`, a human-readable `error` and any
  `warnings`, because the caller is a web request that must answer politely
  rather than return a 500.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024      # 10 MB
MAX_EXTRACTED_CHARS = 60_000           # matches the résumé limit in the API

SUPPORTED_EXTENSIONS = (".txt", ".md", ".text", ".pdf", ".docx", ".odt", ".rtf")

# XML namespaces used by the two ZIP-based office formats.
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"


@dataclass
class ExtractionResult:
    """Outcome of reading one uploaded file."""

    text: str = ""
    fmt: str = "unknown"
    method: str = ""
    ok: bool = True
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    pages: Optional[int] = None
    chars: int = 0
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "format": self.fmt,
            "method": self.method,
            "ok": self.ok,
            "error": self.error,
            "warnings": self.warnings,
            "pages": self.pages,
            "chars": self.chars,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(data: bytes, filename: str = "") -> str:
    """
    Identify the document format.

    Magic bytes win over the file extension: people rename files, and mail
    clients relabel them. Getting this wrong means feeding binary noise to the
    skill extractor, which would produce confident nonsense.
    """
    if not data:
        return "empty"

    head = data[:8]

    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(b"\xd0\xcf\x11\xe0"):       # OLE compound file
        return "doc"
    if head.startswith(b"PK\x03\x04"):             # ZIP container
        return _detect_zip_format(data)

    ext = os.path.splitext(filename.lower())[1]
    if ext in (".txt", ".md", ".text"):
        return "text"
    if ext in (".pdf", ".rtf", ".docx", ".odt", ".doc"):
        return ext.lstrip(".")

    # Fall back to a printability check rather than guessing.
    return "text" if _looks_like_text(data) else "binary"


def _detect_zip_format(data: bytes) -> str:
    """Distinguish DOCX from ODT (and other ZIPs) by their internal layout."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return "binary"

    if "word/document.xml" in names:
        return "docx"
    if "content.xml" in names and any(n.startswith("META-INF") for n in names):
        return "odt"
    if any(n.startswith("ppt/") for n in names):
        return "pptx"
    if any(n.startswith("xl/") for n in names):
        return "xlsx"
    return "zip"


def _looks_like_text(data: bytes, sample: int = 2048) -> bool:
    """True if the leading bytes are overwhelmingly printable."""
    chunk = data[:sample]
    if not chunk:
        return False
    if b"\x00" in chunk:
        return False
    printable = sum(1 for b in chunk if 32 <= b < 127 or b in (9, 10, 13))
    return printable / len(chunk) > 0.90


def decode_text(data: bytes) -> Tuple[str, List[str]]:
    """
    Decode bytes to text, trying the encodings that actually occur.

    Uses `chardet` when it is installed; otherwise walks a short list of
    encodings and finally falls back to lossy latin-1, which never raises, so
    a slightly mangled résumé still gets parsed rather than rejected.
    """
    warnings: List[str] = []

    try:
        import chardet  # optional

        guess = chardet.detect(data[:20000])
        if guess and guess.get("encoding") and (guess.get("confidence") or 0) > 0.7:
            try:
                return data.decode(guess["encoding"]), warnings
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        pass

    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), warnings
        except (UnicodeDecodeError, LookupError):
            continue

    warnings.append("Some characters could not be decoded and were replaced.")
    return data.decode("latin-1", errors="replace"), warnings


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_BLANKS = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def normalise_layout(text: str) -> str:
    """
    Tidy extracted text without destroying structure.

    Section detection in the parser keys off line breaks, so blank lines are
    collapsed but never removed. Runs of spaces — common in PDF column
    extraction — are collapsed to one.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Bullet glyphs carry no meaning once the text is linearised.
    text = re.sub(r"[\u2022\u25cf\u25aa\u00b7\u2023\u2043]", "-", text)
    text = _TRAILING_WS.sub("", text)
    text = _MANY_SPACES.sub(" ", text)
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------


def extract_docx(data: bytes) -> ExtractionResult:
    """
    Read a .docx with the standard library alone.

    A DOCX is a ZIP whose `word/document.xml` holds the body. Paragraphs are
    `<w:p>`, runs are `<w:t>`, and explicit breaks are `<w:br>`/`<w:tab>`.
    Walking that tree directly avoids a dependency and, in practice, handles
    résumés — including tables, whose cells are ordinary paragraphs — perfectly
    well.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "word/document.xml" not in archive.namelist():
                return ExtractionResult(
                    ok=False, fmt="docx",
                    error="This .docx has no document body (word/document.xml is missing).",
                )
            xml_bytes = archive.read("word/document.xml")
    except zipfile.BadZipFile:
        return ExtractionResult(
            ok=False, fmt="docx", error="The file is not a readable .docx archive."
        )

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        return ExtractionResult(ok=False, fmt="docx", error=f"Malformed .docx XML: {exc}")

    lines: List[str] = []
    for paragraph in root.iter(f"{_W_NS}p"):
        pieces: List[str] = []
        for node in paragraph.iter():
            tag = node.tag
            if tag == f"{_W_NS}t" and node.text:
                pieces.append(node.text)
            elif tag == f"{_W_NS}tab":
                pieces.append("\t")
            elif tag == f"{_W_NS}br":
                pieces.append("\n")
        lines.append("".join(pieces))

    text = normalise_layout("\n".join(lines))
    return ExtractionResult(text=text, fmt="docx", method="stdlib zipfile+xml", ok=bool(text))


def extract_odt(data: bytes) -> ExtractionResult:
    """Read an .odt with the standard library alone (same ZIP+XML shape)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "content.xml" not in archive.namelist():
                return ExtractionResult(
                    ok=False, fmt="odt", error="This .odt has no content.xml."
                )
            xml_bytes = archive.read("content.xml")
    except zipfile.BadZipFile:
        return ExtractionResult(
            ok=False, fmt="odt", error="The file is not a readable .odt archive."
        )

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        return ExtractionResult(ok=False, fmt="odt", error=f"Malformed .odt XML: {exc}")

    lines: List[str] = []
    for tag in (f"{_TEXT_NS}p", f"{_TEXT_NS}h"):
        for node in root.iter(tag):
            lines.append("".join(node.itertext()))

    text = normalise_layout("\n".join(lines))
    return ExtractionResult(text=text, fmt="odt", method="stdlib zipfile+xml", ok=bool(text))


_RTF_CONTROL = re.compile(r"\\'([0-9a-fA-F]{2})|\\([a-zA-Z]+)-?\d*\s?|([{}])|(\\\*)")

# RTF "destination" groups whose *contents* are metadata, not document text.
# Their inner text must be discarded wholesale — otherwise font names, colour
# tables and generator strings leak into the résumé and can produce spurious
# skill matches.
_RTF_DESTINATIONS = frozenset(
    {
        "fonttbl", "colortbl", "stylesheet", "info", "pict", "object",
        "themedata", "colorschememapping", "latentstyles", "datastore",
        "generator", "listtable", "listoverridetable", "revtbl", "rsidtbl",
        "xmlnstbl", "filetbl", "header", "footer", "headerl", "headerr",
        "footerl", "footerr", "footnote", "annotation", "bkmkstart", "bkmkend",
    }
)

_DEST_START = re.compile(r"\{\s*\\(?:\*\s*\\)?([a-zA-Z]+)")


def _strip_rtf_destinations(raw: str) -> str:
    """
    Remove metadata groups and their contents, respecting brace nesting.

    Scans left to right tracking depth so a nested group inside a font table is
    removed with it, and an unbalanced document degrades to "drop the rest of
    that group" rather than throwing.
    """
    out: List[str] = []
    i, n = 0, len(raw)

    while i < n:
        char = raw[i]
        if char == "{":
            match = _DEST_START.match(raw, i)
            # A group is a destination if it is \* prefixed or names one.
            starts_star = raw[i:].lstrip("{ ").startswith("\\*")
            if match and (match.group(1).lower() in _RTF_DESTINATIONS or starts_star):
                depth, j = 0, i
                while j < n:
                    if raw[j] == "\\" and j + 1 < n:
                        j += 2          # escaped char: never a real brace
                        continue
                    if raw[j] == "{":
                        depth += 1
                    elif raw[j] == "}":
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
                i = j                    # skip the whole group
                continue
        out.append(char)
        i += 1

    return "".join(out)


def extract_rtf(data: bytes) -> ExtractionResult:
    """
    Strip RTF control words to recover the visible text.

    Handles the constructs résumés actually use — paragraph and line breaks,
    tabs, escaped hex characters, unicode escapes — discards metadata
    destination groups, and drops the rest. Not a complete RTF implementation,
    and does not claim to be.
    """
    raw, warnings = decode_text(data)

    raw = _strip_rtf_destinations(raw)

    # \uN escapes carry the character in N; the following byte is a fallback.
    raw = re.sub(r"\\u(-?\d+)\s?\??", lambda m: _safe_chr(int(m.group(1))), raw)

    def _replace(match: "re.Match[str]") -> str:
        hex_char, word, brace, _star = match.groups()
        if hex_char:
            return _safe_chr(int(hex_char, 16))
        if word:
            if word in ("par", "line"):
                return "\n"
            if word == "tab":
                return "\t"
            return ""
        if brace:
            return ""
        return ""

    text = _RTF_CONTROL.sub(_replace, raw)
    text = normalise_layout(text)
    return ExtractionResult(
        text=text, fmt="rtf", method="control-word stripper",
        ok=bool(text), warnings=warnings,
    )


def _safe_chr(code: int) -> str:
    try:
        return chr(code) if 0 <= code <= 0x10FFFF else ""
    except (ValueError, OverflowError):
        return ""


def extract_pdf(data: bytes) -> ExtractionResult:
    """
    Read a PDF using whichever backend is installed, best fidelity first.

    Order: pdfplumber (best layout handling) → pypdf (lighter) → the
    `pdftotext` binary. If none is available the result names the exact
    install command instead of failing opaquely.
    """
    attempts: List[str] = []

    # 1. pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = normalise_layout("\n\n".join(pages))
        if text.strip():
            return _pdf_result(text, "pdfplumber", len(pages))
        attempts.append("pdfplumber found no text layer")
    except ImportError:
        attempts.append("pdfplumber not installed")
    except Exception as exc:  # noqa: BLE001 - try the next backend
        attempts.append(f"pdfplumber failed ({exc})")

    # 2. pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = normalise_layout("\n\n".join(pages))
        if text.strip():
            return _pdf_result(text, "pypdf", len(pages))
        attempts.append("pypdf found no text layer")
    except ImportError:
        attempts.append("pypdf not installed")
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"pypdf failed ({exc})")

    # 3. pdftotext binary
    if shutil.which("pdftotext"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                completed = subprocess.run(
                    ["pdftotext", "-layout", "-enc", "UTF-8", tmp_path, "-"],
                    capture_output=True, timeout=30, check=False,
                )
                text = normalise_layout(completed.stdout.decode("utf-8", errors="replace"))
                if text.strip():
                    return _pdf_result(text, "pdftotext", None)
                attempts.append("pdftotext found no text layer")
            finally:
                os.unlink(tmp_path)
        except (subprocess.TimeoutExpired, OSError) as exc:
            attempts.append(f"pdftotext failed ({exc})")
    else:
        attempts.append("pdftotext binary not on PATH")

    detail = "; ".join(attempts)
    if "found no text layer" in detail:
        message = (
            "This PDF has no extractable text layer — it is most likely a scan or "
            "an image export. Save it as a text-based PDF, or paste the text instead."
        )
    else:
        message = (
            "No PDF reader is available on this server. Install one with "
            "`pip install pdfplumber` (or `pip install pypdf`), or paste the "
            "resume text instead."
        )
    return ExtractionResult(ok=False, fmt="pdf", error=message, warnings=attempts)


def _pdf_result(text: str, method: str, pages: Optional[int]) -> ExtractionResult:
    warnings: List[str] = []
    if pages and pages > 6:
        warnings.append(f"{pages} pages read; only the first {MAX_EXTRACTED_CHARS:,} characters are used.")
    return ExtractionResult(
        text=text, fmt="pdf", method=method, pages=pages, ok=True, warnings=warnings
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def extract_text(data: bytes, filename: str = "") -> ExtractionResult:
    """
    Extract plain text from an uploaded résumé.

    Never raises for a bad document: an unreadable file comes back as
    `ok=False` with an explanatory `error`, because the caller is an HTTP
    handler that owes the user a clear message rather than a stack trace.
    """
    if not data:
        return ExtractionResult(ok=False, fmt="empty", error="The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        return ExtractionResult(
            ok=False, fmt="unknown",
            error=f"File is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )

    fmt = detect_format(data, filename)

    if fmt in ("text", "txt", "md"):
        text, warnings = decode_text(data)
        result = ExtractionResult(
            text=normalise_layout(text), fmt="text", method="decode", warnings=warnings
        )
    elif fmt == "docx":
        result = extract_docx(data)
    elif fmt == "odt":
        result = extract_odt(data)
    elif fmt == "rtf":
        result = extract_rtf(data)
    elif fmt == "pdf":
        result = extract_pdf(data)
    elif fmt == "doc":
        result = ExtractionResult(
            ok=False, fmt="doc",
            error=(
                "Legacy .doc files are not supported. Open it in Word or "
                "LibreOffice and save as .docx or .pdf."
            ),
        )
    elif fmt in ("pptx", "xlsx"):
        result = ExtractionResult(
            ok=False, fmt=fmt,
            error=f"That looks like a {fmt.upper()} file, not a resume. Upload a PDF, DOCX, ODT or TXT.",
        )
    else:
        result = ExtractionResult(
            ok=False, fmt=fmt,
            error="Unrecognised file type. Upload a PDF, DOCX, ODT, RTF or TXT file.",
        )

    if result.ok:
        if not result.text.strip():
            result.ok = False
            result.error = "No readable text was found in this file."
        else:
            if len(result.text) > MAX_EXTRACTED_CHARS:
                result.text = result.text[:MAX_EXTRACTED_CHARS]
                result.truncated = True
                result.warnings.append(
                    f"Only the first {MAX_EXTRACTED_CHARS:,} characters were kept."
                )
            result.chars = len(result.text)

    return result
