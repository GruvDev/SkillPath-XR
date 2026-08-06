"""
multipart/form-data parsing for the standard-library server.

Python's `cgi.FieldStorage` was the usual answer here; it is deprecated in 3.11
and removed in 3.13, so this module implements the small subset of RFC 7578
that a file upload actually needs.

Kept separate from the server so it can be unit-tested on its own — parsing
untrusted request bodies is exactly the code that should not be tested only
through an HTTP round trip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_BOUNDARY_RE = re.compile(rb'boundary="?([^";,]+)"?', re.IGNORECASE)
_DISPOSITION_NAME = re.compile(rb'name="([^"]*)"', re.IGNORECASE)
_DISPOSITION_FILENAME = re.compile(rb'filename="([^"]*)"', re.IGNORECASE)


class MultipartError(ValueError):
    """Raised when a request body is not valid multipart/form-data."""


@dataclass
class Part:
    """One part of a multipart body."""

    name: str = ""
    filename: Optional[str] = None
    content_type: str = ""
    data: bytes = b""

    @property
    def is_file(self) -> bool:
        return self.filename is not None

    def text(self, encoding: str = "utf-8") -> str:
        return self.data.decode(encoding, errors="replace")


@dataclass
class MultipartForm:
    """Parsed form: named fields and uploaded files."""

    fields: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, Part] = field(default_factory=dict)

    def first_file(self) -> Optional[Part]:
        """The first uploaded file, whatever field name it arrived under."""
        for part in self.files.values():
            return part
        return None


def parse_boundary(content_type: str) -> bytes:
    """Extract the boundary token from a Content-Type header."""
    if not content_type:
        raise MultipartError("Missing Content-Type header.")
    if "multipart/form-data" not in content_type.lower():
        raise MultipartError("Content-Type is not multipart/form-data.")

    match = _BOUNDARY_RE.search(content_type.encode("latin-1", errors="replace"))
    if not match:
        raise MultipartError("Content-Type has no boundary parameter.")

    boundary = match.group(1).strip()
    if not boundary:
        raise MultipartError("Empty multipart boundary.")
    return boundary


def _parse_headers(block: bytes) -> Tuple[str, Optional[str], str]:
    """Return (name, filename, content_type) from a part's header block."""
    name, filename, content_type = "", None, ""

    for line in block.split(b"\r\n"):
        lowered = line.lower()
        if lowered.startswith(b"content-disposition:"):
            found_name = _DISPOSITION_NAME.search(line)
            if found_name:
                name = found_name.group(1).decode("utf-8", errors="replace")
            found_file = _DISPOSITION_FILENAME.search(line)
            if found_file:
                filename = found_file.group(1).decode("utf-8", errors="replace")
        elif lowered.startswith(b"content-type:"):
            content_type = line.split(b":", 1)[1].strip().decode(
                "utf-8", errors="replace"
            )

    return name, filename, content_type


def parse_multipart(body: bytes, content_type: str) -> MultipartForm:
    """
    Parse a multipart/form-data body.

    Splits on the delimiter rather than scanning byte by byte, which keeps the
    implementation short enough to audit. Binary file content is preserved
    exactly: only the CRLF that belongs to the delimiter is removed.
    """
    boundary = parse_boundary(content_type)
    delimiter = b"--" + boundary

    if delimiter not in body:
        raise MultipartError("Request body does not contain the declared boundary.")

    form = MultipartForm()

    for segment in body.split(delimiter):
        # Ignore the preamble, the closing "--" and the epilogue.
        if not segment or segment in (b"--", b"--\r\n", b"\r\n"):
            continue
        if segment.startswith(b"--"):
            continue

        segment = segment[2:] if segment.startswith(b"\r\n") else segment.lstrip(b"\r\n")

        split_at = segment.find(b"\r\n\r\n")
        if split_at == -1:
            continue  # malformed part: no header/body separator

        header_block = segment[:split_at]
        data = segment[split_at + 4 :]

        # The CRLF immediately before the next delimiter belongs to the
        # protocol, not the file. Removing more would corrupt binary content.
        if data.endswith(b"\r\n"):
            data = data[:-2]

        name, filename, part_type = _parse_headers(header_block)
        if not name and filename is None:
            continue

        part = Part(name=name, filename=filename, content_type=part_type, data=data)
        if part.is_file:
            form.files[name or filename or "file"] = part
        else:
            form.fields[name] = part.text()

    if not form.fields and not form.files:
        raise MultipartError("No readable parts were found in the request body.")

    return form
