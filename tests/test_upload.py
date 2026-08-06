"""
Tests for résumé file ingestion and multipart parsing.

Covers the two things most likely to break in an upload path: a document
format that parses to the wrong text, and a request body that is malformed or
hostile.  Fixtures are generated in-process so the suite has no binary blobs
checked in and runs anywhere.
"""

from __future__ import annotations

import io
import os
import sys
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.multipart import (  # noqa: E402
    MultipartError,
    parse_boundary,
    parse_multipart,
)
from skillpath.parsing.documents_io import (  # noqa: E402
    MAX_FILE_BYTES,
    decode_text,
    detect_format,
    extract_text,
    normalise_layout,
)
from skillpath.pipeline import SkillPathPipeline, load_sample_resumes  # noqa: E402

PIPELINE = SkillPathPipeline()
RESUME_TEXT = load_sample_resumes()[0]["text"]


# ---------------------------------------------------------------------------
# Fixture builders — real files, built with the standard library
# ---------------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(text: str) -> bytes:
    """A minimal but genuinely valid .docx."""
    paragraphs = "".join(
        f"<w:p><w:r><w:t xml:space='preserve'>{_xml_escape(line)}</w:t></w:r></w:p>"
        for line in text.split("\n")
    )
    document = (
        f"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<w:document xmlns:w='{W_NS}'><w:body>{paragraphs}</w:body></w:document>"
    )
    content_types = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        "<Default Extension='xml' ContentType='application/xml'/>"
        "<Override PartName='/word/document.xml' ContentType='application/vnd."
        "openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def make_odt(text: str) -> bytes:
    """A minimal but genuinely valid .odt."""
    ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    office = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    paragraphs = "".join(f"<text:p>{_xml_escape(line)}</text:p>" for line in text.split("\n"))
    content = (
        f"<?xml version='1.0' encoding='UTF-8'?>"
        f"<office:document-content xmlns:office='{office}' xmlns:text='{ns}'>"
        f"<office:body><office:text>{paragraphs}</office:text></office:body>"
        f"</office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("META-INF/manifest.xml", "<manifest/>")
        archive.writestr("content.xml", content)
    return buffer.getvalue()


def make_rtf(text: str, with_metadata: bool = True) -> bytes:
    """RTF including the metadata groups real word processors emit."""
    escaped = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    header = r"{\rtf1\ansi\ansicpg1252\deff0"
    if with_metadata:
        header += (
            r"{\fonttbl{\f0\froman Times New Roman;}{\f1 Arial;}}"
            r"{\colortbl;\red0\green0\blue0;}"
            r"{\*\generator Riched20 10.0;}"
            r"{\info{\author Jane Doe}{\company Kubernetes Consulting}}"
        )
    body = escaped.replace("\n", r"\par ")
    return (header + r"\f0\fs22 " + body + "}").encode("utf-8")


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def build_multipart(filename: str, data: bytes, field: str = "file") -> tuple:
    """Return (body, content_type) for a single-file multipart request."""
    boundary = "----SkillPathTestBoundary9f3a"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        data,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------------------


class TestFormatDetection(unittest.TestCase):
    def test_magic_bytes_beat_extension(self):
        """A mislabelled file must be identified by content, not by its name."""
        self.assertEqual(detect_format(b"%PDF-1.7\nstuff", "resume.txt"), "pdf")
        self.assertEqual(detect_format(make_docx("hi"), "resume.pdf"), "docx")
        self.assertEqual(detect_format(rb"{\rtf1 hello}", "resume.docx"), "rtf")

    def test_distinguishes_zip_based_formats(self):
        self.assertEqual(detect_format(make_docx("x"), "a.docx"), "docx")
        self.assertEqual(detect_format(make_odt("x"), "a.odt"), "odt")

    def test_legacy_doc_detected(self):
        self.assertEqual(detect_format(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "old.doc"), "doc")

    def test_plain_text_and_binary(self):
        self.assertEqual(detect_format(b"Just some resume text", "cv.txt"), "text")
        self.assertEqual(detect_format(b"\x00\x01\x02\x03\x04", "x.bin"), "binary")
        self.assertEqual(detect_format(b"", "x.txt"), "empty")


class TestTextExtraction(unittest.TestCase):
    def _skills(self, text: str) -> set:
        return set(PIPELINE.extractor.extract_uris(text))

    def setUp(self):
        self.expected = self._skills(RESUME_TEXT)
        self.assertGreater(len(self.expected), 5, "fixture résumé should yield skills")

    def test_plain_text_roundtrip(self):
        result = extract_text(RESUME_TEXT.encode("utf-8"), "resume.txt")
        self.assertTrue(result.ok)
        self.assertEqual(self._skills(result.text), self.expected)

    def test_docx_preserves_every_skill(self):
        result = extract_text(make_docx(RESUME_TEXT), "resume.docx")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.fmt, "docx")
        self.assertEqual(self._skills(result.text), self.expected)

    def test_odt_preserves_every_skill(self):
        result = extract_text(make_odt(RESUME_TEXT), "resume.odt")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.fmt, "odt")
        self.assertEqual(self._skills(result.text), self.expected)

    def test_rtf_preserves_every_skill(self):
        result = extract_text(make_rtf(RESUME_TEXT), "resume.rtf")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.fmt, "rtf")
        self.assertEqual(self._skills(result.text), self.expected)

    def test_rtf_metadata_never_becomes_a_skill(self):
        """
        Regression: font tables and \\info groups must not leak into the text.

        The fixture's metadata names a company "Kubernetes Consulting"; if the
        destination groups were not stripped, the candidate would be credited
        with Kubernetes.
        """
        result = extract_text(make_rtf("John Smith\nSkills: Python", True), "meta.rtf")
        self.assertTrue(result.ok)
        self.assertNotIn("Kubernetes", result.text)
        self.assertNotIn("Times New Roman", result.text)
        self.assertNotIn("Riched20", result.text)
        labels = {PIPELINE.taxonomy.label(u) for u in self._skills(result.text)}
        self.assertIn("Python", labels)
        self.assertNotIn("Kubernetes", labels)

    def test_docx_newlines_survive_for_section_detection(self):
        """Section parsing keys off line breaks, so they must not be flattened."""
        result = extract_text(make_docx(RESUME_TEXT), "resume.docx")
        from skillpath.parsing.document import parse_document

        doc = parse_document(result.text, "X")
        self.assertIn("skills", doc.sections)

    def test_unicode_survives(self):
        text = "Résumé — João Müller\nSkills: Python, SQL"
        result = extract_text(make_docx(text), "u.docx")
        self.assertTrue(result.ok)
        self.assertIn("João", result.text)
        self.assertIn("Müller", result.text)


class TestExtractionErrors(unittest.TestCase):
    def test_empty_file(self):
        result = extract_text(b"", "empty.txt")
        self.assertFalse(result.ok)
        self.assertIn("empty", result.error.lower())

    def test_oversize_file_rejected(self):
        result = extract_text(b"x" * (MAX_FILE_BYTES + 1), "big.txt")
        self.assertFalse(result.ok)
        self.assertIn("larger", result.error.lower())

    def test_legacy_doc_gives_actionable_message(self):
        result = extract_text(b"\xd0\xcf\x11\xe0" + b"\x00" * 40, "cv.doc")
        self.assertFalse(result.ok)
        self.assertIn(".docx", result.error)

    def test_corrupt_zip(self):
        result = extract_text(b"PK\x03\x04garbage-not-a-zip", "broken.docx")
        self.assertFalse(result.ok)
        self.assertTrue(result.error)

    def test_wrong_office_type_named(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("xl/workbook.xml", "<x/>")
        result = extract_text(buffer.getvalue(), "book.xlsx")
        self.assertFalse(result.ok)
        self.assertIn("XLSX", result.error)

    def test_scanned_pdf_reports_no_text_layer(self):
        """A PDF with no text must not silently return an empty résumé."""
        minimal = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF"
        )
        result = extract_text(minimal, "scan.pdf")
        self.assertFalse(result.ok)
        self.assertTrue(result.error)

    def test_extraction_never_raises(self):
        """Hostile input must come back as ok=False, not as an exception."""
        for payload in [
            b"\xff\xfe\x00\x00garbage",
            b"PK\x03\x04" + os.urandom(200),
            b"%PDF-" + os.urandom(200),
            rb"{\rtf1" + os.urandom(100),
            os.urandom(500),
        ]:
            with self.subTest(prefix=payload[:6]):
                result = extract_text(payload, "x.bin")
                self.assertIsInstance(result.ok, bool)


class TestNormalisation(unittest.TestCase):
    def test_blank_lines_preserved_but_collapsed(self):
        self.assertEqual(normalise_layout("A\n\n\n\n\nB"), "A\n\nB")

    def test_bullets_become_dashes(self):
        self.assertIn("- Python", normalise_layout("\u2022 Python"))

    def test_runs_of_spaces_collapse(self):
        self.assertEqual(normalise_layout("A     B"), "A B")

    def test_decode_handles_bad_bytes(self):
        text, _ = decode_text(b"caf\xe9 Python")
        self.assertIn("Python", text)


class TestMultipart(unittest.TestCase):
    def test_parses_a_single_file(self):
        payload = b"Skills: Python and SQL"
        body, content_type = build_multipart("cv.txt", payload)
        form = parse_multipart(body, content_type)
        part = form.first_file()
        self.assertIsNotNone(part)
        self.assertEqual(part.filename, "cv.txt")
        self.assertEqual(part.data, payload)

    def test_binary_content_is_byte_exact(self):
        """Binary files must survive parsing without a single byte changing."""
        payload = make_docx(RESUME_TEXT)
        body, content_type = build_multipart("cv.docx", payload)
        self.assertEqual(parse_multipart(body, content_type).first_file().data, payload)

    def test_content_with_crlf_is_preserved(self):
        payload = b"line one\r\nline two\r\n\r\nline three"
        body, content_type = build_multipart("cv.txt", payload)
        self.assertEqual(parse_multipart(body, content_type).first_file().data, payload)

    def test_fields_and_files_together(self):
        boundary = "xyzBOUND"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="candidate_id"\r\n\r\n'
            "C009\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\n'
            "Python\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        form = parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(form.fields["candidate_id"], "C009")
        self.assertEqual(form.first_file().data, b"Python")

    def test_quoted_boundary(self):
        self.assertEqual(
            parse_boundary('multipart/form-data; boundary="abc123"'), b"abc123"
        )

    def test_rejects_bad_content_types(self):
        for header in ["", "application/json", "multipart/form-data"]:
            with self.subTest(header=header):
                with self.assertRaises(MultipartError):
                    parse_boundary(header)

    def test_rejects_body_without_boundary(self):
        with self.assertRaises(MultipartError):
            parse_multipart(b"not multipart at all", "multipart/form-data; boundary=abc")

    def test_rejects_empty_parts(self):
        with self.assertRaises(MultipartError):
            parse_multipart(b"--abc--\r\n", "multipart/form-data; boundary=abc")


class TestUploadToRecommendation(unittest.TestCase):
    """The whole point: an uploaded file must produce the same result as text."""

    def test_every_format_yields_identical_recommendations(self):
        baseline = PIPELINE.recommend(RESUME_TEXT, top_k=5)
        expected = [(m.job.job_id, m.score) for m in baseline.matches]

        for name, data in [
            ("resume.txt", RESUME_TEXT.encode("utf-8")),
            ("resume.docx", make_docx(RESUME_TEXT)),
            ("resume.odt", make_odt(RESUME_TEXT)),
            ("resume.rtf", make_rtf(RESUME_TEXT)),
        ]:
            with self.subTest(file=name):
                result = extract_text(data, name)
                self.assertTrue(result.ok, result.error)
                got = PIPELINE.recommend(result.text, top_k=5)
                self.assertEqual(
                    [(m.job.job_id, m.score) for m in got.matches], expected,
                    f"{name} produced different recommendations from the raw text",
                )

    def test_uploaded_resume_still_gets_grounded_explanations(self):
        result = extract_text(make_docx(RESUME_TEXT), "resume.docx")
        recommendation = PIPELINE.recommend(result.text, top_k=3)
        for match in recommendation.matches:
            self.assertGreaterEqual(
                match.explanation.validation.grounding_rate,
                PIPELINE.settings.min_grounding_rate,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
