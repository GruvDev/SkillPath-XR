"""
Skill extraction and normalisation to the taxonomy.

Turns free text into `SkillSpan` objects linked to taxonomy URIs.  Three things
here are load-bearing for correctness downstream:

1. **Longest-match resolution.**  "machine learning engineer" must not be
   reported as both "machine learning" and "engineer"; overlapping candidates
   are resolved in favour of the longest span.

2. **Negation handling.**  "No experience with Kubernetes" must not credit the
   candidate with Kubernetes.  Negated spans are kept (so the UI can show what
   was considered) but flagged and excluded from the skill set.

3. **Offset fidelity.**  Every span carries exact character offsets into the
   source document, so the interface can highlight the literal text that
   produced a piece of evidence.  This is what makes an explanation auditable
   rather than merely plausible.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..ontology.esco import Taxonomy, normalise
from ..parsing.document import ParsedDocument, section_for_offset
from ..types import SkillSpan

# Cues that cancel a skill mention.  Scanned in a window before the mention.
NEGATION_CUES = [
    "no experience with",
    "no experience in",
    "not familiar with",
    "no knowledge of",
    "without experience in",
    "lacking",
    "never used",
    "no exposure to",
    "yet to learn",
    "want to learn",
    "wish to learn",
    "hoping to learn",
    "planning to learn",
    "would like to learn",
]

# Cues that mark a *requirement* rather than a possessed skill; used when the
# same extractor runs over job text so wishlists are not read as capabilities.
NEGATION_WINDOW = 60

PROFICIENCY_CUES: List[Tuple[str, float]] = [
    ("expert in", 1.00),
    ("expert", 1.00),
    ("advanced", 0.95),
    ("proficient in", 0.95),
    ("proficient", 0.95),
    ("strong", 0.92),
    ("experienced in", 0.92),
    ("hands-on", 0.90),
    ("working knowledge", 0.75),
    ("familiar with", 0.70),
    ("exposure to", 0.65),
    ("basic", 0.60),
    ("beginner", 0.55),
    ("currently learning", 0.50),
    ("learning", 0.50),
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9+#.]+", re.UNICODE)


def _tokenise_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokens with their exact (start, end) offsets in the original string.

    The token pattern deliberately admits `.` so names like "Node.js" and
    ".NET" survive tokenisation.  That has a sharp edge: a sentence-final
    "Java." would otherwise tokenise as `java.` and never match the alias
    `java`, silently dropping any skill that happens to end a sentence.
    Trailing dots are therefore trimmed (and the end offset moved with them),
    while internal and leading dots are preserved.
    """
    tokens: List[Tuple[str, int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        token, start, end = m.group(0), m.start(), m.end()
        while len(token) > 1 and token.endswith("."):
            token = token[:-1]
            end -= 1
        if token and token != ".":
            tokens.append((token, start, end))
    return tokens


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - NEGATION_WINDOW) : start].lower()
    # Only treat as negation if the cue is in the same clause: a sentence
    # boundary between the cue and the mention cancels the negation.
    for cue in NEGATION_CUES:
        idx = window.rfind(cue)
        if idx == -1:
            continue
        tail = window[idx + len(cue) :]
        if not re.search(r"[.;!?\n]", tail):
            return True
    return False


def _proficiency(text: str, start: int) -> float:
    window = text[max(0, start - 40) : start].lower()
    for cue, value in PROFICIENCY_CUES:
        if cue in window:
            return value
    return 0.85  # plain, unqualified mention


def _resolve_overlaps(spans: List[SkillSpan]) -> List[SkillSpan]:
    """
    Keep the longest non-overlapping spans.

    Sorts by span length descending (then by position for determinism) and
    greedily accepts spans that do not overlap anything already accepted.
    """
    ordered = sorted(spans, key=lambda s: (-(s.end - s.start), s.start, s.uri or ""))
    accepted: List[SkillSpan] = []
    for span in ordered:
        if any(not (span.end <= a.start or span.start >= a.end) for a in accepted):
            continue
        accepted.append(span)
    return sorted(accepted, key=lambda s: s.start)


class SkillExtractor:
    """Dictionary-driven extractor that links surface mentions to taxonomy URIs."""

    def __init__(self, taxonomy: Taxonomy, max_ngram: Optional[int] = None) -> None:
        self.taxonomy = taxonomy
        self.max_ngram = max_ngram or min(taxonomy.max_alias_words(), 6)

    def extract(
        self,
        text: str,
        document: Optional[ParsedDocument] = None,
    ) -> List[SkillSpan]:
        """Extract linked skill spans from raw text."""
        if not text:
            return []

        tokens = _tokenise_with_offsets(text)
        candidates: List[SkillSpan] = []

        for i in range(len(tokens)):
            upper = min(self.max_ngram, len(tokens) - i)
            for n in range(upper, 0, -1):
                window = tokens[i : i + n]
                start, end = window[0][1], window[-1][2]
                surface = text[start:end]
                key = normalise(surface)
                if not key:
                    continue
                uris = self.taxonomy.alias_index.get(key)
                if not uris:
                    continue
                negated = _is_negated(text, start)
                confidence = _proficiency(text, start)
                # Ambiguous surface: emit each candidate URI, let overlap
                # resolution and downstream scoring settle it.
                for uri in sorted(uris):
                    candidates.append(
                        SkillSpan(
                            surface=surface,
                            start=start,
                            end=end,
                            uri=uri,
                            label=self.taxonomy.label(uri),
                            confidence=confidence,
                            method="alias",
                            section=section_for_offset(document, start) if document else "body",
                            negated=negated,
                        )
                    )

        resolved = _resolve_overlaps(candidates)
        return self._deduplicate(resolved)

    @staticmethod
    def _deduplicate(spans: Sequence[SkillSpan]) -> List[SkillSpan]:
        """
        Collapse repeat mentions of the same URI, keeping the strongest.

        A skill named five times is one capability, but if any mention is
        affirmative the skill counts as held: a résumé saying "no Kubernetes
        experience" in one line and listing Kubernetes in the skills block is
        contradictory, and the affirmative listing wins with reduced
        confidence so the conflict is visible rather than silently resolved.
        """
        by_uri: Dict[str, SkillSpan] = {}
        conflicted: Dict[str, bool] = {}

        for span in spans:
            if not span.uri:
                continue
            existing = by_uri.get(span.uri)
            if existing is None:
                by_uri[span.uri] = span
                continue
            if existing.negated != span.negated:
                conflicted[span.uri] = True
            # Prefer affirmative over negated, then higher confidence.
            better = (not span.negated, span.confidence) > (
                not existing.negated,
                existing.confidence,
            )
            if better:
                by_uri[span.uri] = span

        out: List[SkillSpan] = []
        for uri, span in by_uri.items():
            if conflicted.get(uri) and not span.negated:
                span.confidence = min(span.confidence, 0.70)
                span.method = "alias:conflicted"
            out.append(span)
        return sorted(out, key=lambda s: s.start)

    def extract_uris(self, text: str) -> List[str]:
        """Convenience: affirmative, deduplicated skill URIs only."""
        return [s.uri for s in self.extract(text) if s.uri and not s.negated]

    def mentioned_uris(self, text: str) -> List[str]:
        """
        Every taxonomy concept mentioned, negated or not.

        The faithfulness validator uses this to find what an explanation
        *claims*, independent of whether the claim is true.
        """
        seen, out = set(), []
        for span in self.extract(text):
            if span.uri and span.uri not in seen:
                seen.add(span.uri)
                out.append(span.uri)
        return out
