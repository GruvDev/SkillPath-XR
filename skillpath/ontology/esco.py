"""
Skill taxonomy loading and navigation.

Holds the normalised skill vocabulary and the relations between concepts.
Everything downstream — extraction, graph construction, evidence paths — reads
the taxonomy through this class, so replacing the bundled seed taxonomy with
the real ESCO release is a loader change and nothing else.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..types import Occupation, SkillConcept

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s+#.]", re.UNICODE)


def normalise(text: str) -> str:
    """
    Canonical surface form used for all alias lookups.

    Lowercases, strips punctuation except the characters that carry meaning in
    technology names (`+` in C++, `#` in C#, `.` in Node.js), and collapses
    whitespace.  Extraction and validation must use this same function or
    matching becomes inconsistent between modules.
    """
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


class Taxonomy:
    """An in-memory skill taxonomy with alias, hierarchy and relation indexes."""

    def __init__(
        self,
        skills: Iterable[SkillConcept],
        occupations: Iterable[Occupation] = (),
    ) -> None:
        self.skills: Dict[str, SkillConcept] = {s.uri: s for s in skills}
        self.occupations: Dict[str, Occupation] = {o.uri: o for o in occupations}

        # alias surface -> set of URIs (a surface can be ambiguous)
        self.alias_index: Dict[str, Set[str]] = {}
        # URI -> direct children (inverse of `broader`)
        self.narrower_index: Dict[str, Set[str]] = {}

        self._build_indexes()

    # -- construction --------------------------------------------------------

    def _build_indexes(self) -> None:
        for uri, concept in self.skills.items():
            for surface in (concept.label,) + tuple(concept.alt_labels):
                key = normalise(surface)
                if key:
                    self.alias_index.setdefault(key, set()).add(uri)

        for uri, concept in self.skills.items():
            for parent in concept.broader:
                if parent in self.skills:
                    self.narrower_index.setdefault(parent, set()).add(uri)

    # -- lookup --------------------------------------------------------------

    def get(self, uri: str) -> Optional[SkillConcept]:
        return self.skills.get(uri)

    def label(self, uri: str) -> str:
        concept = self.skills.get(uri)
        return concept.label if concept else uri

    def occupation(self, uri: str) -> Optional[Occupation]:
        return self.occupations.get(uri)

    def lookup_alias(self, surface: str) -> List[str]:
        """Return URIs whose label or alt label matches this surface exactly."""
        return sorted(self.alias_index.get(normalise(surface), set()))

    def max_alias_words(self) -> int:
        """Longest alias in words — bounds the n-gram window during extraction."""
        if not self.alias_index:
            return 1
        return max(len(k.split()) for k in self.alias_index)

    # -- navigation ----------------------------------------------------------

    def broader_of(self, uri: str) -> List[str]:
        concept = self.skills.get(uri)
        return [u for u in concept.broader if u in self.skills] if concept else []

    def narrower_of(self, uri: str) -> List[str]:
        return sorted(self.narrower_index.get(uri, set()))

    def related_of(self, uri: str) -> List[str]:
        concept = self.skills.get(uri)
        if not concept:
            return []
        return [u for u in concept.related if u in self.skills]

    def ancestors(self, uri: str, max_depth: int = 3) -> List[str]:
        """All broader concepts up to `max_depth`, nearest first, cycle-safe."""
        out: List[str] = []
        seen = {uri}
        frontier = [uri]
        for _ in range(max_depth):
            nxt: List[str] = []
            for node in frontier:
                for parent in self.broader_of(node):
                    if parent not in seen:
                        seen.add(parent)
                        out.append(parent)
                        nxt.append(parent)
            if not nxt:
                break
            frontier = nxt
        return out

    def descendants(self, uri: str, max_depth: int = 3) -> List[str]:
        """All narrower concepts down to `max_depth`, nearest first, cycle-safe."""
        out: List[str] = []
        seen = {uri}
        frontier = [uri]
        for _ in range(max_depth):
            nxt: List[str] = []
            for node in frontier:
                for child in self.narrower_of(node):
                    if child not in seen:
                        seen.add(child)
                        out.append(child)
                        nxt.append(child)
            if not nxt:
                break
            frontier = nxt
        return out

    def concept_text(self, uri: str) -> str:
        """Text used to embed a concept: label plus its aliases."""
        concept = self.skills.get(uri)
        if not concept:
            return ""
        parts = [concept.label] + list(concept.alt_labels)
        if concept.description:
            parts.append(concept.description)
        return " ; ".join(parts)

    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, uri: object) -> bool:
        return uri in self.skills


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_seed_taxonomy(data_dir: str = DATA_DIR) -> Taxonomy:
    """Load the bundled seed taxonomy (no network, no downloads)."""
    path = os.path.join(data_dir, "taxonomy.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it with: python -m skillpath.data.build_seed_data"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    skills = [
        SkillConcept(
            uri=s["uri"],
            label=s["label"],
            skill_type=s.get("skill_type", "skill"),
            alt_labels=tuple(s.get("alt_labels", ())),
            broader=tuple(s.get("broader", ())),
            related=tuple(s.get("related", ())),
            description=s.get("description", ""),
        )
        for s in payload["skills"]
    ]
    occupations = [
        Occupation(
            uri=o["uri"],
            label=o["label"],
            essential_skills=tuple(o.get("essential_skills", ())),
            optional_skills=tuple(o.get("optional_skills", ())),
            description=o.get("description", ""),
        )
        for o in payload.get("occupations", [])
    ]
    return Taxonomy(skills, occupations)


def load_esco_csv(
    skills_csv: str,
    relations_csv: Optional[str] = None,
    occupations_csv: Optional[str] = None,
) -> Taxonomy:
    """
    Load the real ESCO release from its published CSV distribution.

    ESCO ships `skills_en.csv` with columns including conceptUri, preferredLabel,
    altLabels (newline separated), skillType and description, plus a
    broader/narrower relations file.  Column names are read defensively so minor
    release-to-release changes do not break the loader.

    This function is the single seam between the bundled seed taxonomy and the
    real one: nothing downstream needs to change.  See docs/02-data-and-ontology.md
    for where to obtain the files.
    """
    import csv

    def _col(row: Dict[str, str], *names: str, default: str = "") -> str:
        for n in names:
            if n in row and row[n]:
                return row[n]
        return default

    concepts: Dict[str, Dict[str, object]] = {}
    with open(skills_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uri = _col(row, "conceptUri", "conceptURI", "uri")
            if not uri:
                continue
            alt_raw = _col(row, "altLabels", "altLabel")
            alts = tuple(
                a.strip() for a in re.split(r"[\r\n|]", alt_raw) if a.strip()
            )
            concepts[uri] = {
                "uri": uri,
                "label": _col(row, "preferredLabel", "label", default=uri),
                "skill_type": _col(row, "skillType", default="skill"),
                "alt_labels": alts,
                "description": _col(row, "description", "definition"),
                "broader": set(),
                "related": set(),
            }

    if relations_csv and os.path.exists(relations_csv):
        with open(relations_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                child = _col(row, "conceptUri", "narrowerUri", "originalUri")
                parent = _col(row, "broaderUri", "targetUri", "parentUri")
                rel = _col(row, "relationType", default="broader").lower()
                if child in concepts and parent in concepts:
                    if "broad" in rel or rel in ("", "hierarchy"):
                        concepts[child]["broader"].add(parent)  # type: ignore[union-attr]
                    else:
                        concepts[child]["related"].add(parent)  # type: ignore[union-attr]
                        concepts[parent]["related"].add(child)  # type: ignore[union-attr]

    skills = [
        SkillConcept(
            uri=str(c["uri"]),
            label=str(c["label"]),
            skill_type=str(c["skill_type"]),
            alt_labels=tuple(c["alt_labels"]),      # type: ignore[arg-type]
            broader=tuple(sorted(c["broader"])),    # type: ignore[arg-type]
            related=tuple(sorted(c["related"])),    # type: ignore[arg-type]
            description=str(c["description"]),
        )
        for c in concepts.values()
    ]

    occupations: List[Occupation] = []
    if occupations_csv and os.path.exists(occupations_csv):
        by_occ: Dict[str, Dict[str, object]] = {}
        with open(occupations_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                occ = _col(row, "occupationUri", "conceptUri")
                skill = _col(row, "skillUri", "relatedSkillUri")
                label = _col(row, "preferredLabel", "occupationLabel", default=occ)
                rel = _col(row, "relationType", default="essential").lower()
                if not occ:
                    continue
                entry = by_occ.setdefault(
                    occ, {"uri": occ, "label": label, "essential": set(), "optional": set()}
                )
                if skill:
                    key = "essential" if "essential" in rel else "optional"
                    entry[key].add(skill)  # type: ignore[union-attr]
        for e in by_occ.values():
            occupations.append(
                Occupation(
                    uri=str(e["uri"]),
                    label=str(e["label"]),
                    essential_skills=tuple(sorted(e["essential"])),  # type: ignore[arg-type]
                    optional_skills=tuple(sorted(e["optional"])),    # type: ignore[arg-type]
                )
            )

    return Taxonomy(skills, occupations)


def load_taxonomy() -> Taxonomy:
    """
    Load whichever taxonomy the environment is configured for.

    Set SKILLPATH_ESCO_SKILLS_CSV (and optionally SKILLPATH_ESCO_RELATIONS_CSV,
    SKILLPATH_ESCO_OCCUPATIONS_CSV) to use the real ESCO release.  Otherwise the
    bundled seed taxonomy is used.
    """
    skills_csv = os.environ.get("SKILLPATH_ESCO_SKILLS_CSV")
    if skills_csv and os.path.exists(skills_csv):
        return load_esco_csv(
            skills_csv,
            os.environ.get("SKILLPATH_ESCO_RELATIONS_CSV"),
            os.environ.get("SKILLPATH_ESCO_OCCUPATIONS_CSV"),
        )
    return load_seed_taxonomy()
