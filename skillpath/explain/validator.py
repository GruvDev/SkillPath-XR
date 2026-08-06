"""
Faithfulness validation.

This is the module the whole system is built around.  An explanation is only
worth showing if every claim in it is backed by an edge in the evidence
subgraph that was retrieved for this match.  Here that is checked mechanically
rather than assumed.

Method
------
1. Split the explanation into sentences.
2. Run the *same* skill extractor used on résumés over each sentence, so a
   claim about a skill is detected identically wherever it appears.
3. A sentence is **grounded** when every skill it names appears in the evidence
   subgraph, and any relation it asserts between two skills exists as an edge.
4. Report the grounding rate and list every unsupported claim.

Two properties make this a meaningful check rather than a rubber stamp:

* It is **generator-agnostic** — the same validator runs over template output
  and LLM output, so the two are measured on identical terms.
* It **can fail**.  The test suite feeds it deliberately hallucinated text and
  asserts that it catches it.  A validator that never rejects anything would
  report a perfect grounding rate and mean nothing.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from ..extraction.skills import SkillExtractor
from ..types import EvidencePath, ValidationResult

# Sentence splitter: breaks on . ! ? followed by whitespace, and on newlines.
# Kept deliberately simple and inspectable; abbreviations are rare in generated
# explanations and a mis-split only ever makes the check stricter, not weaker.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Phrases that assert a *relation* between two skills. If a sentence uses one,
# the validator requires a matching edge, not merely that both skills exist.
RELATION_ASSERTIONS: Dict[str, Sequence[str]] = {
    "exact": ("matches", "meets", "satisfies", "covers directly", "exactly"),
    "narrower": ("is a kind of", "is a specialisation of", "specialises", "a form of"),
    "broader": ("is the broader field of", "is the general", "broader field", "parent field"),
    "related": ("is related to", "is adjacent to", "transfers to", "shares the field"),
}


def split_sentences(text: str) -> List[str]:
    """Split text into non-empty, stripped sentences."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class FaithfulnessValidator:
    """Checks generated explanations against the evidence that licensed them."""

    def __init__(self, extractor: SkillExtractor) -> None:
        self.extractor = extractor

    @staticmethod
    def _evidence_index(evidence: Sequence[EvidencePath]) -> Dict[str, Set[str]]:
        """Build the sets of URIs and relations the evidence licenses."""
        held: Set[str] = set()
        required: Set[str] = set()
        pairs: Set[str] = set()
        relations: Set[str] = set()

        for path in evidence:
            held.add(path.candidate_skill_uri)
            required.add(path.required_skill_uri)
            pairs.add(f"{path.candidate_skill_uri}|{path.required_skill_uri}")
            pairs.add(f"{path.required_skill_uri}|{path.candidate_skill_uri}")
            relations.add(f"{path.candidate_skill_uri}|{path.required_skill_uri}|{path.relation}")

        return {
            "held": held,
            "required": required,
            "all": held | required,
            "pairs": pairs,
            "relations": relations,
        }

    @staticmethod
    def _asserted_relations(sentence: str) -> List[str]:
        lowered = sentence.lower()
        return [
            relation
            for relation, cues in RELATION_ASSERTIONS.items()
            if any(cue in lowered for cue in cues)
        ]

    def validate(
        self,
        text: str,
        evidence: Sequence[EvidencePath],
        allowed_extra_uris: Optional[Sequence[str]] = None,
        min_grounding_rate: float = 0.8,
    ) -> ValidationResult:
        """
        Validate an explanation against its evidence.

        `allowed_extra_uris` whitelists concepts that may legitimately be named
        without being match evidence — specifically the skills identified as
        *gaps*, which an explanation should be able to mention as missing.
        Without this the validator would reject honest statements about what
        the candidate lacks, which is exactly the content users most need.
        """
        sentences = split_sentences(text)
        if not sentences:
            return ValidationResult(
                grounding_rate=0.0,
                total_sentences=0,
                grounded_sentences=0,
                unsupported_claims=[],
                passed=False,
            )

        index = self._evidence_index(evidence)
        permitted = set(index["all"]) | set(allowed_extra_uris or ())

        grounded_count = 0
        unsupported: List[Dict[str, object]] = []

        for position, sentence in enumerate(sentences):
            mentioned = self.extractor.mentioned_uris(sentence)
            ungrounded_entities = [u for u in mentioned if u not in permitted]

            # A relation assertion must correspond to a real edge.
            bad_relations: List[str] = []
            asserted = self._asserted_relations(sentence)
            if asserted and len(mentioned) >= 2:
                for relation in asserted:
                    ok = any(
                        f"{a}|{b}|{relation}" in index["relations"]
                        for a in mentioned
                        for b in mentioned
                        if a != b
                    )
                    if not ok:
                        bad_relations.append(relation)

            if not ungrounded_entities and not bad_relations:
                grounded_count += 1
            else:
                unsupported.append(
                    {
                        "sentence_index": position,
                        "sentence": sentence,
                        "ungrounded_entities": [
                            self.extractor.taxonomy.label(u) for u in ungrounded_entities
                        ],
                        "unsupported_relations": bad_relations,
                    }
                )

        rate = grounded_count / len(sentences)
        return ValidationResult(
            grounding_rate=round(rate, 4),
            total_sentences=len(sentences),
            grounded_sentences=grounded_count,
            unsupported_claims=unsupported,
            passed=rate >= min_grounding_rate,
        )

    def annotate_sentences(
        self,
        text: str,
        evidence: Sequence[EvidencePath],
        allowed_extra_uris: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """
        Per-sentence records with the evidence ids each one rests on.

        This is what lets the interface render a citation chip beside every
        sentence and highlight the matching graph edge on hover.
        """
        sentences = split_sentences(text)
        index = self._evidence_index(evidence)
        permitted = set(index["all"]) | set(allowed_extra_uris or ())

        by_uri: Dict[str, List[str]] = {}
        for path in evidence:
            by_uri.setdefault(path.candidate_skill_uri, []).append(path.evidence_id)
            by_uri.setdefault(path.required_skill_uri, []).append(path.evidence_id)

        records: List[Dict[str, object]] = []
        for position, sentence in enumerate(sentences):
            mentioned = self.extractor.mentioned_uris(sentence)
            cited: List[str] = []
            for uri in mentioned:
                for eid in by_uri.get(uri, []):
                    if eid not in cited:
                        cited.append(eid)
            records.append(
                {
                    "index": position,
                    "text": sentence,
                    "evidence_ids": sorted(cited, key=lambda e: int(e[1:]) if e[1:].isdigit() else 0),
                    "grounded": all(u in permitted for u in mentioned),
                }
            )
        return records
