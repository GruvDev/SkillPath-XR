"""
Explanation generation.

Two providers behind one interface:

* `GroundedTemplateGenerator` (default) composes sentences *only* from the
  fields of evidence objects.  It cannot mention a skill that is not in the
  evidence because it never has access to any other text.  This makes it
  faithful by construction — a useful property, since it gives the system a
  guaranteed-correct baseline and gives the validator a known-good input to be
  tested against.

* `LLMGenerator` sends the evidence subgraph to a language model under a strict
  evidence-only instruction.  Fluency improves; faithfulness stops being
  guaranteed, which is precisely why every output is passed through the
  validator and withheld if it fails.

The pairing is the point.  The template shows what perfect grounding looks
like; the LLM shows what the grounding check is *for*.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

from ..config import SETTINGS, Settings
from ..types import EvidencePath, JobPosting, SkillGap

logger = logging.getLogger(__name__)

RELATION_PHRASES = {
    "exact": "matches",
    "narrower": "is a kind of",
    "broader": "is the broader field of",
    "related": "is related to",
}


def _join(items: Sequence[str], conjunction: str = "and") -> str:
    """Oxford-comma list join."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(items[:-1]) + f", {conjunction} " + items[-1]


class BaseExplanationGenerator:
    name = "base"

    def generate(
        self,
        job: JobPosting,
        evidence: Sequence[EvidencePath],
        gaps: Sequence[SkillGap],
        score: float,
    ) -> str:
        raise NotImplementedError


class GroundedTemplateGenerator(BaseExplanationGenerator):
    """
    Deterministic generator whose every sentence is assembled from evidence.

    Sentences are emitted in a fixed order — direct matches, transferable
    evidence, then gaps — so the reader always encounters the strongest
    justification first and the caveats last.
    """

    name = "grounded_template"

    def generate(
        self,
        job: JobPosting,
        evidence: Sequence[EvidencePath],
        gaps: Sequence[SkillGap],
        score: float,
    ) -> str:
        sentences: List[str] = []

        exact = [e for e in evidence if e.relation == "exact"]
        transfer = [e for e in evidence if e.relation in ("narrower", "broader", "related")]
        essential_exact = [e for e in exact if e.requirement_type == "essential"]

        # 1. Direct evidence on essential requirements.
        if essential_exact:
            labels = [e.required_skill_label for e in essential_exact]
            sentences.append(
                f"This role requires {_join(labels)}, and the resume evidences "
                f"{'each' if len(labels) > 1 else 'it'} directly."
            )
        elif exact:
            labels = [e.required_skill_label for e in exact]
            sentences.append(f"The resume directly evidences {_join(labels)}.")

        # 2. Transferable evidence, grouped by relation so the wording stays
        #    accurate to the relation the graph actually recorded, then merged
        #    into a single sentence so the text does not repeat its own frame.
        clauses: List[str] = []
        for relation in ("narrower", "broader", "related"):
            group = [e for e in transfer if e.relation == relation]
            if not group:
                continue
            phrase = RELATION_PHRASES[relation]
            clauses.extend(
                f"{e.candidate_skill_label} {phrase} {e.required_skill_label}"
                for e in group[:3]
            )
        if clauses:
            sentences.append(f"Partial evidence comes from {_join(clauses[:6])}.")

        # 3. Gaps, named honestly, with agreement matched to the count.
        essential_gaps = [g for g in gaps if g.requirement_type == "essential"]
        optional_gaps = [g for g in gaps if g.requirement_type == "optional"]
        if essential_gaps:
            labels = [g.skill_label for g in essential_gaps]
            sentences.append(
                f"No evidence was found for {_join(labels)}, which the role lists "
                f"as {'an essential requirement' if len(labels) == 1 else 'essential requirements'}."
            )
        if optional_gaps:
            labels = [g.skill_label for g in optional_gaps]
            noun = "requirement" if len(labels) == 1 else "requirements"
            verb = "is" if len(labels) == 1 else "are"
            sentences.append(
                f"The optional {noun} {_join(labels)} {verb} also unevidenced."
            )

        if not sentences:
            sentences.append("No overlapping skill evidence was found for this role.")

        return " ".join(sentences)


class LLMGenerator(BaseExplanationGenerator):
    """
    Language-model generator constrained to the retrieved evidence subgraph.

    Talks to any OpenAI-compatible chat endpoint or to Ollama.  Uses only the
    standard library for HTTP so the package has no hard client dependency.
    On any failure it raises, and the pipeline falls back to the template
    generator rather than showing the user nothing.
    """

    name = "llm"

    SYSTEM_PROMPT = (
        "You explain why a job was recommended to a candidate.\n"
        "You will receive a list of EVIDENCE items and a list of GAPS.\n"
        "Rules you must follow exactly:\n"
        "1. Mention ONLY skills that appear in the EVIDENCE or GAPS lists.\n"
        "2. Never state that the candidate has a skill unless it appears as "
        "candidate_skill in EVIDENCE.\n"
        "3. Describe a relationship between two skills only if an EVIDENCE item "
        "records that exact relation.\n"
        "4. Write 2 to 4 plain sentences. No bullet points, no preamble, no "
        "invented detail about the person or the company.\n"
        "5. If a skill is in GAPS, you may say it is missing, never that it is held."
    )

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or SETTINGS

    def _build_user_prompt(
        self,
        job: JobPosting,
        evidence: Sequence[EvidencePath],
        gaps: Sequence[SkillGap],
        score: float,
    ) -> str:
        payload = {
            "job_title": job.title,
            "match_score": round(score, 3),
            "evidence": [
                {
                    "id": e.evidence_id,
                    "candidate_skill": e.candidate_skill_label,
                    "required_skill": e.required_skill_label,
                    "relation": e.relation,
                    "requirement": e.requirement_type,
                }
                for e in evidence
            ],
            "gaps": [
                {"skill": g.skill_label, "requirement": g.requirement_type} for g in gaps
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def generate(
        self,
        job: JobPosting,
        evidence: Sequence[EvidencePath],
        gaps: Sequence[SkillGap],
        score: float,
    ) -> str:
        prompt = self._build_user_prompt(job, evidence, gaps, score)
        provider = self.settings.explanation_provider
        if provider == "ollama":
            return self._call_ollama(prompt)
        return self._call_openai_compatible(prompt)

    # -- transports ----------------------------------------------------------

    def _post_json(self, url: str, body: Dict[str, object], headers: Dict[str, str]) -> Dict:
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request to {url} failed: {exc}") from exc

    def _call_ollama(self, prompt: str) -> str:
        payload = self._post_json(
            f"{self.settings.llm_base_url.rstrip('/')}/api/chat",
            {
                "model": self.settings.llm_model,
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            {"Content-Type": "application/json"},
        )
        return str(payload.get("message", {}).get("content", "")).strip()

    def _call_openai_compatible(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        payload = self._post_json(
            f"{self.settings.llm_base_url.rstrip('/')}/v1/chat/completions",
            {
                "model": self.settings.llm_model,
                "temperature": 0.2,
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            headers,
        )
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("LLM returned no choices")
        return str(choices[0].get("message", {}).get("content", "")).strip()


def get_generator(settings: Optional[Settings] = None) -> BaseExplanationGenerator:
    """Build the configured generator; template is the safe default."""
    cfg = settings or SETTINGS
    if cfg.explanation_provider in ("ollama", "openai"):
        return LLMGenerator(cfg)
    return GroundedTemplateGenerator()
