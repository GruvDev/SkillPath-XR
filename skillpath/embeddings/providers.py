"""
Embedding providers.

Two implementations behind one interface:

* `HashingEmbedder` — a deterministic character- and word-level hashing
  vectoriser with sub-word character n-grams.  No downloads, no GPU, runs
  anywhere, and gives stable vectors across runs and machines.  This is the
  default so the system is always runnable and the test suite is reproducible.

* `SentenceTransformerEmbedder` — real dense semantic embeddings.  Enable with
  SKILLPATH_EMBEDDINGS=sentence_transformers once the package and model are
  available.

`get_embedder()` selects between them and falls back cleanly with a warning
rather than crashing if the production package is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional, Sequence

import numpy as np

from ..config import SETTINGS, Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


class BaseEmbedder:
    """Interface every embedding provider implements."""

    name: str = "base"
    dim: int = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
        """
        Scale rows to unit length so a dot product is a cosine similarity.

        Zero rows (empty input) are left as zeros rather than divided, which
        would produce NaN and silently poison every downstream comparison.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.where(norms == 0.0, 1.0, norms)
        return matrix / safe


class HashingEmbedder(BaseEmbedder):
    """
    Deterministic hashing vectoriser.

    Each text is mapped into a fixed-dimension space by hashing its word tokens
    and character 3-grams into buckets with signed counts, then L2-normalised.
    Character n-grams give partial credit for morphological overlap
    ("postgres"/"postgresql"), which a pure word-hash would miss entirely.
    """

    name = "hashing"

    def __init__(self, dim: int = 256, char_ngram: int = 3) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim
        self.char_ngram = char_ngram

    @staticmethod
    def _bucket(feature: str, dim: int) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % dim

    @staticmethod
    def _sign(feature: str) -> float:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=1).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    def _features(self, text: str) -> List[str]:
        lowered = text.lower()
        tokens = _TOKEN_RE.findall(lowered)
        features: List[str] = [f"w:{t}" for t in tokens]
        # Word bigrams capture short phrases like "machine learning".
        features.extend(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        for token in tokens:
            if len(token) >= self.char_ngram:
                padded = f"^{token}$"
                features.extend(
                    f"c:{padded[i : i + self.char_ngram]}"
                    for i in range(len(padded) - self.char_ngram + 1)
                )
        return features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            if not text:
                continue
            for feature in self._features(text):
                matrix[row, self._bucket(feature, self.dim)] += self._sign(feature)
        return self._l2_normalise(matrix)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense semantic embeddings from a sentence-transformers checkpoint."""

    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(settings: Optional[Settings] = None) -> BaseEmbedder:
    """
    Build the configured embedder.

    Falls back to the offline hashing embedder if the production provider
    cannot be constructed, logging why.  A recommendation engine that refuses
    to start because a model download failed is worse than one that starts in a
    documented degraded mode and says so in its status output.
    """
    cfg = settings or SETTINGS
    if cfg.embedding_provider == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder(cfg.embedding_model)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, then reported
            logger.warning(
                "sentence-transformers unavailable (%s); using offline hashing embedder",
                exc,
            )
    return HashingEmbedder(dim=cfg.embedding_dim)
