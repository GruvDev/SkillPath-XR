"""
Dense vector index.

An exact cosine-similarity index over normalised embeddings, backed by NumPy.
At the scale this system operates on — tens of thousands of postings — exact
search is fast enough and avoids the approximation error and tuning burden of
an ANN index.  A FAISS adapter is provided for when the corpus outgrows that.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """Exact cosine similarity search over L2-normalised vectors."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Add vectors to the index. Rows must align with `ids`."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected 2-D array, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids ({len(ids)}) and vectors ({vectors.shape[0]}) length mismatch"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected dim {self.dim}, got {vectors.shape[1]}"
            )

        self.ids.extend(ids)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return up to `top_k` (id, similarity) pairs, highest first.

        Similarities are clipped into [0, 1]: with normalised vectors a cosine
        can legitimately be negative, but a negative *relevance* is not
        meaningful to any downstream consumer and would corrupt rank fusion.
        """
        if self._matrix is None or not self.ids:
            return []
        if top_k <= 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} != index dim {self.dim}")

        scores = self._matrix @ query
        k = min(top_k, len(self.ids))
        # argpartition finds the top-k without fully sorting the array.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(self.ids[i], float(np.clip(scores[i], 0.0, 1.0))) for i in top_idx]

    def score_for(self, item_id: str, query: np.ndarray) -> float:
        """Similarity for one specific id, or 0.0 if it is not indexed."""
        if self._matrix is None or item_id not in self.ids:
            return 0.0
        idx = self.ids.index(item_id)
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        return float(np.clip(self._matrix[idx] @ query, 0.0, 1.0))

    def __len__(self) -> int:
        return len(self.ids)


class FaissVectorIndex(VectorIndex):
    """
    FAISS-backed index for large corpora.

    Uses inner product on normalised vectors, which is equivalent to cosine
    similarity.  Falls back to the NumPy implementation if FAISS is missing so
    the system still starts.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._faiss = None
        self._index = None
        try:
            import faiss  # imported lazily

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dim)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable (%s); using NumPy index", exc)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if self._index is None:
            super().add(ids, vectors)
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        self.ids.extend(ids)
        self._index.add(vectors)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._index is None:
            return super().search(query, top_k)
        if not self.ids or top_k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, idx = self._index.search(query, min(top_k, len(self.ids)))
        return [
            (self.ids[i], float(np.clip(s, 0.0, 1.0)))
            for s, i in zip(scores[0], idx[0])
            if i >= 0
        ]
