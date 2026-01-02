from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class RetrievalResult:
    query_index: int
    neighbors: np.ndarray          # (k,) indices
    scores: np.ndarray             # (k,) cosine similarities
    query_label: Optional[int] = None
    neighbor_labels: Optional[np.ndarray] = None


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def cosine_topk(
    Z: np.ndarray,
    query_idx: int,
    k: int = 10,
    exclude_self: bool = True,
    y: Optional[np.ndarray] = None,
) -> RetrievalResult:
    """
    Z: (N, D) embeddings
    query_idx: index in [0..N-1]
    Returns top-k nearest neighbors by cosine similarity.
    """
    assert Z.ndim == 2, "Z must be (N, D)"
    N = Z.shape[0]
    assert 0 <= query_idx < N

    Z = Z.astype(np.float32, copy=False)
    ZN = l2_normalize(Z, axis=1)

    q = ZN[query_idx]  # (D,)
    sims = ZN @ q       # (N,)

    if exclude_self:
        sims[query_idx] = -np.inf

    # top-k indices (unordered), then sort
    k_eff = min(k, N - (1 if exclude_self else 0))
    idx = np.argpartition(-sims, kth=k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-sims[idx])]
    scores = sims[idx]

    res = RetrievalResult(
        query_index=query_idx,
        neighbors=idx.astype(np.int64),
        scores=scores.astype(np.float32),
        query_label=int(y[query_idx]) if y is not None else None,
        neighbor_labels=y[idx].astype(int) if y is not None else None,
    )
    return res


def batch_cosine_topk(
    Z: np.ndarray,
    query_indices: List[int],
    k: int = 10,
    exclude_self: bool = True,
    y: Optional[np.ndarray] = None,
) -> List[RetrievalResult]:
    return [cosine_topk(Z, qi, k=k, exclude_self=exclude_self, y=y) for qi in query_indices]


def class_hit_at_k(res: RetrievalResult, k: int = 10) -> Optional[float]:
    """
    Returns 1.0 if among top-k neighbors there is at least one sample
    with the same label as the query, else 0.0.
    """
    if res.query_label is None or res.neighbor_labels is None:
        return None
    k = min(k, len(res.neighbors))
    return float(np.any(res.neighbor_labels[:k] == res.query_label))


def mean_class_hit_at_k(results: List[RetrievalResult], k: int = 10) -> Optional[float]:
    vals = []
    for r in results:
        v = class_hit_at_k(r, k=k)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))
