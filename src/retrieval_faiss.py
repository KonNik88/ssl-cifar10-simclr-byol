from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.embeddings import load_embeddings_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=True, help="path to .npz embeddings")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    pack, meta = load_embeddings_npz(emb_path)

    Z = pack.z.astype(np.float32)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    try:
        import faiss
    except Exception as e:
        raise RuntimeError(
            "FAISS is not installed. Install one of:\n"
            "  pip install faiss-cpu\n"
            "  pip install faiss-gpu\n"
            f"Original error: {repr(e)}"
        )

    d = Z.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized = cosine
    index.add(Z)

    # demo: query first item
    q = Z[0:1]
    sims, idx = index.search(q, args.k)
    print("Meta:", meta)
    print("Top-k indices:", idx[0].tolist())
    print("Top-k sims   :", sims[0].tolist())


if __name__ == "__main__":
    main()
