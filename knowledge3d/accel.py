from __future__ import annotations

"""
Acceleration utilities for CUDA-first UMAP and KNN.

Auto-detects GPU libraries and falls back to CPU implementations when missing.

Environment variables
- K3D_ACCEL: 'auto' (default), 'gpu', or 'cpu'
"""

import os
from typing import Any, Optional, Tuple

import numpy as np


def _want_gpu() -> bool:
    pref = os.getenv("K3D_ACCEL", "auto").lower().strip()
    if pref == "gpu":
        return True
    if pref == "cpu":
        return False
    # auto: prefer GPU when available
    return True


def _has_cuml() -> bool:
    try:
        import cuml  # noqa: F401
        return True
    except Exception:
        return False


def _has_faiss() -> Tuple[bool, bool]:
    """Return (faiss_available, faiss_has_gpu)."""
    try:
        import faiss  # type: ignore  # noqa: F401
    except Exception:
        return (False, False)
    # Check GPU components
    try:
        import faiss  # type: ignore
        ng = 0
        try:
            ng = int(getattr(faiss, "get_num_gpus", lambda: 0)())
        except Exception:
            ng = 0
        return (True, ng > 0)
    except Exception:
        return (False, False)


def reduce_to_3d(vectors: np.ndarray, method: str = "umap") -> np.ndarray:
    """Reduce dimensionality to 3D.

    STRICT GPU POLICY:
    - For UMAP variants ("umap", "umap_fast", "umap_high"), require RAPIDS cuML on GPU.
      CPU fallbacks (umap-learn/PCA) have been intentionally removed for performance.
      Uncomment the older CPU fallback code if you need non-GPU runs.
    - For non-UMAP methods (e.g., "pca"), PCA is used on CPU (lightweight).
    """
    from sklearn.decomposition import PCA

    red = (method or "pca").lower()
    n = vectors.shape[0]
    if red not in {"umap", "umap_fast", "umap_high"}:
        # Non-UMAP reducers (e.g., PCA) remain CPU-based and fast.
        try:
            from .utils.env_guard import accel_log  # type: ignore
            accel_log("Reduction=non-UMAP -> using PCA (CPU)")
        except Exception:
            pass
        pca = PCA(n_components=3)
        return pca.fit_transform(vectors)

    # UMAP chosen (variants: umap, umap_fast, umap_high) — require GPU cuML
    if n <= 3:
        pca = PCA(n_components=min(3, n))
        projected = pca.fit_transform(vectors)
        if projected.shape[1] < 3:
            pad = np.zeros((projected.shape[0], 3), dtype=np.float32)
            pad[:, : projected.shape[1]] = projected
            return pad
        return projected

    if not (_want_gpu() and _has_cuml()):
        raise RuntimeError(
            "UMAP reduction requires RAPIDS cuML on GPU (CPU fallback disabled)."
        )

    from cuml.manifold import UMAP  # type: ignore
    if red == "umap_fast":
        nn = min(10, max(2, n - 1)); md = 0.5
    elif red == "umap_high":
        nn = min(30, max(2, n - 1)); md = 0.1
    else:
        nn = min(15, max(2, n - 1)); md = 0.3
    um = UMAP(n_components=3, n_neighbors=nn, min_dist=md)
    out = um.fit_transform(vectors)
    try:
        from .utils.env_guard import accel_log  # type: ignore
        accel_log(f"UMAP via RAPIDS cuML (GPU) mode={red}")
    except Exception:
        pass
    return np.asarray(out, dtype=np.float32)


def _faiss_gpu_index(vectors: np.ndarray) -> Optional[Any]:
    """Build a FAISS GPU index (IndexFlatL2) for all-vs-all search.

    Falls back to CPU index if GPU shards are unavailable.
    """
    ok, has_gpu = _has_faiss()
    if not ok:
        return None
    import faiss  # type: ignore

    d = vectors.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    if _want_gpu() and has_gpu:
        try:
            res = []
            for _ in range(faiss.get_num_gpus()):
                res.append(faiss.StandardGpuResources())
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            gpu_index.add(vectors.astype(np.float32))
            try:
                from .utils.env_guard import accel_log  # type: ignore
                accel_log("FAISS IndexFlatL2 (GPU shards)")
            except Exception:
                pass
            return gpu_index
        except Exception:
            # fallback to CPU
            pass
    cpu_index.add(vectors.astype(np.float32))
    try:
        from .utils.env_guard import accel_log  # type: ignore
        accel_log("FAISS IndexFlatL2 (CPU)")
    except Exception:
        pass
    return cpu_index


def _heuristic_ivf_params(n: int) -> tuple[int, int]:
    # nlist ~ 4*sqrt(n), capped
    import math
    nlist = int(max(1024, min(65536, 4 * math.sqrt(max(1, n)))))
    nprobe = int(max(8, min(128, nlist // 32)))
    return nlist, nprobe


def knn_all(
    vectors: np.ndarray,
    k: int,
    ann: str | None = None,
    nlist: int | None = None,
    nprobe: int | None = None,
) -> np.ndarray:
    """Compute k-NN for each row using FAISS on GPU.

    STRICT GPU POLICY:
    - Requires FAISS with GPU; CPU/scikit-learn fallbacks removed.
      Uncomment the old fallback code if CPU support is needed.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    n = vectors.shape[0]
    if k >= n:
        raise ValueError("k must be less than the number of vectors")

    ann_kind = (ann or os.getenv("K3D_FAISS_INDEX", "flat")).lower().strip()
    use_ivf = ann_kind in {"ivf", "ivf-flat", "ivfflat", "ivfpq"}

    # Try FAISS GPU first
    try:
        import faiss  # type: ignore
        ok, has_gpu = _has_faiss()
        if not (_want_faiss_gpu() and ok and has_gpu):
            raise RuntimeError("FAISS GPU unavailable")

        x = np.ascontiguousarray(vectors, dtype=np.float32)
        d = x.shape[1]
        if use_ivf:
            nl, npb = _heuristic_ivf_params(n)
            if nlist:
                nl = int(nlist)
            if nprobe:
                npb = int(nprobe)
            quant = faiss.IndexFlatL2(d)
            if ann_kind == "ivfpq":
                try:
                    M = int(os.getenv("K3D_FAISS_PQ_M", "16"))
                except Exception:
                    M = 16 if d % 16 == 0 else 8
                try:
                    nbits = int(os.getenv("K3D_FAISS_PQ_BITS", "8"))
                except Exception:
                    nbits = 8
                cpu = faiss.IndexIVFPQ(quant, d, nl, M, nbits)
            else:
                cpu = faiss.IndexIVFFlat(quant, d, nl, faiss.METRIC_L2)
            m = min(n, 100_000)
            cpu.train(x[:m])
            gpu = faiss.index_cpu_to_all_gpus(cpu)
            gpu.nprobe = npb
            gpu.add(x)
            index = gpu
        else:
            index = _faiss_gpu_index(x)
            if index is None:
                raise RuntimeError("FAISS GPU IndexFlatL2 unavailable")
        batch = 10000 if n >= 200000 else 20000
        I_list: list[np.ndarray] = []
        for start in range(0, n, batch):
            end = min(n, start + batch)
            _, I = index.search(x[start:end], k + 1)
            I = I[:, 1:]
            I_list.append(I)
        out = np.vstack(I_list)
        return out.astype(np.int64)
    except Exception as e:
        # Fallback to RAPIDS cuML KNN (still GPU) if available
        if not _has_cuml():
            raise
        try:
            from cuml.neighbors import NearestNeighbors  # type: ignore
        except Exception:
            raise
        # cuML expects float32
        x = np.asarray(vectors, dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(x)
        _, I = nn.kneighbors(x)
        I = I[:, 1:]
        return np.asarray(I, dtype=np.int64)


def st_device_kwargs() -> dict:
    """Return kwargs to prefer CUDA for Sentence-Transformers if available."""
    dev = os.getenv("K3D_ACCEL", "auto").lower().strip()
    if dev == "cpu":
        return {"device": "cpu"}
    # auto/gpu: try CUDA
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return {"device": "cuda"}
    except Exception:
        pass
    return {}
def _want_faiss_gpu() -> bool:
    v = os.getenv("K3D_FAISS_DEVICE", "auto").lower().strip()
    if v == "cpu":
        return False
    if v == "gpu":
        return True
    return _want_gpu()
