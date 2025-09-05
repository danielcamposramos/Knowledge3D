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
    """Reduce dimensionality to 3D using (GPU) UMAP when available, else CPU fallback.

    - Uses RAPIDS cuML UMAP when present and GPU is desired.
    - Falls back to umap-learn on CPU, or PCA when UMAP is unavailable.
    """
    from sklearn.decomposition import PCA

    red = (method or "pca").lower()
    n = vectors.shape[0]
    if red != "umap":
        pca = PCA(n_components=3)
        return pca.fit_transform(vectors)

    # UMAP chosen
    if n <= 3:
        pca = PCA(n_components=min(3, n))
        projected = pca.fit_transform(vectors)
        if projected.shape[1] < 3:
            pad = np.zeros((projected.shape[0], 3), dtype=np.float32)
            pad[:, : projected.shape[1]] = projected
            return pad
        return projected

    # Try cuML UMAP if GPU desired and available
    if _want_gpu() and _has_cuml():
        try:
            from cuml.manifold import UMAP  # type: ignore

            um = UMAP(n_components=3, n_neighbors=min(15, max(2, n - 1)))
            out = um.fit_transform(vectors)
            return np.asarray(out, dtype=np.float32)
        except Exception:
            pass

    # CPU umap-learn
    try:
        import umap  # type: ignore

        um = umap.UMAP(n_components=3, n_neighbors=min(15, max(2, n - 1)))
        return np.asarray(um.fit_transform(vectors), dtype=np.float32)
    except Exception:
        # PCA fallback
        pca = PCA(n_components=3)
        return pca.fit_transform(vectors)


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
            return gpu_index
        except Exception:
            # fallback to CPU
            pass
    cpu_index.add(vectors.astype(np.float32))
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
    """Compute k-NN for each row in vectors using FAISS (GPU preferred).

    Parameters
    - ann: 'flat' (default) or 'ivf'
    - nlist, nprobe: IVF parameters; heuristics applied if None

    Returns
    - (N, k) int64 array of neighbor indices excluding self
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    n = vectors.shape[0]
    if k >= n:
        raise ValueError("k must be less than the number of vectors")

    ann_kind = (ann or os.getenv("K3D_FAISS_INDEX", "flat")).lower().strip()
    use_ivf = ann_kind in {"ivf", "ivf-flat", "ivfflat"}

    # FAISS path (GPU preferred)
    try:
        import faiss  # type: ignore

        x = vectors.astype(np.float32)
        d = x.shape[1]
        if use_ivf:
            nl, npb = _heuristic_ivf_params(n)
            if nlist:
                nl = int(nlist)
            if nprobe:
                npb = int(nprobe)
            quant = faiss.IndexFlatL2(d)
            cpu = faiss.IndexIVFFlat(quant, d, nl, faiss.METRIC_L2)
            # IVFFlat requires training
            # Use a sample up to 100k for speed
            m = min(n, 100_000)
            cpu.train(x[:m])
            ok, has_gpu = _has_faiss()
            if _want_gpu() and has_gpu:
                gpu = faiss.index_cpu_to_all_gpus(cpu)
                gpu.nprobe = npb
                gpu.add(x)
                index = gpu
            else:
                cpu.nprobe = npb
                cpu.add(x)
                index = cpu
        else:
            index = _faiss_gpu_index(x)
            if index is None:
                # CPU flat
                cpu = faiss.IndexFlatL2(d)
                cpu.add(x)
                index = cpu
        # Query in batches
        batch = 10000 if n >= 200000 else 20000
        I_list: list[np.ndarray] = []
        for start in range(0, n, batch):
            end = min(n, start + batch)
            _, I = index.search(x[start:end], k + 1)
            I = I[:, 1:]
            I_list.append(I)
        out = np.vstack(I_list)
        return out.astype(np.int64)
    except Exception:
        # sklearn fallback
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(vectors)
        _, idx = nn.kneighbors(vectors)
        return idx[:, 1:].astype(np.int64)

    # sklearn fallback
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(vectors)
        _, idx = nn.kneighbors(vectors)
        return idx[:, 1:].astype(np.int64)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to compute neighbors: {exc}") from exc


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
