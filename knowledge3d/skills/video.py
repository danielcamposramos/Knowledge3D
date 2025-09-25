from __future__ import annotations

"""Video embedding helper for K3D.

Attempts to summarise a video clip by sampling frames and extracting
frame-level statistics. When OpenCLIP is available it reuses the vision
embedding pipeline on sampled frames; otherwise it falls back to
lightweight colour histograms so every clip still yields a meaningful
descriptor.
"""

from pathlib import Path
from typing import List

import numpy as np  # type: ignore


def _frame_stats(frame: np.ndarray, bins: int = 24) -> np.ndarray:
    """Compute a compact descriptor from an HxWxC RGB frame."""
    if frame.ndim == 2:  # grayscale -> add channel axis
        frame = frame[..., None]
    frame = frame.astype(np.float32)
    if frame.max() > 1.0:
        frame /= 255.0
    means = frame.mean(axis=(0, 1))
    stds = frame.std(axis=(0, 1))
    hist = np.histogram(frame, bins=bins, range=(0.0, 1.0))[0]
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    vec = np.concatenate([means, stds, hist])
    return vec


def embed_video(path: str, *, num_samples: int = 8) -> List[float]:
    """Return a deterministic embedding for a video file."""

    video_path = Path(path)
    frames: List[np.ndarray] = []

    try:  # pragma: no cover - dependent on optional imageio
        import imageio.v3 as iio  # type: ignore

        meta = iio.immeta(video_path, exclude_applied=True)
        frame_count = int(meta.get("nframes") or meta.get("n_frames") or 0)
        if frame_count <= 0:
            raise ValueError("Unable to determine frame count")
        indices = np.linspace(0, frame_count - 1, num_samples).astype(int)
        for idx in indices:
            try:
                frame = iio.imread(video_path, index=int(idx))
                frames.append(frame)
            except Exception:
                continue
    except Exception:
        frames = []

    # Compute per-frame statistics; if OpenCLIP is present, promote to embeddings
    vectors: List[np.ndarray] = []
    if frames:
        try:  # pragma: no cover
            import torch  # type: ignore
            import open_clip  # type: ignore
            from PIL import Image  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model = model.to(device)
            for frame in frames:
                img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                vectors.append(feat.squeeze(0).detach().cpu().numpy())
        except Exception:
            vectors = []

    if not vectors:
        for frame in frames:
            vectors.append(_frame_stats(frame))

    if not vectors:
        try:
            import imageio.v3 as iio  # type: ignore

            frame = iio.imread(video_path, index=0)
            vec = _frame_stats(frame)
            return vec.tolist()
        except Exception:
            digest = video_path.read_bytes()[:128]
            if not digest:
                digest = video_path.as_posix().encode("utf-8")
            arr = np.frombuffer(digest, dtype=np.uint8)
            if arr.size < 96:
                arr = np.pad(arr, (0, 96 - arr.size), mode="wrap")
            arr = arr.astype(np.float32) / 255.0
            return arr.tolist()[:96]

    stacked = np.stack([np.array(v, dtype=np.float32).reshape(-1) for v in vectors], axis=0)
    mean_vec = stacked.mean(axis=0)
    if np.linalg.norm(mean_vec) > 0:
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return mean_vec.astype(np.float32).tolist()
