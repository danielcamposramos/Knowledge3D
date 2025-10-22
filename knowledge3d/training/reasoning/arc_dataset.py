"""
ARC-AGI dataset helpers for TRM reasoning training.

The ARC dataset is downloaded into `Knowledge3D.local/datasets/arc_agi/` by
default. We build a cache of (question, answer) embedding pairs so training
loops can iterate quickly without re-embedding grids each run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np

from knowledge3d.cranium.utils.trm import expand_embedding_to_trm, TRM_STATE_DIM

ARC_DATASET_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_ARCHIVE_NAME = "ARC-AGI-master.zip"
ARC_EXTRACTED_DIR = "ARC-AGI-master"
DEFAULT_LOCAL_ROOT = Path("/K3D/Knowledge3D.local/datasets/arc_agi")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_root() -> Path:
    return DEFAULT_LOCAL_ROOT


def ensure_arc_dataset(
    root: Path | None = None,
    url: str = ARC_DATASET_URL,
    force_download: bool = False,
) -> Path:
    """
    Download and extract the ARC-AGI dataset if it is not already present.

    Returns the path to the extracted dataset directory (containing `data/`).
    """
    root = Path(root) if root is not None else _default_root()
    archive_path = root / ARC_ARCHIVE_NAME
    extracted_path = root / ARC_EXTRACTED_DIR

    root.mkdir(parents=True, exist_ok=True)

    if force_download and extracted_path.exists():
        shutil.rmtree(extracted_path)

    if not extracted_path.exists():
        if force_download and archive_path.exists():
            archive_path.unlink()
        if not archive_path.exists():
            _ensure_parent(archive_path)
            print(f"📥 Downloading ARC-AGI dataset from {url} → {archive_path}")
            urllib.request.urlretrieve(url, archive_path)
            print("✅ Download complete")
        print(f"📦 Extracting ARC-AGI archive to {root}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(root)
        print("✅ Extraction complete")

    if not extracted_path.exists():
        raise FileNotFoundError(
            f"ARC-AGI dataset not found after extraction at {extracted_path}"
        )
    return extracted_path


def _detect_dataset_dir(root: Path) -> Path:
    candidates = [
        root / ARC_EXTRACTED_DIR,
        root / "ARC-AGI",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find ARC dataset under {root}. "
        "Run `ensure_arc_dataset` first or set `download=True`."
    )


def _grid_to_text(grid: Sequence[Sequence[int]]) -> str:
    """
    Convert an ARC grid into a token string suitable for the RPN embedding.
    Rows are separated with `|` to preserve coarse spatial structure.
    """
    rows: List[str] = []
    for row in grid:
        row_tokens = " ".join(str(int(cell)) for cell in row)
        rows.append(row_tokens)
    return " | ".join(rows)


def _iter_task_files(dataset_dir: Path, split: str = "training") -> Iterable[Path]:
    split_dir = dataset_dir / "data" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ARC split directory missing: {split_dir}")
    return sorted(p for p in split_dir.iterdir() if p.suffix == ".json")


def _load_task(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class ARCReasoningCache:
    questions: np.ndarray
    answers: np.ndarray
    task_ids: np.ndarray
    example_indices: np.ndarray
    metadata: dict


def prepare_arc_reasoning_cache(
    rpn_embed_sentence: Callable[[str], np.ndarray],
    dataset_root: Path | None = None,
    cache_path: Path | None = None,
    *,
    limit: int | None = None,
    rebuild: bool = False,
    download: bool = True,
    project_fn: Callable[[np.ndarray], np.ndarray] = expand_embedding_to_trm,
) -> Path:
    """
    Build (or reuse) a cached set of ARC reasoning pairs.

    Args:
        rpn_embed_sentence: Callable returning a 128-dim embedding given text.
        dataset_root: Directory containing the ARC dataset root.
        cache_path: Target path for the `.npz` cache.
        limit: Optional cap on the number of (question, answer) pairs.
        rebuild: Force cache regeneration even if it already exists.
        download: Download the dataset if missing.
        project_fn: Function mapping embeddings to the TRM dimension.
    """
    dataset_root = Path(dataset_root) if dataset_root is not None else _default_root()
    if download:
        dataset_dir = ensure_arc_dataset(dataset_root)
    else:
        dataset_dir = _detect_dataset_dir(dataset_root)

    cache_path = Path(cache_path) if cache_path is not None else dataset_root / "arc_reasoning_pairs.npz"
    meta_path = cache_path.with_suffix(".json")

    if cache_path.exists() and not rebuild:
        return cache_path

    task_ids: List[str] = []
    example_indices: List[int] = []
    question_vecs: List[np.ndarray] = []
    answer_vecs: List[np.ndarray] = []

    pair_count = 0
    for task_path in _iter_task_files(dataset_dir, split="training"):
        task_id = task_path.stem
        payload = _load_task(task_path)
        train_pairs = payload.get("train", [])
        for idx, example in enumerate(train_pairs):
            question_grid = example["input"]
            answer_grid = example["output"]
            question_text = _grid_to_text(question_grid)
            answer_text = _grid_to_text(answer_grid)
            q_emb = rpn_embed_sentence(question_text)
            a_emb = rpn_embed_sentence(answer_text)
            question_vecs.append(project_fn(q_emb))
            answer_vecs.append(project_fn(a_emb))
            task_ids.append(task_id)
            example_indices.append(int(idx))
            pair_count += 1
            if limit is not None and pair_count >= limit:
                break
        if limit is not None and pair_count >= limit:
            break

    if not question_vecs:
        raise RuntimeError("No ARC reasoning pairs were generated.")

    questions = np.vstack(question_vecs).astype(np.float32)
    answers = np.vstack(answer_vecs).astype(np.float32)
    max_id_len = max(len(t) for t in task_ids)
    task_id_array = np.array(task_ids, dtype=f"<U{max_id_len}")
    indices_array = np.array(example_indices, dtype=np.int32)

    _ensure_parent(cache_path)
    np.savez_compressed(
        cache_path,
        questions=questions,
        answers=answers,
        task_ids=task_id_array,
        example_indices=indices_array,
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pairs": int(pair_count),
        "limit": limit,
        "dataset_root": str(dataset_dir),
        "embedding_dim": int(TRM_STATE_DIM),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return cache_path


def load_arc_reasoning_cache(cache_path: Path | str) -> ARCReasoningCache:
    """
    Load a previously prepared ARC reasoning cache.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"ARC reasoning cache not found: {cache_path}")

    payload = np.load(cache_path, allow_pickle=False)
    questions = payload["questions"].astype(np.float32)
    answers = payload["answers"].astype(np.float32)
    task_ids = payload["task_ids"]
    example_indices = payload["example_indices"].astype(np.int32)
    payload.close()

    meta_path = cache_path.with_suffix(".json")
    metadata: dict = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    return ARCReasoningCache(
        questions=questions,
        answers=answers,
        task_ids=task_ids.astype(str),
        example_indices=example_indices,
        metadata=metadata,
    )
