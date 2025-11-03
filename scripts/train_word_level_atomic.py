#!/usr/bin/env python3
"""
Train word-context character embeddings using PDFs with OCR layers.

This script complements atomic glyph training by exposing characters to real
document layouts (varied fonts, scales, noise) and fusing visual features with
Matryoshka + RPN text embeddings. The updated embeddings are consolidated back
into the Galaxy checkpoint.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np

from PIL import Image

try:  # Optional dependency; falls back to PIL-based ops if missing
    import cv2  # type: ignore
except Exception:
    cv2 = None

from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM
from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CHECKPOINT_DIR = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")
GALAXY_PATH = CHECKPOINT_DIR / "galaxy_character_embeddings.npz"
MAX_PDFS = 100
TARGET_DIM_HIGH = 128
TARGET_DIM_LOW = 64
EMA_DECAY = 0.9

# Matryoshka + RPN instances (shared)
matryoshka = MatryoshkaTRM(max_dims=2048, min_dims=64)
rpn_engine = RPNEmbeddingEngine(embedding_dim=TARGET_DIM_HIGH)


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec.astype(np.float32)


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize(size, Image.BILINEAR)
    return np.array(pil_img)


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pil_img = Image.fromarray(image)
    return np.array(pil_img.convert("L"))


def project_embedding(vector: np.ndarray, target_dim: int) -> np.ndarray:
    weights = matryoshka.get_base_at_dim(target_dim)
    resized = np.zeros(target_dim, dtype=np.float32)
    length = min(vector.size, target_dim)
    resized[:length] = vector[:length]
    projected = weights @ resized
    return _normalize(projected)


def fuse_visual_text(char: str, visual_embedding: np.ndarray) -> np.ndarray:
    text_embedding = rpn_engine.embed_word(char)
    fused = (visual_embedding + text_embedding) * 0.5
    return _normalize(fused)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
# PDF Extraction
# --------------------------------------------------------------------------- #

def extract_word_patches(pdf_path: Path) -> List[Dict[str, object]]:
    """Extract word patches and metadata from a PDF with OCR layer."""
    doc = fitz.open(pdf_path)
    word_entries: List[Dict[str, object]] = []

    for page_index, page in enumerate(doc):
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    word_text = span.get("text", "").strip()
                    bbox = span.get("bbox")
                    chars = span.get("chars", [])
                    if not word_text or not bbox:
                        continue

                    # Render word patch as grayscale image
                    pix = page.get_pixmap(clip=bbox, alpha=False)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 3:
                        img = _rgb_to_gray(img)
                    else:
                        img = img[:, :, 0]

                    char_bboxes: List[Tuple[int, int, int, int]] = []
                    for ch in chars:
                        cb = ch.get("bbox")
                        if not cb:
                            continue
                        x0, y0, x1, y1 = cb
                        char_bboxes.append((int(x0 - bbox[0]), int(y0 - bbox[1]), int(x1 - bbox[0]), int(y1 - bbox[1])))

                    word_entries.append(
                        {
                            "text": word_text,
                            "word_bbox": bbox,
                            "char_bboxes": char_bboxes,
                            "image": img,
                            "page": page_index,
                        }
                    )

    doc.close()
    return word_entries


def extract_character_patch(word_image: np.ndarray, char_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Extract and normalize a character patch from a word image."""
    x0, y0, x1, y1 = char_bbox
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(word_image.shape[1], x1)
    y1 = min(word_image.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = word_image[y0:y1, x0:x1]
    resized = _resize_image(patch, (64, 64))
    normalized = np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.stack([normalized, normalized, normalized], axis=-1)


# --------------------------------------------------------------------------- #
# Galaxy utilities
# --------------------------------------------------------------------------- #

def load_galaxy_embeddings() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(GALAXY_PATH)
    embeddings = data["embeddings"]
    low_embeddings = data["embeddings_low"]
    char_ids = data["char_ids"]
    return embeddings, low_embeddings, char_ids


def find_similar(
    query: np.ndarray,
    embeddings: np.ndarray,
    char_ids: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, object]]:
    sims = embeddings @ query  # embeddings already normalized
    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append(
            {
                "char_id": int(char_ids[idx]),
                "similarity": float(sims[idx]),
                "index": int(idx),
            }
        )
    return results


def update_galaxy_embedding(
    char_id: int,
    embeddings: np.ndarray,
    low_embeddings: np.ndarray,
    char_ids: np.ndarray,
    high_vec: np.ndarray,
    low_vec: np.ndarray,
):
    mask = np.where(char_ids == char_id)[0]
    if mask.size == 0:
        return
    idx = mask[0]
    embeddings[idx] = _normalize(EMA_DECAY * embeddings[idx] + (1 - EMA_DECAY) * high_vec)
    low_embeddings[idx] = _normalize(EMA_DECAY * low_embeddings[idx] + (1 - EMA_DECAY) * low_vec)


def save_galaxy_embeddings(
    embeddings: np.ndarray,
    low_embeddings: np.ndarray,
    char_ids: np.ndarray,
    metadata: Dict[str, object],
) -> None:
    np.savez(
        GALAXY_PATH,
        embeddings=embeddings,
        embeddings_low=low_embeddings,
        char_ids=char_ids,
        embed_dim_high=embeddings.shape[1],
        embed_dim_low=low_embeddings.shape[1],
        **metadata,
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_word_level(pdf_paths: Iterable[Path]) -> None:
    model = DeepSeekOCRModel()
    embeddings, low_embeddings, char_ids = load_galaxy_embeddings()
    embeddings = embeddings.astype(np.float32)
    low_embeddings = low_embeddings.astype(np.float32)

    # Ensure normalization
    for idx in range(len(embeddings)):
        embeddings[idx] = _normalize(embeddings[idx])
        low_embeddings[idx] = _normalize(low_embeddings[idx])

    processed_pages = 0
    update_counter = defaultdict(int)

    for pdf_path in pdf_paths:
        try:
            word_entries = extract_word_patches(pdf_path)
        except Exception as exc:
            print(f"[WARN] Skipping {pdf_path.name}: {exc}")
            continue

        for entry in word_entries:
            word_image = entry["image"]
            word_text = entry["text"]
            char_bboxes = entry["char_bboxes"]
            if not char_bboxes:
                continue

            for idx, char in enumerate(word_text):
                if not char or not char.isprintable():
                    continue
                if idx >= len(char_bboxes):
                    continue
                char_patch = extract_character_patch(word_image, char_bboxes[idx])
                if char_patch is None:
                    continue

                char_patch = char_patch.astype(np.float32)
                char_patch = np.clip(char_patch, 0.0, 1.0)

                result = model.forward(char_patch, cache_for_backward=True)
                cache = result.get("cache", {})
                conv3_features = cache.get("conv3_out")
                if conv3_features is None:
                    continue

                pooled = conv3_features.mean(axis=(0, 1)).astype(np.float32)
                high_embedding = project_embedding(pooled, TARGET_DIM_HIGH)
                low_embedding = project_embedding(pooled, TARGET_DIM_LOW)
                fused = fuse_visual_text(char, high_embedding)

                matches = find_similar(fused, embeddings, char_ids, top_k=3)
                char_id = ord(char)
                if matches and matches[0]["char_id"] == char_id and matches[0]["similarity"] > 0.7:
                    update_galaxy_embedding(char_id, embeddings, low_embeddings, char_ids, fused, low_embedding)
                    update_counter[char_id] += 1

        processed_pages += 1
        if processed_pages % 5 == 0:
            print(f"[INFO] Processed {processed_pages} PDFs...")

    metadata = {
        "word_level_updates": dict(update_counter),
        "processed_pdfs": processed_pages,
    }
    save_galaxy_embeddings(embeddings, low_embeddings, char_ids, metadata)
    print(f"[DONE] Updated Galaxy embeddings with word-level context ({processed_pages} PDFs)")


def discover_pdfs(root: Path, limit: int) -> List[Path]:
    pdfs = []
    for path in root.rglob("*.pdf"):
        pdfs.append(path)
        if len(pdfs) >= limit:
            break
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Word-level atomic training with PDF OCR context")
    parser.add_argument(
        "--pdf-root",
        type=str,
        default="/mnt/arquivos",
        help="Root directory to search for PDFs",
    )
    parser.add_argument("--limit", type=int, default=MAX_PDFS, help="Maximum number of PDFs to process")
    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    if not pdf_root.exists():
        raise SystemExit(f"PDF root not found: {pdf_root}")

    pdf_paths = discover_pdfs(pdf_root, args.limit)
    print(f"Discovered {len(pdf_paths)} PDFs with potential OCR layers.")

    if not pdf_paths:
        print("No PDFs found; aborting.")
        return

    train_word_level(pdf_paths)


if __name__ == "__main__":
    main()
