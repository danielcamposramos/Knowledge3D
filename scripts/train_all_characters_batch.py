#!/usr/bin/env python3
"""
Batch Atomic Character Training Orchestrator

Trains all 62 alphanumeric characters using atomic binary tasks. Characters
are processed in batches aligned with K3D's 15 RPN stacks to leverage parallel
execution while consolidating embeddings into Galaxy memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import unicodedata

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = Path("/tmp")
FC_ONLY_MODE = True
FONT_COUNT = 200


def get_default_character_set() -> List[str]:
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def get_universal_character_set() -> List[str]:
    chars: List[str] = []

    # ASCII alphanumeric
    chars.extend(get_default_character_set())

    # Common punctuation & symbols
    chars.extend(list("!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\`~"))

    # Simplified Chinese (top frequency)
    chinese_common = [
        '的', '一', '是', '在', '不', '了', '有', '和', '人', '这',
        '中', '大', '为', '上', '个', '国', '我', '以', '要', '他',
        '时', '来', '用', '们', '生', '到', '作', '地', '于', '出',
        '就', '分', '对', '成', '会', '可', '主', '发', '年', '动',
        '同', '工', '也', '能', '下', '过', '子', '说', '产', '种',
        '面', '而', '方', '后', '多', '定', '行', '学', '法', '所',
        '民', '得', '经', '十', '三', '之', '进', '着', '等', '部',
        '度', '家', '电', '力', '里', '如', '水', '化', '高', '自',
        '二', '理', '起', '小', '物', '现', '实', '加', '量', '都',
        '两', '体', '制', '机', '当', '使', '点', '从', '业', '本',
    ]
    chars.extend(chinese_common)

    # Japanese kana + common kanji
    hiragana = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
    katakana = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
    kanji_common = ['日', '本', '人', '国', '年', '大', '十', '二', '社', '会', '円', '行', '金', '東', '京']
    chars.extend(list(hiragana))
    chars.extend(list(katakana))
    chars.extend(kanji_common)

    # Korean Hangul (basic syllables)
    hangul = '한국어기본자모가나다라마바사아자차카타파하'
    chars.extend(list(hangul))

    # Mathematical symbols
    math_symbols = '±×÷∞∑∫√≈≠≤≥∂∇∆Ω∈∉⊂⊃∪∩αβγδε'
    chars.extend(list(math_symbols))

    # Emoji shortlist
    emoji_common = ['😀', '😃', '😄', '😁', '🙂', '😊', '❤️', '👍', '✅', '⭐']
    chars.extend(emoji_common)

    # Deduplicate preserving order
    seen = set()
    unique_chars: List[str] = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            unique_chars.append(ch)
    return unique_chars
def _detect_script(char: str) -> str:
    if not char:
        return "unknown"

    for ch in char:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue

        upper = name.upper()
        if "CJK" in upper or "IDEOGRAPH" in upper or "HAN" in upper:
            return "chinese"
        if any(tok in upper for tok in ["HIRAGANA", "KATAKANA", "KANA"]):
            return "japanese"
        if "HANGUL" in upper:
            return "korean"
        if "ARABIC" in upper:
            return "arabic"
        if "HEBREW" in upper:
            return "hebrew"
        if any(tok in upper for tok in ["DEVANAGARI", "TAMIL", "BENGALI", "GUJARATI", "MALAYALAM", "ORIYA", "TELUGU", "GURMUKHI"]):
            return "indic"
        if any(tok in upper for tok in ["THAI", "LAO", "KHMER"]):
            return "sea"
        if "EMOJI" in upper or "SMILING" in upper:
            return "emoji"
        if "SYMBOL" in upper or "SIGN" in upper:
            return "symbols"
    return "latin"


def _parse_best_accuracy(log_path: Path) -> Optional[float]:
    try:
        for line in reversed(log_path.read_text().splitlines()):
            if line.strip().lower().startswith("best accuracy"):
                parts = line.replace("%", "").split()
                value = float(parts[-1])
                return value
    except Exception:
        return None
    return None


def _train_character_atomic(char: str, learning_rate: float, epochs: int) -> Dict[str, object]:
    """Invoke the atomic training script for a single character."""
    log_path = LOG_DIR / f"atomic_char_{ord(char)}_{char}.log"

    cmd = [
        sys.executable,
        "scripts/train_atomic_character.py",
        "--char",
        char,
        "--lr",
        str(learning_rate),
        "--epochs",
        str(epochs),
        "--fonts",
        str(FONT_COUNT),
    ]
    if FC_ONLY_MODE:
        cmd.append("--fc-only")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[RPN Stack] Training '{char}' → {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    accuracy = _parse_best_accuracy(log_path)

    if result.returncode == 0:
        print(f"✓ '{char}' completed successfully (best accuracy: {accuracy or 'N/A'}%)")
        return {"char": char, "success": True, "accuracy": accuracy, "log_path": log_path}

    print(f"✗ '{char}' failed – see {log_path}")
    return {"char": char, "success": False, "accuracy": accuracy, "log_path": log_path}


def _consolidate_embeddings(chars: List[str]) -> None:
    """Merge per-character embedding files into a Galaxy checkpoint."""
    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    galaxy_path = checkpoint_dir / "galaxy_character_embeddings.npz"

    embeddings: List[np.ndarray] = []
    low_embeddings: List[np.ndarray] = []
    char_ids: List[int] = []
    meta_char_ids: List[int] = []
    meta_chars: List[str] = []
    meta_scripts: List[str] = []
    meta_counts: List[int] = []
    meta_offsets: List[int] = []
    meta_best_acc: List[float] = []
    meta_font_counts: List[int] = []

    for char in chars:
        char_code = ord(char)
        path = checkpoint_dir / f"char_{char_code}_{char}_embeddings.npz"
        if not path.exists():
            continue

        data = np.load(path)
        char_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        char_low = data.get("embeddings_low")
        if char_low is not None:
            char_low = np.asarray(char_low, dtype=np.float32)
        best_acc = float(data.get("best_accuracy", 0.0))
        n_fonts = int(data.get("n_fonts", 0))

        char_label_raw = data.get("char", char)
        if isinstance(char_label_raw, np.ndarray):
            char_label = "".join(char_label_raw.astype(str).flatten().tolist())
        else:
            char_label = str(char_label_raw)
        if not char_label:
            char_label = char

        if char_embeddings.ndim != 2:
            continue

        low_dim = int(data.get("embed_dim_low", 64))
        if isinstance(char_low, np.ndarray) and char_low.ndim == 2 and char_low.shape[0] > 0:
            low_dim = char_low.shape[1]
        else:
            char_low = None

        if char_low is None:
            char_low = np.zeros((char_embeddings.shape[0], low_dim), dtype=np.float32)
        elif char_low.shape[0] < char_embeddings.shape[0]:
            repeats = char_embeddings.shape[0] - char_low.shape[0]
            pad = np.repeat(char_low[-1:], repeats, axis=0)
            char_low = np.concatenate([char_low, pad], axis=0)
        elif char_low.shape[0] > char_embeddings.shape[0]:
            char_low = char_low[: char_embeddings.shape[0]]

        offset = len(embeddings)
        for high_vec, low_vec in zip(char_embeddings, char_low):
            embeddings.append(high_vec.astype(np.float32))
            low_embeddings.append(low_vec.astype(np.float32))
            char_ids.append(char_code)

        count = len(embeddings) - offset
        if count == 0:
            continue

        meta_char_ids.append(char_code)
        meta_chars.append(char_label)
        meta_scripts.append(_detect_script(char_label))
        meta_counts.append(count)
        meta_offsets.append(offset)
        meta_best_acc.append(best_acc * 100.0)
        meta_font_counts.append(n_fonts)

    if not embeddings:
        print("⚠️  No embeddings found to consolidate.")
        return

    embeddings_arr = np.stack(embeddings, axis=0)
    low_embeddings_arr = np.stack(low_embeddings, axis=0)

    char_ids_arr = np.array(char_ids, dtype=np.int32)

    acc_ids = np.array(meta_char_ids, dtype=np.int32)
    acc_values = np.array(meta_best_acc, dtype=np.float32)

    np.savez(
        galaxy_path,
        embeddings=embeddings_arr,
        embeddings_low=low_embeddings_arr,
        char_ids=char_ids_arr,
        n_characters=len(chars),
        n_total_embeddings=len(embeddings_arr),
        embed_dim_high=embeddings_arr.shape[1],
        embed_dim_low=low_embeddings_arr.shape[1],
        accuracy_char_ids=acc_ids,
        accuracy_values=acc_values,
        char_meta_ids=np.array(meta_char_ids, dtype=np.int32),
        char_meta_chars=np.array(meta_chars, dtype="<U32"),
        char_meta_scripts=np.array(meta_scripts, dtype="<U16"),
        char_meta_counts=np.array(meta_counts, dtype=np.int32),
        char_meta_offsets=np.array(meta_offsets, dtype=np.int32),
        char_meta_best_accuracy=np.array(meta_best_acc, dtype=np.float32),
        char_meta_font_counts=np.array(meta_font_counts, dtype=np.int32),
    )

    print(f"✓ Galaxy embeddings saved: {galaxy_path}")
    print(f"  Total embeddings: {len(embeddings_arr)}")
    print(f"  Embedding dimension: {embeddings_arr.shape[1]}")
    print(f"  Characters covered: {len(np.unique(char_ids_arr))}")
    if meta_scripts:
        unique_scripts = sorted(set(meta_scripts))
        print(f"  Scripts covered: {', '.join(unique_scripts)}")


def main() -> None:
    global FONT_COUNT

    parser = argparse.ArgumentParser(description="Batch train atomic characters")
    parser.add_argument("--universal", action="store_true", help="Train universal multi-script set")
    parser.add_argument("--fonts", type=int, default=FONT_COUNT, help="Fonts per character (cap)")
    parser.add_argument(
        "--parallel",
        type=int,
        default=6,
        help="Max characters to train concurrently per batch (default: 6)",
    )
    args = parser.parse_args()

    FONT_COUNT = args.fonts
    parallel_jobs = max(1, args.parallel)

    print("=" * 80)
    title = "UNIVERSAL ATOMIC CHARACTER BATCH TRAINING" if args.universal else "ATOMIC CHARACTER BATCH TRAINING - Phase G"
    print(title)
    print("=" * 80)
    print()

    chars = get_universal_character_set() if args.universal else get_default_character_set()

    print(f"Training {len(chars)} characters atomically")
    if args.universal:
        print("  - Multi-script character set (Latin, CJK, symbols, emoji, etc.)")
    print(f"Using RPN stack architecture with up to {parallel_jobs} concurrent slots")
    print(f"Fonts per script cap: {FONT_COUNT}")
    print()

    batch_size = 15
    batches = [chars[i : i + batch_size] for i in range(0, len(chars), batch_size)]
    print(f"Batches: {len(batches)}")
    for idx, batch in enumerate(batches, start=1):
        preview = " ".join(batch[:10])
        suffix = "..." if len(batch) > 10 else ""
        print(f"  Batch {idx}: {len(batch)} chars - {preview}{suffix}")
    print()

    all_results: List[Dict[str, object]] = []

    for batch_idx, batch_chars in enumerate(batches, start=1):
        print("=" * 80)
        print(f"BATCH {batch_idx}/{len(batches)}: Training {len(batch_chars)} characters in parallel")
        print("=" * 80)
        print()

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(parallel_jobs, len(batch_chars))) as executor:
            futures = {
                executor.submit(_train_character_atomic, char, 0.01, 100): char
                for char in batch_chars
            }

            batch_results: List[Dict[str, object]] = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                all_results.append(result)
                batch_results.append(result)

        successes = sum(1 for res in batch_results if res.get("success"))
        print()
        print(f"Batch {batch_idx} complete: {successes}/{len(batch_chars)} successful")
        print()

    print("=" * 80)
    print("ALL BATCHES COMPLETE")
    print("=" * 80)
    print()

    total_success = sum(1 for res in all_results if res.get("success"))
    accuracies: List[float] = [
        acc for acc in (res.get("accuracy") for res in all_results) if acc is not None
    ]
    avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
    print(f"Total: {total_success}/{len(chars)} characters trained successfully")
    print(f"Average best accuracy: {avg_accuracy:.2f}%")
    print()

    failures = [res["char"] for res in all_results if not res.get("success")]
    if failures:
        print(f"Failed characters ({len(failures)}): {' '.join(failures)}")
        print()

    print("Consolidating embeddings to Galaxy memory...")
    _consolidate_embeddings(chars)

    print()
    print("=" * 80)
    print("PHASE G ATOMIC CHARACTER TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
