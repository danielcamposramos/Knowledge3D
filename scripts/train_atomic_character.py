#!/usr/bin/env python3
"""
Atomic Character Training

Trains the sovereign CNN on a single character as a binary task
("is target character" vs "not target character"). Learned embeddings
are exported to Galaxy memory for later composition.
"""

from __future__ import annotations

import argparse
import ctypes
import pickle
import random
import sys
from io import BytesIO
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import unicodedata

try:  # Optional dependency for richer augmentation
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is optional
    cv2 = None

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM  # noqa: E402
from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel  # noqa: E402
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer  # noqa: E402
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine  # noqa: E402
from knowledge3d.cranium.bridges.trigram_embed_bridge import TrigramEmbedBridge  # noqa: E402
from knowledge3d.cranium.bridges.spatial_pool_bridge import SpatialMeanPooler  # noqa: E402
from knowledge3d.cranium.sovereign import loader  # noqa: E402
from knowledge3d.cranium.adaptive_procedural_bridge import AdaptiveDimensionCompressor  # noqa: E402
from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy  # noqa: E402

MATRY_MAX_DIM = 2048
CHAR_EMBED_DIM = 512  # Adaptive Matryoshka dimension (balanced quality)
LOW_DIM = 64
DEFAULT_FONTS_PER_SCRIPT = 50
FONT_CATEGORY_DIR = Path("/K3D/Knowledge3D.local/font_categories")
CHECKPOINT_DIR = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")
PROCEDURAL_GALAXY_ROOT = Path("/K3D/Knowledge3D.local/procedural_galaxy")

MODEL_STATE_KEYS = [
    "conv1_weight",
    "conv1_bias",
    "bn1_gamma",
    "bn1_beta",
    "conv2_weight",
    "conv2_bias",
    "bn2_gamma",
    "bn2_beta",
    "conv3_weight",
    "conv3_bias",
    "bn3_gamma",
    "bn3_beta",
]

CHINESE_COMMON = [
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

HIRAGANA = list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん')
KATAKANA = list('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン')
KANJI_COMMON = ['日', '本', '人', '国', '年', '大', '十', '二', '社', '会', '円', '行', '金', '東', '京']
HANGUL_BASIC = list('한국어기본자모가나다라마바사아자차카타파하')
MATH_SYMBOLS = list('±×÷∞∑∫√≈≠≤≥∂∇∆Ω∈∉⊂⊃∪∩αβγδε')
EMOJI_COMMON = ['😀', '😃', '😄', '😁', '🙂', '😊', '❤️', '👍', '✅', '⭐']

BASE_ALPHANUM = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
LATIN_PUNCT = list("!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\`~")

NEGATIVE_CHAR_SETS = {
    "latin": BASE_ALPHANUM + LATIN_PUNCT,
    "chinese": CHINESE_COMMON,
    "japanese": HIRAGANA + KATAKANA + KANJI_COMMON,
    "korean": HANGUL_BASIC,
    "arabic": list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي"),
    "hebrew": list("אבגדהוזחטיכלמנסעפצקרשת"),
    "indic": list("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"),
    "sea": list("กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนปผฝพฟภมยรลวศษสห"),
    "emoji": EMOJI_COMMON,
    "symbols": MATH_SYMBOLS + LATIN_PUNCT,
    "math": MATH_SYMBOLS,  # Math symbols use math fonts
}


matryoshka = MatryoshkaTRM(max_dims=MATRY_MAX_DIM, min_dims=LOW_DIM)

# GPU sovereignty: Trigram bridge is REQUIRED - no CPU fallback
try:
    _trigram_bridge = TrigramEmbedBridge()
    print("[INFO] Trigram GPU bridge initialized (GPU-sovereign).")
except Exception as trigram_exc:
    raise RuntimeError(
        f"Failed to initialize GPU trigram bridge: {trigram_exc}\n"
        "RPN embeddings require GPU sovereignty - no CPU fallback allowed. "
        "Ensure CUDA is available and PTX kernels compile successfully."
    ) from trigram_exc

rpn_engine = RPNEmbeddingEngine(embedding_dim=CHAR_EMBED_DIM)
try:
    rpn_engine.attach_gpu_bridge(_trigram_bridge)
    print("[INFO] Trigram GPU bridge attached to RPN engine.")
except Exception as attach_exc:
    raise RuntimeError(
        f"Failed to attach GPU trigram bridge to RPN engine: {attach_exc}\n"
        "GPU sovereignty violation - cannot proceed without GPU bridge."
    ) from attach_exc

spatial_pooler = SpatialMeanPooler()

_FONT_CACHE: Dict[str, List[Path]] = {}


def is_emoji(char: str) -> bool:
    # Handle edge cases: empty string, None, or multi-character strings
    if not char or len(char) != 1:
        return False
    try:
        name = unicodedata.name(char)
    except (ValueError, TypeError):
        return False
    return "EMOJI" in name or ord(char) >= 0x1F300


def get_character_script(char: str) -> str:
    """
    Determine the script of a character to select appropriate fonts.

    Math symbols are detected FIRST to ensure they use math fonts instead
    of falling through to their nominal script (e.g., Greek letters -> 'math' not 'latin').
    """
    # Check math symbols FIRST (before Unicode name lookup)
    # Import here to avoid circular dependencies
    from knowledge3d.cranium.math_symbols_registry import is_math_symbol

    if is_math_symbol(char):
        return "math"

    try:
        name = unicodedata.name(char)
    except ValueError:
        return "latin"

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


def discover_system_fonts(limit: int) -> List[Path]:
    font_dir = Path("/usr/share/fonts")
    fonts = list(font_dir.rglob("*.ttf")) + list(font_dir.rglob("*.otf"))
    if limit and len(fonts) > limit:
        fonts = random.sample(fonts, limit)
    return fonts


def load_fonts_for_script(script: str, n_fonts: int) -> List[Path]:
    cached = _FONT_CACHE.get(script)
    if cached:
        if n_fonts and len(cached) > n_fonts:
            return random.sample(cached, n_fonts)
        return list(cached)

    paths: List[Path] = []
    if FONT_CATEGORY_DIR.exists():
        script_file = FONT_CATEGORY_DIR / f"{script}_fonts.txt"
        if not script_file.exists():
            script_file = FONT_CATEGORY_DIR / "latin_fonts.txt"
        if script_file.exists():
            with script_file.open() as f:
                paths = [Path(line.strip()) for line in f if line.strip()]

    if not paths:
        paths = discover_system_fonts(0)

    unique_paths = [p for p in paths if p.exists()]
    _FONT_CACHE[script] = unique_paths
    if n_fonts and len(unique_paths) > n_fonts:
        return random.sample(unique_paths, n_fonts)
    return list(unique_paths)


def _project_embedding(vector: np.ndarray, target_dim: int = CHAR_EMBED_DIM) -> np.ndarray:
    """Project feature vector using Matryoshka base weights."""
    target_dim = max(LOW_DIM, min(target_dim, MATRY_MAX_DIM))
    projected = matryoshka.project_vector(vector.astype(np.float32), target_dim)
    norm = np.linalg.norm(projected)
    if norm > 1e-6:
        projected = projected / norm

    return projected.astype(np.float32)


def _fuse_visual_text(char: str, visual_embedding: np.ndarray) -> np.ndarray:
    """Fuse visual Matryoshka embedding with RPN trigram embedding (GPU-sovereign)."""
    # GPU sovereignty: Always use GPU path - no conditional fallback
    text_embedding = rpn_engine.embed_word_gpu(char)
    fused = (visual_embedding + text_embedding) * 0.5
    norm = np.linalg.norm(fused)
    if norm > 1e-6:
        fused = fused / norm
    return fused.astype(np.float32)


def negative_characters_for_script(script: str) -> List[str]:
    chars = NEGATIVE_CHAR_SETS.get(script)
    if not chars:
        chars = NEGATIVE_CHAR_SETS.get("latin", BASE_ALPHANUM)
    filtered = [c for c in chars if c]
    return filtered or BASE_ALPHANUM


def render_glyph_image(char: str, font_path: str, size: int = 64) -> Optional[np.ndarray]:
    """Render character from vector font as RGB image."""
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None

    use_rgba = is_emoji(char)
    bg_color = (255, 255, 255, 0) if use_rgba else (255, 255, 255)
    img = Image.new("RGBA" if use_rgba else "RGB", (64, 64), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        bbox = draw.textbbox((0, 0), char, font=font, anchor=None)
    except Exception:
        bbox = draw.textbbox((0, 0), char, font=font)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (64 - text_width) // 2 - bbox[0]
    y = (64 - text_height) // 2 - bbox[1]

    fill = (0, 0, 0, 255) if use_rgba else (0, 0, 0)
    try:
        draw.text((x, y), char, fill=fill, font=font, embedded_color=True)
    except TypeError:
        draw.text((x, y), char, fill=fill, font=font)

    if use_rgba:
        white_bg = Image.new("RGB", (64, 64), (255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])
        img = white_bg

    array = np.array(img, dtype=np.float32) / 255.0
    return array


def _has_glyph(font_meta: Dict[str, object], char: str) -> bool:
    """Return True when font metadata declares that it contains the glyph."""
    glyphs: Optional[Union[Iterable[int], Iterable[str], Dict[object, object]]] = font_meta.get("glyphs")  # type: ignore[assignment]
    if glyphs is None:
        return True

    if isinstance(glyphs, dict):
        keys = glyphs.keys()
    else:
        keys = glyphs

    if char in keys:
        return True

    char_code = ord(char)
    return char_code in keys


def render_contextual_glyph(
    char: str,
    font_path: str,
    size: int = 64,
    context: bool = True,
) -> Optional[np.ndarray]:
    """
    Render a glyph within sentence context to better match PDF patches.

    Falls back to isolated rendering when context rendering fails.
    """
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None

    if not context:
        return render_glyph_image(char, font_path, size=size)

    prefix_chars = "The quick "
    suffix_chars = " jumps over"
    sentence = f"{prefix_chars}{char}{suffix_chars}"

    canvas_width = 256
    canvas_height = 64
    bg_color = (255, 255, 255)
    img = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        draw.text((10, 10), sentence, fill=(0, 0, 0), font=font)
    except Exception:
        return render_glyph_image(char, font_path, size=size)

    try:
        prefix_bbox = draw.textbbox((10, 10), prefix_chars, font=font)
    except Exception:
        prefix_bbox = font.getbbox(prefix_chars)  # type: ignore[attr-defined]
        prefix_bbox = (0, 0, prefix_bbox[2] - prefix_bbox[0], prefix_bbox[3] - prefix_bbox[1])
    prefix_width = prefix_bbox[2] - prefix_bbox[0]

    try:
        char_bbox = draw.textbbox((10 + prefix_width, 10), char, font=font)
    except Exception:
        char_bbox = font.getbbox(char)  # type: ignore[attr-defined]
        char_bbox = (
            10 + prefix_width,
            10,
            10 + prefix_width + (char_bbox[2] - char_bbox[0]),
            10 + (char_bbox[3] - char_bbox[1]),
        )

    char_x0, char_y0, char_x1, char_y1 = char_bbox
    if char_x0 == char_x1 or char_y0 == char_y1:
        return render_glyph_image(char, font_path, size=size)

    char_w = char_x1 - char_x0
    char_h = char_y1 - char_y0
    expand_x = int(char_w * 0.3)
    expand_y = int(char_h * 0.3)

    context_x0 = max(0, char_x0 - expand_x)
    context_x1 = min(canvas_width, char_x1 + expand_x)
    context_y0 = max(0, char_y0 - expand_y)
    context_y1 = min(canvas_height, char_y1 + expand_y)

    if context_x0 >= context_x1 or context_y0 >= context_y1:
        return render_glyph_image(char, font_path, size=size)

    img_array = np.array(img, dtype=np.uint8)
    patch = img_array[context_y0:context_y1, context_x0:context_x1]
    if patch.size == 0:
        return render_glyph_image(char, font_path, size=size)

    patch_pil = Image.fromarray(patch)
    if hasattr(Image, "Resampling"):
        resample_filter = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    else:
        resample_filter = Image.LANCZOS  # type: ignore[attr-defined]
    patch_resized = patch_pil.resize((64, 64), resample_filter)
    return np.array(patch_resized, dtype=np.float32) / 255.0


def augment_pdf_style(img: np.ndarray) -> np.ndarray:
    """
    Apply PDF-like degradation (noise, compression, blur, paper texture).
    """
    degraded = img.astype(np.float32)

    noise = np.random.normal(0, 3, degraded.shape).astype(np.float32)
    degraded = np.clip(degraded + noise, 0, 255)

    if random.random() < 0.5:
        pil_img = Image.fromarray(degraded.astype(np.uint8))
        buffer = BytesIO()
        quality = random.randint(50, 80)
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        try:
            pil_img = Image.open(buffer)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            degraded = np.array(pil_img, dtype=np.float32)
        except Exception:
            degraded = img.astype(np.float32)

    if cv2 is not None and random.random() < 0.3:
        kernel = random.choice([3, 5])
        degraded = cv2.GaussianBlur(degraded.astype(np.uint8), (kernel, kernel), 0.5).astype(np.float32)

    if random.random() < 0.2:
        texture = np.random.randint(248, 256, degraded.shape, dtype=np.uint8)
        degraded = (degraded * 0.95 + texture.astype(np.float32) * 0.05)

    return np.clip(degraded, 0, 255).astype(np.uint8)


def _apply_fc_checkpoint(trainer: GPUCNNTrainer, fc_weight: np.ndarray, fc_bias: np.ndarray) -> None:
    """Load FC weights/bias into trainer and sync GPU buffers."""
    if fc_weight.shape != trainer.fc_weight.shape or fc_bias.shape != trainer.fc_bias.shape:
        raise ValueError(
            f"Checkpoint FC shape mismatch: weight {fc_weight.shape} vs {trainer.fc_weight.shape}, "
            f"bias {fc_bias.shape} vs {trainer.fc_bias.shape}"
        )

    trainer.fc_weight = fc_weight.astype(np.float32).copy()
    trainer.fc_bias = fc_bias.astype(np.float32).copy()
    loader.memcpy_htod(
        trainer.d_fc_weight,
        trainer.fc_weight.ctypes.data_as(ctypes.c_void_p),
        trainer.fc_weight.nbytes,
    )
    loader.memcpy_htod(
        trainer.d_fc_bias,
        trainer.fc_bias.ctypes.data_as(ctypes.c_void_p),
        trainer.fc_bias.nbytes,
    )

    trainer.gpu_backward.zero_gradients(trainer.d_grad_fc_weight, trainer.fc_weight.size)
    trainer.gpu_backward.zero_gradients(trainer.d_grad_fc_bias, trainer.fc_bias.size)
    trainer.gpu_backward.zero_gradients(trainer.d_vel_fc_weight, trainer.fc_weight.size)
    trainer.gpu_backward.zero_gradients(trainer.d_vel_fc_bias, trainer.fc_bias.size)


def augment_character_patch(img: np.ndarray, pdf_augment: bool = True) -> List[np.ndarray]:
    """Augment a glyph image to match patch-level inference distribution."""
    base_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if pdf_augment and random.random() < 0.5:
        base_uint8 = augment_pdf_style(base_uint8)

    augmented: List[np.ndarray] = [base_uint8]

    height, width = base_uint8.shape[:2]

    if cv2 is None:
        # Minimal augmentation fallback without OpenCV
        for _ in range(3):
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            src_img = Image.fromarray(base_uint8)
            if src_img.mode != "L":
                src_img = src_img.convert("L")
            canvas = Image.new("L", (64, 64), color=255)
            canvas.paste(src_img, (dx, dy))
            augmented.append(np.array(canvas.convert("RGB"), dtype=np.uint8))
        # Continue to normalisation stage
        normalized: List[np.ndarray] = []
        for arr in augmented:
            arr_f = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
            if arr_f.ndim == 2:
                arr_f = np.repeat(arr_f[:, :, None], 3, axis=2)
            normalized.append(arr_f.astype(np.float32))
        return normalized

    # Random crops
    for _ in range(3):
        crop_scale = random.uniform(0.7, 0.95)
        crop_h = max(8, int(height * crop_scale))
        crop_w = max(8, int(width * crop_scale))
        if crop_h > height or crop_w > width:
            continue
        y0 = random.randint(0, height - crop_h)
        x0 = random.randint(0, width - crop_w)
        crop = base_uint8[y0:y0 + crop_h, x0:x0 + crop_w]
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        augmented.append(resized)

    # Scaling variations
    for scale in [0.6, 0.8, 1.2, 1.4]:
        scaled = cv2.resize(base_uint8, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        sh, sw = scaled.shape[:2]
        if sh < 64 or sw < 64:
            scaled = cv2.resize(scaled, (64, 64), interpolation=cv2.INTER_AREA)
        else:
            y0 = max(0, (sh - 64) // 2)
            x0 = max(0, (sw - 64) // 2)
            scaled = scaled[y0:y0 + 64, x0:x0 + 64]
        augmented.append(scaled)

    # Rotations
    center = (32, 32)
    for angle in [-12, -6, 6, 12]:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(base_uint8, matrix, (64, 64), borderValue=(255, 255, 255))
        augmented.append(rotated.astype(np.uint8))

    # Noise
    for _ in range(2):
        noise = np.random.normal(0.0, 8.0, base_uint8.shape)
        noisy = np.clip(base_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)

    normalized: List[np.ndarray] = []
    for arr in augmented:
        arr_f = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
        if arr_f.ndim == 2:
            arr_f = np.repeat(arr_f[:, :, None], 3, axis=2)
        normalized.append(arr_f.astype(np.float32))
    return normalized


def _prepare_negative_groups(target_char: str, script: str, max_extra_scripts: int = 4) -> Dict[str, List[str]]:
    """Build script-aware pools of negative characters."""
    groups: Dict[str, List[str]] = {}

    primary = [c for c in negative_characters_for_script(script) if c != target_char]
    if primary:
        groups[script] = list(dict.fromkeys(primary))

    if script != "latin":
        latin_chars = [c for c in NEGATIVE_CHAR_SETS.get("latin", BASE_ALPHANUM) if c != target_char]
        if latin_chars:
            groups["latin"] = list(dict.fromkeys(latin_chars))

    available_scripts = [name for name in NEGATIVE_CHAR_SETS.keys() if name not in groups]
    random.shuffle(available_scripts)
    for extra_script in available_scripts[:max_extra_scripts]:
        chars = NEGATIVE_CHAR_SETS.get(extra_script, [])
        if not chars:
            continue
        sample_size = min(16, len(chars))
        chosen = chars if sample_size == len(chars) else random.sample(chars, sample_size)
        chosen = [c for c in chosen if c != target_char]
        if chosen:
            groups[extra_script] = list(dict.fromkeys(chosen))

    return groups


def _prepare_font_buckets(
    negative_groups: Dict[str, List[str]],
    target_script: str,
    target_fonts: List[Path],
    n_fonts: int,
) -> Dict[str, List[Path]]:
    """Return fonts per script, falling back to target fonts when needed."""
    buckets: Dict[str, List[Path]] = {target_script: target_fonts}
    extra_font_cap = max(10, n_fonts // 2) if n_fonts else DEFAULT_FONTS_PER_SCRIPT

    for script_name in negative_groups.keys():
        if script_name == target_script:
            continue
        fonts = load_fonts_for_script(script_name, extra_font_cap)
        if not fonts:
            fonts = target_fonts
        buckets[script_name] = fonts

    return buckets


def _build_dataset(
    target_char: str,
    target_script: str,
    positive_fonts: List[Path],
    negative_groups: Dict[str, List[str]],
    font_buckets: Dict[str, List[Path]],
    augmentations_per_font: int = 4,
) -> Dict[str, np.ndarray]:
    """Create balanced binary dataset for target character."""
    base_positives: List[np.ndarray] = []

    for font_path in positive_fonts:
        glyph = render_contextual_glyph(target_char, str(font_path), context=True)
        if glyph is None:
            glyph = render_glyph_image(target_char, str(font_path))
        if glyph is not None:
            base_positives.append(glyph)

    if not base_positives:
        raise RuntimeError(f"No positive samples rendered for '{target_char}'")

    positives: List[np.ndarray] = []
    for glyph in base_positives:
        augmented = augment_character_patch(glyph)
        positives.extend(augmented[: augmentations_per_font + 1])

    n_positive = len(positives)
    if n_positive == 0:
        raise RuntimeError(f"No usable positive samples for '{target_char}'")

    negatives: List[np.ndarray] = []
    script_weights: List[str] = []
    for script_name in negative_groups.keys():
        weight = 3 if script_name == target_script else 1
        script_weights.extend([script_name] * weight)
    if not script_weights:
        script_weights.append(target_script)

    script_cycle = cycle(script_weights)
    char_cycles = {name: cycle(chars) for name, chars in negative_groups.items() if chars}
    font_cycles = {
        name: cycle(font_buckets.get(name, positive_fonts))
        for name in negative_groups.keys()
        if font_buckets.get(name, positive_fonts)
    }

    max_attempts = n_positive * 25
    attempts = 0
    while len(negatives) < n_positive and attempts < max_attempts:
        attempts += 1
        script_name = next(script_cycle)
        char_iter = char_cycles.get(script_name)
        font_iter = font_cycles.get(script_name)
        if char_iter is None or font_iter is None:
            continue
        char_candidate = next(char_iter)
        if char_candidate == target_char:
            continue
        font_path = next(font_iter)
        glyph = render_contextual_glyph(char_candidate, str(font_path), context=True)
        if glyph is None:
            glyph = render_glyph_image(char_candidate, str(font_path))
        if glyph is None:
            continue
        augmented = augment_character_patch(glyph)
        negatives.extend(augmented[: augmentations_per_font + 1])

    if len(negatives) < n_positive and negatives:
        shortfall = n_positive - len(negatives)
        negatives.extend(random.choices(negatives, k=shortfall))

    if not negatives:
        raise RuntimeError("No negative samples generated for negatives")

    if len(negatives) > n_positive:
        negatives = random.sample(negatives, n_positive)
    else:
        negatives = negatives[:n_positive]

    images = np.array(positives + negatives, dtype=np.float32)
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

    return {
        "images": images,
        "labels": labels,
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }


def train_single_character(
    target_char: str,
    learning_rate: float = 0.6,
    n_epochs: int = 1500,
    n_fonts: int = 0,
    fc_only: bool = False,
    max_epochs: int = 3000,
    compressor: Optional[AdaptiveDimensionCompressor] = None,
    galaxy: Optional[ProceduralGalaxy] = None,
) -> Dict[str, object]:
    """Train CNN to recognize single character across fonts."""
    print("=" * 80)
    print(f"ATOMIC CHARACTER TRAINING: '{target_char}' (ord={ord(target_char)})")
    print("=" * 80)
    script = get_character_script(target_char)
    print(f"Script: {script}")
    print()

    char_code = ord(target_char)
    checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / f"char_{char_code}_{target_char}_weights.npz"

    print("[1/6] Loading font database (all available fonts)...")
    fonts: List[Path] = []
    font_db_path = Path("/K3D/Knowledge3D.local/font_db.pkl")
    if font_db_path.exists():
        try:
            with font_db_path.open("rb") as handle:
                font_db = pickle.load(handle)
        except Exception as exc:
            print(f"       ⚠️  Failed to load font_db.pkl ({exc}); falling back to script fonts.")
        else:
            for font_meta in font_db.values():
                font_path = font_meta.get("font_path")  # type: ignore[assignment]
                if not font_path:
                    continue
                font_path_obj = Path(font_path)
                if not font_path_obj.exists():
                    continue
                if font_meta.get("is_symbol_font"):  # type: ignore[call-arg]
                    continue
                if not _has_glyph(font_meta, target_char):
                    continue
                fonts.append(font_path_obj)

    fonts = list(dict.fromkeys(fonts))
    if not fonts:
        fonts = load_fonts_for_script(script, n_fonts if n_fonts else DEFAULT_FONTS_PER_SCRIPT)
        fonts = list(dict.fromkeys(fonts))

    if n_fonts and n_fonts > 0 and len(fonts) > n_fonts:
        fonts = random.sample(fonts, n_fonts)

    if not fonts:
        raise RuntimeError(f"No fonts available for script '{script}' and character '{target_char}'")

    print(f"       Using {len(fonts)} fonts for script '{script}'")
    print()

    print("[2/6] Initializing binary CNN model...")
    model = DeepSeekOCRModel()
    resume_best_accuracy = 0.0
    resume_fc_weight: Optional[np.ndarray] = None
    resume_fc_bias: Optional[np.ndarray] = None

    if weights_path.exists():
        try:
            with np.load(weights_path, allow_pickle=True) as data:
                state_dict = {name: data[name] for name in MODEL_STATE_KEYS if name in data}
                if state_dict:
                    model.load_state_dict(state_dict, strict=False)
                    print(f"       ✓ Loaded {len(state_dict)} CNN parameter tensors from checkpoint")
                if "fc_weight" in data and "fc_bias" in data:
                    resume_fc_weight = np.array(data["fc_weight"])
                    resume_fc_bias = np.array(data["fc_bias"])
                if "accuracy" in data:
                    resume_best_accuracy = float(np.squeeze(data["accuracy"]))
            print(
                "       Resuming from checkpoint "
                f"(best accuracy: {resume_best_accuracy * 100:.2f}%)"
            )
        except Exception as exc:
            print(f"       ⚠️  Failed to load checkpoint ({exc}); starting from scratch.")

    trainer = GPUCNNTrainer(
        model,
        num_classes=2,
        learning_rate=learning_rate,
        momentum=0.9,
        normalize_gradients=not fc_only,
        fc_only=fc_only,
    )
    if resume_fc_weight is not None and resume_fc_bias is not None:
        try:
            _apply_fc_checkpoint(trainer, resume_fc_weight, resume_fc_bias)
            print("       ✓ Loaded FC classifier weights from checkpoint")
        except Exception as exc:
            print(f"       ⚠️  Failed to load FC weights ({exc}); reinitializing FC layer.")

    mode_desc = "fc-only (frozen CNN)" if fc_only else "full CNN fine-tuning"
    print(f"       Task: Binary classification (is '{target_char}' vs not)")
    print(f"       Learning rate: {learning_rate}")
    print(f"       Trainer mode: {mode_desc}")
    print()

    print("[3/6] Rendering glyphs...")
    negative_groups = _prepare_negative_groups(target_char, script)
    font_buckets = _prepare_font_buckets(negative_groups, script, fonts, n_fonts)
    dataset = _build_dataset(
        target_char=target_char,
        target_script=script,
        positive_fonts=fonts,
        negative_groups=negative_groups,
        font_buckets=font_buckets,
    )
    images = dataset["images"]
    labels = dataset["labels"]
    print(f"       Positive samples ('{target_char}'): {dataset['n_positive']}")
    print(f"       Negative samples (other chars): {dataset['n_negative']}")
    print(f"       Total samples: {len(images)}")
    print()

    print("[4/6] Training binary CNN...")
    print("=" * 80)

    # Increased from 32 to 148 for better GPU utilization
    # With fc-only mode, memory usage is low enough for larger batches
    # This reduces kernel launch overhead and improves throughput
    batch_size = 148
    n_samples = len(images)
    best_accuracy = max(0.0, resume_best_accuracy)

    epoch = 0
    target_epochs = max(1, n_epochs)
    max_epochs = max(max_epochs, target_epochs)
    current_limit = target_epochs
    extend_applied = False
    accuracy_at_epoch_50: Optional[float] = None

    while epoch < current_limit:
        epoch += 1
        indices = np.random.permutation(n_samples)
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]

        epoch_losses: List[float] = []
        epoch_accs: List[float] = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_imgs = [img for img in images_shuffled[start:end]]
            batch_labels = [int(lbl) for lbl in labels_shuffled[start:end]]

            loss, acc = trainer.train_batch(batch_imgs, batch_labels)

            if np.isnan(loss) or np.isinf(loss):
                print(f"\n⚠️  NaN/inf detected at epoch {epoch}, batch {start // batch_size}")
                raise RuntimeError("Training produced NaN/inf loss")

            epoch_losses.append(loss)
            epoch_accs.append(acc)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        avg_acc = float(np.mean(epoch_accs)) if epoch_accs else 0.0

        print(f"Epoch {epoch:3d}/{current_limit} | Loss: {avg_loss:.4f} | Acc: {avg_acc * 100:5.2f}%", end="")

        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            state_dict = model.get_state_dict()
            char_code = ord(target_char)
            state_dict["fc_weight"] = trainer.fc_weight.copy()
            state_dict["fc_bias"] = trainer.fc_bias.copy()
            np.savez(
                checkpoint_dir / f"char_{char_code}_{target_char}_weights.npz",
                **state_dict,
                accuracy=avg_acc,
                character=target_char,
            )
            print(" ✓ [saved]")
        else:
            print()

        if accuracy_at_epoch_50 is None and epoch >= 50:
            accuracy_at_epoch_50 = best_accuracy

        if epoch % 50 == 0:
            est_msg = ""
            if accuracy_at_epoch_50 is not None and epoch > 50:
                recent_improvement = best_accuracy - accuracy_at_epoch_50
                epochs_remaining = current_limit - epoch
                if epochs_remaining > 0:
                    estimated_final = best_accuracy + (recent_improvement / max(1, epoch - 50)) * epochs_remaining
                    est_msg = f"\n            Estimated final accuracy: {estimated_final * 100:.2f}%"
            print(
                f"            Progress checkpoint | Best so far: {best_accuracy * 100:5.2f}%{est_msg}"
            )

        if best_accuracy >= 0.85:
            print(f"🎯 Target accuracy {best_accuracy * 100:.2f}% reached at epoch {epoch}!")
            print("   Early stopping (target: 85%)")
            break

        if (
            epoch == current_limit
            and best_accuracy < 0.80
            and not extend_applied
            and max_epochs > current_limit
        ):
            print(f"⚠️  Accuracy {best_accuracy * 100:.2f}% below target at epoch {epoch}")
            print(f"   Extending training to {max_epochs} epochs...")
            current_limit = max_epochs
            extend_applied = True

    print()
    print("=" * 80)
    print("[5/6] Extracting character embeddings...")

    fused_embeddings: List[np.ndarray] = []
    low_dim_embeddings: List[np.ndarray] = []
    for idx, image in enumerate(images):
        if labels[idx] == 1:
            result = model.forward(image, cache_for_backward=True)
            cache = result.get("cache", {})
            shapes = cache.get("shapes", {})
            conv3_shape = shapes.get("after_conv3")
            conv3_device = cache.get("conv3_out_device")

            if conv3_device is not None and conv3_shape is not None:
                pooled = spatial_pooler.mean_pool_host(conv3_device, conv3_shape[0], conv3_shape[1], conv3_shape[2])
            else:
                conv3_features = cache.get("conv3_out")
                if conv3_features is None:
                    continue
                pooled = conv3_features.mean(axis=(0, 1)).astype(np.float32)

            visual_embedding = _project_embedding(pooled, CHAR_EMBED_DIM)
            fused = _fuse_visual_text(target_char, visual_embedding)
            low_dim = _project_embedding(pooled, LOW_DIM)
            fused_embeddings.append(fused)
            low_dim_embeddings.append(low_dim)

    if not fused_embeddings:
        raise RuntimeError("No embeddings extracted for target character")

    embeddings_array = np.stack(fused_embeddings, axis=0)
    low_dim_array = np.stack(low_dim_embeddings, axis=0)
    print(f"       Extracted {len(embeddings_array)} embeddings, shape: {embeddings_array.shape}")
    print()

    print("[6/6] Saving embeddings to Galaxy checkpoint...")
    char_code = ord(target_char)
    embedding_path = checkpoint_dir / f"char_{char_code}_{target_char}_embeddings.npz"
    np.savez(
        embedding_path,
        embeddings=embeddings_array,
        embeddings_low=low_dim_array,
        char_id=char_code,
        char=target_char,
        n_fonts=len(fonts),
        best_accuracy=best_accuracy,
        embed_dim_high=CHAR_EMBED_DIM,
        embed_dim_low=LOW_DIM,
    )
    print(f"       ✓ Saved to {embedding_path}")
    print()

    if compressor is not None and galaxy is not None:
        try:
            canonical_embedding = embeddings_array.mean(axis=0).astype(np.float32)
            # Adaptive quality based on embedding complexity (Matryoshka principle)
            quality = "balanced"  # 512D adaptive compression (vs fixed 128D "fast")
            program, proc_meta = compressor.compress(canonical_embedding, quality=quality, return_metadata=True)
            galaxy_key = f"char_{char_code}_{target_char}"
            galaxy.store_program(galaxy_key, program, proc_meta["actual_compression"])
            reconstructed = compressor.decompress(program, proc_meta["target_dim"])
            denom = float(np.linalg.norm(canonical_embedding) * np.linalg.norm(reconstructed) + 1e-9)
            cosine = float(np.dot(canonical_embedding, reconstructed) / denom)
            overall_ratio = proc_meta["actual_compression"] * (2048 / proc_meta["target_dim"])
            print(
                f"       Procedural capture: {overall_ratio:.1f}:1 @ {cosine:.6f} fidelity "
                f"(quality=fast, key={galaxy_key})"
            )
        except Exception as exc:
            print(f"       ⚠️ Procedural compression skipped due to error: {exc}")
    elif compressor is None or galaxy is None:
        print("       Procedural compression disabled (raw embeddings only).")

    print("=" * 80)
    print(f"CHARACTER '{target_char}' TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    print(f"Embeddings: {embeddings_array.shape}")
    print()

    result = {
        "char": target_char,
        "char_id": char_code,
        "best_accuracy": best_accuracy,
        "embeddings": embeddings_array,
        "checkpoint_path": embedding_path,
    }
    if compressor is not None and galaxy is not None:
        result["procedural_key"] = f"char_{char_code}_{target_char}"
    return result


def main() -> None:
    # ========================================================================
    # CUDA CONTEXT ISOLATION (Fork-Safety Fix)
    # ========================================================================
    # When Python forks child processes, CUDA contexts don't properly clone.
    # This causes gradient vanishing and GPU synchronization deadlocks.
    # Solution: Explicitly reset and reinitialize CUDA in each child process.
    # ========================================================================

    import os

    # Detect if we're in a forked child process
    current_pid = os.getpid()
    parent_pid = os.getppid()
    is_forked = hasattr(os, '_initial_pid') and current_pid != getattr(os, '_initial_pid', current_pid)

    try:
        import cupy as cp

        # Force CUDA initialization in this process
        # This creates a fresh CUDA context instead of inheriting broken state
        device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])

        # Reset any inherited CUDA state
        try:
            cp.cuda.Device(device_id).use()
            # Force synchronization to ensure clean state
            cp.cuda.Stream.null.synchronize()
            print(f"[CUDA] Context initialized for device {device_id} (PID={current_pid}, PPID={parent_pid})")
        except Exception as cuda_exc:
            print(f"[WARN] CUDA context reset failed: {cuda_exc}")
            print("[WARN] Proceeding anyway - may experience gradient issues")

    except ImportError:
        # CuPy not available - training will use loader.py's CUDA initialization
        print("[INFO] CuPy not available - using loader.py CUDA initialization")

    parser = argparse.ArgumentParser(description="Train CNN on a single character (binary task).")
    parser.add_argument("--char", type=str, required=True, help="Target character (e.g., 'A')")
    parser.add_argument("--lr", type=float, default=0.6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1500, help="Number of epochs (default: 1500)")
    parser.add_argument(
        "--fonts",
        type=int,
        default=0,
        help="Number of fonts to sample (0 uses all available fonts).",
    )
    parser.add_argument("--fc-only", action="store_true", help="Train only the final FC layer (freeze CNN)")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=3000,
        help="Maximum epochs when extension is required (default: 3000).",
    )
    parser.add_argument(
        "--disable-procedural",
        action="store_true",
        help="Disable adaptive procedural compression (raw embeddings only).",
    )

    args = parser.parse_args()
    if len(args.char) != 1:
        raise ValueError("--char argument must be a single character")

    compressor: Optional[AdaptiveDimensionCompressor] = None
    galaxy: Optional[ProceduralGalaxy] = None
    if not args.disable_procedural:
        try:
            compressor = AdaptiveDimensionCompressor()
            galaxy = ProceduralGalaxy(PROCEDURAL_GALAXY_ROOT)
            print("[INFO] Adaptive procedural compression enabled (Matryoshka 64D-2048D, quality=balanced, 512D).")
        except Exception as exc:
            print(f"[WARN] Failed to initialise adaptive compressor ({exc}); proceeding without procedural capture.")
            compressor = None
            galaxy = None

    result = train_single_character(
        target_char=args.char,
        learning_rate=args.lr,
        n_epochs=args.epochs,
        n_fonts=args.fonts,
        fc_only=args.fc_only,
        max_epochs=args.max_epochs,
        compressor=compressor,
        galaxy=galaxy,
    )

    print(f"✓ Successfully trained '{result['char']}' with {result['best_accuracy'] * 100:.2f}% accuracy")


if __name__ == "__main__":
    main()
