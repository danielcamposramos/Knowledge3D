"""
DeepSeekOCRBridge: K3D integration of DeepSeek-OCR techniques.

Implements "Contexts Optical Compression" for dual-texture generation.
Maps DeepSeek-OCR's two-stage vision encoder to K3D's sovereign stack.

Pipeline:
    Image → LocalPerception (SAM-base) → ConvCompressor (16×) →
    Text Extraction → GlobalContext (CLIP-large) → Dual Textures

Phase E: CPU stubs with simple OCR (Tesseract/PyMuPDF)
Phase F: Full PTX sovereign implementation
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from knowledge3d.cranium.ocr.local_perception import LocalPerceptionEncoder
from knowledge3d.cranium.ocr.conv_compressor import ConvolutionalCompressor
from knowledge3d.cranium.ocr.global_context import GlobalContextEncoder
from knowledge3d.cranium.ocr.resolution_controller import MultiResolutionController

# Phase F.1: GPU-accelerated OCR model
try:
    from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
    GPU_OCR_AVAILABLE = True
except ImportError:
    GPU_OCR_AVAILABLE = False


class DeepSeekOCRBridge:
    """
    K3D integration of DeepSeek-OCR techniques.

    Implements the dual-texture paradigm:
    - Human Texture: Pretty, game-style rendering (512×512 RGB)
    - AI Texture: Dense text-as-image compression (256×256 RGB)

    DeepSeek architecture mapping:
        Stage 1: SAM-base (local perception)      → LocalPerceptionEncoder
        Stage 2: 16× Conv Compressor              → ConvolutionalCompressor
        Stage 3: Text extraction (OCR)            → PyMuPDF + Tesseract
        Stage 4: CLIP-large (global context)      → GlobalContextEncoder

    Target: 7-20× compression, 97% fidelity at <10× compression
    """

    def __init__(self, mode: str = 'small', use_gpu_ocr: bool = True):
        """
        Initialize DeepSeek OCR bridge.

        Args:
            mode: Resolution mode (tiny, small, base, large, gundam)
                  Default: 'small' optimized for House storage
            use_gpu_ocr: Use GPU-accelerated OCR model (Phase F.1) if available
        """
        # DeepSeek components
        self.local_encoder = LocalPerceptionEncoder(window_size=16)
        self.compressor = ConvolutionalCompressor(compression_ratio=16)
        self.global_encoder = GlobalContextEncoder()
        self.resolution_ctrl = MultiResolutionController(mode=mode)

        # Target compression (DeepSeek achieves 7-20×)
        self.compression_target = self.resolution_ctrl.get_compression_target()

        # Mode configuration
        self.mode = mode
        self.texture_size = self.resolution_ctrl.get_texture_size()

        # Phase F.1: Initialize GPU OCR model if available and requested
        self.gpu_ocr_model = None
        self.use_gpu_ocr = use_gpu_ocr and GPU_OCR_AVAILABLE

        if self.use_gpu_ocr:
            try:
                print("[PHASE_F.1] Initializing GPU-accelerated OCR model...")
                self.gpu_ocr_model = DeepSeekOCRModel(
                    num_glyphs=256,
                    input_channels=3,
                    use_micro_trm=False  # Disable for stability
                )
                print(f"[PHASE_F.1] ✓ GPU OCR model ready (mode={mode})")
            except Exception as exc:
                print(f"[PHASE_F.1] WARNING: Could not initialize GPU OCR model - {exc}")
                self.use_gpu_ocr = False

    def extract(self, image: np.ndarray, pdf_path: Optional[Path] = None, page_num: int = 0) -> Dict[str, Any]:
        """
        Extract text and features from image using DeepSeek pipeline.

        Args:
            image: Input image (H, W, 3) RGB uint8
            pdf_path: Optional PDF path for structured text extraction
            page_num: Page number if extracting from PDF

        Returns:
            Dictionary containing:
            - full_text: Extracted text
            - compressed_features: Compressed visual features
            - global_context: Global document embedding
            - compression_ratio: Achieved compression ratio
            - fidelity: Estimated accuracy (0.0-1.0)
        """
        # Resize to target resolution
        target_w, target_h = self.resolution_ctrl.resize_input(image.shape[1], image.shape[0])

        try:
            from skimage.transform import resize  # type: ignore
            resized = resize(
                image,
                (target_h, target_w),
                order=1,
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.uint8)
        except ImportError:
            # Fallback: use original image
            resized = image

        # Stage 1: Local perception (SAM-base equivalent)
        local_features = self.local_encoder.encode_local_features(resized)

        # Stage 2: Convolutional compression (16× reduction)
        compressed = self.compressor.compress(local_features)

        # Stage 3: Extract text
        # Phase F.1: Try GPU OCR model first, fallback to traditional OCR
        if self.use_gpu_ocr and self.gpu_ocr_model is not None:
            # Convert to float32 [0, 1] range for GPU model
            resized_float = resized.astype(np.float32) / 255.0
            gpu_results = self.gpu_ocr_model.forward(resized_float)
            feature_map = gpu_results.get("feature_map")

            # TODO: Implement character detection from feature map
            # For now, fallback to PDF text extraction if available
            if pdf_path and pdf_path.exists():
                text = self._extract_text_from_pdf(pdf_path, page_num)
                if not text:
                    text = self._extract_text_simple(resized)
            else:
                text = self._extract_text_simple(resized)

            print(f"[PHASE_F.1] GPU OCR feature extraction: {gpu_results['output_shape']}")
        elif pdf_path and pdf_path.exists():
            text = self._extract_text_from_pdf(pdf_path, page_num)
        else:
            text = self._extract_text_simple(resized)

        # Stage 4: Global context (CLIP-large equivalent)
        global_context = self.global_encoder.encode_global_context(
            compressed, text
        )

        # Calculate compression ratio
        input_tokens = (resized.shape[0] // 4) * (resized.shape[1] // 4)  # After local encoder
        output_tokens = compressed.shape[0] * compressed.shape[1]
        compression_ratio = float(input_tokens) / max(1, output_tokens)

        feature_map_dim = feature_map.shape[2] if isinstance(feature_map, np.ndarray) and feature_map.ndim == 3 else 0
        complexity_score = self._estimate_image_complexity(resized)
        if feature_map_dim <= 0:
            feature_dim = 0
        elif complexity_score >= 0.75:
            feature_dim = min(feature_map_dim, 256)
        elif complexity_score >= 0.55:
            feature_dim = min(feature_map_dim, 192)
        elif complexity_score >= 0.35:
            feature_dim = min(feature_map_dim, 128)
        else:
            feature_dim = min(feature_map_dim, 64)

        # Estimate fidelity (DeepSeek: 97% at <10× compression)
        if compression_ratio <= 10.0:
            fidelity = 0.97
        elif compression_ratio <= 15.0:
            fidelity = 0.85
        else:
            fidelity = 0.60

        return {
            'full_text': text,
            'compressed_features': compressed,
            'global_context': global_context,
            'compression_ratio': compression_ratio,
            'fidelity': fidelity,
            'mode': self.mode,
            'texture_size': self.texture_size,
            'feature_map': feature_map,
            'feature_dim': feature_dim,
            'feature_complexity': complexity_score,
        }

    def encode_ai_texture(self, compressed_features: np.ndarray, text: str) -> np.ndarray:
        """
        Generate AI texture from compressed features.

        DeepSeek approach: Text-as-image compression (dense text rendering)

        Args:
            compressed_features: Compressed visual features
            text: Full extracted text

        Returns:
            AI texture (texture_size, texture_size, 3) uint8 RGB
        """
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except ImportError:
            # Fallback: Return white texture
            return np.full((self.texture_size, self.texture_size, 3), 255, dtype=np.uint8)

        # Phase E: Render text densely as image
        # DeepSeek approach: Maximize text density (7× more compact than normal)

        img = Image.new('RGB', (self.texture_size, self.texture_size), 'white')
        draw = ImageDraw.Draw(img)

        # Use tiny font for maximum density
        try:
            # DejaVu Sans Mono is common on Linux systems
            font = ImageFont.truetype("DejaVuSansMono.ttf", 6)
        except Exception:
            try:
                # Fallback fonts
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 6)
            except Exception:
                # Use default font
                font = ImageFont.load_default()

        # Dense text rendering (7× compression)
        y = 2
        line_height = 7  # Very tight spacing
        max_chars = self.texture_size // 4  # ~64 chars per line for 256px

        for i in range(0, len(text), max_chars):
            line = text[i:i+max_chars]
            try:
                draw.text((2, y), line, fill='black', font=font)
            except Exception:
                # Font might not support some characters
                safe_line = ''.join(c if ord(c) < 128 else '?' for c in line)
                draw.text((2, y), safe_line, fill='black', font=font)

            y += line_height
            if y > self.texture_size - line_height:
                break  # Texture full

        return np.array(img, dtype=np.uint8)

    def encode_human_texture(self, image: np.ndarray, text: str) -> np.ndarray:
        """
        Generate human-readable texture (pretty, game-style).

        Args:
            image: Original page image
            text: Extracted text (optional overlay)

        Returns:
            Human texture (512, 512, 3) uint8 RGB
        """
        try:
            from skimage.transform import resize  # type: ignore
        except ImportError:
            # Fallback: Return original or white
            if image.size > 0:
                return image[:512, :512] if image.shape[0] >= 512 and image.shape[1] >= 512 else image
            return np.full((512, 512, 3), 255, dtype=np.uint8)

        # Resize to 512×512 (human texture is always 512×512 for consistency)
        human_texture = resize(
            image,
            (512, 512),
            order=1,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)

        return human_texture

    def _extract_text_from_pdf(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text from PDF using PyMuPDF (structured text).

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to extract

        Returns:
            Extracted text
        """
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as doc:
                if page_num < 0 or page_num >= len(doc):
                    return ""

                page = doc[page_num]
                text = page.get_text("text")
                return text.strip()

        except Exception:
            return ""

    def _extract_text_simple(self, image: np.ndarray) -> str:
        """
        Simple OCR fallback using Tesseract.

        Args:
            image: Image to OCR (H, W, 3) uint8

        Returns:
            Extracted text
        """
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            # Convert numpy to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

            pil_img = Image.fromarray(image)
            text = pytesseract.image_to_string(pil_img, lang='eng')
            return text.strip()

        except Exception:
            # No OCR available
            return ""

    @staticmethod
    def _estimate_image_complexity(image: np.ndarray) -> float:
        """
        Estimate visual complexity of the page to choose embedding dimension.
        """
        if image.ndim == 3:
            gray = image.mean(axis=2)
        else:
            gray = image.astype(np.float32)

        gray = gray.astype(np.float32)
        if gray.max() > 0.0:
            gray = gray / gray.max()

        gx = np.mean(np.abs(np.diff(gray, axis=1))) if gray.shape[1] > 1 else 0.0
        gy = np.mean(np.abs(np.diff(gray, axis=0))) if gray.shape[0] > 1 else 0.0

        score = (gx + gy) * 1.5
        if np.isnan(score):
            score = 0.0
        return float(max(0.0, min(1.0, score)))

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics for current mode.

        Returns:
            Dictionary with compression metrics
        """
        return {
            'mode': self.mode,
            'compression_target': self.compression_target,
            'texture_size_human': 512,
            'texture_size_ai': self.texture_size,
            'expected_fidelity': 0.97 if self.compression_target <= 10.0 else 0.85,
            'token_budget': self.resolution_ctrl.get_token_budget()
        }
