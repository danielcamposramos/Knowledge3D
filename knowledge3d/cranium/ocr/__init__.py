"""
Phase E: DeepSeek-OCR integration for K3D dual-texture paradigm.

This package implements the "Contexts Optical Compression" approach from
DeepSeek-OCR, mapping their architecture to K3D's sovereign stack:

- LocalPerceptionEncoder: SAM-base equivalent (window attention)
- ConvolutionalCompressor: 16× spatial token reduction
- GlobalContextEncoder: CLIP-large equivalent (dense attention)
- MultiResolutionController: Token budget management
- DeepSeekOCRBridge: Complete integration bridge

Architecture alignment:
    DeepSeek-OCR                K3D Phase E
    ════════════════            ═══════════
    SAM-base (80M)     →        LocalPerceptionEncoder (PTX kernels)
    16× Conv Compress  →        ConvolutionalCompressor (strided conv)
    CLIP-large (300M)  →        GlobalContextEncoder (GalaxyResonance)
    Multi-resolution   →        MultiResolutionController (LOD system)

Target performance:
- 7-20× compression ratio
- 97% accuracy at <10× compression
- Dual-texture output: Human (512×512) + AI (256×256)
"""

from knowledge3d.cranium.ocr.local_perception import LocalPerceptionEncoder
from knowledge3d.cranium.ocr.conv_compressor import ConvolutionalCompressor
from knowledge3d.cranium.ocr.global_context import GlobalContextEncoder
from knowledge3d.cranium.ocr.resolution_controller import MultiResolutionController
from knowledge3d.cranium.ocr.deepseek_bridge import DeepSeekOCRBridge

__all__ = [
    "LocalPerceptionEncoder",
    "ConvolutionalCompressor",
    "GlobalContextEncoder",
    "MultiResolutionController",
    "DeepSeekOCRBridge",
]
