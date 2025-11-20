"""Codecs package for ternary procedural audio/video compression."""

from .ternary_quantization import (
    quantize_ternary,
    dequantize_ternary,
    compute_sparsity,
    entropy_encode_ternary,
    entropy_decode_ternary,
)
from .procedural_audio import ProceduralAudioSynthesizer
from .ternary_audio_codec import TernaryAudioCodec
from .procedural_video import ProceduralVideoGenerator

__all__ = [
    "quantize_ternary",
    "dequantize_ternary",
    "compute_sparsity",
    "entropy_encode_ternary",
    "entropy_decode_ternary",
    "ProceduralAudioSynthesizer",
    "TernaryAudioCodec",
    "ProceduralVideoGenerator",
]
