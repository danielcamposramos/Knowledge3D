"""Codecs package — sovereign PTX kernel sources.

Post-purge surface (2026-04-18):
    All Python codec wrappers were moved to ``Old_Attempts/2026-04-18/``
    per ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`` §4.1:

        ternary_quantization             (quantize/dequantize helpers)
        procedural_audio                 (ProceduralAudioSynthesizer)
        ternary_audio_codec              (TernaryAudioCodec)
        ternary_video_codec              (TernaryVideoCodec)
        procedural_video                 (ProceduralVideoGenerator)
        ternary_codec_ops                (TernaryCodecOps)
        sovereign_ternary_video_codec    (SovereignTernaryVideoCodec)
        sovereign_ternary_audio_codec    (SovereignTernaryAudioCodec)

    The CUDA sources that back these codecs still live on disk at:

        knowledge3d/cranium/codecs/kernels/procedural_synthesis.cu
        knowledge3d/cranium/codecs/kernels/procedural_texture.cu
        knowledge3d/cranium/codecs/kernels/ternary_dct_2d.cu
        knowledge3d/cranium/codecs/kernels/ternary_mdct.cu
        knowledge3d/cranium/ptx/codec_ops.ptx
        knowledge3d/cranium/ptx/ternary_ops.ptx

    Sovereign successors drive those kernels directly through
    ``sovereign/loader.py``. Daniel's Ruling A (2026-04-18) folds image /
    audio / video into a single procedural-image head with an audio
    dot-vector-map overlay and video-as-frame-sequence — there is no
    per-kind Python dispatch anymore. Do not reintroduce one.
"""

__all__: list[str] = []
