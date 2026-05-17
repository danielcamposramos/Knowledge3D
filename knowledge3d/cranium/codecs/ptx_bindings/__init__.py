"""PTX bindings for ternary codecs — post-purge stub.

Post-purge surface (2026-04-18):
    All four Python bindings were moved to ``Old_Attempts/2026-04-18/``
    per ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`` §4.1:

        ternary_mdct_binding      (TernaryMDCTKernel)
        ternary_dct8x8_binding    (TernaryDCT8x8Kernel)
        ternary_quant_binding     (TernaryQuantizer)
        audio_harmonic_binding    (AudioHarmonicGPU)

    The underlying PTX kernels remain on disk; drive them directly via
    ``sovereign/loader.py``. Daniel's Ruling A collapses audio into a
    dot-vector-map overlay on the procedural-image head — there is no
    separate audio codec dispatch path.
"""

__all__: list[str] = []
