"""
Opcode constants for ternary codec operations (DCT/MDCT + ternary quantization).

These map to PTX kernels in knowledge3d/cranium/kernels/codec_ops.cu (to be
implemented). They are declared here for integration with ModularRPNEngine.
"""

OP_TERNARY_QUANT = 0xC0
OP_TERNARY_DEQUANT = 0xC1
OP_TERNARY_ADD = 0xC2
OP_TERNARY_MUL = 0xC3
OP_DCT8X8 = 0xC4
OP_IDCT8X8 = 0xC5
OP_MDCT_FRAME = 0xC6
OP_IMDCT_FRAME = 0xC7
OP_BATCH_DCT = 0xC8
OP_BATCH_MDCT = 0xC9
OP_RESHAPE_TO_BLOCKS = 0xCA
OP_BLOCKS_TO_GRID = 0xCB

__all__ = [
    "OP_TERNARY_QUANT",
    "OP_TERNARY_DEQUANT",
    "OP_TERNARY_ADD",
    "OP_TERNARY_MUL",
    "OP_DCT8X8",
    "OP_IDCT8X8",
    "OP_MDCT_FRAME",
    "OP_IMDCT_FRAME",
    "OP_BATCH_DCT",
    "OP_BATCH_MDCT",
    "OP_RESHAPE_TO_BLOCKS",
    "OP_BLOCKS_TO_GRID",
]
