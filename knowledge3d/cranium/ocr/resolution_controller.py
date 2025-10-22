"""
MultiResolutionController: DeepSeek-inspired multi-resolution processing.

Controls token budgets via explicit resolution modes.

Matches DeepSeek-OCR modes:
- Tiny: 64 tokens, 512px resolution
- Small: 100 tokens, 640px resolution
- Base: 256 tokens, 1024px resolution
- Large: 400 tokens, 1280px resolution
- Gundam: Variable tokens, multi-scale pyramid
"""

from __future__ import annotations

from typing import Dict, Tuple


class MultiResolutionController:
    """
    DeepSeek-inspired multi-resolution processing controller.

    Manages token budgets and resolution settings for different use cases:
    - Tiny: Fast preview, minimal tokens
    - Small: Balanced speed/quality
    - Base: Default K3D mode
    - Large: High-detail documents
    - Gundam: Adaptive multi-scale (Phase F)
    """

    MODES = {
        'tiny': {
            'resolution': 512,
            'tokens': 64,
            'texture_size': 256,
            'compression_target': 20.0,  # 20× compression
            'description': 'Fast preview mode'
        },
        'small': {
            'resolution': 640,
            'tokens': 100,
            'texture_size': 256,
            'compression_target': 12.0,  # 12× compression
            'description': 'Balanced mode for House storage'
        },
        'base': {
            'resolution': 1024,
            'tokens': 256,
            'texture_size': 512,
            'compression_target': 7.0,   # 7× compression (optimal)
            'description': 'Default K3D mode'
        },
        'large': {
            'resolution': 1280,
            'tokens': 400,
            'texture_size': 512,
            'compression_target': 5.0,   # 5× compression
            'description': 'High-detail mode for critical documents'
        },
        'gundam': {
            'resolution': 'multi',
            'tokens': 'variable',
            'texture_size': 'pyramid',
            'compression_target': 'adaptive',
            'description': 'Adaptive multi-scale pyramid (Phase F)'
        }
    }

    def __init__(self, mode: str = 'small'):
        """
        Initialize multi-resolution controller.

        Args:
            mode: Resolution mode (tiny, small, base, large, gundam)
                  Default: 'small' (optimized for House storage)
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(self.MODES.keys())}")

        self.mode = mode
        self.config = self.MODES[mode]

    def get_resolution(self) -> int:
        """Get target resolution for current mode."""
        res = self.config['resolution']
        if isinstance(res, str):
            return 1024  # Default fallback for 'multi'
        return int(res)

    def get_token_budget(self) -> int:
        """Get token budget for current mode."""
        tokens = self.config['tokens']
        if isinstance(tokens, str):
            return 256  # Default fallback for 'variable'
        return int(tokens)

    def get_texture_size(self) -> int:
        """Get texture size for AI texture generation."""
        size = self.config['texture_size']
        if isinstance(size, str):
            return 512  # Default fallback for 'pyramid'
        return int(size)

    def get_compression_target(self) -> float:
        """Get target compression ratio."""
        comp = self.config['compression_target']
        if isinstance(comp, str):
            return 7.0  # Default fallback for 'adaptive'
        return float(comp)

    def resize_input(self, width: int, height: int) -> Tuple[int, int]:
        """
        Resize input dimensions to match resolution target.

        Args:
            width: Input image width
            height: Input image height

        Returns:
            (target_width, target_height) maintaining aspect ratio
        """
        target_res = self.get_resolution()

        # Maintain aspect ratio
        max_dim = max(width, height)
        if max_dim == 0:
            return (target_res, target_res)

        scale = target_res / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)

        return (new_width, new_height)

    def should_compress(self, current_tokens: int) -> bool:
        """
        Check if compression is needed based on token budget.

        Args:
            current_tokens: Current token count

        Returns:
            True if compression needed, False otherwise
        """
        budget = self.get_token_budget()
        if isinstance(budget, str):
            return False  # Gundam mode handles adaptively

        return current_tokens > budget

    def get_mode_info(self) -> Dict[str, object]:
        """Get current mode configuration."""
        return {
            'mode': self.mode,
            'resolution': self.get_resolution(),
            'tokens': self.get_token_budget(),
            'texture_size': self.get_texture_size(),
            'compression_target': self.get_compression_target(),
            'description': self.config['description']
        }

    @classmethod
    def recommend_mode(cls, page_count: int, detail_level: str = 'medium') -> str:
        """
        Recommend resolution mode based on document characteristics.

        Args:
            page_count: Number of pages in document
            detail_level: Required detail (low, medium, high)

        Returns:
            Recommended mode name
        """
        if detail_level == 'high' or page_count < 10:
            return 'base'
        elif detail_level == 'medium' or page_count < 100:
            return 'small'
        else:
            return 'tiny'
