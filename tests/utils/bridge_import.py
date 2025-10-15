"""
Robust import utility for ThinkingTagBridge across different repository structures.

This module provides a unified way to import ThinkingTagBridge, handling:
- Development environment (sovereign_bridges location)
- Public repo structure (ptx_runtime location)
- Test-only environments (mock fallback)
"""
import importlib
from unittest.mock import Mock


def get_thinking_tag_bridge():
    """
    Robust import for ThinkingTagBridge across development and public repo structures.

    Import priority:
    1. knowledge3d.cranium.bridges.sovereign_bridges (development)
    2. knowledge3d.cranium.ptx_runtime.thinking_tag_bridge (public)
    3. Mock (test-only environments)

    Returns:
        ThinkingTagBridge class or Mock
    """
    try:
        # Primary location (development environment)
        from knowledge3d.cranium.bridges.sovereign_bridges import ThinkingTagBridge
        return ThinkingTagBridge
    except ImportError:
        try:
            # Fallback location (public repo structure)
            from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge
            return ThinkingTagBridge
        except ImportError:
            # Mock for testing environments without actual implementation
            return Mock


__all__ = ['get_thinking_tag_bridge']
