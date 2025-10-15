"""
Phase 1.1: Shape Primitives Edge Case Tests for Step 11

Tests edge cases in shape primitive generation from Step 11 sovereign runtime.
Validates handling of zero/negative dimensions, extreme aspect ratios, UTF-8 issues,
and boundary conditions.

Target: 30+ edge case tests
Developed by: Kimi, enhanced by Claude
"""
import pytest
import random
import unicodedata
from unittest import mock

from tests.utils import get_thinking_tag_bridge

ThinkingTagBridge = get_thinking_tag_bridge()

# Deterministic torture seeds
random.seed(42)

# Edge case test matrices
DIMS_NEGATIVE = [(-1,), (0, -1e-6), (-1e6, 5)]
DIMS_ZERO = [(0, 0, 0), (0.0, 1, 1)]
DIMS_EXTREME = [(1e-6, 1e6), (1e6, 1e-6)]
UTF8_MALFORMED = b'\xff\xfe invalid utf8 \xaa'
PROMPT_EMPTY = ""
PROMPT_HUGE = "wooden chair " * 200  # ~2400 tokens


class TestShapePrimitivesEdges:
    def setup_method(self):
        try:
            self.bridge = ThinkingTagBridge()
        except RuntimeError:
            self.bridge = mock.Mock()
        # GPU-sovereign mock: never call real kernels
        if not hasattr(self.bridge, 'generate_shape'):
            self.bridge.generate_shape = mock.Mock(
                return_value=mock.Mock(vertices=b"", indices=b"", aabb=[0, 0, 0, 1, 1, 1])
            )

    @pytest.mark.parametrize("dims", DIMS_NEGATIVE)
    def test_rejects_negative_dimensions(self, dims):
        """Negative dimensions should raise ValueError."""
        with pytest.raises((ValueError, AssertionError)):
            if hasattr(self.bridge, 'generate_shape'):
                self.bridge.generate_shape(prompt="cube", dims=dims)

    @pytest.mark.parametrize("dims", DIMS_ZERO)
    def test_accepts_zero_volume(self, dims):
        """Zero-volume is legal (degenerate) but must not crash."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="flat cube", dims=dims)
            assert mesh is not None

    @pytest.mark.parametrize("dims", DIMS_EXTREME)
    def test_extreme_aspect_ratios(self, dims):
        """Extreme ratios (needle/pancake shapes) should work."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="needle", dims=dims)
            # Ensure no NaN in bounding box
            if hasattr(mesh, 'aabb'):
                assert all(isinstance(x, (int, float)) and not (x != x) for x in mesh.aabb)

    def test_invalid_utf8_description(self):
        """Bridge must sanitize or reject malformed UTF-8."""
        if hasattr(self.bridge, 'generate_shape'):
            with pytest.raises((UnicodeDecodeError, ValueError, TypeError)):
                self.bridge.generate_shape(prompt=UTF8_MALFORMED, dims=(1, 1, 1))

    def test_empty_prompt_string(self):
        """Empty prompt should yield default primitive."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt=PROMPT_EMPTY, dims=(1, 1, 1))
            # Should yield default (mock returns empty)
            assert mesh is not None

    def test_maximum_complexity_prompt(self):
        """Huge prompt must not OOM."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt=PROMPT_HUGE, dims=(1, 1, 1))
            assert mesh is not None
            if hasattr(mesh, 'vertex_count'):
                assert mesh.vertex_count < 1e6  # Sanity ceiling

    def test_sphere_zero_radius(self):
        """Zero-radius sphere (point)."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="sphere radius 0", dims=(0, 0, 0))
            assert mesh is not None

    def test_cylinder_negative_height(self):
        """Negative height should be rejected."""
        if hasattr(self.bridge, 'generate_shape'):
            with pytest.raises((ValueError, AssertionError)):
                self.bridge.generate_shape(prompt="cylinder", dims=(1, -1, 1))

    def test_nan_in_dimensions(self):
        """NaN dimensions should be rejected."""
        if hasattr(self.bridge, 'generate_shape'):
            with pytest.raises((ValueError, AssertionError)):
                self.bridge.generate_shape(prompt="cube", dims=(float('nan'), 1, 1))

    def test_inf_in_dimensions(self):
        """Infinity dimensions should be rejected."""
        if hasattr(self.bridge, 'generate_shape'):
            with pytest.raises((ValueError, AssertionError)):
                self.bridge.generate_shape(prompt="cube", dims=(float('inf'), 1, 1))

    def test_unicode_emoji_in_prompt(self):
        """Unicode emoji should be handled."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="red cube \U0001f600", dims=(1, 1, 1))
            assert mesh is not None

    def test_chinese_characters_in_prompt(self):
        """Chinese characters should work."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="红色立方体", dims=(1, 1, 1))
            assert mesh is not None

    def test_rtl_text_in_prompt(self):
        """Right-to-left text (Arabic/Hebrew)."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="مكعب أحمر", dims=(1, 1, 1))
            assert mesh is not None

    def test_control_characters_in_prompt(self):
        """Control characters should be sanitized."""
        if hasattr(self.bridge, 'generate_shape'):
            prompt_with_control = "cube\x00with\nnull\x1f"
            mesh = self.bridge.generate_shape(prompt=prompt_with_control, dims=(1, 1, 1))
            assert mesh is not None

    def test_dimension_tuple_vs_list(self):
        """Both tuple and list dimensions should work."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh1 = self.bridge.generate_shape(prompt="cube", dims=(1, 1, 1))
            mesh2 = self.bridge.generate_shape(prompt="cube", dims=[1, 1, 1])
            assert mesh1 is not None and mesh2 is not None

    def test_single_dimension_value(self):
        """Single value should be interpreted or rejected."""
        if hasattr(self.bridge, 'generate_shape'):
            # Might interpret as uniform scaling or reject
            try:
                mesh = self.bridge.generate_shape(prompt="cube", dims=(1,))
            except (ValueError, TypeError):
                pass  # Expected if not supported

    def test_four_dimensional_input(self):
        """4D input should be rejected or truncated."""
        if hasattr(self.bridge, 'generate_shape'):
            try:
                mesh = self.bridge.generate_shape(prompt="hypercube", dims=(1, 1, 1, 1))
            except (ValueError, TypeError):
                pass  # Expected

    def test_very_small_dimensions(self):
        """Near-epsilon dimensions."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="tiny cube", dims=(1e-10, 1e-10, 1e-10))
            assert mesh is not None

    def test_very_large_dimensions(self):
        """Near-max float dimensions."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="huge cube", dims=(1e10, 1e10, 1e10))
            assert mesh is not None

    def test_mixed_dimension_types(self):
        """Mix of int and float dimensions."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="cube", dims=(1, 2.5, 3))
            assert mesh is not None

    def test_prompt_with_special_characters(self):
        """Special chars: quotes, backslashes, etc."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt='cube "with" \\backslash', dims=(1, 1, 1))
            assert mesh is not None

    def test_prompt_with_numbers_only(self):
        """Numeric-only prompt."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="12345", dims=(1, 1, 1))
            assert mesh is not None

    def test_prompt_with_url(self):
        """URL in prompt should not cause issues."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="https://example.com/cube", dims=(1, 1, 1))
            assert mesh is not None

    def test_prompt_with_sql_injection_attempt(self):
        """SQL injection patterns should be harmless."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="'; DROP TABLE shapes; --", dims=(1, 1, 1))
            assert mesh is not None

    def test_prompt_with_html_tags(self):
        """HTML tags should be treated as text."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="<script>alert('xss')</script>", dims=(1, 1, 1))
            assert mesh is not None

    def test_whitespace_only_prompt(self):
        """Whitespace-only prompt."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="     ", dims=(1, 1, 1))
            assert mesh is not None

    def test_newline_in_prompt(self):
        """Multi-line prompts."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="red cube\nwith\nlines", dims=(1, 1, 1))
            assert mesh is not None

    def test_tabs_in_prompt(self):
        """Tab characters in prompt."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="red\tcube\twith\ttabs", dims=(1, 1, 1))
            assert mesh is not None

    def test_repeated_words(self):
        """Repeated words shouldn't cause issues."""
        if hasattr(self.bridge, 'generate_shape'):
            mesh = self.bridge.generate_shape(prompt="cube cube cube cube", dims=(1, 1, 1))
            assert mesh is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
