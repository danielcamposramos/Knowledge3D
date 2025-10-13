"""
Comprehensive test suite for Step 11 ShapePrimitives with RPN integration.
Tests primitive generation, LOD levels, semantic adaptation, and RPN acceleration.
"""
import pytest
import numpy as np
from pathlib import Path

from knowledge3d.cranium.ptx_runtime.shape_primitives import ShapePrimitives


class TestShapePrimitivesBasic:
    """Test basic primitive generation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_initialization(self):
        """Test ShapePrimitives initializes correctly."""
        assert self.primitives is not None
        assert hasattr(self.primitives, 'rpn')
        assert hasattr(self.primitives, 'templates')
        assert len(self.primitives.templates) == 5  # cube, sphere, cylinder, cone, torus

    def test_semantic_geometry_map(self):
        """Test semantic-to-geometry mapping exists."""
        semantic_map = self.primitives.semantic_geometry_map
        assert 'architectural' in semantic_map
        assert 'organic' in semantic_map
        assert 'mechanical' in semantic_map
        assert 'natural' in semantic_map


class TestLODGeneration:
    """Test LOD (Level of Detail) generation for all primitives."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_cube_lod_levels(self):
        """Test cube generation at all 3 LOD levels."""
        for lod in [0, 1, 2]:
            vertices, indices = self.primitives.generate_cube(size=1.0, lod_level=lod)

            assert vertices.shape[0] > 0, f"LOD {lod} should have vertices"
            assert indices.shape[0] > 0, f"LOD {lod} should have indices"
            assert vertices.dtype == np.float32
            assert indices.dtype == np.uint32

    def test_lod_vertex_count_decreases(self):
        """Test that LOD 2 has fewer vertices than LOD 0."""
        v0, _ = self.primitives.generate_cube(size=1.0, lod_level=0)
        v1, _ = self.primitives.generate_cube(size=1.0, lod_level=1)
        v2, _ = self.primitives.generate_cube(size=1.0, lod_level=2)

        # LOD 2 should have fewer or equal vertices than LOD 1
        assert v2.shape[0] <= v1.shape[0]
        # LOD 1 should have fewer or equal vertices than LOD 0
        assert v1.shape[0] <= v0.shape[0]

    def test_sphere_lod_levels(self):
        """Test sphere generation at all LOD levels."""
        for lod in [0, 1, 2]:
            vertices, indices = self.primitives.generate_sphere(
                radius=1.0, subdivisions=2, lod_level=lod
            )

            assert vertices.shape[0] > 0
            assert indices.shape[0] > 0
            # Sphere vertices should be roughly normalized
            norms = np.linalg.norm(vertices, axis=1)
            assert np.allclose(norms, 1.0, atol=0.1), "Sphere vertices should be on unit sphere"

    def test_cylinder_lod_levels(self):
        """Test cylinder generation at all LOD levels."""
        for lod in [0, 1, 2]:
            vertices, indices = self.primitives.generate_cylinder(
                radius=1.0, height=2.0, segments=16, lod_level=lod
            )

            assert vertices.shape[0] > 0
            assert indices.shape[0] > 0

    def test_cone_lod_levels(self):
        """Test cone generation at all LOD levels."""
        for lod in [0, 1, 2]:
            vertices, indices = self.primitives.generate_cone(
                radius=1.0, height=2.0, segments=16, lod_level=lod
            )

            assert vertices.shape[0] > 0
            assert indices.shape[0] > 0

    def test_torus_lod_levels(self):
        """Test torus generation at all LOD levels."""
        for lod in [0, 1, 2]:
            vertices, indices = self.primitives.generate_torus(
                major_radius=2.0, minor_radius=0.5, lod_level=lod
            )

            assert vertices.shape[0] > 0
            assert indices.shape[0] > 0


class TestRPNAcceleration:
    """Test RPN-accelerated operations in primitives."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_rpn_scaling(self):
        """Test RPN is used for scaling operations."""
        # Generate cube with size 2.0
        vertices, _ = self.primitives.generate_cube(size=2.0)

        # Vertices should be in range [-1, 1] after scaling
        assert vertices.max() <= 1.1, "Scaled vertices should fit in unit cube"
        assert vertices.min() >= -1.1

    def test_rpn_transform_batch(self):
        """Test RPN can handle batch operations."""
        # Generate multiple primitives
        v1, _ = self.primitives.generate_cube(size=0.5)
        v2, _ = self.primitives.generate_cube(size=1.0)
        v3, _ = self.primitives.generate_cube(size=2.0)

        # All should succeed
        assert v1.shape[0] > 0
        assert v2.shape[0] > 0
        assert v3.shape[0] > 0


class TestSemanticAdaptation:
    """Test semantic adaptation of primitives with modal features."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_organic_deformation(self):
        """Test organic semantic adaptation."""
        base_verts, _ = self.primitives.generate_sphere(radius=1.0)
        features = np.random.rand(32).astype(np.float32)

        adapted = self.primitives.adapt_primitive_from_modal(
            base_verts, features, {'category': 'organic', 'strength': 0.5}
        )

        assert adapted.shape == base_verts.shape
        # Adapted vertices should differ from base
        assert not np.array_equal(adapted, base_verts)

    def test_mechanical_precision(self):
        """Test mechanical semantic adaptation."""
        base_verts, _ = self.primitives.generate_cube(size=1.0)
        features = np.random.rand(32).astype(np.float32)

        adapted = self.primitives.adapt_primitive_from_modal(
            base_verts, features, {'category': 'mechanical', 'strength': 0.7}
        )

        assert adapted.shape == base_verts.shape
        # Mechanical adaptation should quantize vertices
        assert not np.array_equal(adapted, base_verts)

    def test_architectural_constraints(self):
        """Test architectural semantic adaptation."""
        base_verts, _ = self.primitives.generate_cylinder(radius=1.0, height=2.0)
        features = np.random.rand(32).astype(np.float32)

        adapted = self.primitives.adapt_primitive_from_modal(
            base_verts, features, {'category': 'architectural', 'strength': 0.6}
        )

        assert adapted.shape == base_verts.shape
        assert not np.array_equal(adapted, base_verts)

    def test_no_features_returns_unchanged(self):
        """Test that empty features returns unchanged vertices."""
        base_verts, _ = self.primitives.generate_cube(size=1.0)

        adapted = self.primitives.adapt_primitive_from_modal(
            base_verts, np.array([]), None
        )

        assert np.array_equal(adapted, base_verts)


class TestSemanticSuggestions:
    """Test semantic shape suggestions from embeddings."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_semantic_suggestions_format(self):
        """Test semantic suggestions return correct format."""
        embedding = np.random.rand(512).astype(np.float32)

        suggestions = self.primitives.get_semantic_suggestions(embedding)

        # Should return list of tuples (shape_type, confidence)
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3  # Top 3

        for shape_type, confidence in suggestions:
            assert isinstance(shape_type, str)
            assert isinstance(confidence, (float, np.floating))
            assert 0.0 <= confidence <= 1.0

    def test_geometric_pattern_detection(self):
        """Test detection of geometric patterns in embeddings."""
        # Create embedding with high first 3 components (geometric)
        embedding = np.zeros(512, dtype=np.float32)
        embedding[:3] = 0.8

        suggestions = self.primitives.get_semantic_suggestions(embedding)

        # Should suggest geometric shapes
        shape_types = [s[0] for s in suggestions]
        assert 'cube' in shape_types or 'cylinder' in shape_types


class TestPrimitiveQuality:
    """Test quality and correctness of generated primitives."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_cube_has_correct_topology(self):
        """Test cube has 8 vertices and 12 triangles at LOD 0."""
        vertices, indices = self.primitives.generate_cube(size=1.0, lod_level=0)

        # Base cube should have 8 vertices
        assert vertices.shape[0] == 8
        # 12 triangles (6 faces * 2 triangles per face)
        assert indices.shape[0] == 12

    def test_sphere_subdivision_increases_vertices(self):
        """Test sphere subdivision increases vertex count."""
        v0, _ = self.primitives.generate_sphere(radius=1.0, subdivisions=0)
        v1, _ = self.primitives.generate_sphere(radius=1.0, subdivisions=1)
        v2, _ = self.primitives.generate_sphere(radius=1.0, subdivisions=2)

        # More subdivisions = more vertices
        assert v1.shape[0] > v0.shape[0]
        assert v2.shape[0] > v1.shape[0]

    def test_cylinder_has_caps(self):
        """Test cylinder has top and bottom caps."""
        vertices, indices = self.primitives.generate_cylinder(
            radius=1.0, height=2.0, segments=16
        )

        # Check z-coordinates cover both caps
        z_coords = vertices[:, 2]
        assert z_coords.max() > 0.9  # Top cap around z=1
        assert z_coords.min() < -0.9  # Bottom cap around z=-1

    def test_torus_has_donut_shape(self):
        """Test torus has correct donut topology."""
        vertices, indices = self.primitives.generate_torus(
            major_radius=2.0, minor_radius=0.5
        )

        # Torus should have significant number of vertices
        assert vertices.shape[0] > 100

        # Check that vertices form a ring
        xy_distances = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
        # Major radius should dominate
        assert xy_distances.mean() > 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_zero_size_cube(self):
        """Test cube with zero size."""
        vertices, indices = self.primitives.generate_cube(size=0.0)

        # Should still generate, just with zero-sized vertices
        assert vertices.shape[0] > 0
        assert indices.shape[0] > 0

    def test_negative_size_cube(self):
        """Test cube with negative size (should work via RPN)."""
        vertices, indices = self.primitives.generate_cube(size=-1.0)

        # Should generate inverted cube
        assert vertices.shape[0] > 0
        assert indices.shape[0] > 0

    def test_very_large_size(self):
        """Test primitive with very large size."""
        vertices, indices = self.primitives.generate_sphere(radius=1000.0)

        assert vertices.shape[0] > 0
        # Vertices should be scaled appropriately
        norms = np.linalg.norm(vertices, axis=1)
        assert norms.max() > 500.0

    def test_invalid_lod_level(self):
        """Test that invalid LOD level is handled."""
        # LOD level 3 doesn't exist, should default to LOD 2 or raise
        try:
            vertices, indices = self.primitives.generate_cube(size=1.0, lod_level=3)
            # If it doesn't raise, it should return valid geometry
            assert vertices.shape[0] > 0
        except (KeyError, IndexError):
            # It's acceptable to raise an error for invalid LOD
            pass


class TestPerformance:
    """Test performance characteristics of primitive generation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.primitives = ShapePrimitives()

    def test_generation_speed_lod_0(self):
        """Test LOD 0 generation completes quickly."""
        import time

        start = time.perf_counter()
        for _ in range(10):
            self.primitives.generate_cube(size=1.0, lod_level=0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete 10 generations in under 50ms (target: <5ms each)
        assert elapsed_ms < 50, f"10 LOD 0 cubes took {elapsed_ms:.2f}ms, expected <50ms"

    def test_generation_speed_lod_2(self):
        """Test LOD 2 generation is faster than LOD 0."""
        import time

        # LOD 0
        start = time.perf_counter()
        for _ in range(10):
            self.primitives.generate_cube(size=1.0, lod_level=0)
        lod0_time = time.perf_counter() - start

        # LOD 2
        start = time.perf_counter()
        for _ in range(10):
            self.primitives.generate_cube(size=1.0, lod_level=2)
        lod2_time = time.perf_counter() - start

        # LOD 2 should be faster or equal
        assert lod2_time <= lod0_time * 1.5  # Allow 50% margin


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
