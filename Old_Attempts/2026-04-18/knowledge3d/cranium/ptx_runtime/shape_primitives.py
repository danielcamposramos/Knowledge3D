"""
Sovereign shape primitives with advanced multi-modal adaptation.
Implements GPU-native geometry generation with semantic understanding.
"""
import numpy as np
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as ModularRPNEngine
from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh

class ShapePrimitives:
    """
    Advanced GPU-accelerated primitive shape generation with multi-modal adaptation.
    Features semantic understanding, LOD-aware generation, and modal-specific optimizations.
    """
    
    def __init__(self):
        # Load shape generation kernel; tests may run without compiled PTX.
        try:
            self.shape_kernel = load_ptx_file(
                "knowledge3d/cranium/ptx/gre_shape_generator.ptx",
                "generate_adaptive_primitive"
            )
        except (FileNotFoundError, RuntimeError):
            self.shape_kernel = None

        try:
            self.rpn = ModularRPNEngine()
            self._rpn_available = True
        except RuntimeError:
            self.rpn = None
            self._rpn_available = False
        self.templates = self._init_enhanced_templates()
        
        # Semantic-to-geometry mapping
        self.semantic_geometry_map = {
            'architectural': ['cube', 'cylinder', 'prism'],
            'organic': ['sphere', 'blob', 'fractal'],
            'mechanical': ['gear', 'cylinder', 'cone'],
            'natural': ['sphere', 'fractal', 'organic_blob']
        }
        
    def _init_enhanced_templates(self):
        """Initialize enhanced primitive templates with semantic metadata."""
        templates = {
            "cube": {
                "vertices": np.array([
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                ], dtype=np.float32),
                "indices": np.array([
                    [0, 1, 2], [0, 2, 3],  # Front
                    [4, 5, 6], [4, 6, 7],  # Back
                    [0, 4, 7], [0, 7, 3],  # Left
                    [1, 5, 6], [1, 6, 2],  # Right
                    [3, 2, 6], [3, 6, 7],  # Top
                    [0, 1, 5], [0, 5, 4],  # Bottom
                ], dtype=np.uint32),
                "semantic_tags": ["geometric", "architectural", "stable"],
                "uv_seams": [(0, 1), (1, 2), (2, 3), (3, 0)]  # UV seam edges
            },
            "sphere": {
                "vertices": self._icosahedron_vertices(),
                "indices": self._icosahedron_indices(),
                "semantic_tags": ["organic", "natural", "smooth"],
                "uv_seams": []
            },
            "cylinder": {
                "vertices": self._cylinder_vertices(),
                "indices": self._cylinder_indices(),
                "semantic_tags": ["mechanical", "architectural", "symmetric"],
                "uv_seams": [(0, 2)]  # Vertical seam
            },
            "cone": {
                "vertices": self._cone_vertices(),
                "indices": self._cone_indices(),
                "semantic_tags": ["geometric", "pointed", "directional"],
                "uv_seams": [(0, 1)]
            },
            "torus": {
                "vertices": self._torus_vertices(),
                "indices": self._torus_indices(),
                "semantic_tags": ["mechanical", "organic", "looped"],
                "uv_seams": []
            }
        }
        
        # Add LOD variants for each template
        for shape_name, template in templates.items():
            template["lod_variants"] = self._generate_lod_variants(template)
            
        return templates
    
    def _icosahedron_vertices(self):
        """Generate icosahedron vertices for sphere approximation."""
        t = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=np.float32)
        return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    def _icosahedron_indices(self):
        """Generate icosahedron indices."""
        return np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.uint32)
    
    def _cylinder_vertices(self, segments=16):
        """Generate cylinder vertices with enhanced topology."""
        vertices = []
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            vertices.extend([
                [np.cos(theta), np.sin(theta), -1],  # Bottom circle
                [np.cos(theta), np.sin(theta), 1]    # Top circle
            ])
        vertices.append([0, 0, -1])  # Bottom center
        vertices.append([0, 0, 1])   # Top center
        return np.array(vertices, dtype=np.float32)
    
    def _cylinder_indices(self, segments=16):
        """Generate cylinder indices with proper topology."""
        indices = []
        for i in range(segments):
            next_i = (i + 1) % segments
            # Side faces
            indices.extend([
                [i * 2, next_i * 2, next_i * 2 + 1],
                [i * 2, next_i * 2 + 1, i * 2 + 1]
            ])
            # Bottom cap
            indices.append([i * 2, next_i * 2, segments * 2])
            # Top cap
            indices.append([i * 2 + 1, next_i * 2 + 1, segments * 2 + 1])
        return np.array(indices, dtype=np.uint32)
    
    def _cone_vertices(self, segments=16):
        """Generate cone vertices with enhanced topology."""
        vertices = [[0, 0, 1]]  # Apex
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            vertices.append([np.cos(theta), np.sin(theta), -1])  # Base circle
        vertices.append([0, 0, -1])  # Base center
        return np.array(vertices, dtype=np.float32)
    
    def _cone_indices(self, segments=16):
        """Generate cone indices with proper topology."""
        indices = []
        for i in range(segments):
            next_i = (i + 1) % segments
            # Side faces
            indices.append([0, i + 1, next_i + 1])
            # Base cap
            indices.append([i + 1, next_i + 1, segments + 1])
        return np.array(indices, dtype=np.uint32)
    
    def _torus_vertices(self, major_segments=16, minor_segments=8):
        """Generate torus vertices."""
        vertices = []
        for i in range(major_segments):
            theta = 2.0 * np.pi * i / major_segments
            for j in range(minor_segments):
                phi = 2.0 * np.pi * j / minor_segments
                x = (2 + np.cos(phi)) * np.cos(theta)
                y = (2 + np.cos(phi)) * np.sin(theta)
                z = np.sin(phi)
                vertices.append([x, y, z])
        return np.array(vertices, dtype=np.float32)
    
    def _torus_indices(self, major_segments=16, minor_segments=8):
        """Generate torus indices."""
        indices = []
        for i in range(major_segments):
            next_i = (i + 1) % major_segments
            for j in range(minor_segments):
                next_j = (j + 1) % minor_segments
                current = i * minor_segments + j
                next_major = next_i * minor_segments + j
                next_both = next_i * minor_segments + next_j
                next_minor = i * minor_segments + next_j
                
                indices.extend([
                    [current, next_major, next_both],
                    [current, next_both, next_minor]
                ])
        return np.array(indices, dtype=np.uint32)
    
    def _generate_lod_variants(self, template):
        """Generate LOD variants for a template."""
        base_vertices = template["vertices"]
        base_indices = template["indices"]
        
        lod_variants = {}
        
        # LOD 0: Original (highest quality)
        lod_variants[0] = {
            "vertices": base_vertices,
            "indices": base_indices,
            "vertex_count": len(base_vertices),
            "triangle_count": len(base_indices)
        }
        
        # LOD 1: Medium quality (50% vertices)
        medium_vertices = self._simplify_mesh(base_vertices, base_indices, 0.5)
        lod_variants[1] = {
            "vertices": medium_vertices["vertices"],
            "indices": medium_vertices["indices"],
            "vertex_count": len(medium_vertices["vertices"]),
            "triangle_count": len(medium_vertices["indices"])
        }
        
        # LOD 2: Low quality (25% vertices)
        low_vertices = self._simplify_mesh(base_vertices, base_indices, 0.25)
        lod_variants[2] = {
            "vertices": low_vertices["vertices"],
            "indices": low_vertices["indices"],
            "vertex_count": len(low_vertices["vertices"]),
            "triangle_count": len(low_vertices["indices"])
        }
        
        return lod_variants
    
    def _simplify_mesh(self, vertices, indices, reduction_factor):
        """Simplify mesh using edge collapse algorithm."""
        # Simplified implementation - in production would use quadric error metrics
        target_vertices = int(len(vertices) * reduction_factor)
        
        if target_vertices < 4:  # Minimum vertices for a tetrahedron
            target_vertices = 4
            
        # For now, use uniform sampling
        step = max(1, len(vertices) // target_vertices)
        simplified_vertices = vertices[::step]
        
        # Regenerate indices for simplified vertices
        simplified_indices = []
        for i in range(0, len(simplified_vertices) - 2, 3):
            simplified_indices.append([i, i + 1, i + 2])
            
        return {
            "vertices": simplified_vertices,
            "indices": np.array(simplified_indices, dtype=np.uint32)
        }
    
    def generate_cube(self, size=1.0, lod_level=0):
        """Generate cube with RPN scaling and LOD support."""
        template = self.templates["cube"]
        lod_variant = template["lod_variants"][lod_level]
        
        if self._rpn_available:
            opcodes = np.array([0x03], dtype=np.uint16)  # MUL
            scalars = np.array([size / 2.0], dtype=np.float32)
            scaled_vertices = self.rpn.execute_batch(opcodes, scalars, lod_variant["vertices"])
        else:
            scaled_vertices = lod_variant["vertices"].copy() * (size / 2.0)
        
        return scaled_vertices, lod_variant["indices"]
    
    def generate_sphere(self, radius=1.0, subdivisions=2, lod_level=0):
        """Generate sphere via icosahedron subdivision with LOD support."""
        template = self.templates["sphere"]
        lod_variant = template["lod_variants"][lod_level]
        
        # For higher LOD levels, apply additional subdivisions
        vertices, indices = lod_variant["vertices"], lod_variant["indices"]
        for _ in range(subdivisions):
            vertices, indices = self._subdivide_mesh(vertices, indices)
            
        # Normalize to sphere radius
        vertices = self._normalize_to_sphere(vertices, radius)
        
        return vertices, indices
    
    def generate_cylinder(self, radius=1.0, height=2.0, segments=16, lod_level=0):
        """Generate cylinder with RPN scaling and LOD support."""
        template = self.templates["cylinder"]
        lod_variant = template["lod_variants"][lod_level]
        
        # Scale using RPN for both radius and height
        opcodes = np.array([0x03, 0x03], dtype=np.uint16)  # MUL, MUL
        scalars = np.array([radius, height / 2.0], dtype=np.float32)
        
        # Apply scaling to x,y for radius and z for height
        scaled_vertices = lod_variant["vertices"].copy()
        scaled_vertices[:, 0] *= radius  # X component
        scaled_vertices[:, 1] *= radius  # Y component
        scaled_vertices[:, 2] *= height / 2.0  # Z component
        
        return scaled_vertices, lod_variant["indices"]
    
    def generate_cone(self, radius=1.0, height=2.0, segments=16, lod_level=0):
        """Generate cone with RPN scaling and LOD support."""
        template = self.templates["cone"]
        lod_variant = template["lod_variants"][lod_level]
        
        # Scale using RPN
        scaled_vertices = lod_variant["vertices"].copy()
        scaled_vertices[:, 0] *= radius  # X component
        scaled_vertices[:, 1] *= radius  # Y component
        scaled_vertices[:, 2] *= height / 2.0  # Z component
        
        return scaled_vertices, lod_variant["indices"]
    
    def generate_torus(self, major_radius=2.0, minor_radius=0.5, lod_level=0):
        """Generate torus with RPN scaling and LOD support."""
        template = self.templates["torus"]
        lod_variant = template["lod_variants"][lod_level]
        
        # Scale using RPN
        scaled_vertices = lod_variant["vertices"].copy()
        # Scale major radius (x,y) and minor radius (all components)
        scaled_vertices[:, 0] *= major_radius
        scaled_vertices[:, 1] *= major_radius
        scaled_vertices[:, 2] *= minor_radius
        
        return scaled_vertices, lod_variant["indices"]
    
    def _subdivide_mesh(self, vertices, indices):
        """Subdivide mesh for smoother sphere."""
        edge_midpoints = {}
        new_vertices = list(vertices)
        new_indices = []

        def get_midpoint(v1, v2):
            key = tuple(sorted([v1, v2]))
            if key not in edge_midpoints:
                mid = (vertices[v1] + vertices[v2]) / 2
                edge_midpoints[key] = len(new_vertices)
                new_vertices.append(mid)
            return edge_midpoints[key]

        for face in indices:
            v0, v1, v2 = face
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            new_indices.extend([
                [v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]
            ])
        return np.array(new_vertices, dtype=np.float32), np.array(new_indices, dtype=np.uint32)
    
    def _normalize_to_sphere(self, vertices, radius):
        """Normalize vertices to sphere surface using RPN."""
        mags = np.linalg.norm(vertices, axis=1, keepdims=True)
        if self._rpn_available:
            opcodes = np.array([0x04, 0x03], dtype=np.uint16)  # DIV, MUL
            scalars = np.concatenate([mags.flatten(), np.full(len(vertices), radius, dtype=np.float32)])
            return self.rpn.execute_batch(opcodes, scalars, vertices)
        mags[mags == 0] = 1.0
        normalized = vertices / mags
        normalized *= radius
        zero_mask = np.linalg.norm(normalized, axis=1) == 0
        if np.any(zero_mask):
            normalized[zero_mask] = np.array([radius, 0.0, 0.0], dtype=np.float32)
        return normalized
    
    def adapt_primitive_from_modal(self, base_verts, modal_features, semantic_context=None):
        """
        Adapt primitive vertices from multi-modal features with semantic understanding.
        
        Args:
            base_verts: Base primitive vertices
            modal_features: Multi-modal feature vector
            semantic_context: Optional semantic context for adaptation
            
        Returns:
            Adapted vertices with modal influence
        """
        if len(modal_features) == 0:
            return base_verts
            
        # Extract semantic context if provided
        if semantic_context:
            semantic_category = semantic_context.get('category', 'generic')
            adaptation_strength = semantic_context.get('strength', 0.5)
        else:
            semantic_category = 'generic'
            adaptation_strength = 0.5
            
        # Apply semantic-specific adaptations
        adapted_verts = base_verts.copy()
        
        if semantic_category == 'organic':
            # Apply organic deformation
            adapted_verts = self._apply_organic_deformation(adapted_verts, modal_features, adaptation_strength)
        elif semantic_category == 'mechanical':
            # Apply mechanical precision
            adapted_verts = self._apply_mechanical_precision(adapted_verts, modal_features, adaptation_strength)
        elif semantic_category == 'architectural':
            # Apply architectural constraints
            adapted_verts = self._apply_architectural_constraints(adapted_verts, modal_features, adaptation_strength)
        else:
            # Generic adaptation
            scales = modal_features[:3] if len(modal_features) >= 3 else np.ones(3)
            scalars = scales.astype(np.float32) * adaptation_strength + (1 - adaptation_strength)
            if self._rpn_available:
                opcodes = np.array([0x03, 0x03, 0x03], dtype=np.uint16)  # MUL x3
                adapted_verts = self.rpn.execute_batch(opcodes, scalars, adapted_verts)
            else:
                adapted_verts = adapted_verts * scalars
            
        return adapted_verts
    
    def _apply_organic_deformation(self, vertices, features, strength):
        """Apply organic deformation to vertices."""
        # Use features to drive organic deformation
        deform_params = features[:6] if len(features) >= 6 else np.zeros(6)
        
        # Apply sine-based deformation for organic look
        for i in range(len(vertices)):
            x, y, z = vertices[i]
            
            # Deformation based on feature parameters
            vertices[i, 0] += np.sin(y * deform_params[0]) * deform_params[3] * strength
            vertices[i, 1] += np.cos(x * deform_params[1]) * deform_params[4] * strength
            vertices[i, 2] += np.sin(z * deform_params[2]) * deform_params[5] * strength
            
        return vertices
    
    def _apply_mechanical_precision(self, vertices, features, strength):
        """Apply mechanical precision to vertices."""
        # Quantize vertices for mechanical precision
        precision = 0.1 * (1 - strength) + 0.01 * strength  # Adjust precision based on strength
        
        # Extract precision factor from features
        if len(features) > 0:
            precision *= (1 - features[0] * 0.5)  # Feature influences precision
            
        # Quantize vertices
        vertices = np.round(vertices / precision) * precision
        
        return vertices
    
    def _apply_architectural_constraints(self, vertices, features, strength):
        """Apply architectural constraints to vertices."""
        # Enforce right angles and planar faces for architectural look
        
        # Extract constraint parameters from features
        if len(features) >= 3:
            angle_constraint = features[0] * strength
            planar_constraint = features[1] * strength
            scale_constraint = features[2] * strength
        else:
            angle_constraint = planar_constraint = scale_constraint = strength
            
        # Apply constraints
        # Simplified implementation - in production would use more sophisticated algorithms
        vertices = self._enforce_planar_faces(vertices, planar_constraint)
        vertices = self._enforce_right_angles(vertices, angle_constraint)
        
        return vertices
    
    def _enforce_planar_faces(self, vertices, strength):
        """Enforce planar faces for architectural look."""
        # Simplified implementation - project vertices onto dominant planes
        # In production would use PCA to find dominant planes
        
        # For now, just flatten z-component slightly
        vertices[:, 2] *= (1 - strength * 0.3)
        
        return vertices
    
    def _enforce_right_angles(self, vertices, strength):
        """Enforce right angles for architectural look."""
        # Simplified implementation - quantize angles to 90-degree multiples
        
        # For now, just snap coordinates to grid
        grid_size = 0.5 * (1 - strength) + 0.1 * strength
        vertices = np.round(vertices / grid_size) * grid_size
        
        return vertices
    
    def get_semantic_suggestions(self, embedding):
        """
        Get semantic shape suggestions based on embedding.
        
        Args:
            embedding: Semantic embedding vector
            
        Returns:
            List of suggested shape types with confidence scores
        """
        # Simplified semantic analysis - in production would use trained model
        suggestions = []
        
        # Analyze embedding for semantic patterns
        if len(embedding) >= 10:
            # Check for geometric patterns
            geometric_score = np.mean(embedding[:3])
            if geometric_score > 0.5:
                suggestions.append(("cube", geometric_score))
                suggestions.append(("cylinder", geometric_score * 0.8))
                
            # Check for organic patterns
            organic_score = np.mean(embedding[3:6])
            if organic_score > 0.5:
                suggestions.append(("sphere", organic_score))
                suggestions.append(("torus", organic_score * 0.7))
                
            # Check for mechanical patterns
            mechanical_score = np.mean(embedding[6:9])
            if mechanical_score > 0.5:
                suggestions.append(("cone", mechanical_score))
                suggestions.append(("torus", mechanical_score * 0.9))
                
        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions[:3]  # Return top 3 suggestions
