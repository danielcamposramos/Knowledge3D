# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
"""
Knowledge Garden Fractal Growth (φ-Constrained Ontology Trees)

Part of Week 3-4 implementation from Step7.1_FINAL.txt
Swarm collaboration:
- Grok: Space colonization algorithm
- Codex: Python implementation
- Kimi: PTX kernel conversion (future)
- GLM: Golden ratio (φ) proofs
- Qwen: Quadrant layout integration
- Claude: φ-validation tests

Key Features:
- Fractal tree growth using space colonization algorithm
- Golden ratio (φ ≈ 1.618) constraints on ALL parameters:
  - Branch angle: θ = 2π/φ ≈ 137.5° (golden angle spiral)
  - Recursion depth: d = int(φ × honesty × 10)
  - Branch thickness: t = base / φ^depth
  - Branching density: φ ratio between levels
- RPN-powered φ calculations for GPU acceleration
- Circular quadrant layout (North/East/South/West domains)
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
import cupy as cp

# Golden ratio constant
PHI = 1.618033988749895


class GardenFractalGrower:
    """
    Grows fractal ontology trees in Knowledge Garden using φ constraints.

    Uses RPN kernel for:
    - Golden ratio calculations (angle, depth, thickness)
    - Branch constraint validation
    - Fractal parameter optimization
    """

    def __init__(
        self,
        house_path: str,
        base_thickness: float = 1.0,
        attraction_distance: float = 10.0,
        kill_distance: float = 2.0,
        segment_length: float = 1.0,
        max_iterations: int = 1000
    ):
        """
        Initialize fractal grower with space colonization parameters.

        Args:
            house_path: Path to house.glb
            base_thickness: Base trunk thickness
            attraction_distance: Attraction radius for influence points
            kill_distance: Distance at which influence points are "consumed"
            segment_length: Length of each branch segment
            max_iterations: Maximum growth iterations
        """
        self.house_path = Path(house_path)
        self.base_thickness = base_thickness
        self.attraction_distance = attraction_distance
        self.kill_distance = kill_distance
        self.segment_length = segment_length
        self.max_iterations = max_iterations

        # RPN executor for φ calculations
        try:
            from knowledge3d.tools.garden_fractal_rpn import (
                compute_golden_angle_rpn,
                compute_max_depth_rpn,
                compute_thickness_rpn,
                compute_branching_density_rpn,
                compute_fractal_constraints_batch_rpn
            )
            self.compute_golden_angle = compute_golden_angle_rpn
            self.compute_max_depth = compute_max_depth_rpn
            self.compute_thickness = compute_thickness_rpn
            self.compute_branching_density = compute_branching_density_rpn
            self.compute_batch_constraints = compute_fractal_constraints_batch_rpn
            self._use_rpn = True
        except Exception as e:
            print(f"⚠️  RPN not available for φ calculations, using CPU fallback: {e}")
            self._use_rpn = False

        # Quadrant definitions (from Step7.1_FINAL.txt)
        self.quadrants = {
            'North': {'name': 'Mathematics/Physics', 'center': np.array([0.0, 10.0, 0.0])},
            'East': {'name': 'AI/CS', 'center': np.array([10.0, 0.0, 0.0])},
            'South': {'name': 'Humanities', 'center': np.array([0.0, -10.0, 0.0])},
            'West': {'name': 'Languages', 'center': np.array([-10.0, 0.0, 0.0])}
        }

    class TreeNode:
        """Represents a node in the fractal tree."""

        def __init__(
            self,
            position: np.ndarray,
            parent: Optional['TreeNode'] = None,
            depth: int = 0
        ):
            self.position = position
            self.parent = parent
            self.children = []
            self.depth = depth
            self.thickness = 1.0
            self.is_tip = True

    def assign_cluster_to_quadrant(
        self,
        cluster_centroid: np.ndarray,
        cluster_metadata: Dict[str, Any]
    ) -> str:
        """
        Assign cluster to appropriate quadrant based on semantic domain.

        Args:
            cluster_centroid: Cluster centroid embedding
            cluster_metadata: Cluster metadata (may contain domain hints)

        Returns:
            Quadrant name ('North', 'East', 'South', 'West')
        """
        # For now, use simple distance-based assignment
        # Future: Use semantic classification

        min_dist = float('inf')
        assigned_quadrant = 'North'  # Default

        for quadrant_name, quadrant_data in self.quadrants.items():
            center = quadrant_data['center']
            # Use first 3 dimensions of embedding as 3D position proxy
            dist = np.linalg.norm(cluster_centroid[:3] - center)

            if dist < min_dist:
                min_dist = dist
                assigned_quadrant = quadrant_name

        return assigned_quadrant

    def generate_influence_points(
        self,
        cluster_embeddings: np.ndarray,
        quadrant_center: np.ndarray,
        num_points: int = 100
    ) -> np.ndarray:
        """
        Generate influence points for space colonization from cluster embeddings.

        Args:
            cluster_embeddings: Cluster member embeddings (N, D)
            quadrant_center: Quadrant center position
            num_points: Number of influence points to generate

        Returns:
            Influence points (num_points, 3)
        """
        # Use PCA to reduce embeddings to 3D
        from sklearn.decomposition import PCA

        if len(cluster_embeddings) < num_points:
            # Sample with replacement if needed
            indices = np.random.choice(len(cluster_embeddings), num_points, replace=True)
            sampled = cluster_embeddings[indices]
        else:
            # Sample without replacement
            indices = np.random.choice(len(cluster_embeddings), num_points, replace=False)
            sampled = cluster_embeddings[indices]

        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(sampled)

        # Normalize and scale around quadrant center
        points_3d = (points_3d - points_3d.mean(axis=0))  # Center at origin
        points_3d = points_3d / (np.linalg.norm(points_3d, axis=1, keepdims=True) + 1e-8)  # Normalize
        points_3d = points_3d * 5.0  # Scale to reasonable size
        points_3d = points_3d + quadrant_center  # Offset to quadrant

        return points_3d

    def grow_tree_space_colonization(
        self,
        root_position: np.ndarray,
        influence_points: np.ndarray,
        honesty_score: float
    ) -> List[TreeNode]:
        """
        Grow fractal tree using space colonization algorithm with φ constraints.

        Args:
            root_position: Tree root position (3D)
            influence_points: Target influence points (N, 3)
            honesty_score: Honesty score for depth calculation

        Returns:
            List of tree nodes
        """
        # Compute max depth using RPN (φ constraint)
        if self._use_rpn:
            max_depth = self.compute_max_depth(honesty_score)
        else:
            max_depth = int(PHI * honesty_score * 10)

        # Initialize tree with root
        root = self.TreeNode(position=root_position, parent=None, depth=0)
        nodes = [root]
        active_influence_points = influence_points.copy()

        # Compute golden angle using RPN
        if self._use_rpn:
            golden_angle = self.compute_golden_angle()
        else:
            golden_angle = 2 * np.pi / PHI  # ≈ 2.4 radians ≈ 137.5°

        # Space colonization iterations
        for iteration in range(self.max_iterations):
            if len(active_influence_points) == 0:
                break

            # For each influence point, find closest tree node
            influence_to_node = {}  # influence_idx → closest_node

            for inf_idx, inf_point in enumerate(active_influence_points):
                min_dist = float('inf')
                closest_node = None

                for node in nodes:
                    if not node.is_tip:
                        continue

                    dist = np.linalg.norm(inf_point - node.position)

                    if dist < self.attraction_distance and dist < min_dist:
                        min_dist = dist
                        closest_node = node

                if closest_node is not None:
                    influence_to_node[inf_idx] = (closest_node, min_dist)

            # Remove consumed influence points
            consumed_indices = []
            for inf_idx, (node, dist) in influence_to_node.items():
                if dist < self.kill_distance:
                    consumed_indices.append(inf_idx)

            active_influence_points = np.delete(
                active_influence_points,
                consumed_indices,
                axis=0
            )

            # Grow new branches toward influence points
            growth_vectors = {}  # node → average_direction

            for inf_idx, (node, dist) in influence_to_node.items():
                if inf_idx in consumed_indices:
                    continue

                inf_point = active_influence_points[inf_idx] if inf_idx < len(active_influence_points) else None
                if inf_point is None:
                    continue

                direction = inf_point - node.position
                direction = direction / (np.linalg.norm(direction) + 1e-8)

                if node not in growth_vectors:
                    growth_vectors[node] = []

                growth_vectors[node].append(direction)

            # Create new nodes
            new_nodes = []
            for node, directions in growth_vectors.items():
                # Average all attraction directions
                avg_direction = np.mean(directions, axis=0)
                avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-8)

                # Apply golden angle spiral rotation (φ constraint)
                # Rotate around Y-axis by golden angle
                angle = golden_angle * (node.depth % 10)  # Spiral pattern
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
                rotated_direction = rotation_matrix @ avg_direction

                # New position
                new_position = node.position + rotated_direction * self.segment_length

                # Create new node
                new_depth = node.depth + 1

                # Stop if exceeding max depth (φ constraint)
                if new_depth > max_depth:
                    continue

                new_node = self.TreeNode(
                    position=new_position,
                    parent=node,
                    depth=new_depth
                )

                # Compute thickness using RPN (φ constraint: t = base / φ^depth)
                if self._use_rpn:
                    new_node.thickness = self.compute_thickness(self.base_thickness, new_depth)
                else:
                    new_node.thickness = self.base_thickness / (PHI ** new_depth)

                node.children.append(new_node)
                node.is_tip = False
                new_nodes.append(new_node)

            nodes.extend(new_nodes)

        return nodes

    def tree_to_glb_representation(
        self,
        nodes: List[TreeNode],
        tree_id: str
    ) -> Dict[str, Any]:
        """
        Convert tree nodes to GLB-compatible representation.

        Args:
            nodes: Tree nodes
            tree_id: Unique tree identifier

        Returns:
            Tree representation for GLB extras
        """
        # Convert nodes to simple dict representation
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                'position': node.position.tolist(),
                'depth': node.depth,
                'thickness': float(node.thickness),
                'parent_id': nodes.index(node.parent) if node.parent else -1,
                'is_tip': node.is_tip
            })

        return {
            'type': 'fractal_tree',
            'tree_id': tree_id,
            'num_nodes': len(nodes),
            'nodes': nodes_data,
            'golden_ratio': PHI,
            'computation_method': 'space_colonization_rpn' if self._use_rpn else 'space_colonization_cpu'
        }

    def grow_garden_from_clusters(
        self,
        clusters: List[Dict[str, Any]],
        cluster_embeddings: List[np.ndarray],
        cluster_qualities: List[float]
    ) -> Dict[str, Any]:
        """
        Grow entire garden from Galaxy clusters.

        Args:
            clusters: List of cluster metadata dicts
            cluster_embeddings: List of embedding arrays (one per cluster)
            cluster_qualities: List of quality scores (honesty)

        Returns:
            Garden representation with all trees
        """
        start_time = time.time()

        garden_data = {
            'quadrants': {},
            'trees': [],
            'total_nodes': 0,
            'total_trees': 0
        }

        print("🌳 Growing Knowledge Garden fractals...")

        for cluster_idx, (cluster_meta, embeddings, quality) in enumerate(
            zip(clusters, cluster_embeddings, cluster_qualities)
        ):
            # Assign to quadrant
            centroid = np.mean(embeddings, axis=0)
            quadrant_name = self.assign_cluster_to_quadrant(centroid, cluster_meta)
            quadrant_center = self.quadrants[quadrant_name]['center']

            # Generate influence points from embeddings
            influence_points = self.generate_influence_points(
                embeddings,
                quadrant_center,
                num_points=min(100, len(embeddings) * 5)
            )

            # Grow tree with φ constraints
            tree_root = quadrant_center.copy()
            tree_nodes = self.grow_tree_space_colonization(
                root_position=tree_root,
                influence_points=influence_points,
                honesty_score=quality
            )

            # Convert to GLB representation
            tree_id = f"tree_{quadrant_name}_{cluster_idx}"
            tree_data = self.tree_to_glb_representation(tree_nodes, tree_id)
            tree_data['quadrant'] = quadrant_name
            tree_data['cluster_id'] = cluster_idx
            tree_data['honesty_score'] = float(quality)

            garden_data['trees'].append(tree_data)
            garden_data['total_nodes'] += len(tree_nodes)

            # Add to quadrant
            if quadrant_name not in garden_data['quadrants']:
                garden_data['quadrants'][quadrant_name] = {
                    'name': self.quadrants[quadrant_name]['name'],
                    'center': self.quadrants[quadrant_name]['center'].tolist(),
                    'trees': []
                }

            garden_data['quadrants'][quadrant_name]['trees'].append(tree_id)

            print(f"   → {quadrant_name}: {tree_id} ({len(tree_nodes)} nodes, quality={quality:.2f})")

        garden_data['total_trees'] = len(garden_data['trees'])

        elapsed = time.time() - start_time

        print(f"\n✅ Garden growth complete!")
        print(f"   → {garden_data['total_trees']} trees")
        print(f"   → {garden_data['total_nodes']} total nodes")
        print(f"   → {elapsed:.2f}s elapsed")

        return garden_data


def grow_fractal_trees(
    clusters: List[Dict[str, Any]],
    cluster_embeddings: List[np.ndarray],
    cluster_qualities: List[float],
    house_path: str = "viewer/public/house/house_memory.glb",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to grow fractal trees from clusters.

    Args:
        clusters: Cluster metadata
        cluster_embeddings: Cluster embeddings
        cluster_qualities: Cluster honesty scores
        house_path: Path to House GLB
        **kwargs: Additional arguments for GardenFractalGrower

    Returns:
        Garden data
    """
    grower = GardenFractalGrower(house_path, **kwargs)
    return grower.grow_garden_from_clusters(
        clusters,
        cluster_embeddings,
        cluster_qualities
    )
