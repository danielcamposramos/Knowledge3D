"""
Kernel Splitter - Handles >48KB kernels by splitting into semantic domains

When a dependency kernel exceeds 48KB, it spills out of L2 cache and latency
jumps from <0.3ms to ~1.2ms. This module implements automatic splitting.

Strategy:
1. Cluster nodes by semantic domain (embedding k-means)
2. Split kernel into N domains (each <48KB)
3. Navigation switches domains when crossing boundaries
4. Total latency: <0.3ms intra-domain, +0.5ms domain switch

Kimi's insight: Most paths stay within one semantic domain (95%+),
so domain switches are rare in practice.

Author: Claude (K3D Core Team), based on Kimi K2's constraint analysis
Date: 2025-10-04
License: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    import cupy as cp
    from sklearn.cluster import KMeans
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    cp = None
    KMeans = None

from knowledge3d.spatial.led_pathfinder import DependencyKernel, KERNEL_SIZE_LIMIT_BYTES

_logger = logging.getLogger(__name__)


class KernelSplitter:
    """
    Splits large dependency kernels into semantic domains.

    Each domain fits in 48KB L2 cache for optimal performance.
    """

    def __init__(self, target_size_bytes: int = KERNEL_SIZE_LIMIT_BYTES):
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("CuPy and scikit-learn required for kernel splitting")

        self.target_size = target_size_bytes
        self.domains: List[DependencyKernel] = []
        self.domain_labels: Optional[np.ndarray] = None  # Node → domain ID
        self.domain_boundaries: List[Tuple[int, int]] = []  # Edges crossing domains

    def split_kernel(
        self,
        edges: np.ndarray,
        embeddings: np.ndarray,
        positions: np.ndarray,
        num_domains: Optional[int] = None
    ) -> List[DependencyKernel]:
        """
        Split edges into multiple domain kernels.

        Algorithm:
        1. Cluster embeddings into semantic domains (k-means)
        2. Assign edges to domains (both endpoints in same domain)
        3. Build separate kernel for each domain
        4. Track boundary edges (cross-domain connections)

        Args:
            edges: Source-dest pairs (E, 2)
            embeddings: Semantic embeddings (N, 256)
            positions: Geometric positions (N, 3)
            num_domains: Number of domains (auto-computed if None)

        Returns:
            List of domain kernels (each <48KB)
        """
        num_nodes = len(embeddings)
        num_edges = len(edges)

        # Auto-compute number of domains if not specified
        if num_domains is None:
            # Estimate kernel size for full graph
            estimated_size = self._estimate_full_kernel_size(num_nodes, num_edges)
            num_domains = max(2, int(np.ceil(estimated_size / self.target_size)))
            _logger.info(
                f"Auto-computed {num_domains} domains "
                f"(estimated size: {estimated_size/1024:.1f}KB)"
            )

        # Cluster nodes into semantic domains
        _logger.info(f"Clustering {num_nodes} nodes into {num_domains} semantic domains...")
        kmeans = KMeans(n_clusters=num_domains, random_state=1973, n_init=10)
        self.domain_labels = kmeans.fit_predict(embeddings)

        # Split edges by domain
        domain_edges = [[] for _ in range(num_domains)]
        boundary_edges = []

        for src, dst in edges:
            src_domain = self.domain_labels[src]
            dst_domain = self.domain_labels[dst]

            if src_domain == dst_domain:
                # Intra-domain edge
                domain_edges[src_domain].append([src, dst])
            else:
                # Boundary edge (cross-domain)
                boundary_edges.append([src, dst])

        _logger.info(
            f"Split edges: {sum(len(d) for d in domain_edges)} intra-domain, "
            f"{len(boundary_edges)} boundary"
        )

        # Build kernel for each domain
        self.domains = []
        for domain_id in range(num_domains):
            _logger.info(f"\nBuilding kernel for domain {domain_id}...")

            # Get nodes in this domain
            domain_nodes = np.where(self.domain_labels == domain_id)[0]
            domain_edge_array = np.array(domain_edges[domain_id], dtype=np.uint32)

            if len(domain_edge_array) == 0:
                _logger.warning(f"Domain {domain_id} has no edges, skipping")
                continue

            # Create local node mapping (global ID → local ID)
            global_to_local = {global_id: local_id for local_id, global_id in enumerate(domain_nodes)}

            # Remap edges to local IDs
            local_edges = np.array([
                [global_to_local.get(src, -1), global_to_local.get(dst, -1)]
                for src, dst in domain_edge_array
            ])
            # Filter out edges with missing nodes
            valid_mask = (local_edges[:, 0] >= 0) & (local_edges[:, 1] >= 0)
            local_edges = local_edges[valid_mask]

            # Build domain kernel
            domain_kernel = DependencyKernel(num_vertices=len(domain_nodes))
            domain_kernel.build_from_edges(
                local_edges,
                embeddings[domain_nodes],
                positions[domain_nodes],
                similarity_threshold=0.7,
                enable_semantic_highways=True
            )

            self.domains.append(domain_kernel)

            # Validate size
            kernel_size = domain_kernel._estimate_kernel_size(
                len(domain_nodes), domain_kernel.num_edges
            )
            _logger.info(
                f"  Domain {domain_id}: {len(domain_nodes)} nodes, "
                f"{domain_kernel.num_edges} edges, "
                f"{kernel_size/1024:.1f}KB"
            )

            if kernel_size > self.target_size:
                _logger.error(
                    f"  Domain {domain_id} still exceeds 48KB! "
                    f"Need more domains or higher threshold."
                )

        # Store boundary edges for cross-domain navigation
        self.domain_boundaries = boundary_edges

        _logger.info(
            f"\nKernel splitting complete: {len(self.domains)} domains, "
            f"{len(boundary_edges)} boundary edges"
        )

        return self.domains

    def _estimate_full_kernel_size(self, num_nodes: int, num_edges: int) -> int:
        """Estimate kernel size if built as single monolith."""
        # Assume 30% edges survive similarity filtering
        filtered_edges = int(num_edges * 0.3)

        row_offsets = (num_nodes + 1) * 4
        col_indices = filtered_edges * 4
        packed_costs = filtered_edges * 4
        lazy_bitmask = num_nodes * 8
        query_salt = 8 * 8

        return row_offsets + col_indices + packed_costs + lazy_bitmask + query_salt

    def navigate_cross_domain(
        self,
        start: int,
        goal: int,
        pathfinder_class
    ) -> Tuple[List[int], float]:
        """
        Navigate across domain boundaries.

        Strategy:
        1. Find start domain and goal domain
        2. If same domain: Use intra-domain pathfinder
        3. If different domains:
           a. Find boundary node in start domain (closest to goal domain)
           b. Navigate start → boundary (intra-domain)
           c. Find boundary node in goal domain (closest to start domain)
           d. Navigate boundary → goal (intra-domain)
           e. Combine paths

        Args:
            start: Start node (global ID)
            goal: Goal node (global ID)
            pathfinder_class: LEDPathfinder class to use

        Returns:
            (path, total_cost)
        """
        start_domain = self.domain_labels[start]
        goal_domain = self.domain_labels[goal]

        if start_domain == goal_domain:
            # Intra-domain navigation (fast path)
            domain_kernel = self.domains[start_domain]
            pathfinder = pathfinder_class()
            pathfinder.kernel = domain_kernel

            # Convert to local IDs
            domain_nodes = np.where(self.domain_labels == start_domain)[0]
            global_to_local = {g: l for l, g in enumerate(domain_nodes)}

            local_start = global_to_local[start]
            local_goal = global_to_local[goal]

            local_path, cost = pathfinder.find_path(local_start, local_goal)

            # Convert back to global IDs
            global_path = [domain_nodes[local_id] for local_id in local_path]
            return global_path, cost

        else:
            # Cross-domain navigation (requires boundary switch)
            _logger.info(
                f"Cross-domain navigation: domain {start_domain} → {goal_domain}"
            )

            # Find boundary nodes
            start_boundary = self._find_boundary_node(start, goal_domain)
            goal_boundary = self._find_boundary_node(goal, start_domain)

            # Navigate start → boundary in start domain
            path1, cost1 = self._navigate_intra_domain(
                start, start_boundary, start_domain, pathfinder_class
            )

            # Navigate boundary → goal in goal domain
            path2, cost2 = self._navigate_intra_domain(
                goal_boundary, goal, goal_domain, pathfinder_class
            )

            # Combine paths
            combined_path = path1 + path2[1:]  # Avoid duplicate boundary node
            total_cost = cost1 + cost2

            _logger.info(
                f"  Cross-domain path: {len(path1)} + {len(path2)} nodes, "
                f"cost: {cost1:.3f} + {cost2:.3f} = {total_cost:.3f}"
            )

            return combined_path, total_cost

    def _find_boundary_node(self, from_node: int, target_domain: int) -> int:
        """Find closest boundary node leading to target domain."""
        # Find all boundary edges from current domain to target domain
        current_domain = self.domain_labels[from_node]

        boundary_candidates = [
            src for src, dst in self.domain_boundaries
            if self.domain_labels[src] == current_domain and
               self.domain_labels[dst] == target_domain
        ]

        if not boundary_candidates:
            raise ValueError(
                f"No boundary between domain {current_domain} and {target_domain}"
            )

        # Return first candidate (TODO: optimize by distance)
        return boundary_candidates[0]

    def _navigate_intra_domain(
        self,
        start: int,
        goal: int,
        domain_id: int,
        pathfinder_class
    ) -> Tuple[List[int], float]:
        """Navigate within a single domain."""
        domain_kernel = self.domains[domain_id]
        pathfinder = pathfinder_class()
        pathfinder.kernel = domain_kernel

        # Get domain nodes and create mapping
        domain_nodes = np.where(self.domain_labels == domain_id)[0]
        global_to_local = {g: l for l, g in enumerate(domain_nodes)}

        local_start = global_to_local[start]
        local_goal = global_to_local[goal]

        local_path, cost = pathfinder.find_path(local_start, local_goal)

        # Convert to global IDs
        global_path = [domain_nodes[local_id] for local_id in local_path]

        return global_path, cost
