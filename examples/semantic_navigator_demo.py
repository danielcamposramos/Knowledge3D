#!/usr/bin/env python3
"""
Semantic Navigator Demo - Integration Example

Demonstrates the complete Morton Octree + LED-A* pipeline for semantic spatial navigation.

This example shows:
1. Building Morton octree from House positions
2. Extracting dependency kernel during sleep-time
3. Runtime semantic navigation (<0.3ms)
4. Label-to-label navigation workflow

Usage:
    python examples/semantic_navigator_demo.py --house viewer/public/my_house.glb
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cupy as cp
import numpy as np
from pygltflib import GLTF2

from knowledge3d.spatial.morton_octree import MortonOctree
from knowledge3d.spatial.led_pathfinder import LEDPathfinder


class SemanticNavigator:
    """
    Unified spatial + semantic navigation system.

    Combines Morton octree (spatial indexing) with LED-A* (semantic pathfinding)
    for <1ms label-to-label navigation.
    """

    def __init__(self):
        self.octree = MortonOctree()
        self.pathfinder = LEDPathfinder()

        self.positions_gpu: Optional[cp.ndarray] = None
        self.embeddings_gpu: Optional[cp.ndarray] = None
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}

    def load_house_from_glb(self, glb_path: str) -> None:
        """
        Load House knowledge from embedded glTF.

        Extracts:
        - Node positions (spatial data)
        - Embeddings (semantic data)
        - Labels/metadata (human-readable IDs)
        """
        gltf = GLTF2().load(glb_path)

        mesh = gltf.meshes[0]
        primitive = mesh.primitives[0]

        position_accessor = gltf.accessors[primitive.attributes['POSITION']]
        positions = self._read_accessor(gltf, position_accessor)
        self.positions_gpu = cp.array(positions, dtype=cp.float32)

        k3d_data = primitive.extras.get('k3d', {})

        if 'embeddingsView' in k3d_data:
            embedding_view_idx = k3d_data['embeddingsView']
            embedding_dims = k3d_data['embeddingDims']
            embeddings = self._read_buffer_view(gltf, embedding_view_idx,
                                                shape=(len(positions), embedding_dims))
            self.embeddings_gpu = cp.array(embeddings, dtype=cp.float32)
        else:
            print("Warning: No embeddings found, using positions as proxy")
            self.embeddings_gpu = self.positions_gpu.copy()

        self.labels = k3d_data.get('labels', [f"node_{i}" for i in range(len(positions))])
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        print(f"Loaded {len(self.labels)} nodes from {glb_path}")

    def _read_accessor(self, gltf, accessor) -> np.ndarray:
        """Read data from glTF accessor."""
        buffer_view = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]

        if buffer.uri.startswith('data:'):
            import base64
            data_str = buffer.uri.split(',', 1)[1]
            data_bytes = base64.b64decode(data_str)
        else:
            with open(buffer.uri, 'rb') as f:
                data_bytes = f.read()

        offset = buffer_view.byteOffset + accessor.byteOffset
        count = accessor.count

        if accessor.type == 'VEC3':
            shape = (count, 3)
        elif accessor.type == 'SCALAR':
            shape = (count,)
        else:
            raise ValueError(f"Unsupported accessor type: {accessor.type}")

        dtype = np.float32 if accessor.componentType == 5126 else np.uint32
        return np.frombuffer(data_bytes, dtype=dtype, count=np.prod(shape), offset=offset).reshape(shape)

    def _read_buffer_view(self, gltf, view_idx, shape) -> np.ndarray:
        """Read raw buffer view data."""
        buffer_view = gltf.bufferViews[view_idx]
        buffer = gltf.buffers[buffer_view.buffer]

        if buffer.uri.startswith('data:'):
            import base64
            data_str = buffer.uri.split(',', 1)[1]
            data_bytes = base64.b64decode(data_str)
        else:
            with open(buffer.uri, 'rb') as f:
                data_bytes = f.read()

        offset = buffer_view.byteOffset
        return np.frombuffer(data_bytes, dtype=np.float32, count=np.prod(shape), offset=offset).reshape(shape)

    def build_octree(self) -> None:
        """Build Morton octree from loaded positions (sleep-time operation)."""
        print("Building Morton octree...")
        t0 = time.perf_counter()
        self.octree.build_from_gpu_positions(self.positions_gpu)
        t1 = time.perf_counter()
        print(f"  Octree built in {(t1-t0)*1000:.2f}ms")

    def extract_dependency_kernel(self, k_neighbors: int = 8, similarity_threshold: float = 0.7) -> None:
        """
        Extract dependency kernel for LED-A* (sleep-time operation).

        Args:
            k_neighbors: Number of neighbors per node for kernel
            similarity_threshold: Minimum semantic similarity to include edge
        """
        print(f"Extracting dependency kernel (k={k_neighbors}, threshold={similarity_threshold})...")
        t0 = time.perf_counter()

        edges_list = []
        num_nodes = len(self.labels)

        for i in range(num_nodes):
            center = self.positions_gpu[i].get()
            radius = 2.0
            neighbor_ids = self.octree.query_radius_gpu(center, radius, refine_euclidean=True).get()

            neighbor_ids = neighbor_ids[neighbor_ids != i][:k_neighbors]

            for j in neighbor_ids:
                edges_list.append([i, j])

        edges = np.array(edges_list, dtype=np.uint32)

        embeddings_cpu = self.embeddings_gpu.get()
        positions_cpu = self.positions_gpu.get()

        self.pathfinder.build_kernel_from_octree(edges, embeddings_cpu, positions_cpu,
                                                 threshold=similarity_threshold)

        t1 = time.perf_counter()
        print(f"  Kernel extracted in {(t1-t0)*1000:.2f}ms ({len(edges)} edges → {self.pathfinder.kernel.nnz} kernel edges)")

    def navigate(self, start_label: str, goal_label: str,
                 alpha: float = 0.7, beta: float = 0.3) -> Tuple[List[str], float]:
        """
        Navigate from start label to goal label (runtime operation, <1ms target).

        Args:
            start_label: Starting node label
            goal_label: Goal node label
            alpha: Semantic cost weight
            beta: Geometric cost weight

        Returns:
            (path_labels, total_cost)
        """
        if start_label not in self.label_to_idx:
            raise ValueError(f"Start label '{start_label}' not found")
        if goal_label not in self.label_to_idx:
            raise ValueError(f"Goal label '{goal_label}' not found")

        start_idx = self.label_to_idx[start_label]
        goal_idx = self.label_to_idx[goal_label]

        t0 = time.perf_counter()
        path_indices, cost = self.pathfinder.find_path(start_idx, goal_idx, alpha, beta)
        t1 = time.perf_counter()

        path_labels = [self.labels[i] for i in path_indices]

        print(f"  Navigation: {start_label} → {goal_label}")
        print(f"  Path: {' → '.join(path_labels)}")
        print(f"  Cost: {cost:.3f}, Time: {(t1-t0)*1000:.3f}ms")

        return path_labels, cost

    def query_spatial_neighbors(self, label: str, radius: float = 2.0) -> List[str]:
        """
        Query spatial neighbors using Morton octree (runtime operation, <10ms target).

        Args:
            label: Node label to query around
            radius: Search radius

        Returns:
            List of neighbor labels
        """
        if label not in self.label_to_idx:
            raise ValueError(f"Label '{label}' not found")

        idx = self.label_to_idx[label]
        center = self.positions_gpu[idx].get()

        t0 = time.perf_counter()
        neighbor_ids = self.octree.query_radius_gpu(center, radius, refine_euclidean=True).get()
        t1 = time.perf_counter()

        neighbor_labels = [self.labels[i] for i in neighbor_ids if i != idx]

        print(f"  Spatial query: {label} (r={radius})")
        print(f"  Neighbors: {neighbor_labels}")
        print(f"  Time: {(t1-t0)*1000:.3f}ms")

        return neighbor_labels


def main():
    parser = argparse.ArgumentParser(description="Semantic Navigator Demo")
    parser.add_argument("--house", required=True, help="Path to House GLB file")
    parser.add_argument("--start", default=None, help="Start label for navigation")
    parser.add_argument("--goal", default=None, help="Goal label for navigation")
    parser.add_argument("--k", type=int, default=8, help="k-neighbors for kernel")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--alpha", type=float, default=0.7, help="Semantic weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Geometric weight")
    args = parser.parse_args()

    print("=== Semantic Navigator Demo ===\n")

    navigator = SemanticNavigator()

    print("1. Loading House...")
    navigator.load_house_from_glb(args.house)

    print("\n2. Building Morton Octree (sleep-time)...")
    navigator.build_octree()

    print("\n3. Extracting Dependency Kernel (sleep-time)...")
    navigator.extract_dependency_kernel(k_neighbors=args.k, similarity_threshold=args.threshold)

    print("\n4. Runtime Operations:")

    if args.start and args.goal:
        print(f"\n  [Navigation Test]")
        navigator.navigate(args.start, args.goal, alpha=args.alpha, beta=args.beta)
    else:
        available_labels = navigator.labels[:10]
        print(f"\n  Available labels: {available_labels}")

        if len(navigator.labels) >= 2:
            start = navigator.labels[0]
            goal = navigator.labels[-1]
            print(f"\n  [Auto Navigation Test]")
            navigator.navigate(start, goal, alpha=args.alpha, beta=args.beta)

    if navigator.labels:
        test_label = navigator.labels[0]
        print(f"\n  [Spatial Query Test]")
        navigator.query_spatial_neighbors(test_label, radius=3.0)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
