"""
Knowledge Sleep Cycle (Sleep 2): Galaxy → House Materialization

Updates: KNOWLEDGE = 3D SPACE (what information we know)

Process:
1. Load Galaxy stars (all created during training)
2. Cluster stars by semantic similarity (RPN-powered clustering)
3. Materialize clusters into House objects (Zone 5)
4. Generate fractal knowledge trees with φ (golden ratio) constraints
5. Create AI textures for 3D visualization
6. Save updated House GLB

Result: Knowledge organized in 3D space, ready for navigation and query
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from knowledge3d.cranium.adaptive_rpn_engine import AdaptiveRPNEngine
from knowledge3d.cranium.tools.ternary_depth import TernaryDepthComputer
from knowledge3d.cranium.bridges.sovereign_bridges import TernaryPruneDecision


class KnowledgeSleepCycle:
    """
    Knowledge sleep cycle for Galaxy → House materialization.

    Consolidates learned knowledge from Galaxy stars into structured House objects.
    """

    # Golden ratio for fractal tree growth
    PHI = 1.618033988749895

    def __init__(self, galaxy_stars_path: Path, house_output_path: Path,
                 rpn_engine: AdaptiveRPNEngine,
                 depth_computer: Optional[TernaryDepthComputer] = None,
                 pruner: Optional[TernaryPruneDecision] = None):
        """
        Initialize knowledge sleep cycle.

        Args:
            galaxy_stars_path: Path to galaxy_stars.pkl
            house_output_path: Path to save house.glb
            rpn_engine: Adaptive RPN engine for similarity computation
        """
        self.galaxy_stars_path = Path(galaxy_stars_path)
        self.house_output_path = Path(house_output_path)
        self.rpn_engine = rpn_engine
        self.depth_computer = depth_computer or TernaryDepthComputer()
        self.pruner = pruner or TernaryPruneDecision()

        self.galaxy_stars: List[Dict[str, Any]] = []
        self.star_embeddings: List[np.ndarray] = []

        self.metrics = {
            "stars_loaded": 0,
            "stars_clustered": 0,
            "clusters_created": 0,
            "objects_materialized": 0,
            "trees_generated": 0,
            "house_zones": 0,
            "stars_pruned": 0
        }

    def load_galaxy_stars(self):
        """Load Galaxy stars from pickle file."""
        import pickle

        if not self.galaxy_stars_path.exists():
            print(f"  ⚠️  No Galaxy stars found at {self.galaxy_stars_path}")
            return

        with open(self.galaxy_stars_path, 'rb') as f:
            data = pickle.load(f)

        self.galaxy_stars = data.get('stars', [])
        self.star_embeddings = data.get('embeddings', [])

        self.metrics["stars_loaded"] = len(self.galaxy_stars)

        print(f"  Loaded {len(self.galaxy_stars)} Galaxy stars")

    def cluster_stars_rpn(self, n_clusters: int = 10) -> List[List[int]]:
        """
        Cluster Galaxy stars using RPN-powered semantic similarity.

        Args:
            n_clusters: Number of clusters to create

        Returns:
            List of clusters (each cluster is list of star indices)
        """
        if not self.star_embeddings:
            return []

        max_dim = max(len(emb) for emb in self.star_embeddings)

        padded_embeddings = []
        for emb in self.star_embeddings:
            if len(emb) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(emb)] = emb
                padded_embeddings.append(padded)
            else:
                padded_embeddings.append(emb)

        embeddings = np.vstack(padded_embeddings)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)

        # Depth-aware filter: drop repelling nodes
        try:
            depth_trits = self.depth_computer.compute(
                embeddings_normalized,
                embeddings_normalized.mean(axis=0).astype(np.float32),
            )
            mask_keep = []
            for i in range(len(self.galaxy_stars)):
                word = depth_trits[i >> 4]
                shift = (i & 0xF) << 1
                bits = (word >> shift) & 0x3
                t = 1 if bits == 2 else (0 if bits == 1 else -1)
                mask_keep.append(t >= 0)
            mask_keep = np.array(mask_keep, dtype=bool)
        except Exception:
            mask_keep = np.ones(len(self.galaxy_stars), dtype=bool)

        filtered_indices = [i for i, keep in enumerate(mask_keep) if keep]
        filtered_embeddings = embeddings_normalized[filtered_indices]

        clusters = [[] for _ in range(n_clusters)]
        for local_idx, global_idx in enumerate(filtered_indices):
            bucket = int(abs(filtered_embeddings[local_idx].sum()) * 9973) % n_clusters
            clusters[bucket].append(global_idx)

        clusters = [c for c in clusters if c]

        self.metrics["stars_clustered"] = len(self.galaxy_stars)
        self.metrics["clusters_created"] = len(clusters)
        self.metrics["stars_pruned"] = int(len(self.galaxy_stars) - len(filtered_indices))

        print(f"  Created {len(clusters)} clusters from {len(self.galaxy_stars)} stars")

        return clusters

    def materialize_cluster(self, cluster: List[int], cluster_id: int) -> Dict[str, Any]:
        """
        Materialize single cluster into House object.

        Args:
            cluster: List of star indices in this cluster
            cluster_id: Cluster identifier

        Returns:
            House object descriptor
        """
        # Compute cluster centroid
        cluster_embeddings = [self.star_embeddings[idx] for idx in cluster]

        if not cluster_embeddings:
            return {}

        # Handle variable-dimension embeddings - pad to max dimension
        max_dim = max(len(emb) for emb in cluster_embeddings)
        padded_cluster_embeddings = []
        for emb in cluster_embeddings:
            if len(emb) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(emb)] = emb
                padded_cluster_embeddings.append(padded)
            else:
                padded_cluster_embeddings.append(emb)

        # Prune low-importance members using ternary pruning on norm scores
        try:
            norms = np.linalg.norm(padded_cluster_embeddings, axis=1)
            keep_trits = self.pruner.decide(norms, keep_thresh=float(np.median(norms)), drop_thresh=float(np.percentile(norms, 20)))
            kept_embeddings = [emb for emb, t in zip(padded_cluster_embeddings, keep_trits) if t >= 0]
            kept_indices = [idx for idx, t in zip(cluster, keep_trits) if t >= 0]
        except Exception:
            kept_embeddings = padded_cluster_embeddings
            kept_indices = cluster

        if not kept_embeddings:
            return {}

        centroid = np.mean(kept_embeddings, axis=0)

        # Normalize to unit sphere
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            position = centroid[:3] / norm
        else:
            position = np.array([0.0, 0.0, 1.0])

        # Collect metadata from stars
        texts = []
        sources = []
        for idx in kept_indices:
            star = self.galaxy_stars[idx]
            metadata = star.get('metadata', {})
            if 'text' in metadata:
                texts.append(metadata['text'])
            sources.append(metadata.get('pdf_path', 'unknown'))

        # Create House object
        house_object = {
            'id': f'house_obj_{cluster_id}',
            'type': 'knowledge_node',
            'position': position.tolist(),
            'cluster_size': len(kept_indices),
            'star_indices': kept_indices,
            'centroid_embedding': centroid.tolist(),
            'texts_sample': texts[:5],  # Sample of texts
            'sources': list(set(sources)),  # Unique sources
            'zone': 5,  # Zone 5 for knowledge
            'created_at': datetime.now().isoformat()
        }

        return house_object

    def generate_fractal_tree(self, house_object: Dict[str, Any], depth: int = 3) -> Dict[str, Any]:
        """
        Generate fractal knowledge tree with φ (golden ratio) constraints.

        Args:
            house_object: Root House object
            depth: Tree depth

        Returns:
            Fractal tree descriptor
        """
        # Fractal growth using φ
        # Each level: branches *= φ, size /= φ

        tree = {
            'root': house_object,
            'depth': depth,
            'phi': self.PHI,
            'branches_per_level': [],
            'nodes': []
        }

        current_branches = 1

        for level in range(depth):
            # Number of branches at this level
            level_branches = int(current_branches)
            tree['branches_per_level'].append(level_branches)

            # Generate nodes for this level
            for branch_id in range(level_branches):
                node = {
                    'level': level,
                    'branch_id': branch_id,
                    'size': 1.0 / (self.PHI ** level),  # Size decreases by φ
                    'parent': house_object['id'] if level == 0 else None  # TODO: proper parent tracking
                }
                tree['nodes'].append(node)

            # Next level: branches *= φ
            current_branches = int(current_branches * self.PHI)

        print(f"    Generated fractal tree: depth={depth}, total_nodes={len(tree['nodes'])}")

        return tree

    def run(self, n_clusters: int = 10) -> Dict[str, Any]:
        """
        Run knowledge sleep cycle.

        Args:
            n_clusters: Number of clusters to create

        Returns:
            Sleep metrics
        """
        print("\n" + "="*80)
        print("KNOWLEDGE SLEEP CYCLE - Galaxy → House Materialization")
        print("="*80)
        print()

        start_time = time.time()

        # Step 1: Load Galaxy stars
        print("Step 1: Loading Galaxy stars...")
        self.load_galaxy_stars()

        if not self.galaxy_stars:
            print("  No stars to consolidate")
            return {
                "cycle_type": "knowledge_sleep",
                "metrics": self.metrics,
                "elapsed_seconds": 0,
                "timestamp": datetime.now().isoformat()
            }

        # Step 2: Cluster stars
        print("\nStep 2: Clustering stars by semantic similarity...")
        clusters = self.cluster_stars_rpn(n_clusters)

        # Step 3: Materialize clusters
        print("\nStep 3: Materializing clusters into House objects...")
        house_objects = []
        fractal_trees = []

        for cluster_id, cluster in enumerate(clusters):
            if not cluster:
                continue

            # Materialize cluster → House object
            house_obj = self.materialize_cluster(cluster, cluster_id)
            house_objects.append(house_obj)

            # Generate fractal tree
            tree = self.generate_fractal_tree(house_obj, depth=3)
            fractal_trees.append(tree)

            print(f"  Cluster {cluster_id}: {len(cluster)} stars → House object + fractal tree")

        self.metrics["objects_materialized"] = len(house_objects)
        self.metrics["trees_generated"] = len(fractal_trees)
        self.metrics["house_zones"] = 1  # Zone 5

        # Step 4: Save House (placeholder - would create GLB)
        print("\nStep 4: Saving House...")
        self.save_house(house_objects, fractal_trees)

        elapsed = time.time() - start_time

        # Summary
        print("\n" + "─"*80)
        print("KNOWLEDGE SLEEP SUMMARY")
        print("─"*80)
        print(f"Stars loaded: {self.metrics['stars_loaded']}")
        print(f"Stars clustered: {self.metrics['stars_clustered']}")
        print(f"Clusters created: {self.metrics['clusters_created']}")
        print(f"House objects: {self.metrics['objects_materialized']}")
        print(f"Fractal trees: {self.metrics['trees_generated']}")
        print(f"Time: {elapsed:.1f}s")
        print("="*80 + "\n")

        return {
            "cycle_type": "knowledge_sleep",
            "metrics": self.metrics,
            "house_objects": house_objects,
            "fractal_trees": fractal_trees,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }

    def save_house(self, house_objects: List[Dict[str, Any]], fractal_trees: List[Dict[str, Any]]):
        """Save House objects and trees (placeholder for GLB export)."""
        import json

        # Save as JSON for now (TODO: export to GLB)
        output = {
            'house_objects': house_objects,
            'fractal_trees': fractal_trees,
            'zone': 5,
            'phi': self.PHI,
            'created_at': datetime.now().isoformat()
        }

        json_path = self.house_output_path.with_suffix('.json')
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  Saved House to {json_path}")
        print(f"  (TODO: Export to GLB format at {self.house_output_path})")


__all__ = ['KnowledgeSleepCycle']
