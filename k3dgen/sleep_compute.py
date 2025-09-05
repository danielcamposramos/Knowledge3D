#!/usr/bin/env python3
"""
Sleep-Time Compute Simulation for K3D
Implements background optimization and galaxy pruning as described in Gemini's analysis.
"""

import json
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np


class SleepTimeCompute:
    """Handles offline memory consolidation and galaxy pruning."""

    def __init__(self, data_file: str = None):
        self.data = []
        self.load_data(data_file)

    def load_data(self, data_file: str = None) -> None:
        """Load knowledge graph data from JSON file."""
        if data_file:
            try:
                with open(data_file, 'r') as f:
                    self.data = json.load(f)
                print(f"📋 Loaded {len(self.data)} nodes for sleep-time compute")
            except FileNotFoundError:
                print(f"⚠️  Data file {data_file} not found, using sample data")
                self.generate_sample_data()
        else:
            # No data file specified, generate sample data for demonstration
            self.generate_sample_data()

    def generate_sample_data(self, num_nodes: int = 1000) -> None:
        """Generate sample knowledge graph data for demonstration."""
        print(f"🎯 Generating sample data with {num_nodes} nodes")

        np.random.seed(42)
        self.data = []

        for i in range(num_nodes):
            node = {
                "id": f"node_{i:04d}",
                "type": np.random.choice(["content", "concept", "relation"]),
                "metadata": {
                    "label": f"Concept {i}",
                    "domain": np.random.choice(["science", "technology", "philosophy", "art"]),
                    "confidence": np.random.uniform(0.5, 1.0)
                },
                "compression": {
                    "level": np.random.randint(0, 5),
                    "last_pruned": None,
                    "compression_score": np.random.uniform(0, 1),
                    "should_compress": False
                },
                "ai_native": {
                    "cognition": {
                        "memory_strength": np.random.uniform(0, 1),
                        "focus_level": np.random.uniform(0, 1)
                    },
                    "training": {
                        "observation_count": np.random.randint(0, 100),
                        "utility_score": np.random.uniform(0, 1)
                    }
                },
                "spatial": {
                    "last_accessed": (
                        datetime.now() - timedelta(days=np.random.randint(0, 365))
                    ).isoformat()
                }
            }
            self.data.append(node)

    def memory_consolidation(self) -> Dict[str, int]:
        """Phase 1: Strengthen frequently accessed memories."""
        print("🧠 Phase 1: Memory consolidation")

        strengthened = 0
        total_processed = 0

        for node in self.data:
            total_processed += 1
            current_strength = node["ai_native"]["cognition"]["memory_strength"]

            # Strengthen based on observation count and utility
            observations = node["ai_native"]["training"]["observation_count"]
            utility = node["ai_native"]["training"]["utility_score"]

            # Memory strengthening formula
            strengthening_factor = min(0.15, (observations * 0.001)) + (utility * 0.1)

            new_strength = min(1.0, current_strength + strengthening_factor)

            if new_strength > current_strength:
                node["ai_native"]["cognition"]["memory_strength"] = new_strength
                strengthened += 1

        print(f"   Strengthened {strengthened}/{total_processed} memory nodes")
        return {"strengthened": strengthened, "processed": total_processed}

    def galaxy_pruning(self) -> Dict[str, int]:
        """Phase 2: Remove redundant and low-utility nodes."""
        print("✂️  Phase 2: Galaxy pruning")

        pruned = 0
        compressed = 0

        # Sort by compression score (highest first)
        sorted_nodes = sorted(
            self.data,
            key=lambda x: x["compression"]["compression_score"],
            reverse=True
        )

        for node in sorted_nodes:
            compression_score = node["compression"]["compression_score"]
            memory_strength = node["ai_native"]["cognition"]["memory_strength"]
            current_compression = node["compression"]["level"]

            # Pruning decision algorithm
            prune_threshold = 0.8  # High compression score
            memory_bonus = memory_strength * 0.2  # Protect strong memories

            if compression_score > (prune_threshold - memory_bonus) and current_compression < 10:
                # Compress rather than delete
                node["compression"]["level"] += 1
                node["compression"]["last_pruned"] = datetime.now().isoformat()
                compressed += 1

        print(f"   Compressed {compressed} redundant nodes")
        return {"compressed": compressed}

    def vector_optimization(self) -> Dict[str, int]:
        """Phase 3: Optimize embeddings for better storage/processing."""
        print("🔄 Phase 3: Vector optimization")

        optimized = 0
        total_candidates = 0

        for node in self.data:
            last_accessed = node["spatial"]["last_accessed"]
            if not last_accessed:
                continue

            last_accessed_date = datetime.fromisoformat(last_accessed)
            days_since_access = (datetime.now() - last_accessed_date).days

            total_candidates += 1

            # Optimize nodes not accessed in 30+ days
            if days_since_access > 30:
                current_compression = node["compression"]["level"]

                # Apply opportunistic compression optimization
                if current_compression < 8:
                    # Simulate vector optimization by increasing compression
                    optimization_level = min(8, current_compression + 2)
                    node["compression"]["level"] = optimization_level
                    optimized += 1
                    print(f"   Optimized node {node['id']}: compression {current_compression} → {optimization_level}")

        print(f"   Optimized {optimized}/{total_candidates} stale nodes")
        return {"optimized": optimized, "candidates": total_candidates}

    def relationship_inference(self) -> Dict[str, int]:
        """Phase 4: Discover new semantic relationships."""
        print("🔗 Phase 4: Relationship inference")

        # Simple clustering-based relationship discovery
        high_confidence_nodes = [
            node for node in self.data
            if node["metadata"]["confidence"] > 0.8
        ]

        relationships_discovered = 0

        # Group by domain and create relationships
        domains = {}
        for node in high_confidence_nodes:
            domain = node["metadata"]["domain"]
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(node)

        # Link nodes within domains
        for domain, nodes in domains.items():
            if len(nodes) < 3:
                continue

            # Sort by utility score
            nodes.sort(key=lambda x: x["ai_native"]["training"]["utility_score"])

            # Create hierarchical relationships
            for i, node in enumerate(nodes[:3]):  # Top 3 utility nodes
                if "relations" not in node:
                    node["relations"] = {"semantic": []}

                new_relation = {
                    "target_id": "hierarchy_root",
                    "relationship_type": "parent" if i == 0 else ("child" if i == 1 else "related"),
                    "strength": 0.7 - (i * 0.2),
                    "bidirectional": i != 0
                }

                node["relations"]["semantic"].append(new_relation)
                relationships_discovered += 1

        print(f"   Discovered {relationships_discovered} new semantic relationships")
        return {"relationships": relationships_discovered}

    def save_results(self, output_file: str = None) -> None:
        """Save the optimized knowledge graph."""
        output_path = output_file or "data/optimized_knowledge_graph.json"

        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

        print(f"💾 Results saved to {output_path}")

    def run_full_cycle(self, phases: List[str] = None) -> Dict[str, Any]:
        """Run complete sleep-time compute cycle."""
        start_time = time.time()

        if phases is None:
            phases = ["memory", "pruning", "optimization", "inference"]

        results = {
            "start_time": datetime.now().isoformat(),
            "cycles": {}
        }

        print("🌙 Starting sleep-time compute cycle\n")

        if "memory" in phases:
            results["cycles"]["memory_consolidation"] = self.memory_consolidation()

        if "pruning" in phases:
            results["cycles"]["galaxy_pruning"] = self.galaxy_pruning()

        if "optimization" in phases:
            results["cycles"]["vector_optimization"] = self.vector_optimization()

        if "inference" in phases:
            results["cycles"]["relationship_inference"] = self.relationship_inference()

        # Calculate compression improvements
        total_compression = sum(node["compression"]["level"] for node in self.data)
        avg_memory_strength = sum(
            node["ai_native"]["cognition"]["memory_strength"] for node in self.data
        ) / len(self.data) if self.data else 0

        results["summary"] = {
            "total_nodes": len(self.data),
            "total_compression": total_compression,
            "average_memory_strength": round(avg_memory_strength, 3),
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

        print(f"\n✅ Sleep-time compute completed in {results['summary']['processing_time_seconds']}s")
        print(f"📊 Summary: {results['summary']['total_nodes']} nodes processed")
        print(".3f")
        print(f"🗜️  Total compression level: {total_compression}")

        # Save results to disk for monitoring
        with open("data/sleep_compute_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results


def main():
    parser = argparse.ArgumentParser(description="K3D Sleep-Time Compute")
    parser.add_argument("--data", help="Path to knowledge graph data file")
    parser.add_argument("--phases", nargs="+",
                       choices=["memory", "pruning", "optimization", "inference"],
                       default=["memory", "pruning", "optimization", "inference"],
                       help="Specify which phases to run")
    parser.add_argument("--output", help="Output file for optimized data")

    args = parser.parse_args()

    compute = SleepTimeCompute(args.data)
    results = compute.run_full_cycle(args.phases)
    compute.save_results(args.output)

    print(f"\n🎉 Sleep-time compute cycle complete!")
    print("🌌 Knowledge universe optimized and ready for next runtime")


if __name__ == "__main__":
    main()
