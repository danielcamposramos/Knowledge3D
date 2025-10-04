import unittest
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Tuple

from knowledge3d.spatial.led_pathfinder import DependencyKernel, LEDPathfinder


class TestDependencyKernel(unittest.TestCase):

    def setUp(self):
        self.kernel = DependencyKernel()

    def test_kernel_construction_simple_graph(self):
        """Test CSR construction from edge list."""
        edges = np.array([
            [0, 1], [0, 2],
            [1, 2], [1, 3],
            [2, 3]
        ], dtype=np.uint32)

        embeddings = np.random.randn(4, 128).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        positions = np.random.randn(4, 3).astype(np.float32)

        self.kernel.build_from_edges(edges, embeddings, positions, similarity_threshold=0.5)

        self.assertEqual(self.kernel.node_count, 4)
        self.assertIsNotNone(self.kernel.row_offsets_gpu)
        self.assertIsNotNone(self.kernel.col_indices_gpu)
        self.assertIsNotNone(self.kernel.packed_costs_gpu)

    def test_packed_cost_format(self):
        """Verify semantic and geometric costs are packed correctly."""
        edges = np.array([[0, 1]], dtype=np.uint32)

        embeddings = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        self.kernel.build_from_edges(edges, embeddings, positions, similarity_threshold=0.0)

        packed_costs = self.kernel.packed_costs_gpu.get()
        self.assertEqual(len(packed_costs), 1)

        semantic_cost = (packed_costs[0] >> 16) & 0xFFFF
        geometric_cost = packed_costs[0] & 0xFFFF

        self.assertGreater(semantic_cost, 0)
        self.assertGreater(geometric_cost, 0)

    def test_similarity_threshold_filtering(self):
        """Verify low-similarity edges are filtered out."""
        edges = np.array([
            [0, 1], [0, 2], [0, 3]
        ], dtype=np.uint32)

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],  # High similarity
            [0.5, 0.5, 0.5],   # Medium similarity
            [0.0, 0.0, 1.0]    # Low similarity
        ], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        positions = np.random.randn(4, 3).astype(np.float32)

        self.kernel.build_from_edges(edges, embeddings, positions, similarity_threshold=0.8)

        nnz = int(self.kernel.row_offsets_gpu[-1].get())
        self.assertLess(nnz, 3, "High threshold should filter low-similarity edges")


class TestLEDPathfinder(unittest.TestCase):

    def setUp(self):
        self.pathfinder = LEDPathfinder()

    def _create_simple_graph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a simple 5-node graph for testing."""
        edges = np.array([
            [0, 1], [1, 0],
            [1, 2], [2, 1],
            [2, 3], [3, 2],
            [3, 4], [4, 3],
            [0, 4], [4, 0]
        ], dtype=np.uint32)

        embeddings = np.random.randn(5, 128).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0]
        ], dtype=np.float32)

        return edges, embeddings, positions

    def test_simple_path_finding(self):
        """Test basic pathfinding from start to goal."""
        edges, embeddings, positions = self._create_simple_graph()
        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.0)

        path, cost = self.pathfinder.find_path(start=0, goal=4)

        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 4)
        self.assertGreater(cost, 0)

    def test_path_exactness_vs_dijkstra(self):
        """Verify LED-A* finds exact optimal path (matches Dijkstra)."""
        edges, embeddings, positions = self._create_simple_graph()
        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.0)

        path_led, cost_led = self.pathfinder.find_path(start=0, goal=4, alpha=0.0, beta=1.0)

        path_dijkstra, cost_dijkstra = self._dijkstra_baseline(edges, positions, start=0, goal=4)

        self.assertAlmostEqual(cost_led, cost_dijkstra, places=2,
                               msg="LED-A* cost should match Dijkstra (exact optimality)")

    def _dijkstra_baseline(self, edges, positions, start, goal):
        """CPU Dijkstra implementation for ground truth."""
        import heapq

        num_nodes = positions.shape[0]
        dist = np.full(num_nodes, np.inf, dtype=np.float32)
        parent = np.full(num_nodes, -1, dtype=np.int32)
        dist[start] = 0

        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == goal:
                break

            for i in range(len(edges)):
                if edges[i, 0] == u:
                    v = edges[i, 1]
                    edge_cost = np.linalg.norm(positions[v] - positions[u])
                    if dist[u] + edge_cost < dist[v]:
                        dist[v] = dist[u] + edge_cost
                        parent[v] = u
                        heapq.heappush(pq, (dist[v], v))

        path = []
        node = goal
        while node != -1:
            path.append(int(node))
            node = parent[node]
        path.reverse()

        return path, float(dist[goal])

    def test_no_path_case(self):
        """Test behavior when no path exists."""
        edges = np.array([[0, 1], [1, 0]], dtype=np.uint32)
        embeddings = np.random.randn(3, 128).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        positions = np.random.randn(3, 3).astype(np.float32)

        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.0)

        path, cost = self.pathfinder.find_path(start=0, goal=2)

        self.assertEqual(len(path), 0, "Should return empty path when unreachable")
        self.assertEqual(cost, float('inf'), "Cost should be infinity when unreachable")

    def test_alpha_beta_weighting(self):
        """Test semantic vs geometric weighting."""
        edges, embeddings, positions = self._create_simple_graph()

        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.0)

        _, cost_semantic = self.pathfinder.find_path(start=0, goal=4, alpha=1.0, beta=0.0)
        _, cost_geometric = self.pathfinder.find_path(start=0, goal=4, alpha=0.0, beta=1.0)
        _, cost_balanced = self.pathfinder.find_path(start=0, goal=4, alpha=0.5, beta=0.5)

        self.assertNotEqual(cost_semantic, cost_geometric,
                           "Semantic and geometric costs should differ")
        self.assertGreater(cost_balanced, 0)


class TestPerformanceBenchmarks(unittest.TestCase):

    def setUp(self):
        self.pathfinder = LEDPathfinder()

    def _create_dense_graph(self, num_nodes: int, avg_degree: int = 8):
        """Create a dense random graph for benchmarking."""
        edges_list = []
        for i in range(num_nodes):
            neighbors = np.random.choice(num_nodes, size=avg_degree, replace=False)
            for j in neighbors:
                if i != j:
                    edges_list.append([i, j])

        edges = np.array(edges_list, dtype=np.uint32)
        embeddings = np.random.randn(num_nodes, 128).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        positions = np.random.randn(num_nodes, 3).astype(np.float32)

        return edges, embeddings, positions

    def test_performance_1k_nodes(self):
        """Benchmark LED-A* on 1K nodes (target: <0.15ms)."""
        edges, embeddings, positions = self._create_dense_graph(1000, avg_degree=8)
        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.6)

        start = 0
        goal = 999

        num_trials = 100
        times = []

        for _ in range(num_trials):
            import time
            t0 = time.perf_counter()
            path, cost = self.pathfinder.find_path(start, goal)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\n1K nodes: avg={avg_time:.3f}ms, p95={p95_time:.3f}ms")
        self.assertLess(avg_time, 0.20, "1K nodes should complete <0.15ms (allowing margin)")

    def test_performance_10k_nodes(self):
        """Benchmark LED-A* on 10K nodes (target: <0.28ms)."""
        edges, embeddings, positions = self._create_dense_graph(10000, avg_degree=8)
        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.6)

        start = 0
        goal = 9999

        num_trials = 50
        times = []

        for _ in range(num_trials):
            import time
            t0 = time.perf_counter()
            path, cost = self.pathfinder.find_path(start, goal)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\n10K nodes: avg={avg_time:.3f}ms, p95={p95_time:.3f}ms")
        self.assertLess(avg_time, 0.35, "10K nodes should complete <0.28ms (allowing margin)")

    @unittest.skipUnless(cp.cuda.runtime.getDeviceCount() > 0, "Requires GPU with >8GB VRAM")
    def test_performance_100k_nodes(self):
        """Benchmark LED-A* on 100K nodes (target: <0.45ms)."""
        edges, embeddings, positions = self._create_dense_graph(100000, avg_degree=8)
        self.pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.6)

        start = 0
        goal = 99999

        num_trials = 20
        times = []

        for _ in range(num_trials):
            import time
            t0 = time.perf_counter()
            path, cost = self.pathfinder.find_path(start, goal)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\n100K nodes: avg={avg_time:.3f}ms, p95={p95_time:.3f}ms")
        self.assertLess(avg_time, 0.60, "100K nodes should complete <0.45ms (allowing margin)")


if __name__ == "__main__":
    unittest.main()
