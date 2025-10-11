"""
Sovereign Bridges - Pure ctypes + libcuda.so bridges for all Step8 kernels

This module provides Python bridges for all 15 Step8 kernels using the sovereign
loader (pure ctypes + CUDA Driver API). Zero dependencies beyond Python stdlib.

Usage:
    from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard, ARCReasoner, ...

    guard = LatencyGuard(threshold_us=95.0)
    guard.start()
    # ... GPU work ...
    elapsed_ns, breached = guard.stop()

Architecture:
    - All bridges use knowledge3d.cranium.sovereign.loader
    - All memory management via gpu_malloc/gpu_free
    - All kernel launches via sovereign launch()
    - No CuPy, no cuda-python, no external dependencies
"""

import numpy as np
import ctypes
from pathlib import Path
from typing import Tuple, Optional

from knowledge3d.cranium.sovereign.loader import (
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)

# Base paths
KERNELS_DIR = Path(__file__).parent.parent / "kernels"


# ============================================================================
# Kimi's Kernels
# ============================================================================

class LatencyGuard:
    """Sovereign Latency Guard - Records GPU timing with %globaltimer

    Uses gre_sub100micro_gate.ptx to measure kernel execution time directly
    on GPU, avoiding CPU timer overhead.

    Args:
        threshold_us: Maximum allowed latency in microseconds (default: 100.0)
    """

    def __init__(self, threshold_us: float = 100.0):
        self.threshold_us = float(threshold_us)
        self.threshold_ns = int(threshold_us * 1_000.0)

        # Load PTX kernel
        ptx_path = KERNELS_DIR / "gre_sub100micro_gate.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_sub100micro_gate")

        # Allocate device buffers (reused across calls)
        self.d_timestamps = gpu_malloc(2 * 8)  # 2 x uint64
        self.d_flag = gpu_malloc(4)            # 1 x uint32

        # Host buffers for readback
        self.timestamps = np.zeros(2, dtype=np.uint64)
        self.flag = np.zeros(1, dtype=np.uint32)

    def start(self):
        """Record start timestamp on GPU"""
        launch(
            self.kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                ctypes.c_uint64(self.d_timestamps.value),
                ctypes.c_uint64(self.d_flag.value),
                ctypes.c_uint64(self.threshold_ns),
                ctypes.c_uint32(0),  # mode=0 (start)
            ],
        )
        synchronize()

    def stop(self) -> Tuple[int, bool]:
        """Record stop timestamp and check threshold

        Returns:
            (elapsed_ns, breached): Elapsed time in ns and whether threshold was exceeded
        """
        launch(
            self.kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                ctypes.c_uint64(self.d_timestamps.value),
                ctypes.c_uint64(self.d_flag.value),
                ctypes.c_uint64(self.threshold_ns),
                ctypes.c_uint32(1),  # mode=1 (stop)
            ],
        )
        synchronize()

        # Copy results back
        memcpy_dtoh(self.timestamps.ctypes.data_as(ctypes.c_void_p),
                   self.d_timestamps, self.timestamps.nbytes)
        memcpy_dtoh(self.flag.ctypes.data_as(ctypes.c_void_p),
                   self.d_flag, self.flag.nbytes)

        elapsed_ns = int(self.timestamps[1] - self.timestamps[0])
        breached = bool(self.flag[0] == 0xDEADBEEF)

        return elapsed_ns, breached

    def cleanup(self):
        """Free GPU memory"""
        gpu_free(self.d_timestamps)
        gpu_free(self.d_flag)

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


class ARCReasoner:
    """Sovereign ARC Reasoner - Extracts rules from ARC grids

    Uses gre_arc_reasoner.ptx to analyze ARC-AGI grids and extract
    compact rule representations.

    Example:
        reasoner = ARCReasoner()
        grid = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.int32)
        rule_id, rotation, color_checksum = reasoner.extract_rules(grid)
    """

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_arc_reasoner.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_arc_reasoner")

    def extract_rules(self, grid: np.ndarray) -> Tuple[int, int, int]:
        """Extract rules from ARC grid

        Args:
            grid: 2D int32 array (will be flattened)

        Returns:
            (rule_id, rotation_count, color_checksum): Extracted rule parameters
        """
        # Flatten grid
        grid_flat = grid.flatten().astype(np.int32)
        grid_size = len(grid_flat)

        # Allocate GPU memory
        d_grid = gpu_malloc(grid_flat.nbytes)
        d_output = gpu_malloc(3 * 4)  # 3 x int32

        try:
            # Copy grid to GPU
            memcpy_htod(d_grid, grid_flat.ctypes.data_as(ctypes.c_void_p), grid_flat.nbytes)

            # Launch kernel
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(d_grid.value),
                    ctypes.c_uint32(grid_size),
                    ctypes.c_uint64(d_output.value),
                ],
            )
            synchronize()

            # Copy results back
            output = np.zeros(3, dtype=np.int32)
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)

            return int(output[0]), int(output[1]), int(output[2])

        finally:
            gpu_free(d_grid)
            gpu_free(d_output)


class OOMSpillManager:
    """Sovereign OOM Spill Manager - Memory overflow protection

    Uses gre_oom_spill.ptx to compute spill plans when GPU memory is low.
    """

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_oom_spill.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_oom_spill")

    def compute_spill_plan(
        self,
        oldest_index: int,
        atom_size_bytes: int,
        available_bytes: int,
        request_count: int
    ) -> Tuple[int, int]:
        """Compute how many atoms to spill given available memory

        Args:
            oldest_index: Index of oldest atom in memory
            atom_size_bytes: Size of each atom in bytes
            available_bytes: Available GPU memory
            request_count: Number of atoms requested

        Returns:
            (atoms_to_spill, bytes_required): Spill plan
        """
        # Prepare stats input
        stats = np.array([oldest_index, atom_size_bytes], dtype=np.uint64)
        output = np.zeros(2, dtype=np.uint64)

        # Allocate GPU memory
        d_stats = gpu_malloc(stats.nbytes)
        d_output = gpu_malloc(output.nbytes)

        try:
            # Copy stats to GPU
            memcpy_htod(d_stats, stats.ctypes.data_as(ctypes.c_void_p), stats.nbytes)

            # Launch kernel
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(d_stats.value),
                    ctypes.c_uint64(available_bytes),
                    ctypes.c_uint32(request_count),
                    ctypes.c_uint64(d_output.value),
                ],
            )
            synchronize()

            # Copy results back
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)

            return int(output[0]), int(output[1])

        finally:
            gpu_free(d_stats)
            gpu_free(d_output)


# ============================================================================
# Qwen's Kernel
# ============================================================================

class GalaxyResonanceEngine:
    """Sovereign Galaxy Resonance Engine - Recursive core blending

    Uses galaxy_resonance_engine.ptx to blend embeddings with latent state
    using alpha-weighted combination (RPN-style lerp).
    """

    def __init__(self):
        ptx_path = KERNELS_DIR / "galaxy_resonance_engine.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "galaxy_resonance_engine")

    def resonate(
        self,
        embeddings: np.ndarray,
        latent: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Blend embeddings with latent state

        Args:
            embeddings: Input embeddings [batch_size, vector_dim]
            latent: Latent state [batch_size, vector_dim]
            alpha: Blend factor (0.0 to 1.0)

        Returns:
            output: Blended result [batch_size, vector_dim]
        """
        assert embeddings.shape == latent.shape
        assert embeddings.dtype == latent.dtype == np.float32

        batch_size, vector_dim = embeddings.shape
        output = np.zeros_like(embeddings)

        # Allocate GPU memory
        d_embeddings = gpu_malloc(embeddings.nbytes)
        d_latent = gpu_malloc(latent.nbytes)
        d_output = gpu_malloc(output.nbytes)

        try:
            # Copy inputs to GPU
            memcpy_htod(d_embeddings, embeddings.ctypes.data_as(ctypes.c_void_p), embeddings.nbytes)
            memcpy_htod(d_latent, latent.ctypes.data_as(ctypes.c_void_p), latent.nbytes)

            # Launch kernel (one block per batch element)
            launch(
                self.kernel,
                grid=(batch_size, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_embeddings.value),
                    ctypes.c_uint64(d_latent.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(vector_dim),
                    ctypes.c_uint32(batch_size),
                    ctypes.c_float(alpha),
                ],
            )
            synchronize()

            # Copy result back
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)

            return output

        finally:
            gpu_free(d_embeddings)
            gpu_free(d_latent)
            gpu_free(d_output)


__all__ = [
    "LatencyGuard",
    "ARCReasoner",
    "OOMSpillManager",
    "GalaxyResonanceEngine",
]


# ============================================================================
# Deep Seek's Kernels
# ============================================================================

class GeometryRouter:
    """Sovereign Geometry Router - Media-type dispatch and scaling"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_geometry_router.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_geometry_router")

    def route(self, input_data: np.ndarray, shape_id: int) -> np.ndarray:
        """Route and scale data based on media geometry
        
        Args:
            input_data: Input float32 array
            shape_id: 0=text, 1=image, 2=audio, 3=video, 4=mixed
        
        Returns:
            Scaled output array
        """
        assert input_data.dtype == np.float32
        output = np.zeros_like(input_data)
        
        d_input = gpu_malloc(input_data.nbytes)
        d_output = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_input, input_data.ctypes.data_as(ctypes.c_void_p), input_data.nbytes)
            
            launch(
                self.kernel,
                grid=((len(input_data) + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_input.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(len(input_data)),
                    ctypes.c_uint32(shape_id),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
            return output
        finally:
            gpu_free(d_input)
            gpu_free(d_output)


class FractalEmitter:
    """Sovereign Fractal Emitter - Knowledge Garden coordinate generation"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_fractal_emitter.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_fractal_emitter")

    def emit(self, atoms: np.ndarray, base_scale: float = 1.0) -> np.ndarray:
        """Generate fractal coordinates for atoms
        
        Args:
            atoms: Atom values (float32)
            base_scale: Coordinate scaling factor
        
        Returns:
            Coordinates array [count, 3] (x, y, z)
        """
        assert atoms.dtype == np.float32
        count = len(atoms)
        coords = np.zeros((count, 3), dtype=np.float32)
        
        d_atoms = gpu_malloc(atoms.nbytes)
        d_coords = gpu_malloc(coords.nbytes)
        
        try:
            memcpy_htod(d_atoms, atoms.ctypes.data_as(ctypes.c_void_p), atoms.nbytes)
            
            launch(
                self.kernel,
                grid=((count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_atoms.value),
                    ctypes.c_uint64(d_coords.value),
                    ctypes.c_uint32(count),
                    ctypes.c_float(base_scale),
                ],
            )
            synchronize()
            
            memcpy_dtoh(coords.ctypes.data_as(ctypes.c_void_p), d_coords, coords.nbytes)
            return coords
        finally:
            gpu_free(d_atoms)
            gpu_free(d_coords)


# ============================================================================
# GLM's Kernels
# ============================================================================

class ResonanceField:
    """Sovereign Resonance Field - Energetic field management"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_resonance_field.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_resonance_field")

    def compute(self, positions: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Compute resonance strengths from positions and density
        
        Args:
            positions: Position array [count, 3] (x, y, z)
            density: Density values [count]
        
        Returns:
            Resonance strengths [count]
        """
        assert positions.dtype == density.dtype == np.float32
        count = len(density)
        assert positions.shape == (count, 3)
        
        output = np.zeros(count, dtype=np.float32)
        
        d_positions = gpu_malloc(positions.nbytes)
        d_density = gpu_malloc(density.nbytes)
        d_output = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_positions, positions.ctypes.data_as(ctypes.c_void_p), positions.nbytes)
            memcpy_htod(d_density, density.ctypes.data_as(ctypes.c_void_p), density.nbytes)
            
            launch(
                self.kernel,
                grid=((count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_positions.value),
                    ctypes.c_uint64(d_density.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(count),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
            return output
        finally:
            gpu_free(d_positions)
            gpu_free(d_density)
            gpu_free(d_output)


class AtomicFissionFusion:
    """Sovereign Atomic Fission/Fusion - Atom compress/expand operations"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_atomic_fission_fusion.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_atomic_fission_fusion")

    def transform(self, atoms: np.ndarray, mode: int, ratio: float) -> np.ndarray:
        """Transform atoms via fission or fusion
        
        Args:
            atoms: Atom values (float32)
            mode: 0=fusion (compress), 1=fission (expand)
            ratio: Transformation ratio
        
        Returns:
            Transformed atoms
        """
        assert atoms.dtype == np.float32
        output = np.zeros_like(atoms)
        
        d_input = gpu_malloc(atoms.nbytes)
        d_output = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_input, atoms.ctypes.data_as(ctypes.c_void_p), atoms.nbytes)
            
            launch(
                self.kernel,
                grid=((len(atoms) + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_input.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(len(atoms)),
                    ctypes.c_uint32(mode),
                    ctypes.c_float(ratio),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
            return output
        finally:
            gpu_free(d_input)
            gpu_free(d_output)


class TemporalReasoning:
    """Sovereign Temporal Reasoning - Sequential delta computation"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_temporal_reasoning.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_temporal_reasoning")

    def compute_deltas(self, sequence: np.ndarray) -> np.ndarray:
        """Compute frame-to-frame deltas
        
        Args:
            sequence: Sequence array [sequence_length, feature_dim]
        
        Returns:
            Delta array [sequence_length, feature_dim]
        """
        assert sequence.dtype == np.float32
        assert len(sequence.shape) == 2
        
        seq_length, feat_dim = sequence.shape
        output = np.zeros_like(sequence)
        
        d_sequence = gpu_malloc(sequence.nbytes)
        d_output = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_sequence, sequence.ctypes.data_as(ctypes.c_void_p), sequence.nbytes)
            
            launch(
                self.kernel,
                grid=((feat_dim + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_sequence.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(seq_length),
                    ctypes.c_uint32(feat_dim),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
            return output
        finally:
            gpu_free(d_sequence)
            gpu_free(d_output)


# ============================================================================
# Grok's Kernels
# ============================================================================

class VectorResonator:
    """Sovereign Vector Resonator - Recursive ANN search with alpha blend"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_vector_resonator.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_vector_resonator")

    def resonate(self, vec_a: np.ndarray, vec_b: np.ndarray, alpha: float) -> np.ndarray:
        """Blend two vectors using alpha
        
        Args:
            vec_a, vec_b: Input vectors (float32)
            alpha: Blend factor (0.0 to 1.0)
        
        Returns:
            Blended vector
        """
        assert vec_a.shape == vec_b.shape
        assert vec_a.dtype == vec_b.dtype == np.float32
        
        output = np.zeros_like(vec_a)
        
        d_a = gpu_malloc(vec_a.nbytes)
        d_b = gpu_malloc(vec_b.nbytes)
        d_out = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_a, vec_a.ctypes.data_as(ctypes.c_void_p), vec_a.nbytes)
            memcpy_htod(d_b, vec_b.ctypes.data_as(ctypes.c_void_p), vec_b.nbytes)
            
            launch(
                self.kernel,
                grid=((len(vec_a) + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_a.value),
                    ctypes.c_uint64(d_b.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_uint32(len(vec_a)),
                    ctypes.c_float(alpha),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_out, output.nbytes)
            return output
        finally:
            gpu_free(d_a)
            gpu_free(d_b)
            gpu_free(d_out)


class GraphCrystallizer:
    """Sovereign Graph Crystallizer - Recursive GNN with EMA"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_graph_crystallizer.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_graph_crystallizer")

    def crystallize(self, nodes: np.ndarray, neighbors: np.ndarray, ema_rate: float = 0.999) -> np.ndarray:
        """Aggregate neighbor contributions with EMA
        
        Args:
            nodes: Current node values (float32)
            neighbors: Aggregated neighbor values (float32)
            ema_rate: EMA rate for stability (0.999 for TRM)
        
        Returns:
            Updated node values
        """
        assert nodes.shape == neighbors.shape
        assert nodes.dtype == neighbors.dtype == np.float32
        
        output = np.zeros_like(nodes)
        
        d_nodes = gpu_malloc(nodes.nbytes)
        d_neighbors = gpu_malloc(neighbors.nbytes)
        d_output = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_nodes, nodes.ctypes.data_as(ctypes.c_void_p), nodes.nbytes)
            memcpy_htod(d_neighbors, neighbors.ctypes.data_as(ctypes.c_void_p), neighbors.nbytes)
            
            launch(
                self.kernel,
                grid=((len(nodes) + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_nodes.value),
                    ctypes.c_uint64(d_neighbors.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(len(nodes)),
                    ctypes.c_float(ema_rate),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
            return output
        finally:
            gpu_free(d_nodes)
            gpu_free(d_neighbors)
            gpu_free(d_output)


class MultimodalHaltingGate:
    """Sovereign Multimodal Halting Gate - Geometry-aware halting"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_multimodal_halting_gate.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_multimodal_halting_gate")

    def check_halt(self, logits: np.ndarray, masks: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Check halting conditions with modality masks
        
        Args:
            logits: Halting logits (float32)
            masks: Modality bitmasks (uint32, 0=inactive)
            threshold: Halting threshold
        
        Returns:
            Halt flags (uint32: 1=continue, 0=halt)
        """
        assert logits.dtype == np.float32
        assert masks.dtype == np.uint32
        assert logits.shape == masks.shape
        
        output = np.zeros_like(masks)
        
        d_logits = gpu_malloc(logits.nbytes)
        d_masks = gpu_malloc(masks.nbytes)
        d_output = gpu_malloc(output.nbytes)
        
        try:
            memcpy_htod(d_logits, logits.ctypes.data_as(ctypes.c_void_p), logits.nbytes)
            memcpy_htod(d_masks, masks.ctypes.data_as(ctypes.c_void_p), masks.nbytes)
            
            launch(
                self.kernel,
                grid=((len(logits) + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_logits.value),
                    ctypes.c_uint64(d_masks.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(len(logits)),
                    ctypes.c_float(threshold),
                ],
            )
            synchronize()
            
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_output, output.nbytes)
            return output
        finally:
            gpu_free(d_logits)
            gpu_free(d_masks)
            gpu_free(d_output)


# Update __all__
__all__ = [
    # Kimi's
    "LatencyGuard",
    "ARCReasoner",
    "OOMSpillManager",
    # Qwen's
    "GalaxyResonanceEngine",
    # Deep Seek's
    "GeometryRouter",
    "FractalEmitter",
    # GLM's
    "ResonanceField",
    "AtomicFissionFusion",
    "TemporalReasoning",
    # Grok's
    "VectorResonator",
    "GraphCrystallizer",
    "MultimodalHaltingGate",
]
