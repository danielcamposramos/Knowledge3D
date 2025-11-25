"""
Sovereign Bridges - Pure ctypes + libcuda.so bridges for all Step8 kernels

This module provides Python bridges for all 15 Step8 kernels using the sovereign
loader (pure ctypes + CUDA Driver API). For the RPN bridge used in the hot
path we avoid NumPy entirely; other bridges may lazily import NumPy inside
helper functions when used outside the sovereignty‑critical loop.

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
    - No CuPy, no cuda-python in hot path
"""

import ctypes
from pathlib import Path
from typing import Tuple, Optional, Iterable, Sequence, List

from knowledge3d.cranium.sovereign.loader import (
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
    CUdeviceptr,
)
from knowledge3d.cranium.bridges.rpn_config import RPN_GRID_DIM, TIER2_BLOCK_DIM


def _np():
    """Lazy NumPy accessor for non‑hot‑path helpers."""
    import numpy as _numpy  # type: ignore[import]
    return _numpy

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
        self.timestamps = _np().zeros(2, dtype=_np().uint64)
        self.flag = _np().zeros(1, dtype=_np().uint32)

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

    def extract_rules(self, grid) -> Tuple[int, int, int]:
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
            np_mod = _np()
            output = np_mod.zeros(3, dtype=np_mod.int32)
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
        # Prepare stats input (uint64[2])
        StatsArray = ctypes.c_uint64 * 2
        stats = StatsArray(ctypes.c_uint64(oldest_index), ctypes.c_uint64(atom_size_bytes))
        OutputArray = ctypes.c_uint64 * 2
        output = OutputArray()

        # Allocate GPU memory
        d_stats = gpu_malloc(ctypes.sizeof(stats))
        d_output = gpu_malloc(ctypes.sizeof(output))

        try:
            # Copy stats to GPU
            memcpy_htod(d_stats, ctypes.cast(stats, ctypes.c_void_p), ctypes.sizeof(stats))

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
            memcpy_dtoh(ctypes.cast(output, ctypes.c_void_p), d_output, ctypes.sizeof(output))

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
        embeddings,
        latent,
        alpha: float = 0.5
    ):
        """Blend embeddings with latent state

        Args:
            embeddings: Input embeddings [batch_size, vector_dim]
            latent: Latent state [batch_size, vector_dim]
            alpha: Blend factor (0.0 to 1.0)

        Returns:
            output: Blended result [batch_size, vector_dim]
        """
        assert getattr(embeddings, "shape", None) == getattr(latent, "shape", None)
        batch_size, vector_dim = embeddings.shape

        # Compute element counts in bytes assuming float32 inputs
        elem_count = batch_size * vector_dim
        byte_count = elem_count * 4

        # Allocate GPU memory
        d_embeddings = gpu_malloc(byte_count)
        d_latent = gpu_malloc(byte_count)
        d_output = gpu_malloc(byte_count)

        try:
            # Copy inputs to GPU
            memcpy_htod(d_embeddings, embeddings.ctypes.data_as(ctypes.c_void_p), byte_count)
            memcpy_htod(d_latent, latent.ctypes.data_as(ctypes.c_void_p), byte_count)

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
            # Allocate host buffer and copy result back
            OutArray = ctypes.c_float * elem_count
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, byte_count)

            # Repackage into the same shape as embeddings using the caller's type
            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32).reshape(embeddings.shape)
            except Exception:
                # Fallback: return a nested Python list
                flat = [float(out_host[i]) for i in range(elem_count)]
                rows: List[List[float]] = []
                for b in range(batch_size):
                    start = b * vector_dim
                    rows.append(flat[start:start + vector_dim])
                return rows

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

    def route(self, input_data, shape_id: int):
        """Route and scale data based on media geometry
        
        Args:
            input_data: Input float32 array
            shape_id: 0=text, 1=image, 2=audio, 3=video, 4=mixed
        
        Returns:
            Scaled output array
        """
        vector_len = len(input_data)
        byte_count = vector_len * 4

        d_input = gpu_malloc(byte_count)
        d_output = gpu_malloc(byte_count)
        
        try:
            memcpy_htod(d_input, input_data.ctypes.data_as(ctypes.c_void_p), byte_count)
            
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
            
            OutArray = ctypes.c_float * vector_len
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, byte_count)

            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32).reshape(input_data.shape)
            except Exception:
                return [float(out_host[i]) for i in range(vector_len)]
        finally:
            gpu_free(d_input)
            gpu_free(d_output)


class FractalEmitter:
    """Sovereign Fractal Emitter - Knowledge Garden coordinate generation"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_fractal_emitter.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_fractal_emitter")

    def emit(self, atoms, base_scale: float = 1.0):
        """Generate fractal coordinates for atoms
        
        Args:
            atoms: Atom values (float32)
            base_scale: Coordinate scaling factor
        
        Returns:
            Coordinates array [count, 3] (x, y, z)
        """
        count = len(atoms)
        atoms_bytes = count * 4
        coords_bytes = count * 3 * 4

        d_atoms = gpu_malloc(atoms_bytes)
        d_coords = gpu_malloc(coords_bytes)
        
        try:
            memcpy_htod(d_atoms, atoms.ctypes.data_as(ctypes.c_void_p), atoms_bytes)
            
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
            
            CoordArray = ctypes.c_float * (count * 3)
            coords_host = CoordArray()
            memcpy_dtoh(ctypes.cast(coords_host, ctypes.c_void_p), d_coords, coords_bytes)

            try:
                np_mod = _np()
                return np_mod.asarray(coords_host, dtype=np_mod.float32).reshape((count, 3))
            except Exception:
                rows: List[List[float]] = []
                for i in range(count):
                    base = i * 3
                    rows.append(
                        [
                            float(coords_host[base]),
                            float(coords_host[base + 1]),
                            float(coords_host[base + 2]),
                        ]
                    )
                return rows
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

    def compute(self, positions, density):
        """Compute resonance strengths from positions and density
        
        Args:
            positions: Position array [count, 3] (x, y, z)
            density: Density values [count]
        
        Returns:
            Resonance strengths [count]
        """
        count = len(density)
        assert positions.shape == (count, 3)

        pos_bytes = count * 3 * 4
        density_bytes = count * 4
        output_bytes = count * 4

        d_positions = gpu_malloc(pos_bytes)
        d_density = gpu_malloc(density_bytes)
        d_output = gpu_malloc(output_bytes)
        
        try:
            memcpy_htod(d_positions, positions.ctypes.data_as(ctypes.c_void_p), pos_bytes)
            memcpy_htod(d_density, density.ctypes.data_as(ctypes.c_void_p), density_bytes)
            
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
            
            OutArray = ctypes.c_float * count
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, output_bytes)

            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32)
            except Exception:
                return [float(out_host[i]) for i in range(count)]
        finally:
            gpu_free(d_positions)
            gpu_free(d_density)
            gpu_free(d_output)


class AtomicFissionFusion:
    """Sovereign Atomic Fission/Fusion - Atom compress/expand operations"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_atomic_fission_fusion.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_atomic_fission_fusion")

    def transform(self, atoms, mode: int, ratio: float):
        """Transform atoms via fission or fusion
        
        Args:
            atoms: Atom values (float32)
            mode: 0=fusion (compress), 1=fission (expand)
            ratio: Transformation ratio
        
        Returns:
            Transformed atoms
        """
        count = len(atoms)
        byte_count = count * 4

        d_input = gpu_malloc(byte_count)
        d_output = gpu_malloc(byte_count)
        
        try:
            memcpy_htod(d_input, atoms.ctypes.data_as(ctypes.c_void_p), byte_count)
            
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

            OutArray = ctypes.c_float * count
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, byte_count)

            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32).reshape(atoms.shape)
            except Exception:
                return [float(out_host[i]) for i in range(count)]
        finally:
            gpu_free(d_input)
            gpu_free(d_output)

    def create_sparse(self, weights, sparsity_level: float, preserve_important: bool = True) -> dict:
        """Create sparse weight representation for efficient GPU computation.

        This method converts dense weights into sparse format, keeping only the most
        important values based on magnitude. Used for adaptive sparsity in thinking tags.

        Args:
            weights: Weight arrays (can be dict or ndarray)
            sparsity_level: Target sparsity (0.0 = dense, 1.0 = maximally sparse)
            preserve_important: If True, keep high-magnitude values

        Returns:
            Sparse weight dictionary with same keys as input
        """
        if isinstance(weights, dict):
            # Process each weight matrix
            sparse_dict = {}
            for key, W in weights.items():
                if not isinstance(W, np.ndarray):
                    sparse_dict[key] = W
                    continue

                W = W.astype(np.float32) if W.dtype != np.float32 else W

                if preserve_important:
                    # Keep top-k values by magnitude
                    threshold_percentile = sparsity_level * 100.0
                    threshold = np.percentile(np.abs(W), threshold_percentile)
                    sparse_W = np.where(np.abs(W) >= threshold, W, 0.0)
                else:
                    # Random sparsification
                    mask = np.random.rand(*W.shape) > sparsity_level
                    sparse_W = W * mask

                sparse_dict[key] = sparse_W.astype(np.float32)
            return sparse_dict

        elif isinstance(weights, np.ndarray):
            # Process single array
            W = weights.astype(np.float32) if weights.dtype != np.float32 else weights

            if preserve_important:
                threshold_percentile = sparsity_level * 100.0
                threshold = np.percentile(np.abs(W), threshold_percentile)
                return np.where(np.abs(W) >= threshold, W, 0.0).astype(np.float32)
            else:
                mask = np.random.rand(*W.shape) > sparsity_level
                return (W * mask).astype(np.float32)

        else:
            # Unknown type, return as-is
            return weights


class TemporalReasoning:
    """Sovereign Temporal Reasoning - Sequential delta computation"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_temporal_reasoning.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_temporal_reasoning")

    def compute_deltas(self, sequence):
        """Compute frame-to-frame deltas
        
        Args:
            sequence: Sequence array [sequence_length, feature_dim]
        
        Returns:
            Delta array [sequence_length, feature_dim]
        """
        assert len(sequence.shape) == 2

        seq_length, feat_dim = sequence.shape
        total = seq_length * feat_dim
        in_bytes = total * 4
        out_bytes = total * 4

        d_sequence = gpu_malloc(in_bytes)
        d_output = gpu_malloc(out_bytes)
        
        try:
            memcpy_htod(d_sequence, sequence.ctypes.data_as(ctypes.c_void_p), in_bytes)
            
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

            OutArray = ctypes.c_float * total
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, out_bytes)

            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32).reshape(sequence.shape)
            except Exception:
                rows: List[List[float]] = []
                for t in range(seq_length):
                    base = t * feat_dim
                    rows.append([float(out_host[base + j]) for j in range(feat_dim)])
                return rows
        finally:
            gpu_free(d_sequence)
            gpu_free(d_output)

    def compute_coherence(self, crystallized, temporal_context):
        """Compute temporal coherence scores.

        Measures how well the crystallized output aligns with temporal context.
        Used in thinking tag inference for coherence scoring.

        Args:
            crystallized: Crystallized output vector
            temporal_context: Temporal context vector

        Returns:
            Coherence scores (per dimension)
        """
        np_mod = _np()
        if not isinstance(crystallized, np_mod.ndarray):
            crystallized = np_mod.array(crystallized, dtype=np_mod.float32)
        if not isinstance(temporal_context, np_mod.ndarray):
            temporal_context = np_mod.array(temporal_context, dtype=np_mod.float32)

        # Ensure same shape for comparison
        if crystallized.shape != temporal_context.shape:
            # Broadcast or truncate to match
            min_len = min(len(crystallized.flatten()), len(temporal_context.flatten()))
            crystallized_flat = crystallized.flatten()[:min_len]
            context_flat = temporal_context.flatten()[:min_len]
        else:
            crystallized_flat = crystallized.flatten()
            context_flat = temporal_context.flatten()

        # Compute element-wise coherence (similarity measure)
        # High coherence when values are similar
        diff = np_mod.abs(crystallized_flat - context_flat)
        max_diff = np_mod.max(diff) if np_mod.max(diff) > 0 else 1.0
        coherence = 1.0 - (diff / max_diff)

        return coherence.astype(np_mod.float32)

    def estimate_coherence(self, context):
        """Estimate coherence from temporal context alone.

        Simplified version that estimates coherence without comparing to output.
        Useful for fallback paths.

        Args:
            context: Temporal context vector

        Returns:
            Estimated coherence scores
        """
        np_mod = _np()
        if not isinstance(context, np_mod.ndarray):
            context = np_mod.array(context, dtype=np_mod.float32)

        # Use temporal stability (low variance = high coherence)
        context_flat = context.flatten()
        if len(context_flat) > 1:
            variance = np_mod.var(context_flat)
            # Normalize variance to 0-1 range (assuming typical variance < 1.0)
            normalized_var = min(variance, 1.0)
            coherence_score = 1.0 - normalized_var
        else:
            coherence_score = 1.0

        # Return uniform coherence scores
        return np.full_like(context_flat, coherence_score, dtype=np.float32)


# ============================================================================
# Grok's Kernels
# ============================================================================

class VectorResonator:
    """Sovereign Vector Resonator - Recursive ANN search with alpha blend"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_vector_resonator.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_vector_resonator")

    def resonate(self, vec_a, vec_b, alpha: float):
        """Blend two vectors using alpha
        
        Args:
            vec_a, vec_b: Input vectors (float32)
            alpha: Blend factor (0.0 to 1.0)
        
        Returns:
            Blended vector
        """
        assert vec_a.shape == vec_b.shape
        length = len(vec_a)
        byte_count = length * 4

        d_a = gpu_malloc(byte_count)
        d_b = gpu_malloc(byte_count)
        d_out = gpu_malloc(byte_count)
        
        try:
            memcpy_htod(d_a, vec_a.ctypes.data_as(ctypes.c_void_p), byte_count)
            memcpy_htod(d_b, vec_b.ctypes.data_as(ctypes.c_void_p), byte_count)
            
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

            OutArray = ctypes.c_float * length
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_out, byte_count)

            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32).reshape(vec_a.shape)
            except Exception:
                return [float(out_host[i]) for i in range(length)]
        finally:
            gpu_free(d_a)
            gpu_free(d_b)
            gpu_free(d_out)

    def calculate_complexity(self, input_embedding, modal_signature: list) -> float:
        """Calculate input complexity for adaptive sparsity decisions.

        Uses vector magnitude and modal diversity as complexity indicators.
        This is a heuristic for determining whether to use sparse or dense operations.

        Args:
            input_embedding: Input vector (float32)
            modal_signature: List of modality names (e.g., ['text', 'image'])

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Normalize input embedding if needed
        np_mod = _np()
        if getattr(input_embedding, "dtype", None) != np_mod.float32:
            input_embedding = np_mod.asarray(input_embedding, dtype=np_mod.float32)

        # Calculate vector magnitude (normalized)
        magnitude = np_mod.linalg.norm(input_embedding)
        max_magnitude = np_mod.sqrt(len(input_embedding))  # Maximum possible for unit components
        normalized_magnitude = min(magnitude / max_magnitude, 1.0)

        # Calculate modal diversity score (more modalities = more complex)
        modal_diversity = len(set(modal_signature)) / 3.0  # Normalize by max 3 modalities
        modal_diversity = min(modal_diversity, 1.0)

        # Combine factors (weighted average)
        complexity = 0.7 * normalized_magnitude + 0.3 * modal_diversity

        return float(complexity)

    def cosine_similarity(self, vec_a, vec_b) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a, vec_b: Input vectors

        Returns:
            Cosine similarity (-1.0 to 1.0)
        """
        np_mod = _np()
        dot_product = np_mod.dot(vec_a.flatten(), vec_b.flatten())
        norm_a = np_mod.linalg.norm(vec_a)
        norm_b = np_mod.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def compute(self, confidence_vector):
        """Compute confidence rays from crystallized output.

        This is used in thinking tag inference to generate per-tag confidence scores.

        Args:
            confidence_vector: Crystallized output vector

        Returns:
            Confidence scores (one per dimension)
        """
        # Sigmoid activation for confidence scores
        np_mod = _np()
        vec = np_mod.asarray(confidence_vector, dtype=np_mod.float32)
        return 1.0 / (1.0 + np_mod.exp(-vec))


class GraphCrystallizer:
    """Sovereign Graph Crystallizer - Recursive GNN with EMA"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_graph_crystallizer.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_graph_crystallizer")

    def crystallize(self, nodes, neighbors, ema_rate: float = 0.999):
        """Aggregate neighbor contributions with EMA
        
        Args:
            nodes: Current node values (float32)
            neighbors: Aggregated neighbor values (float32)
            ema_rate: EMA rate for stability (0.999 for TRM)
        
        Returns:
            Updated node values
        """
        assert nodes.shape == neighbors.shape

        length = len(nodes)
        byte_count = length * 4

        d_nodes = gpu_malloc(byte_count)
        d_neighbors = gpu_malloc(byte_count)
        d_output = gpu_malloc(byte_count)
        
        try:
            memcpy_htod(d_nodes, nodes.ctypes.data_as(ctypes.c_void_p), byte_count)
            memcpy_htod(d_neighbors, neighbors.ctypes.data_as(ctypes.c_void_p), byte_count)
            
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

            OutArray = ctypes.c_float * length
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, byte_count)

            try:
                np_mod = _np()
                return np_mod.asarray(out_host, dtype=np_mod.float32).reshape(nodes.shape)
            except Exception:
                return [float(out_host[i]) for i in range(length)]
        finally:
            gpu_free(d_nodes)
            gpu_free(d_neighbors)
            gpu_free(d_output)

    def smooth_intermediate(self, output, ema_buffer, warp_level: bool = True):
        """Smooth intermediate outputs using EMA buffer.

        This is used in thinking tag inference for dynamic crystallization.
        Applies EMA-based smoothing to reduce high-frequency noise.

        Args:
            output: Intermediate output vector
            ema_buffer: GPU buffer containing EMA state
            warp_level: If True, use warp-level synchronization

        Returns:
            Smoothed output vector
        """
        np_mod = _np()
        if not isinstance(output, np_mod.ndarray):
            output = np_mod.array(output, dtype=np_mod.float32)
        elif output.dtype != np_mod.float32:
            output = output.astype(np_mod.float32)

        # For now, use simple EMA on CPU (can be optimized with GPU kernel later)
        # This maintains the interface while providing functional smoothing
        alpha = 0.999 if warp_level else 0.99

        # Read current EMA state from GPU buffer
        ema_state = np_mod.zeros_like(output)
        if ema_buffer is not None and hasattr(ema_buffer, 'value'):
            try:
                from knowledge3d.cranium.sovereign.loader import memcpy_dtoh
                import ctypes
                memcpy_dtoh(ema_state.ctypes.data_as(ctypes.c_void_p), ema_buffer, output.nbytes)
            except:
                pass  # First call, EMA state is zeros

        # Apply EMA: new_state = alpha * old_state + (1 - alpha) * new_value
        smoothed = alpha * ema_state + (1.0 - alpha) * output

        # Write updated EMA state back to GPU buffer
        if ema_buffer is not None and hasattr(ema_buffer, 'value'):
            try:
                from knowledge3d.cranium.sovereign.loader import memcpy_htod
                import ctypes
                memcpy_htod(ema_buffer, smoothed.ctypes.data_as(ctypes.c_void_p), smoothed.nbytes)
            except:
                pass

        return smoothed

    def apply(self, output, ema_buffer):
        """Alias for smooth_intermediate() with default parameters.

        Args:
            output: Intermediate output vector
            ema_buffer: GPU buffer containing EMA state

        Returns:
            Smoothed output vector
        """
        return self.smooth_intermediate(output, ema_buffer, warp_level=True)


class MultimodalHaltingGate:
    """Sovereign Multimodal Halting Gate - Geometry-aware halting"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_multimodal_halting_gate.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_multimodal_halting_gate")

    def check_halt(self, logits, masks, threshold: float = 0.5):
        """Check halting conditions with modality masks
        
        Args:
            logits: Halting logits (float32)
            masks: Modality bitmasks (uint32, 0=inactive)
            threshold: Halting threshold
        
        Returns:
            Halt flags (uint32: 1=continue, 0=halt)
        """
        np_mod = _np()
        assert logits.dtype == np_mod.float32
        assert masks.dtype == np_mod.uint32
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


class ModularRPNEngine:
    """Sovereign Modular RPN Engine - GPU-native RPN execution

    Uses modular_rpn_kernel.ptx for geometric and semantic computations.
    Supports 15 parallel instances with 64-deep stacks (float4 elements).

    Operations:
        - Literals: scalar (op 0), vector (op 1)
        - Arithmetic: add(10), sub(11), mul(12), div(13), pow(14), neg(15)
        - Advanced: sqrt(20), exp(21), log(22), sin(24), cos(25), tan(26)
        - Comparison: gt(40), lt(42), eq(44), max(46), min(47)
        - Stack: dup(50), swap(51), drop(52), over(53), rot(54), clear(55)
        - Vector: dot(60), cross(61), mag(62), norm(63), rotate(70), scale(71), translate(72)
        - Conditional: ifelse(80)

    Example:
        engine = ModularRPNEngine()
        result = engine.execute_single(
            instance_id=0,
            op_codes=np.array([0, 0, 10], dtype=np.uint16),  # push 2.0, push 3.0, add
            scalars=np.array([2.0, 3.0, 0.0], dtype=np.float32),
            vectors=np.zeros((3, 3), dtype=np.float32)
        )
        # result = 5.0
    """

    MAX_INSTANCES = 18  # Tesla 3-6-9: 18/3=6 (ternary resonance)
    STACK_DEPTH = 69    # Tesla 6-9: 6+9=15→6, 6×9=54→9, Yin-Yang balance
    INSTANCE_STRIDE = 1040  # bytes per instance state

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"RPN PTX kernel not found: {ptx_path}")

        self.kernel = load_ptx_file(str(ptx_path), "modular_rpn_geometric_kernel")
        self.extract_kernel = load_ptx_file(str(ptx_path), "modular_rpn_extract_top")

        # Allocate persistent state buffer (18 instances × 1040 bytes, Tesla 3-6-9 resonance)
        total_bytes = self.MAX_INSTANCES * self.INSTANCE_STRIDE
        self.d_state = gpu_malloc(total_bytes)

        # Zero-initialize state buffer using ctypes (no NumPy)
        ZerosArray = ctypes.c_uint8 * total_bytes
        zeros = ZerosArray()
        memcpy_htod(self.d_state, ctypes.cast(zeros, ctypes.c_void_p), total_bytes)

    def execute_single(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Sequence[float],
        vectors: Sequence[Sequence[float]],
    ) -> float:
        """Execute single RPN program on specified instance

        Args:
            instance_id: Instance slot (0-14)
            op_codes: RPN operation codes (uint16 array)
            scalars: Scalar literal pool (float32 array)
            vectors: Vector literal pool (float32 array, shape N×3)

        Returns:
            Result from top of stack (float32 scalar)
        """
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id} (must be 0-14)")

        # Prepare inputs as ctypes arrays (no NumPy on hot path)
        op_list = [int(o) for o in op_codes]
        OpArray = ctypes.c_uint16 * len(op_list)
        op_arr = OpArray(*op_list)

        scalar_list = [float(s) for s in scalars]
        ScalarArray = ctypes.c_float * len(scalar_list) if scalar_list else ctypes.c_float * 1
        scalar_arr = ScalarArray(*scalar_list) if scalar_list else None

        flat_vec: List[float] = [float(c) for vec in vectors for c in vec]
        VecArray = ctypes.c_float * len(flat_vec) if flat_vec else ctypes.c_float * 1
        vec_arr = VecArray(*flat_vec) if flat_vec else None

        # Allocate GPU memory
        d_op_codes = gpu_malloc(ctypes.sizeof(op_arr))
        d_scalars = gpu_malloc(ctypes.sizeof(scalar_arr)) if scalar_arr is not None else None
        d_vectors = gpu_malloc(ctypes.sizeof(vec_arr)) if vec_arr is not None else None

        try:
            # Copy inputs to GPU
            memcpy_htod(d_op_codes, ctypes.cast(op_arr, ctypes.c_void_p), ctypes.sizeof(op_arr))
            if d_scalars is not None and scalar_arr is not None:
                memcpy_htod(d_scalars, ctypes.cast(scalar_arr, ctypes.c_void_p), ctypes.sizeof(scalar_arr))
            if d_vectors is not None and vec_arr is not None and len(flat_vec):
                memcpy_htod(d_vectors, ctypes.cast(vec_arr, ctypes.c_void_p), ctypes.sizeof(vec_arr))

            # Launch kernel
            launch(
                self.kernel,
                grid=(RPN_GRID_DIM, 1, 1),
                block=(TIER2_BLOCK_DIM, 1, 1),
                params=[
                    ctypes.c_uint32(instance_id),
                    ctypes.c_uint64(d_op_codes.value),
                    ctypes.c_uint64(d_scalars.value if d_scalars is not None else 0),
                    ctypes.c_uint64(d_vectors.value if d_vectors is not None else 0),
                    ctypes.c_uint64(self.d_state.value),
                    ctypes.c_uint32(len(op_codes)),
                ],
            )
            synchronize()

            # Read result from instance stack (top element)
            # Stack layout: header (16 bytes: head, size, error, reserved) + stack[64] (64 × 16 bytes of float4)
            instance_offset = instance_id * self.INSTANCE_STRIDE

            # First, read head and size to find stack top
            HeaderArray = ctypes.c_uint32 * 4
            header_bytes = HeaderArray()
            memcpy_dtoh(
                ctypes.cast(header_bytes, ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + instance_offset),
                16,
            )

            head = int(header_bytes[0])
            size = int(header_bytes[1])
            error_code = int(header_bytes[2])

            if error_code != 0:
                raise RuntimeError(f"RPN execution error: code {error_code}")

            if size == 0:
                raise RuntimeError("RPN stack underflow - no result available")

            # Calculate position of top element
            # Stack top is at (head + size - 1) & 63
            stack_top_index = (head + size - 1) & 63

            # Read float4 from stack[stack_top_index]
            stack_base_offset = instance_offset + 16
            element_offset = stack_base_offset + (stack_top_index * 16)  # 16 bytes per float4

            ResultArray = ctypes.c_float * 4
            result_bytes = ResultArray()
            memcpy_dtoh(
                ctypes.cast(result_bytes, ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + element_offset),
                16,
            )

            return float(result_bytes[0])

        finally:
            gpu_free(d_op_codes)
            if d_scalars is not None:
                gpu_free(d_scalars)
            if d_vectors is not None:
                gpu_free(d_vectors)

    def execute_batch(
        self,
        programs: List[dict],
        max_instances: int = 15
    ) -> List[float]:
        """Execute batch of RPN programs in parallel across instances

        Args:
            programs: List of dicts with keys 'op_codes', 'scalars', 'vectors'
            max_instances: Max parallel instances (default 15)

        Returns:
            List of results (length = len(programs))
        """
        results: List[float] = []

        # Process in batches of max_instances
        for batch_start in range(0, len(programs), max_instances):
            batch = programs[batch_start:batch_start + max_instances]

            # Execute programs sequentially (kernel is single-threaded per instance)
            for i, program in enumerate(batch):
                result = self.execute_single(
                    instance_id=i,
                    op_codes=program["op_codes"],
                    scalars=program["scalars"],
                    vectors=program["vectors"],
                )
                results.append(result)

        return results

    def execute_batch_device(
        self,
        programs: List[dict],
    ) -> tuple[CUdeviceptr, int]:
        """Execute batch of RPN programs and write results to a device buffer.

        Returns (device_pointer, count). Caller owns the device buffer and must free it.
        """
        count = len(programs)
        if count == 0:
            return CUdeviceptr(0), 0

        d_out = gpu_malloc(count * 4)

        # Process sequentially per instance slot (reusing instance 0..MAX_INSTANCES-1)
        for i, program in enumerate(programs):
            instance_id = i % self.MAX_INSTANCES
            op_list = [int(o) for o in program["op_codes"]]
            OpArray = ctypes.c_uint16 * len(op_list)
            op_arr = OpArray(*op_list)

            scalar_list = [float(s) for s in program["scalars"]]
            ScalarArray = ctypes.c_float * len(scalar_list) if scalar_list else ctypes.c_float * 1
            scalar_arr = ScalarArray(*scalar_list) if scalar_list else None

            flat_vec: List[float] = [float(c) for vec in program["vectors"] for c in vec]
            VecArray = ctypes.c_float * len(flat_vec) if flat_vec else ctypes.c_float * 1
            vec_arr = VecArray(*flat_vec) if flat_vec else None

            d_op_codes = gpu_malloc(ctypes.sizeof(op_arr))
            d_scalars = gpu_malloc(ctypes.sizeof(scalar_arr)) if scalar_arr is not None else None
            d_vectors = gpu_malloc(ctypes.sizeof(vec_arr)) if vec_arr is not None else None

            try:
                memcpy_htod(d_op_codes, ctypes.cast(op_arr, ctypes.c_void_p), ctypes.sizeof(op_arr))
                if d_scalars is not None and scalar_arr is not None:
                    memcpy_htod(d_scalars, ctypes.cast(scalar_arr, ctypes.c_void_p), ctypes.sizeof(scalar_arr))
                if d_vectors is not None and vec_arr is not None and flat_vec:
                    memcpy_htod(d_vectors, ctypes.cast(vec_arr, ctypes.c_void_p), ctypes.sizeof(vec_arr))

                launch(
                    self.kernel,
                    grid=(RPN_GRID_DIM, 1, 1),
                    block=(TIER2_BLOCK_DIM, 1, 1),
                    params=[
                        ctypes.c_uint32(instance_id),
                        ctypes.c_uint64(d_op_codes.value),
                        ctypes.c_uint64(d_scalars.value if d_scalars is not None else 0),
                        ctypes.c_uint64(d_vectors.value if d_vectors is not None else 0),
                        ctypes.c_uint64(self.d_state.value),
                        ctypes.c_uint32(len(op_codes)),
                    ],
                )
                launch(
                    self.extract_kernel,
                    grid=(1, 1, 1),
                    block=(1, 1, 1),
                    params=[
                        ctypes.c_uint32(instance_id),
                        ctypes.c_uint64(self.d_state.value),
                        ctypes.c_uint64(d_out.value),
                        ctypes.c_uint32(i),
                    ],
                )
            finally:
                gpu_free(d_op_codes)
                if d_scalars is not None:
                    gpu_free(d_scalars)
                if d_vectors is not None:
                    gpu_free(d_vectors)

        synchronize()
        return d_out, count

    def reset_instance(self, instance_id: int):
        """Reset instance state (clear stack, reset head/size)"""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id}")

        # Zero out instance state
        instance_offset = instance_id * self.INSTANCE_STRIDE
        ZerosArray = ctypes.c_uint8 * self.INSTANCE_STRIDE
        zeros = ZerosArray()
        memcpy_htod(
            ctypes.c_void_p(self.d_state.value + instance_offset),
            ctypes.cast(zeros, ctypes.c_void_p),
            self.INSTANCE_STRIDE,
        )

    def cleanup(self):
        """Free GPU memory"""
        gpu_free(self.d_state)

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


class GalaxyMemoryUpdater:
    """Sovereign Galaxy Memory Updater - Blend embeddings on GPU

    Uses galaxy_memory_updater.ptx to blend old and teacher embeddings
    with exponential moving average (EMA) on GPU.

    Formula: new = old * (1 - blend_factor) + teacher * blend_factor

    Example:
        updater = GalaxyMemoryUpdater()
        old_emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        teacher_emb = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        new_emb = updater.blend(old_emb, teacher_emb, blend_factor=0.3)
        # new_emb ≈ [1.9, 2.9, 3.9]
    """

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "galaxy_memory_updater.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Galaxy memory PTX kernel not found: {ptx_path}")

        self.kernel = load_ptx_file(str(ptx_path), "update_star_embedding_kernel")

    def blend(self, old, teacher, blend_factor: float):
        """Blend old and teacher embeddings with GPU acceleration.

        Args:
            old: Old embedding (float32 array)
            teacher: Teacher embedding (float32 array, same shape as old)
            blend_factor: Blend factor (0.0 = keep old, 1.0 = use teacher)

        Returns:
            Blended embedding (float32 array, same shape as inputs)
        """
        # Prepare inputs
        np_mod = _np()
        old_arr = np_mod.ascontiguousarray(np_mod.asarray(old, dtype=np_mod.float32).flatten())
        teacher_arr = np_mod.ascontiguousarray(np_mod.asarray(teacher, dtype=np_mod.float32).flatten())

        if old_arr.shape != teacher_arr.shape:
            raise ValueError(f"Shape mismatch: old {old_arr.shape} vs teacher {teacher_arr.shape}")

        dim = int(old_arr.size)
        if dim == 0:
            return np_mod.array([], dtype=np_mod.float32)

        # Allocate GPU memory
        d_old = gpu_malloc(old_arr.nbytes)
        d_teacher = gpu_malloc(teacher_arr.nbytes)
        d_out = gpu_malloc(old_arr.nbytes)

        try:
            # Copy inputs to GPU
            memcpy_htod(d_old, old_arr.ctypes.data_as(ctypes.c_void_p), old_arr.nbytes)
            memcpy_htod(d_teacher, teacher_arr.ctypes.data_as(ctypes.c_void_p), teacher_arr.nbytes)

            # Launch kernel
            threads = 256
            blocks = (dim + threads - 1) // threads

            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_old.value),
                    ctypes.c_uint64(d_teacher.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_float(blend_factor),
                    ctypes.c_uint32(dim),
                ],
            )
            synchronize()

            # Copy result back
            output = np_mod.zeros_like(old_arr)
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_out, output.nbytes)

            return output.reshape(old_arr.shape)

        finally:
            gpu_free(d_old)
            gpu_free(d_teacher)
            gpu_free(d_out)

    def blend_sequence(
        self,
        base,
        teachers: list,
        blend_factor: float = 0.3
    ):
        """Blend base embedding with sequence of teacher embeddings.

        Args:
            base: Base embedding (float32 array)
            teachers: List of teacher embeddings
            blend_factor: Blend factor for each step

        Returns:
            Final blended embedding
        """
        np_mod = _np()
        out = np_mod.array(base, dtype=np_mod.float32)
        if not teachers:
            return out

        for teacher in teachers:
            out = self.blend(out, np_mod.array(teacher, dtype=np_mod.float32), blend_factor)

        return out


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
    # Runtime Engines
    "ModularRPNEngine",
    "GalaxyMemoryUpdater",
    # GLM's World Model
    "WorldModelBridge",
]


class WorldModelBridge:
    """
    Sovereign bridge for world model operations.
    Enables temporal coherence, multi-modal fusion, and dynamic mesh generation.
    
    GLM's World Model Integration - Multi-modal temporal generation.
    """
    def __init__(self):
        from pathlib import Path
        ptx_dir = Path(__file__).parent.parent / "ptx"
        
        # Load world model kernels
        self.temporal_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "compute_temporal_coherence"
        )
        self.fusion_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "fuse_multimodal_features"
        )
        self.prediction_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "predict_world_state"
        )
        self.dynamic_mesh_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "generate_dynamic_mesh"
        )
        self.resonance_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "enhance_galaxy_resonance"
        )
    
    def compute_temporal_coherence(
        self,
        frame_features,  # (N_frames * feature_dim,) flattened
        n_frames: int,
        feature_dim: int
    ):
        """Compute temporal coherence scores across video frames."""
        # Allocate GPU memory
        d_features = gpu_malloc(frame_features.nbytes)
        d_coherence = gpu_malloc(feature_dim * 4)  # float32
        
        try:
            # Copy to GPU
            memcpy_htod(d_features, frame_features.ctypes.data_as(ctypes.c_void_p), frame_features.nbytes)
            
            # Launch kernel
            threads = 256
            blocks = (feature_dim + threads - 1) // threads
            
            launch(
                self.temporal_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_features.value),
                    ctypes.c_uint64(d_coherence.value),
                    ctypes.c_int32(n_frames),
                    ctypes.c_int32(feature_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            np_mod = _np()
            coherence = np_mod.zeros(feature_dim, dtype=np_mod.float32)
            memcpy_dtoh(coherence.ctypes.data_as(ctypes.c_void_p), d_coherence, coherence.nbytes)
            
            return coherence
        
        finally:
            gpu_free(d_features)
            gpu_free(d_coherence)
    
    def fuse_multimodal_features(
        self,
        text_features,
        visual_features,
        text_weight: float = 0.5
    ):
        """Fuse text and visual features with attention weighting."""
        feature_dim = len(text_features)
        visual_weight = 1.0 - text_weight
        
        # Allocate GPU memory
        d_text = gpu_malloc(text_features.nbytes)
        d_visual = gpu_malloc(visual_features.nbytes)
        d_weights = gpu_malloc(8)  # 2 floats
        d_fused = gpu_malloc(text_features.nbytes)
        
        try:
            # Copy to GPU
            memcpy_htod(d_text, text_features.ctypes.data_as(ctypes.c_void_p), text_features.nbytes)
            memcpy_htod(d_visual, visual_features.ctypes.data_as(ctypes.c_void_p), visual_features.nbytes)
            
            np_mod = _np()
            weights = np_mod.array([text_weight, visual_weight], dtype=np_mod.float32)
            memcpy_htod(d_weights, weights.ctypes.data_as(ctypes.c_void_p), weights.nbytes)
            
            # Launch kernel
            threads = 256
            blocks = (feature_dim + threads - 1) // threads
            
            launch(
                self.fusion_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_text.value),
                    ctypes.c_uint64(d_visual.value),
                    ctypes.c_uint64(d_weights.value),
                    ctypes.c_uint64(d_fused.value),
                    ctypes.c_int32(feature_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            fused = np_mod.zeros_like(text_features)
            memcpy_dtoh(fused.ctypes.data_as(ctypes.c_void_p), d_fused, fused.nbytes)
            
            return fused
        
        finally:
            gpu_free(d_text)
            gpu_free(d_visual)
            gpu_free(d_weights)
            gpu_free(d_fused)
    
    def predict_world_state(
        self,
        current_state,
        action_vector
    ):
        """Predict next world state given current state and action."""
        state_dim = len(current_state)
        action_dim = len(action_vector)
        
        # Allocate GPU memory
        d_current = gpu_malloc(current_state.nbytes)
        d_action = gpu_malloc(action_vector.nbytes)
        d_predicted = gpu_malloc(current_state.nbytes)
        
        try:
            # Copy to GPU
            memcpy_htod(d_current, current_state.ctypes.data_as(ctypes.c_void_p), current_state.nbytes)
            memcpy_htod(d_action, action_vector.ctypes.data_as(ctypes.c_void_p), action_vector.nbytes)
            
            # Launch kernel
            threads = 256
            blocks = (state_dim + threads - 1) // threads
            
            launch(
                self.prediction_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_current.value),
                    ctypes.c_uint64(d_action.value),
                    ctypes.c_uint64(d_predicted.value),
                    ctypes.c_int32(state_dim),
                    ctypes.c_int32(action_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            np_mod = _np()
            predicted = np_mod.zeros_like(current_state)
            memcpy_dtoh(predicted.ctypes.data_as(ctypes.c_void_p), d_predicted, predicted.nbytes)
            
            return predicted
        
        finally:
            gpu_free(d_current)
            gpu_free(d_action)
            gpu_free(d_predicted)
    
    def generate_dynamic_mesh(
        self,
        world_state,
        base_vertices  # (N, 3)
    ):
        """Generate dynamic mesh based on world state."""
        vertex_count = len(base_vertices)
        state_dim = len(world_state)
        
        # Flatten vertices
        np_mod = _np()
        vertices_flat = base_vertices.flatten().astype(np_mod.float32)
        
        # Allocate GPU memory
        d_state = gpu_malloc(world_state.nbytes)
        d_base = gpu_malloc(vertices_flat.nbytes)
        d_dynamic = gpu_malloc(vertices_flat.nbytes)
        
        try:
            # Copy to GPU
            memcpy_htod(d_state, world_state.ctypes.data_as(ctypes.c_void_p), world_state.nbytes)
            memcpy_htod(d_base, vertices_flat.ctypes.data_as(ctypes.c_void_p), vertices_flat.nbytes)
            
            # Launch kernel
            threads = 256
            blocks = (vertex_count + threads - 1) // threads
            
            launch(
                self.dynamic_mesh_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_state.value),
                    ctypes.c_uint64(d_base.value),
                    ctypes.c_uint64(d_dynamic.value),
                    ctypes.c_int32(vertex_count),
                    ctypes.c_int32(state_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            dynamic_flat = np_mod.zeros_like(vertices_flat)
            memcpy_dtoh(dynamic_flat.ctypes.data_as(ctypes.c_void_p), d_dynamic, dynamic_flat.nbytes)
            
            return dynamic_flat.reshape(base_vertices.shape)
        
        finally:
            gpu_free(d_state)
            gpu_free(d_base)
            gpu_free(d_dynamic)
    
    def enhance_galaxy_resonance(
        self,
        query_embedding,
        galaxy_embeddings  # (N, embedding_dim)
    ):
        """Enhance galaxy query with temperature-scaled similarity."""
        n_embeddings = galaxy_embeddings.shape[0]
        embedding_dim = galaxy_embeddings.shape[1]
        
        # Flatten galaxy embeddings
        galaxy_flat = galaxy_embeddings.flatten().astype(np.float32)
        
        # Allocate GPU memory
        d_query = gpu_malloc(query_embedding.nbytes)
        d_galaxy = gpu_malloc(galaxy_flat.nbytes)
        d_resonance = gpu_malloc(n_embeddings * 4)  # float32
        
        try:
            # Copy to GPU
            memcpy_htod(d_query, query_embedding.ctypes.data_as(ctypes.c_void_p), query_embedding.nbytes)
            memcpy_htod(d_galaxy, galaxy_flat.ctypes.data_as(ctypes.c_void_p), galaxy_flat.nbytes)
            
            # Launch kernel
            threads = 256
            blocks = (n_embeddings + threads - 1) // threads
            
            launch(
                self.resonance_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_query.value),
                    ctypes.c_uint64(d_galaxy.value),
                    ctypes.c_uint64(d_resonance.value),
                    ctypes.c_int32(n_embeddings),
                    ctypes.c_int32(embedding_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            resonance = np.zeros(n_embeddings, dtype=np.float32)
            memcpy_dtoh(resonance.ctypes.data_as(ctypes.c_void_p), d_resonance, resonance.nbytes)
            
            return resonance
        
        finally:
            gpu_free(d_query)
            gpu_free(d_galaxy)
            gpu_free(d_resonance)


# ============================================================================
# Trit Overlay + Inspector (Balanced Ternary Diagnostics)
# ============================================================================


class TritOverlayGenerator:
    """Generate RGBA8 overlays from packed ternary fields."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "trit_overlay_generator.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "trit_overlay_generator")
        self.guard = LatencyGuard(threshold_us=500.0)

    def generate(
        self,
        trits_packed,
        grid_shape: Tuple[int, int, int],
        field_stride: int,
        field_type: int = 0,
        threshold: float = 0.0,
    ):
        """Render ternary field overlay to RGBA8."""
        gx, gy, gz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        np_mod = _np()
        trits = np_mod.ascontiguousarray(trits_packed, dtype=np_mod.uint32)
        rgba = np_mod.zeros((gx * gy * gz * 4,), dtype=np_mod.uint8)

        d_trits = gpu_malloc(trits.nbytes)
        d_rgba = gpu_malloc(rgba.nbytes)
        try:
            memcpy_htod(d_trits, trits.ctypes.data_as(ctypes.c_void_p), trits.nbytes)
            self.guard.start()
            launch(
                self.kernel,
                grid=(
                    (gx + 7) // 8,
                    (gy + 7) // 8,
                    (gz + 7) // 8,
                ),
                block=(8, 8, 8),
                params=[
                    ctypes.c_uint64(d_trits.value),
                    ctypes.c_uint64(d_rgba.value),
                    ctypes.c_int32(gx),
                    ctypes.c_int32(gy),
                    ctypes.c_int32(gz),
                    ctypes.c_int32(int(field_stride)),
                    ctypes.c_int32(int(field_type)),
                    ctypes.c_float(float(threshold)),
                ],
            )
            synchronize()
            self.guard.stop()
            memcpy_dtoh(rgba.ctypes.data_as(ctypes.c_void_p), d_rgba, rgba.nbytes)
            return rgba
        finally:
            gpu_free(d_trits)
            gpu_free(d_rgba)


class TritInspectorBridge:
    """Inspect packed ternary fields for specific nodes."""

    _dtype = None

    @classmethod
    def _ensure_dtype(cls):
        if cls._dtype is None:
            np_mod = _np()
            cls._dtype = np_mod.dtype(
                [
                    ("count", np_mod.int32),
                    ("sum", np_mod.int32),
                    ("mean", np_mod.float32),
                    ("var", np_mod.float32),
                    ("bottlenecks", np_mod.int32),
                ]
            )
        return cls._dtype

    def __init__(self):
        ptx_path = KERNELS_DIR / "trit_inspector.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "trit_inspector")
        self.guard = LatencyGuard(threshold_us=500.0)

    def inspect(
        self,
        trits_packed,
        node_indices,
        field_stride: int,
    ):
        """Inspect ternary fields at node_indices."""
        trits = np.ascontiguousarray(trits_packed, dtype=np.uint32)
        nodes = np.ascontiguousarray(node_indices, dtype=np.int32)
        n = int(nodes.shape[0])
        np_mod = _np()
        out = np_mod.zeros(n, dtype=self._ensure_dtype())

        d_trits = gpu_malloc(trits.nbytes)
        d_nodes = gpu_malloc(nodes.nbytes)
        d_out = gpu_malloc(out.nbytes)
        try:
            memcpy_htod(d_trits, trits.ctypes.data_as(ctypes.c_void_p), trits.nbytes)
            memcpy_htod(d_nodes, nodes.ctypes.data_as(ctypes.c_void_p), nodes.nbytes)
            self.guard.start()
            threads = 128
            blocks = (n + threads - 1) // threads
            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_trits.value),
                    ctypes.c_uint64(d_nodes.value),
                    ctypes.c_int32(n),
                    ctypes.c_int32(int(field_stride)),
                    ctypes.c_uint64(d_out.value),
                ],
            )
            synchronize()
            self.guard.stop()
            memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            gpu_free(d_trits)
            gpu_free(d_nodes)
            gpu_free(d_out)


class TernaryDepthField:
    """Compute ternary attract/neutral/repel field for a query embedding."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "ternary_depth_field.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "ternary_depth_field")
        self.guard = LatencyGuard(threshold_us=500.0)

    def compute(
        self,
        embeddings,
        query,
        attract_thresh: float = 0.35,
        repel_thresh: float = -0.05,
    ):
        """Return packed 2-bit trits indicating near/neutral/far (per node)."""
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        q = np.ascontiguousarray(query, dtype=np.float32)
        n_nodes, dim = emb.shape
        assert q.shape[0] == dim, "query dim mismatch"
        n_words = (n_nodes + 15) // 16
        out_host = np.zeros(n_words, dtype=np.uint32)

        d_emb = gpu_malloc(emb.nbytes)
        d_query = gpu_malloc(q.nbytes)
        d_out = gpu_malloc(out_host.nbytes)
        try:
            memcpy_htod(d_emb, emb.ctypes.data_as(ctypes.c_void_p), emb.nbytes)
            memcpy_htod(d_query, q.ctypes.data_as(ctypes.c_void_p), q.nbytes)
            # zero output buffer (host prepared zeros)
            memcpy_htod(d_out, out_host.ctypes.data_as(ctypes.c_void_p), out_host.nbytes)
            self.guard.start()
            threads = 128
            blocks = (n_nodes + threads - 1) // threads
            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_emb.value),
                    ctypes.c_uint64(d_query.value),
                    ctypes.c_int32(int(n_nodes)),
                    ctypes.c_int32(int(dim)),
                    ctypes.c_float(float(attract_thresh)),
                    ctypes.c_float(float(repel_thresh)),
                    ctypes.c_uint64(d_out.value),
                ],
            )
            synchronize()
            self.guard.stop()
            memcpy_dtoh(out_host.ctypes.data_as(ctypes.c_void_p), d_out, out_host.nbytes)
            return out_host
        finally:
            gpu_free(d_emb)
            gpu_free(d_query)
            gpu_free(d_out)


class TernaryPruneDecision:
    """Map scores to ternary keep/discard signals on GPU."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "ternary_prune_decision.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "ternary_prune_decision")
        self.guard = LatencyGuard(threshold_us=500.0)

    def decide(
        self,
        scores,
        keep_thresh: float = 0.5,
        drop_thresh: float = 0.05,
    ):
        np_mod = _np()
        scores_np = np_mod.ascontiguousarray(scores, dtype=np_mod.float32)
        n = int(scores_np.shape[0])
        out = np.zeros(n, dtype=np.int8)
        d_scores = gpu_malloc(scores_np.nbytes)
        d_out = gpu_malloc(out.nbytes)
        try:
            memcpy_htod(d_scores, scores_np.ctypes.data_as(ctypes.c_void_p), scores_np.nbytes)
            self.guard.start()
            threads = 256
            blocks = (n + threads - 1) // threads
            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int32(n),
                    ctypes.c_float(float(keep_thresh)),
                    ctypes.c_float(float(drop_thresh)),
                ],
            )
            synchronize()
            self.guard.stop()
            memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            gpu_free(d_scores)
            gpu_free(d_out)


class TernaryAttentionMask:
    """Compute ternary attention masks (packed 2-bit trits) from Q·K."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "ternary_attention_mask.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "ternary_attention_mask")
        self.threshold_kernel = load_ptx_file(str(ptx_path), "compute_adaptive_thresholds")
        self.guard = LatencyGuard(threshold_us=500.0)

    def compute(
        self,
        Q,
        K,
        attract_thresh: float,
        repel_thresh: float,
    ):
        """Return packed ternary masks for Q·K."""
        if Q.shape != K.shape:
            raise ValueError(f"Q and K must match; got {Q.shape} vs {K.shape}")
        np_mod = _np()
        batch_size, seq_len, embed_dim = Q.shape
        n_words = (seq_len * seq_len + 15) // 16

        q = np_mod.ascontiguousarray(Q, dtype=np_mod.float32)
        k = np_mod.ascontiguousarray(K, dtype=np_mod.float32)
        masks = np_mod.zeros((batch_size, n_words), dtype=np_mod.uint32)

        d_q = gpu_malloc(q.nbytes)
        d_k = gpu_malloc(k.nbytes)
        d_masks = gpu_malloc(masks.nbytes)
        try:
            memcpy_htod(d_q, q.ctypes.data_as(ctypes.c_void_p), q.nbytes)
            memcpy_htod(d_k, k.ctypes.data_as(ctypes.c_void_p), k.nbytes)
            memcpy_htod(d_masks, masks.ctypes.data_as(ctypes.c_void_p), masks.nbytes)  # zero out

            block = (1, 1, 1)
            grid = (seq_len, seq_len, batch_size)
            self.guard.start()
            launch(
                self.kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_k.value),
                    ctypes.c_uint64(d_masks.value),
                    ctypes.c_float(float(attract_thresh)),
                    ctypes.c_float(float(repel_thresh)),
                    ctypes.c_int32(int(batch_size)),
                    ctypes.c_int32(int(seq_len)),
                    ctypes.c_int32(int(embed_dim)),
                ],
            )
            synchronize()
            self.guard.stop()
            memcpy_dtoh(masks.ctypes.data_as(ctypes.c_void_p), d_masks, masks.nbytes)
            return masks
        finally:
            gpu_free(d_q)
            gpu_free(d_k)
            gpu_free(d_masks)

    def compute_adaptive_thresholds(
        self,
        Q,
        K,
        percentile_attract: float = 75.0,
        percentile_repel: float = 25.0,
    ) -> tuple[float, float]:
        """Compute approximate thresholds per batch, return averaged attract/repel."""
        if Q.shape != K.shape:
            raise ValueError(f"Q and K must match; got {Q.shape} vs {K.shape}")
        np_mod = _np()
        batch_size, seq_len, embed_dim = Q.shape
        q = np_mod.ascontiguousarray(Q, dtype=np_mod.float32)
        k = np_mod.ascontiguousarray(K, dtype=np_mod.float32)
        thresholds = np_mod.zeros((batch_size, 2), dtype=np_mod.float32)

        d_q = gpu_malloc(q.nbytes)
        d_k = gpu_malloc(k.nbytes)
        d_thr = gpu_malloc(thresholds.nbytes)
        try:
            memcpy_htod(d_q, q.ctypes.data_as(ctypes.c_void_p), q.nbytes)
            memcpy_htod(d_k, k.ctypes.data_as(ctypes.c_void_p), k.nbytes)
            memcpy_htod(d_thr, thresholds.ctypes.data_as(ctypes.c_void_p), thresholds.nbytes)

            block = (256, 1, 1)
            grid = (batch_size, 1, 1)
            self.guard.start()
            launch(
                self.threshold_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_k.value),
                    ctypes.c_uint64(d_thr.value),
                    ctypes.c_float(float(percentile_attract)),
                    ctypes.c_float(float(percentile_repel)),
                    ctypes.c_int32(int(batch_size)),
                    ctypes.c_int32(int(seq_len)),
                    ctypes.c_int32(int(embed_dim)),
                ],
            )
            synchronize()
            self.guard.stop()
            memcpy_dtoh(thresholds.ctypes.data_as(ctypes.c_void_p), d_thr, thresholds.nbytes)
            attract = float(thresholds[:, 0].mean())
            repel = float(thresholds[:, 1].mean())
            return attract, repel
        finally:
            gpu_free(d_q)
            gpu_free(d_k)
            gpu_free(d_thr)
