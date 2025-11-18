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
from knowledge3d.cranium.bridges.rpn_config import RPN_GRID_DIM, TIER2_BLOCK_DIM

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

    def compute_coherence(self, crystallized: np.ndarray, temporal_context: np.ndarray) -> np.ndarray:
        """Compute temporal coherence scores.

        Measures how well the crystallized output aligns with temporal context.
        Used in thinking tag inference for coherence scoring.

        Args:
            crystallized: Crystallized output vector
            temporal_context: Temporal context vector

        Returns:
            Coherence scores (per dimension)
        """
        if not isinstance(crystallized, np.ndarray):
            crystallized = np.array(crystallized, dtype=np.float32)
        if not isinstance(temporal_context, np.ndarray):
            temporal_context = np.array(temporal_context, dtype=np.float32)

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
        diff = np.abs(crystallized_flat - context_flat)
        max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
        coherence = 1.0 - (diff / max_diff)

        return coherence.astype(np.float32)

    def estimate_coherence(self, context: np.ndarray) -> np.ndarray:
        """Estimate coherence from temporal context alone.

        Simplified version that estimates coherence without comparing to output.
        Useful for fallback paths.

        Args:
            context: Temporal context vector

        Returns:
            Estimated coherence scores
        """
        if not isinstance(context, np.ndarray):
            context = np.array(context, dtype=np.float32)

        # Use temporal stability (low variance = high coherence)
        context_flat = context.flatten()
        if len(context_flat) > 1:
            variance = np.var(context_flat)
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

    def calculate_complexity(self, input_embedding: np.ndarray, modal_signature: list) -> float:
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
        if input_embedding.dtype != np.float32:
            input_embedding = input_embedding.astype(np.float32)

        # Calculate vector magnitude (normalized)
        magnitude = np.linalg.norm(input_embedding)
        max_magnitude = np.sqrt(len(input_embedding))  # Maximum possible for unit components
        normalized_magnitude = min(magnitude / max_magnitude, 1.0)

        # Calculate modal diversity score (more modalities = more complex)
        modal_diversity = len(set(modal_signature)) / 3.0  # Normalize by max 3 modalities
        modal_diversity = min(modal_diversity, 1.0)

        # Combine factors (weighted average)
        complexity = 0.7 * normalized_magnitude + 0.3 * modal_diversity

        return float(complexity)

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a, vec_b: Input vectors

        Returns:
            Cosine similarity (-1.0 to 1.0)
        """
        dot_product = np.dot(vec_a.flatten(), vec_b.flatten())
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def compute(self, confidence_vector: np.ndarray) -> np.ndarray:
        """Compute confidence rays from crystallized output.

        This is used in thinking tag inference to generate per-tag confidence scores.

        Args:
            confidence_vector: Crystallized output vector

        Returns:
            Confidence scores (one per dimension)
        """
        # Sigmoid activation for confidence scores
        return 1.0 / (1.0 + np.exp(-confidence_vector.astype(np.float32)))


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

    def smooth_intermediate(self, output: np.ndarray, ema_buffer, warp_level: bool = True) -> np.ndarray:
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
        if not isinstance(output, np.ndarray):
            output = np.array(output, dtype=np.float32)
        elif output.dtype != np.float32:
            output = output.astype(np.float32)

        # For now, use simple EMA on CPU (can be optimized with GPU kernel later)
        # This maintains the interface while providing functional smoothing
        alpha = 0.999 if warp_level else 0.99

        # Read current EMA state from GPU buffer
        ema_state = np.zeros_like(output)
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

    def apply(self, output: np.ndarray, ema_buffer) -> np.ndarray:
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

        # Allocate persistent state buffer (18 instances × 1040 bytes, Tesla 3-6-9 resonance)
        self.d_state = gpu_malloc(self.MAX_INSTANCES * self.INSTANCE_STRIDE)

        # Zero-initialize state buffer
        state_zeros = np.zeros(self.MAX_INSTANCES * self.INSTANCE_STRIDE, dtype=np.uint8)
        memcpy_htod(self.d_state, state_zeros.ctypes.data_as(ctypes.c_void_p), state_zeros.nbytes)

    def execute_single(
        self,
        instance_id: int,
        op_codes: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray
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

        # Prepare inputs
        op_codes = np.ascontiguousarray(op_codes, dtype=np.uint16)
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        vectors = np.ascontiguousarray(vectors.flatten(), dtype=np.float32)

        # Allocate GPU memory
        d_op_codes = gpu_malloc(op_codes.nbytes)
        d_scalars = gpu_malloc(scalars.nbytes) if scalars.nbytes else None
        d_vectors = gpu_malloc(vectors.nbytes) if vectors.nbytes else None

        try:
            # Copy inputs to GPU
            memcpy_htod(d_op_codes, op_codes.ctypes.data_as(ctypes.c_void_p), op_codes.nbytes)
            if d_scalars is not None:
                memcpy_htod(d_scalars, scalars.ctypes.data_as(ctypes.c_void_p), scalars.nbytes)
            if d_vectors is not None and vectors.nbytes:
                memcpy_htod(d_vectors, vectors.ctypes.data_as(ctypes.c_void_p), vectors.nbytes)

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
            header_bytes = np.zeros(4, dtype=np.uint32)
            memcpy_dtoh(
                header_bytes.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + instance_offset),
                16
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

            result_bytes = np.zeros(4, dtype=np.float32)
            memcpy_dtoh(
                result_bytes.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + element_offset),
                16
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
        programs: list,
        max_instances: int = 15
    ) -> np.ndarray:
        """Execute batch of RPN programs in parallel across instances

        Args:
            programs: List of dicts with keys 'op_codes', 'scalars', 'vectors'
            max_instances: Max parallel instances (default 15)

        Returns:
            NumPy array of results (length = len(programs))
        """
        results = []

        # Process in batches of max_instances
        for batch_start in range(0, len(programs), max_instances):
            batch = programs[batch_start:batch_start + max_instances]

            # Execute programs sequentially (kernel is single-threaded per instance)
            for i, program in enumerate(batch):
                result = self.execute_single(
                    instance_id=i,
                    op_codes=program['op_codes'],
                    scalars=program['scalars'],
                    vectors=program['vectors']
                )
                results.append(result)

        return np.array(results, dtype=np.float32)

    def reset_instance(self, instance_id: int):
        """Reset instance state (clear stack, reset head/size)"""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id}")

        # Zero out instance state
        instance_offset = instance_id * self.INSTANCE_STRIDE
        zeros = np.zeros(self.INSTANCE_STRIDE, dtype=np.uint8)
        memcpy_htod(
            ctypes.c_void_p(self.d_state.value + instance_offset),
            zeros.ctypes.data_as(ctypes.c_void_p),
            self.INSTANCE_STRIDE
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

    def blend(self, old: np.ndarray, teacher: np.ndarray, blend_factor: float) -> np.ndarray:
        """Blend old and teacher embeddings with GPU acceleration.

        Args:
            old: Old embedding (float32 array)
            teacher: Teacher embedding (float32 array, same shape as old)
            blend_factor: Blend factor (0.0 = keep old, 1.0 = use teacher)

        Returns:
            Blended embedding (float32 array, same shape as inputs)
        """
        # Prepare inputs
        old = np.ascontiguousarray(old.flatten(), dtype=np.float32)
        teacher = np.ascontiguousarray(teacher.flatten(), dtype=np.float32)

        if old.shape != teacher.shape:
            raise ValueError(f"Shape mismatch: old {old.shape} vs teacher {teacher.shape}")

        dim = len(old)
        if dim == 0:
            return np.array([], dtype=np.float32)

        # Allocate GPU memory
        d_old = gpu_malloc(old.nbytes)
        d_teacher = gpu_malloc(teacher.nbytes)
        d_out = gpu_malloc(old.nbytes)

        try:
            # Copy inputs to GPU
            memcpy_htod(d_old, old.ctypes.data_as(ctypes.c_void_p), old.nbytes)
            memcpy_htod(d_teacher, teacher.ctypes.data_as(ctypes.c_void_p), teacher.nbytes)

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
            output = np.zeros_like(old)
            memcpy_dtoh(output.ctypes.data_as(ctypes.c_void_p), d_out, output.nbytes)

            return output

        finally:
            gpu_free(d_old)
            gpu_free(d_teacher)
            gpu_free(d_out)

    def blend_sequence(
        self,
        base: np.ndarray,
        teachers: list,
        blend_factor: float = 0.3
    ) -> np.ndarray:
        """Blend base embedding with sequence of teacher embeddings.

        Args:
            base: Base embedding (float32 array)
            teachers: List of teacher embeddings
            blend_factor: Blend factor for each step

        Returns:
            Final blended embedding
        """
        out = np.array(base, dtype=np.float32)
        if not teachers:
            return out

        for teacher in teachers:
            out = self.blend(out, np.array(teacher, dtype=np.float32), blend_factor)

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
        frame_features: np.ndarray,  # (N_frames * feature_dim,) flattened
        n_frames: int,
        feature_dim: int
    ) -> np.ndarray:
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
            coherence = np.zeros(feature_dim, dtype=np.float32)
            memcpy_dtoh(coherence.ctypes.data_as(ctypes.c_void_p), d_coherence, coherence.nbytes)
            
            return coherence
        
        finally:
            gpu_free(d_features)
            gpu_free(d_coherence)
    
    def fuse_multimodal_features(
        self,
        text_features: np.ndarray,
        visual_features: np.ndarray,
        text_weight: float = 0.5
    ) -> np.ndarray:
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
            
            weights = np.array([text_weight, visual_weight], dtype=np.float32)
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
            fused = np.zeros_like(text_features)
            memcpy_dtoh(fused.ctypes.data_as(ctypes.c_void_p), d_fused, fused.nbytes)
            
            return fused
        
        finally:
            gpu_free(d_text)
            gpu_free(d_visual)
            gpu_free(d_weights)
            gpu_free(d_fused)
    
    def predict_world_state(
        self,
        current_state: np.ndarray,
        action_vector: np.ndarray
    ) -> np.ndarray:
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
            predicted = np.zeros_like(current_state)
            memcpy_dtoh(predicted.ctypes.data_as(ctypes.c_void_p), d_predicted, predicted.nbytes)
            
            return predicted
        
        finally:
            gpu_free(d_current)
            gpu_free(d_action)
            gpu_free(d_predicted)
    
    def generate_dynamic_mesh(
        self,
        world_state: np.ndarray,
        base_vertices: np.ndarray  # (N, 3)
    ) -> np.ndarray:
        """Generate dynamic mesh based on world state."""
        vertex_count = len(base_vertices)
        state_dim = len(world_state)
        
        # Flatten vertices
        vertices_flat = base_vertices.flatten().astype(np.float32)
        
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
            dynamic_flat = np.zeros_like(vertices_flat)
            memcpy_dtoh(dynamic_flat.ctypes.data_as(ctypes.c_void_p), d_dynamic, dynamic_flat.nbytes)
            
            return dynamic_flat.reshape(base_vertices.shape)
        
        finally:
            gpu_free(d_state)
            gpu_free(d_base)
            gpu_free(d_dynamic)
    
    def enhance_galaxy_resonance(
        self,
        query_embedding: np.ndarray,
        galaxy_embeddings: np.ndarray  # (N, embedding_dim)
    ) -> np.ndarray:
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
        trits_packed: np.ndarray,
        grid_shape: Tuple[int, int, int],
        field_stride: int,
        field_type: int = 0,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Render ternary field overlay to RGBA8."""
        gx, gy, gz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        trits = np.ascontiguousarray(trits_packed, dtype=np.uint32)
        rgba = np.zeros((gx * gy * gz * 4,), dtype=np.uint8)

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

    _dtype = np.dtype(
        [
            ("count", np.int32),
            ("sum", np.int32),
            ("mean", np.float32),
            ("var", np.float32),
            ("bottlenecks", np.int32),
        ]
    )

    def __init__(self):
        ptx_path = KERNELS_DIR / "trit_inspector.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "trit_inspector")
        self.guard = LatencyGuard(threshold_us=500.0)

    def inspect(
        self,
        trits_packed: np.ndarray,
        node_indices: np.ndarray,
        field_stride: int,
    ) -> np.ndarray:
        """Inspect ternary fields at node_indices."""
        trits = np.ascontiguousarray(trits_packed, dtype=np.uint32)
        nodes = np.ascontiguousarray(node_indices, dtype=np.int32)
        n = int(nodes.shape[0])
        out = np.zeros(n, dtype=self._dtype)

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
        embeddings: np.ndarray,
        query: np.ndarray,
        attract_thresh: float = 0.35,
        repel_thresh: float = -0.05,
    ) -> np.ndarray:
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
        scores: np.ndarray,
        keep_thresh: float = 0.5,
        drop_thresh: float = 0.05,
    ) -> np.ndarray:
        scores_np = np.ascontiguousarray(scores, dtype=np.float32)
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
