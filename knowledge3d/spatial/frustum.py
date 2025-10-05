"""
Frustum Culling - GPU-native spatial attention filtering

This module implements Kimi's SIMD frustum culling kernel, which filters
nodes based on camera view frustum. It serves as the "avatar's eyelid" -
pre-filtering ~80% of candidates before semantic navigation even begins.

Performance Target: <0.02ms for 28k nodes
Reduction Target: >80% candidate reduction
Integration: Morton Octree -> Frustum Cull -> LED-A* Pathfinding

Author: The Swarm (Grok's interface + Kimi's SIMD kernel)
Branch: phase4-frustum-simd-v1
"""

import cupy as cp
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FrustumCuller:
    """
    GPU-native frustum culling using Kimi's SIMD bit-mask kernel.

    This class wraps the frustum_cull_simd.ptx kernel and provides:
    - Constant memory plane caching (upload once, reuse infinitely)
    - View-projection matrix management
    - Bit-mask to indices expansion
    - Performance monitoring hooks

    Usage:
        culler = FrustumCuller()
        culler.upload_view_projection(view_proj_matrix)
        visible_indices = culler.cull_nodes(positions_gpu, candidate_indices)
    """

    def __init__(self, enable_profiling: bool = False):
        """
        Initialize frustum culler.

        Args:
            enable_profiling: Enable CuPy events for latency measurement
        """
        self.enable_profiling = enable_profiling
        self.kernel = None
        self._module: Optional[cp.RawModule] = None
        self._const_view_proj_ptr: Optional[int] = None
        self._const_view_ptr: Optional[int] = None
        self._view_proj_mem: Optional[cp.cuda.memory.MemoryPointer] = None
        self._view_mem: Optional[cp.cuda.memory.MemoryPointer] = None
        self._kernel_loaded = False

        # Performance buffers
        self.visible_flags = cp.zeros(0, dtype=cp.uint8)
        self._cached_view_proj: Optional[np.ndarray] = None
        self._cached_view: Optional[np.ndarray] = None

        # Statistics tracking
        self.total_culls = 0
        self.total_input_nodes = 0
        self.total_output_nodes = 0
        self.total_time_ms = 0.0

        # Load the PTX kernel
        self._load_kernel()

    def _load_kernel(self):
        """Load the frustum_cull_simd.ptx kernel via CuPy RawModule."""
        try:
            ptx_path = Path(__file__).parent.parent / "cranium" / "ptx" / "frustum_cull_simd.ptx"

            if not ptx_path.exists():
                raise FileNotFoundError(f"PTX kernel not found: {ptx_path}")

            module = cp.RawModule(path=str(ptx_path))

            self.kernel = module.get_function('warp_frustum_cull_simd')
            self._module = module
            self._kernel_loaded = True

            # Cache constant memory addresses for fast uploads
            view_proj_mem_raw = module.get_global('view_proj')
            view_mem_raw = module.get_global('view_matrix')

            self._view_proj_mem = view_proj_mem_raw if hasattr(view_proj_mem_raw, 'ptr') else view_proj_mem_raw[0]
            self._view_mem = view_mem_raw if hasattr(view_mem_raw, 'ptr') else view_mem_raw[0]

            self._const_view_proj_ptr = int(self._view_proj_mem.ptr)
            self._const_view_ptr = int(self._view_mem.ptr)

            logger.info(f"✓ Loaded frustum culling kernel from {ptx_path}")

        except Exception as e:
            logger.error(f"Failed to load frustum kernel: {e}")
            raise

    def upload_view_projection(self, view_proj: np.ndarray, view: Optional[np.ndarray] = None):
        """
        Upload view-projection and view matrices to constant memory.

        This uploads matrices to constant memory on the GPU, where they're
        cached per SM and reused across all warps. Upload once per frame/query.

        Args:
            view_proj: 4x4 f32 view-projection matrix (projection @ view)
            view: Optional 4x4 f32 view matrix (for depth test). If None, extracted from view_proj
        """
        if view_proj.shape != (4, 4):
            raise ValueError(f"View-projection must be 4x4, got {view_proj.shape}")

        if self._const_view_proj_ptr is None:
            raise RuntimeError("Frustum kernel not initialised")

        # Upload view-projection matrix
        view_proj_flat = np.asarray(view_proj, dtype=np.float32).ravel()
        assert self._view_proj_mem is not None
        dest_vp = cp.ndarray((16,), dtype=cp.float32, memptr=self._view_proj_mem)
        cp.copyto(dest_vp, cp.asarray(view_proj_flat))
        self._cached_view_proj = view_proj_flat.reshape(4, 4).copy()

        # Upload view matrix (if not provided, use view_proj as approximation)
        # Note: This is a simplification - ideally view should be passed separately
        if view is None:
            view = view_proj  # Fallback - tests should provide proper view matrix

        if view.shape != (4, 4):
            raise ValueError(f"View matrix must be 4x4, got {view.shape}")

        view_flat = np.asarray(view, dtype=np.float32).ravel()
        assert self._view_mem is not None
        dest_v = cp.ndarray((16,), dtype=cp.float32, memptr=self._view_mem)
        cp.copyto(dest_v, cp.asarray(view_flat))
        self._cached_view = view_flat.reshape(4, 4).copy()

    def upload_frustum_planes(self, planes: np.ndarray):
        """
        Upload frustum plane equations to constant memory.

        Frustum is defined by 6 planes (near, far, left, right, top, bottom).
        Each plane is represented as [nx, ny, nz, d] where:
            nx*x + ny*y + nz*z + d > 0  =>  point is inside

        Args:
            planes: (6, 4) f32 array of plane equations
        """
        if planes.shape != (6, 4):
            raise ValueError(f"Frustum planes must be (6, 4), got {planes.shape}")

        planes_flat = np.asarray(planes, dtype=np.float32).ravel()
        self._cached_planes = planes_flat.reshape(6, 4).copy()

    def extract_frustum_planes_from_matrix(self, view_proj: np.ndarray) -> np.ndarray:
        """
        Extract 6 frustum planes from view-projection matrix.

        Standard Gribb-Hartmann extraction method:
        - Left:   row3 + row0
        - Right:  row3 - row0
        - Bottom: row3 + row1
        - Top:    row3 - row1
        - Near:   row3 + row2
        - Far:    row3 - row2

        Args:
            view_proj: 4x4 view-projection matrix

        Returns:
            (6, 4) array of normalized plane equations
        """
        vp = view_proj
        planes = np.zeros((6, 4), dtype=np.float32)

        # Extract planes (Gribb-Hartmann method)
        planes[0] = vp[3] + vp[0]  # Left
        planes[1] = vp[3] - vp[0]  # Right
        planes[2] = vp[3] + vp[1]  # Bottom
        planes[3] = vp[3] - vp[1]  # Top
        planes[4] = vp[3] + vp[2]  # Near
        planes[5] = vp[3] - vp[2]  # Far

        # Normalize planes (nx, ny, nz, d) -> (nx/len, ny/len, nz/len, d/len)
        for i in range(6):
            length = np.sqrt(planes[i, 0]**2 + planes[i, 1]**2 + planes[i, 2]**2)
            if length > 1e-6:
                planes[i] /= length

        return planes

    def cull_nodes(self,
                   positions_gpu: cp.ndarray,
                   candidate_indices: Optional[cp.ndarray] = None,
                   view_proj: Optional[np.ndarray] = None,
                   view: Optional[np.ndarray] = None) -> cp.ndarray:
        """
        Cull nodes using frustum test.

        This is the main entry point for frustum culling. It:
        1. Uploads view-projection matrix if provided
        2. Runs SIMD frustum kernel with view-space depth test
        3. Returns visible node indices

        Args:
            positions_gpu: (N, 3) f32 node positions on GPU
            candidate_indices: Optional pre-filtered indices (from Morton)
                             If None, tests all positions
            view_proj: Optional 4x4 view-projection matrix
                      If None, uses cached matrix
            view: Optional 4x4 view matrix for depth test
                 If None, computed from view_proj

        Returns:
            Array of visible node indices (subset of input candidates)
        """
        if not self._kernel_loaded:
            raise RuntimeError("Frustum kernel not loaded")

        # Update view-projection if provided
        if view_proj is not None:
            self.upload_view_projection(view_proj, view)

        if self._cached_view_proj is None:
            raise RuntimeError(
                "View-projection matrix not uploaded – call upload_view_projection() before culling"
            )

        # If no candidates provided, test all positions
        if candidate_indices is None:
            candidate_indices = cp.arange(len(positions_gpu), dtype=cp.uint32)

        n_candidates = len(candidate_indices)
        if n_candidates == 0:
            return cp.array([], dtype=cp.uint32)

        # Start profiling if enabled
        if self.enable_profiling:
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()

        if candidate_indices.dtype != cp.uint32:
            candidate_indices = candidate_indices.astype(cp.uint32, copy=False)

        if self.visible_flags.size < n_candidates:
            self.visible_flags = cp.zeros(int(n_candidates * 1.1) + 32, dtype=cp.uint8)
        else:
            self.visible_flags[:n_candidates] = 0

        block_size = 128
        grid_size = ((n_candidates + block_size - 1) // block_size,)

        if self.enable_profiling:
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()

        self.kernel(
            grid_size,
            (block_size,),
            (
                positions_gpu,
                candidate_indices,
                np.uint32(n_candidates),
                self.visible_flags
            )
        )

        if self.enable_profiling:
            end_event.record()
            end_event.synchronize()
            elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            self.total_time_ms += elapsed_ms
        else:
            elapsed_ms = 0.0

        flags = self.visible_flags[:n_candidates]
        visible_idx = cp.where(flags != 0)[0]
        visible_indices = candidate_indices[visible_idx]

        self.total_culls += 1
        self.total_input_nodes += int(n_candidates)
        self.total_output_nodes += int(visible_indices.size)
        if self.enable_profiling:
            logger.debug(
                "Frustum cull: %d -> %d (%.1f%% visible) in %.4fms",
                n_candidates,
                int(visible_indices.size),
                100.0 * float(visible_indices.size) / float(n_candidates),
                elapsed_ms,
            )

        return visible_indices

    def cull_from_octree(self,
                        candidates_gpu: cp.ndarray,
                        positions_gpu: cp.ndarray,
                        view_proj: np.ndarray) -> cp.ndarray:
        """
        Chain from Morton octree query to frustum culling.

        This is the integration point with the Morton octree.
        Typical flow:
            1. Morton octree query by radius -> candidates (10-30% of nodes)
            2. Frustum culling -> visible (2-5% of nodes, 80-90% reduction)
            3. LED-A* pathfinding on visible set

        Args:
            candidates_gpu: Candidate indices from Morton query
            positions_gpu: Full position buffer (N, 3)
            view_proj: 4x4 camera view-projection matrix

        Returns:
            Visible node indices (subset of candidates)
        """
        return self.cull_nodes(positions_gpu, candidates_gpu, view_proj)

    def get_statistics(self) -> dict:
        """
        Get culling performance statistics.

        Returns:
            Dictionary with:
                - total_culls: Number of cull operations
                - avg_input_size: Average input candidates
                - avg_output_size: Average visible nodes
                - avg_reduction: Average reduction ratio (0-1)
                - avg_time_ms: Average cull time in ms
        """
        if self.total_culls == 0:
            return {
                'total_culls': 0,
                'avg_input_size': 0,
                'avg_output_size': 0,
                'avg_reduction': 0.0,
                'avg_time_ms': 0.0
            }

        avg_input = self.total_input_nodes / self.total_culls
        avg_output = self.total_output_nodes / self.total_culls
        avg_reduction = 1.0 - (avg_output / avg_input) if avg_input > 0 else 0.0
        avg_time = self.total_time_ms / self.total_culls

        return {
            'total_culls': self.total_culls,
            'avg_input_size': int(avg_input),
            'avg_output_size': int(avg_output),
            'avg_reduction': avg_reduction,
            'avg_time_ms': avg_time
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        self.total_culls = 0
        self.total_input_nodes = 0
        self.total_output_nodes = 0
        self.total_time_ms = 0.0


def create_perspective_matrix(fov_degrees: float,
                             aspect_ratio: float,
                             near: float,
                             far: float) -> np.ndarray:
    """
    Create perspective projection matrix.

    Standard OpenGL-style perspective projection.

    Args:
        fov_degrees: Vertical field of view in degrees
        aspect_ratio: Width / height
        near: Near clipping plane distance
        far: Far clipping plane distance

    Returns:
        4x4 perspective projection matrix
    """
    fov_rad = np.radians(fov_degrees)
    f = 1.0 / np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect_ratio
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0

    return proj


def create_view_matrix(eye: np.ndarray,
                      target: np.ndarray,
                      up: np.ndarray) -> np.ndarray:
    """
    Create view (camera) matrix using lookAt convention.

    Args:
        eye: Camera position (3,)
        target: Look-at point (3,)
        up: Up vector (3,)

    Returns:
        4x4 view matrix
    """
    # Compute camera basis
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_actual = np.cross(right, forward)

    # Build view matrix
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up_actual
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up_actual, eye)
    view[2, 3] = np.dot(forward, eye)

    return view
