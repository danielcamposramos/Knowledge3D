"""TRM (Tiny Recursive Model) Launcher - Sovereign GPU Execution

Implements the TRM recursive refinement architecture from:
"Less is More: Recursive Reasoning with Tiny Networks"

Architecture:
- 2-layer MLP with SwiGLU activation (512 → 1024 → 512)
- Recursive refinement: z ← f(x + y + z), y ← f(y + z)
- n=6 recursive steps with drift halting (eps=1e-4)

Usage:
    from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher

    trm = TRMLauncher()
    y, z = trm.refine(q=question, y=answer, z=latent, n_steps=6)
"""

import numpy as np
import ctypes
from pathlib import Path
from typing import Tuple, Optional

from .loader import (
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)


class TRMLauncher:
    """Sovereign TRM launcher using pure PTX kernels.

    This launcher manages TRM recursive refinement with zero-copy GPU execution.
    All mathematical operations run in PTX kernels loaded via sovereign loader.

    Architecture:
        - Input: q (question, 512-dim), y (answer, 512-dim), z (latent, 512-dim)
        - Recursion: For n steps (default 6):
            1. temp = q + y + z
            2. hidden = W1 @ temp  (512 → 1024)
            3. hidden = swiglu(hidden)
            4. z_new = W2 @ hidden  (1024 → 512)
            5. temp2 = y + z_new
            6. hidden2 = W3 @ temp2  (512 → 1024)
            7. hidden2 = swiglu(hidden2)
            8. y_new = W4 @ hidden2  (1024 → 512)
            9. If ||z_new - z|| < eps: halt
            10. z ← z_new, y ← y_new
        - Output: refined (y, z)

    Attributes:
        ptx_path: Path to trm_extensions.ptx
        kernels: Dict of loaded kernel functions
        workspace: GPU memory for intermediate results
    """

    def __init__(self, ptx_path: Optional[str] = None):
        """Initialize TRM launcher and load PTX kernels.

        Args:
            ptx_path: Path to trm_extensions.ptx (default: auto-detect)
        """
        if ptx_path is None:
            # Auto-detect PTX path
            ptx_path = str(Path(__file__).parent.parent / "ptx" / "trm_extensions.ptx")

        self.ptx_path = ptx_path
        self.kernels = {}

        # Load all required kernels
        print(f"🔥 TRM Launcher: Loading sovereign PTX kernels from {ptx_path}")
        self._load_kernels()

        # Allocate persistent GPU workspace for intermediate results
        # We need: temp (512), hidden (1024), temp2 (512), hidden2 (1024)
        # Total: 3072 floats = 12,288 bytes
        self.d_temp = gpu_malloc(512 * 4)
        self.d_hidden = gpu_malloc(1024 * 4)
        self.d_temp2 = gpu_malloc(512 * 4)
        self.d_hidden2 = gpu_malloc(1024 * 4)

        print("✅ TRM Launcher initialized!")
        print(f"   GPU workspace allocated: {(512 + 1024 + 512 + 1024) * 4} bytes")

    def _load_kernels(self):
        """Load all TRM PTX kernels."""
        kernel_names = [
            "swiglu_vec_512",
            "swiglu_vec_1024",
            "vec_add_512",
            "vec_add3_512",
            "matvec_512x1024",
            "matvec_1024x512",
        ]

        for name in kernel_names:
            try:
                self.kernels[name] = load_ptx_file(self.ptx_path, name)
                print(f"   ✓ Loaded {name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load kernel {name}: {e}")

    def refine_step(
        self,
        d_q: int,  # Device pointer to question (512)
        d_y: int,  # Device pointer to answer (512)
        d_z: int,  # Device pointer to latent (512)
        d_W1: int,  # Device pointer to W1 (1024 x 512)
        d_W2: int,  # Device pointer to W2 (512 x 1024)
        d_W3: int,  # Device pointer to W3 (1024 x 512)
        d_W4: int,  # Device pointer to W4 (512 x 1024)
        d_z_new: int,  # Device pointer to z_new output (512)
        d_y_new: int,  # Device pointer to y_new output (512)
    ) -> None:
        """Execute one TRM refinement step.

        This performs the complete forward pass:
            z_new = W2 @ swiglu(W1 @ (q + y + z))
            y_new = W4 @ swiglu(W3 @ (y + z_new))

        All operations run on GPU with zero CPU involvement.

        Args:
            d_q, d_y, d_z: Input device pointers (512-dim)
            d_W1, d_W2, d_W3, d_W4: Weight device pointers
            d_z_new, d_y_new: Output device pointers (512-dim)
        """
        # Step 1: temp = q + y + z
        launch(
            self.kernels["vec_add3_512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_q),
                ctypes.c_uint64(d_y),
                ctypes.c_uint64(d_z),
                ctypes.c_uint64(self.d_temp.value),
            ],
        )

        # Step 2: hidden = W1 @ temp  (512 → 1024)
        launch(
            self.kernels["matvec_512x1024"],
            grid=(1, 1, 1),
            block=(1024, 1, 1),
            params=[
                ctypes.c_uint64(d_W1),
                ctypes.c_uint64(self.d_temp.value),
                ctypes.c_uint64(self.d_hidden.value),
            ],
        )

        # Step 3: hidden = swiglu(hidden)  (element-wise)
        launch(
            self.kernels["swiglu_vec_1024"],
            grid=(4, 1, 1),  # 4 blocks × 256 threads = 1024
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(self.d_hidden.value),
                ctypes.c_uint64(self.d_hidden.value),  # in-place
            ],
        )

        # Step 4: z_new = W2 @ hidden  (1024 → 512)
        launch(
            self.kernels["matvec_1024x512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_W2),
                ctypes.c_uint64(self.d_hidden.value),
                ctypes.c_uint64(d_z_new),
            ],
        )

        # Step 5: temp2 = y + z_new
        launch(
            self.kernels["vec_add_512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_y),
                ctypes.c_uint64(d_z_new),
                ctypes.c_uint64(self.d_temp2.value),
            ],
        )

        # Step 6: hidden2 = W3 @ temp2  (512 → 1024)
        launch(
            self.kernels["matvec_512x1024"],
            grid=(1, 1, 1),
            block=(1024, 1, 1),
            params=[
                ctypes.c_uint64(d_W3),
                ctypes.c_uint64(self.d_temp2.value),
                ctypes.c_uint64(self.d_hidden2.value),
            ],
        )

        # Step 7: hidden2 = swiglu(hidden2)  (element-wise)
        launch(
            self.kernels["swiglu_vec_1024"],
            grid=(4, 1, 1),
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(self.d_hidden2.value),
                ctypes.c_uint64(self.d_hidden2.value),  # in-place
            ],
        )

        # Step 8: y_new = W4 @ hidden2  (1024 → 512)
        launch(
            self.kernels["matvec_1024x512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_W4),
                ctypes.c_uint64(self.d_hidden2.value),
                ctypes.c_uint64(d_y_new),
            ],
        )

        synchronize()

    def refine(
        self,
        q: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        W1: np.ndarray,
        W2: np.ndarray,
        W3: np.ndarray,
        W4: np.ndarray,
        n_steps: int = 6,
        eps: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run TRM recursive refinement for n steps.

        Args:
            q: Question embedding (512,)
            y: Initial answer embedding (512,)
            z: Initial latent embedding (512,)
            W1, W2, W3, W4: Weight matrices
            n_steps: Number of recursive steps (default: 6)
            eps: Drift threshold for early stopping (default: 1e-4)

        Returns:
            (y_refined, z_refined): Tuple of refined embeddings

        Example:
            >>> trm = TRMLauncher()
            >>> q = np.random.randn(512).astype(np.float32)
            >>> y = np.random.randn(512).astype(np.float32)
            >>> z = np.random.randn(512).astype(np.float32)
            >>> # ... initialize weights ...
            >>> y_refined, z_refined = trm.refine(q, y, z, W1, W2, W3, W4)
        """
        # Validate inputs
        assert q.dtype == y.dtype == z.dtype == np.float32
        assert q.shape == y.shape == z.shape == (512,)
        assert W1.shape == (1024, 512) and W1.dtype == np.float32
        assert W2.shape == (512, 1024) and W2.dtype == np.float32
        assert W3.shape == (1024, 512) and W3.dtype == np.float32
        assert W4.shape == (512, 1024) and W4.dtype == np.float32

        # Allocate GPU memory for inputs/outputs
        d_q = gpu_malloc(q.nbytes)
        d_y = gpu_malloc(y.nbytes)
        d_z = gpu_malloc(z.nbytes)
        d_W1 = gpu_malloc(W1.nbytes)
        d_W2 = gpu_malloc(W2.nbytes)
        d_W3 = gpu_malloc(W3.nbytes)
        d_W4 = gpu_malloc(W4.nbytes)
        d_z_new = gpu_malloc(512 * 4)
        d_y_new = gpu_malloc(512 * 4)

        try:
            # Copy inputs to GPU
            memcpy_htod(d_q, q.ctypes.data_as(ctypes.c_void_p), q.nbytes)
            memcpy_htod(d_y, y.ctypes.data_as(ctypes.c_void_p), y.nbytes)
            memcpy_htod(d_z, z.ctypes.data_as(ctypes.c_void_p), z.nbytes)
            memcpy_htod(d_W1, W1.ctypes.data_as(ctypes.c_void_p), W1.nbytes)
            memcpy_htod(d_W2, W2.ctypes.data_as(ctypes.c_void_p), W2.nbytes)
            memcpy_htod(d_W3, W3.ctypes.data_as(ctypes.c_void_p), W3.nbytes)
            memcpy_htod(d_W4, W4.ctypes.data_as(ctypes.c_void_p), W4.nbytes)

            # Recursive refinement loop
            z_old = np.zeros(512, dtype=np.float32)
            for step in range(n_steps):
                # Copy current z to z_old for drift check
                memcpy_dtoh(z_old.ctypes.data_as(ctypes.c_void_p), d_z, z.nbytes)

                # Execute one refinement step
                self.refine_step(
                    d_q.value, d_y.value, d_z.value,
                    d_W1.value, d_W2.value, d_W3.value, d_W4.value,
                    d_z_new.value, d_y_new.value
                )

                # Copy results back for drift check
                z_new = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(z_new.ctypes.data_as(ctypes.c_void_p), d_z_new, z.nbytes)

                # Check drift (early stopping)
                drift = np.max(np.abs(z_new - z_old))
                if drift < eps:
                    print(f"   🛑 TRM halted at step {step + 1}/{n_steps} (drift={drift:.6f} < {eps})")
                    # Copy final y
                    y_final = np.zeros(512, dtype=np.float32)
                    memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y.nbytes)
                    return y_final, z_new

                # Update z and y for next iteration (copy z_new → z, y_new → y)
                memcpy_htod(d_z, z_new.ctypes.data_as(ctypes.c_void_p), z.nbytes)
                y_tmp = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_tmp.ctypes.data_as(ctypes.c_void_p), d_y_new, y.nbytes)
                memcpy_htod(d_y, y_tmp.ctypes.data_as(ctypes.c_void_p), y.nbytes)

            # All steps completed, return final results
            y_final = np.zeros(512, dtype=np.float32)
            z_final = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y.nbytes)
            memcpy_dtoh(z_final.ctypes.data_as(ctypes.c_void_p), d_z_new, z.nbytes)

            return y_final, z_final

        finally:
            # Cleanup
            gpu_free(d_q)
            gpu_free(d_y)
            gpu_free(d_z)
            gpu_free(d_W1)
            gpu_free(d_W2)
            gpu_free(d_W3)
            gpu_free(d_W4)
            gpu_free(d_z_new)
            gpu_free(d_y_new)

    def cleanup(self):
        """Free persistent GPU workspace."""
        gpu_free(self.d_temp)
        gpu_free(self.d_hidden)
        gpu_free(self.d_temp2)
        gpu_free(self.d_hidden2)

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


__all__ = ["TRMLauncher"]
