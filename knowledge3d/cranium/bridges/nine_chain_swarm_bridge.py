"""
Nine-Chain Swarm Prototype Bridge.

Provides a minimal Python interface for the Step 14 swarm proof-of-concept.
Each call launches the CUDA kernel that orchestrates nine parallel chains,
collects their states, and exposes resonance diagnostics.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from knowledge3d.cranium.sovereign import loader


class NineChainSwarmBridge:
    """Python orchestrator for the nine-chain swarm prototype."""

    NUM_CHAINS = 9
    CHAIN_STATE_DIM = 64

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "ptx" / "nine_chain_swarm_kernel.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(
                "PTX for nine-chain swarm kernel not found: "
                f"{ptx_path}. Compile with:\n"
                "  cd knowledge3d/cranium/kernels &&\n"
                "  nvcc -ptx -arch=sm_86 -O3 nine_chain_swarm_kernel.cu "
                "-o ../ptx/nine_chain_swarm_kernel.ptx"
            )

        self._module = loader.load_module_from_file(str(ptx_path))
        self._kernel = loader.get_function(self._module, "nine_chain_swarm_kernel")

        state_bytes = self.NUM_CHAINS * self.CHAIN_STATE_DIM * 4
        self._d_chain_states = loader.gpu_malloc(state_bytes)
        self._d_resonance = loader.gpu_malloc(self.NUM_CHAINS * 4)
        self._d_input = loader.gpu_malloc(self.CHAIN_STATE_DIM * 4)
        self._d_output = loader.gpu_malloc(self.CHAIN_STATE_DIM * 4)

        self._h_chain_states = np.zeros((self.NUM_CHAINS, self.CHAIN_STATE_DIM), dtype=np.float32)
        self._h_resonance = np.zeros(self.NUM_CHAINS, dtype=np.float32)

    def execute_swarm(
        self,
        input_embedding: np.ndarray,
        num_iterations: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the nine-chain swarm prototype.

        Args:
            input_embedding: Input vector of length 64.
            num_iterations: Number of intra-kernel adaptation iterations.

        Returns:
            (output_embedding, chain_states, resonance_scores)
        """
        vec = np.asarray(input_embedding, dtype=np.float32)
        if vec.shape != (self.CHAIN_STATE_DIM,):
            raise ValueError(
                f"Input embedding must have shape ({self.CHAIN_STATE_DIM},), "
                f"got {vec.shape}"
            )

        loader.memcpy_htod(
            self._d_input,
            vec.ctypes.data_as(ctypes.c_void_p),
            self.CHAIN_STATE_DIM * 4,
        )

        loader.launch(
            self._kernel,
            grid=(self.NUM_CHAINS, 1, 1),
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(self._d_input.value),
                ctypes.c_uint64(self._d_chain_states.value),
                ctypes.c_uint64(self._d_output.value),
                ctypes.c_uint64(self._d_resonance.value),
                ctypes.c_int32(int(num_iterations)),
            ],
        )
        loader.synchronize()

        output = np.zeros(self.CHAIN_STATE_DIM, dtype=np.float32)
        loader.memcpy_dtoh(
            output.ctypes.data_as(ctypes.c_void_p),
            self._d_output,
            self.CHAIN_STATE_DIM * 4,
        )

        loader.memcpy_dtoh(
            self._h_chain_states.ctypes.data_as(ctypes.c_void_p),
            self._d_chain_states,
            self.NUM_CHAINS * self.CHAIN_STATE_DIM * 4,
        )
        loader.memcpy_dtoh(
            self._h_resonance.ctypes.data_as(ctypes.c_void_p),
            self._d_resonance,
            self.NUM_CHAINS * 4,
        )

        return output, self._h_chain_states.copy(), self._h_resonance.copy()

    def get_chain_diagnostics(self) -> Dict[str, np.ndarray | float]:
        """Return the most recent chain diagnostics."""
        norms = np.linalg.norm(self._h_chain_states, axis=1)
        mean_res = float(np.mean(self._h_resonance))
        res_var = float(np.var(self._h_resonance))
        return {
            "chain_states": self._h_chain_states.copy(),
            "resonance_scores": self._h_resonance.copy(),
            "chain_norms": norms,
            "mean_resonance": mean_res,
            "resonance_variance": res_var,
        }

    def cleanup(self) -> None:
        """Release GPU buffers."""
        loader.gpu_free(self._d_chain_states)
        loader.gpu_free(self._d_resonance)
        loader.gpu_free(self._d_input)
        loader.gpu_free(self._d_output)
