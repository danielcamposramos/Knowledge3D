"""Boot-only launch wrapper for the persistent TRM tick.

One `cudaLaunchCooperativeKernel` call. Returns immediately. Kernel runs on
every SM until `shutdown_flag` is set. Zero hot-path logic lives here.
See: TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §2.3.
"""
from __future__ import annotations

from typing import Any


def launch_persistent_trm(ctx: Any) -> None:
    ctx.cuda.launch_cooperative_kernel(
        ctx.kernels.trm_step_fused,
        grid_dim=ctx.device.sm_count,
        block_dim=256,
        shared_bytes=49152,
        stream=ctx.stream,
        args=ctx.kernel_args,
    )
