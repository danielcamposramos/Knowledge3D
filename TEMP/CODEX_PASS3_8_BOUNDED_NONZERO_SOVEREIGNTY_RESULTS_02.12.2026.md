# Pass 3.8 Bounded Non-Zero Sovereignty Validation (2026-02-12)

## Run Type
- Daemon path (game lifecycle): `knowledge3d.daemon.main --mode tcp`
- Bounded non-zero validation: 10 math tasks via `benchmarks.math_sender`-equivalent ROUTE commands.

## Sovereignty Summary
- Total tasks: `10`
- Solved: `10`
- Tasks using GPU: `10/10` (`100%`)
- Tasks solved without GPU: `0`
- Fallback triggered: `0`
- Average GPU calls per solved task: `1.0`
- Daemon status before shutdown:
  - `command_count=12`
  - `gpu_calls_total=11`
  - `require_ptx_query=true`

## Violations
- `None`
- No `SovereigntyViolation` condition observed.

## Daemon Stability
- Daemon stayed alive for all bounded tasks and returned STATUS successfully.
- Graceful SHUTDOWN succeeded.
- Process exited after shutdown.

## Artifacts
- Full JSON trace: `/tmp/k3d_bounded_nonzero_sovereignty_validation.json`
- Daemon log: `/tmp/k3d_daemon_pass3_8_retry.log`

## Important Runtime Note
- Initial attempt failed with NVRTC include error (`cuda_fp16.h` not found), which prevented GPU launches.
- Validation run succeeded after launching daemon with CUDA include env:
  - `CPATH=/usr/include`
  - `CPLUS_INCLUDE_PATH=/usr/include`
  - `CUDA_PATH=/usr`
