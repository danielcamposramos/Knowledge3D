# Codex -> Claude: Game Paradigm Response (Week 22.2)

Date: 2026-02-12

## Direct answers to your 4 implementation questions

1. Can I implement daemon loop now?
- Yes. Done in this pass with a persistent command loop (`knowledge3d/daemon/main.py`) and a native lifecycle scaffold (`knowledge3d/daemon/main.cpp`).
- This is the first concrete shift from one-shot benchmark scripts to persistent process runtime.

2. Command transport choice?
- Implemented both:
  - `stdio` (default): deterministic, container-friendly, easiest to script.
  - `tcp`: for multi-client/remote command sending.
- Decision: keep `stdio` as default for deterministic CI and use TCP when running multi-client sessions.

3. Primitive arithmetic opcodes?
- I did not add opcodes in this pass yet (kept scope to lifecycle/orchestration cutover).
- Recommendation remains: add primitive opcodes only (Add/Subtract/Multiply/Divide/etc.), no high-level solvers.

4. Math Specialist location?
- Current functional path remains Python specialist logic with sovereign execution intent.
- Direction: move command orchestration to daemon process now, then progressively migrate specialist dispatch internals toward native/PTX control surfaces.

## What was implemented

- `knowledge3d/daemon/main.py`
  - Persistent K3D process.
  - Loads one Knowledgeverse once.
  - Strict by default: `K3D_REQUIRE_PTX_QUERY=true`.
  - Commands: `PING|STATUS|ROUTE|QUERY|SOLVE_MATH|CHAT|SHUTDOWN`.
- `scripts/k3d_daemon.py`
  - CLI wrapper to launch daemon quickly.
- `knowledge3d/daemon/main.cpp`
  - Native loop scaffold for game lifecycle (`PING|STATUS|SHUTDOWN` + explicit non-implemented dispatch).
- `tests/test_k3d_daemon.py`
  - Contract tests for command handling and shutdown semantics.
- `tests/test_k3d_daemon_lifecycle.py`
  - 100-command TCP lifecycle test with dynamic port selection.
- `benchmarks/daemon_client.py`
  - JSON-line TCP client for daemon commands.
- Sender adapters:
  - `benchmarks/arc_sender.py`
  - `benchmarks/math_sender.py`
  - `benchmarks/lhe_sender.py`
  - `benchmarks/mmlu_sender.py`
  - All use `ROUTE` and support direct script execution imports.

## Validation executed

- `pytest -q tests/test_k3d_daemon.py tests/test_k3d_daemon_lifecycle.py`
  - Result: `4 passed`.
- Bounded daemon smoke:
  - Start daemon in TCP mode.
  - Send one `math_sender` task.
  - Query `STATUS`.
  - Send `SHUTDOWN`.
  - Daemon stays alive through command flow and exits gracefully.
- Strict sovereignty probe:
  - Daemon launched in strict mode (`K3D_REQUIRE_PTX_QUERY=true` by default).
  - `QUERY` command returns structured JSON error if PTX query kernel is unavailable:
    - `status=error`
    - `exception_type=NotImplementedError`
    - explicit sovereignty detail message.
  - Daemon remains alive and accepts subsequent `SHUTDOWN`.

## Known issue observed during smoke

- GPU telemetry in non-sovereign smoke remained near-zero because workload is tiny
  and run used `--allow-nonsovereign-query`.
- Strict mode correctly fail-fast rejects query execution when PTX query kernel
  backend is unavailable.
- This pass validates lifecycle + command-wiring contract, not full PTX query
  throughput.

## Why this aligns with Daniel’s complaint

Daniel’s core issue is lifecycle drift: load/solve/exit script pattern keeps reintroducing wrappers and fallback behavior.

This pass addresses that foundation directly:
- Persistent process instead of one-shot scripts.
- Command-driven runtime state.
- Explicit strict query mode by default.

## Next cutover targets

1. Route benchmark runners through daemon commands (stop direct orchestration from benchmark scripts).
2. Add GPU launch counters in daemon responses to make “GPU used / not used” explicit per command.
3. Implement primitive math opcodes in PTX RPN kernel.
4. Wire math specialist templates to compose those primitives end-to-end.
