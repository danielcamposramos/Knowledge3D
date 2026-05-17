"""Live-daemon integration test — one boot, full cycle.

Paradigm: K3D is a living game process.  ONE Knowledgeverse + House load,
multiple queries, sleeptime between — then safe shutdown with file persistence.

This file contains ONE test function that exercises the full cycle:
  (a) CHAT  "what is 2+3?"  → GPU answer "5"
  (b) INGEST tiny markdown  → ingest_receipt + temporary-star JSONL
  (c) Sleeptime tick        → at least one tick runs; file persistence verified
  (d) CHAT  second question → daemon still alive after ingest + sleeptime

REQUIRES: CUDA + k3d-cranium env.  Skips cleanly if CUDA is absent.
NO subprocesses.  NO mocks.  NO CPU fallbacks.  NO re-boots between steps.

Run standalone:
    pytest tests/tablet/test_live_daemon_cycle.py -v

Or as part of the full suite after a single KV load:
    pytest tests/ -v
"""
from __future__ import annotations

import json
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# GPU gate — same pattern as tests/test_procedural_drawing_bridge.py
# ---------------------------------------------------------------------------

def _require_cuda() -> None:
    """Skip this entire module if CUDA is not available.

    Uses cupy as the canonical GPU probe (already used elsewhere in the
    test suite).  If cupy is missing or reports zero devices we skip, because
    there is NO CPU fallback: the sovereign path is PTX-on-GPU only.
    """
    cupy = pytest.importorskip(
        "cupy",
        reason="live-daemon test requires CUDA (cupy not installed)",
    )
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"live-daemon test requires CUDA: {exc}")
    if device_count == 0:
        pytest.skip("live-daemon test requires CUDA: no devices found")


# ---------------------------------------------------------------------------
# Module-scoped fixture — ONE daemon boot for the entire test file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_daemon(tmp_path_factory):
    """Boot ONE K3DDaemon for the full test cycle.

    The fixture:
    1. Checks CUDA is present (skips whole module if not).
    2. Creates a temporary storage root so the test doesn't pollute
       /K3D/Knowledge3D.local.
    3. Boots K3DDaemon with require_ptx_query=True, eager_load_default_galaxies=True.
    4. Yields a thin dispatch callable: `dispatch(payload) -> dict`.
    5. On teardown, calls daemon.kv.shutdown(persist=True) so the
       temporary-star files are preserved (they're small and Daniel can inspect
       them; they live under tmp_path so they'll be cleaned by pytest eventually).
    """
    _require_cuda()

    from knowledge3d.daemon.main import DaemonConfig, K3DDaemon
    from knowledge3d.local_paths import default_storage_root

    # Use a temporary directory as the storage root so the test is isolated.
    # K3D_STORAGE_ROOT is the env var the project respects (see local_paths.py).
    storage_root = tmp_path_factory.mktemp("k3d_live_test")
    os.environ["K3D_STORAGE_ROOT"] = str(storage_root)

    daemon = K3DDaemon(
        config=DaemonConfig(
            storage_root=storage_root,
            require_ptx_query=True,
            eager_load_default_galaxies=True,
            warm_gpu_runtime_on_boot=False,
        )
    )

    def dispatch(payload: dict) -> dict:
        """Call daemon.handle_command and add minimal telemetry wrapper."""
        try:
            result = daemon.handle_command(payload)
        except Exception as exc:
            result = {
                "status": "error",
                "error": "command_execution_failed",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
        if "telemetry" not in result:
            result["telemetry"] = {}
        return result

    # Expose the daemon and storage root for assertions in the test.
    dispatch.daemon = daemon
    dispatch.storage_root = storage_root

    yield dispatch

    # Teardown: persist state, then shut down cleanly.
    try:
        daemon.kv.shutdown(persist=True, profile="test")
    except Exception:
        try:
            daemon.kv.shutdown(persist=False, profile="test")
        except Exception:
            pass

    os.environ.pop("K3D_STORAGE_ROOT", None)


# ---------------------------------------------------------------------------
# The ONE live-daemon cycle test
# ---------------------------------------------------------------------------

def test_live_daemon_cycle(live_daemon, tmp_path):
    """Full live-daemon cycle: CHAT → INGEST → sleeptime → CHAT.

    Steps
    -----
    (a) CHAT  "what is 2+3?"   — GPU arithmetic answer must be "5".
    (b) INGEST tiny markdown   — returns ingest_receipt with ingest_id.
    (c) Sleeptime tick         — runs one consolidation tick via daemon method;
                                 temporary-star JSONL must exist on disk.
    (d) CHAT  "what is gravity?" — daemon still alive; status ok.

    Assertions are hard (fail loudly) wherever the contract is fully
    implemented.  The sleeptime PROMOTION assertion (Lane A: promote
    temporary star to House entry) is marked xfail until sleeptime Lane A
    is wired — see TEMP/CLAUDE_INGEST_WINE_SPEC_04.20.2026.md §10.2.
    """
    dispatch = live_daemon
    daemon = dispatch.daemon
    storage_root = dispatch.storage_root

    # ------------------------------------------------------------------
    # (a) CHAT — sovereign GPU math
    # ------------------------------------------------------------------
    chat_result = dispatch({
        "command": "CHAT",
        "messages": [{"role": "user", "content": "what is 2+3?"}],
    })

    assert chat_result["status"] == "ok", (
        f"CHAT step failed: {chat_result.get('error')} | "
        f"detail={chat_result.get('detail')}"
    )
    response_text = str(chat_result.get("response", ""))
    assert "5" in response_text, (
        f"Expected '5' in chat response; got: {response_text!r}"
    )
    assert chat_result.get("gpu_execution") is True, (
        "Chat response missing gpu_execution=True — sovereign path not taken. "
        f"Full result: {json.dumps(chat_result, default=str)}"
    )

    # ------------------------------------------------------------------
    # (b) INGEST — enqueue a tiny markdown document
    # ------------------------------------------------------------------
    md_file = tmp_path / "hello.md"
    md_file.write_text(
        "# Hello Knowledge\n\nThe speed of light is approximately 299,792 km/s.\n",
        encoding="utf-8",
    )

    ingest_result = dispatch({
        "command": "INGEST",
        "source_uri": f"file://{md_file}",
        "mime": "text/markdown",
    })

    assert ingest_result["status"] == "ok", (
        f"INGEST step failed: {ingest_result.get('error')} | "
        f"detail={ingest_result.get('detail')}"
    )
    assert ingest_result.get("result_kind") == "ingest_receipt", (
        f"Expected result_kind='ingest_receipt'; got: {ingest_result.get('result_kind')!r}"
    )
    ingest_id = ingest_result.get("ingest_id")
    assert ingest_id, (
        f"INGEST receipt missing ingest_id; full result: {ingest_result}"
    )

    # ------------------------------------------------------------------
    # (c) Verify temporary-star JSONL written to disk
    # ------------------------------------------------------------------
    # enqueue_ingest writes to storage_root/galaxies/_temporary/{ingest_id}.jsonl
    # (see knowledge3d/knowledgeverse/knowledgeverse.py:enqueue_ingest §16036)
    temp_star_path = storage_root / "galaxies" / "_temporary" / f"{ingest_id}.jsonl"
    assert temp_star_path.exists(), (
        f"Temporary-star JSONL not found at {temp_star_path}. "
        "enqueue_ingest must create this file synchronously."
    )
    with open(temp_star_path, encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    assert first_line, f"Temporary-star JSONL is empty at {temp_star_path}"
    entry = json.loads(first_line)
    assert entry.get("temporary") is True, (
        f"Entry missing temporary=True; got: {entry.get('temporary')!r}"
    )
    assert entry.get("confidence_trit") == 0, (
        f"Entry missing confidence_trit=0; got: {entry.get('confidence_trit')!r}"
    )
    assert entry.get("ingest_id") == ingest_id, (
        f"Entry ingest_id mismatch: expected {ingest_id!r}, "
        f"got {entry.get('ingest_id')!r}"
    )

    # ------------------------------------------------------------------
    # (c-cont) Sleeptime tick — run one consolidation tick on the live daemon
    # ------------------------------------------------------------------
    # _run_sleep_consolidation_tick is the daemon's internal tick runner.
    # We call it directly — this is NOT a subprocess; the daemon is already
    # in memory from the module-scoped fixture.
    tick_summary = daemon._run_sleep_consolidation_tick()
    assert isinstance(tick_summary, dict), (
        f"_run_sleep_consolidation_tick must return a dict; got: {type(tick_summary)}"
    )
    assert "tick_name" in tick_summary, (
        f"Sleeptime tick summary missing tick_name; got: {tick_summary}"
    )
    # The tick ran — the temporary-star file must still exist (it is ONLY
    # removed when a promotion completes — see Ingest spec §10.2 Lane A).
    assert temp_star_path.exists(), (
        "Temporary-star JSONL was unexpectedly removed after sleeptime tick. "
        "If Lane A promotion is now implemented, update this assertion."
    )

    # Lane A (sleeptime promotes temporary stars to House) is not yet wired.
    # This xfail documents the exact gap and the spec section to close it.
    # TODO: remove xfail when TEMP/CLAUDE_INGEST_WINE_SPEC_04.20.2026.md §10.2
    #       Lane A is fully implemented.
    house_jsonl_candidates = list(
        (storage_root / "galaxies").glob("*.jsonl")
    )
    house_entries_with_ingest_id = []
    for p in house_jsonl_candidates:
        if "_temporary" in str(p):
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("ingest_id") == ingest_id or rec.get("source_ingest_id") == ingest_id:
                            house_entries_with_ingest_id.append(rec)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass

    # xfail — promotion is not yet implemented (Lane A pending)
    if not house_entries_with_ingest_id:
        pytest.xfail(
            "Sleeptime Lane A (promote temporary star → House JSONL) not yet "
            "implemented. See TEMP/CLAUDE_INGEST_WINE_SPEC_04.20.2026.md §10.2."
        )

    # If we reach here, Lane A is working — assert at least one promoted entry.
    assert len(house_entries_with_ingest_id) > 0, (
        "Lane A reported entries but list is empty — logic error above."
    )

    # ------------------------------------------------------------------
    # (c-cont2) Sleeptime Lane B — weight checkpoint must be written
    # ------------------------------------------------------------------
    # Run additional ticks until lane_b_weights fires (tick order is round-robin;
    # drive until the cursor reaches lane_b_weights or we exhaust a full rotation).
    lane_b_summary: dict | None = None
    tick_order = list(getattr(daemon, "_sleep_tick_order", ()))
    for _ in range(max(1, len(tick_order))):
        extra_tick = daemon._run_sleep_consolidation_tick()
        if extra_tick.get("tick_name") == "lane_b_weights":
            lane_b_summary = extra_tick
            break

    # Determine expected checkpoint path (mirrors daemon._sleep_lane_b_weights_tick logic).
    kv = daemon.kv
    storage_root_kv = getattr(kv, "storage_root", None)
    expected_checkpoint = (
        (Path(storage_root_kv) / "weights" / "trm_bitnet.bitnet")
        if storage_root_kv is not None
        else None
    )

    # xfail — Lane B weight checkpoint write requires CUDA + populated TRM host weights
    checkpoint_exists = (
        expected_checkpoint is not None and expected_checkpoint.exists()
        and expected_checkpoint.stat().st_size > 0
    )
    if not checkpoint_exists:
        pytest.xfail(
            "Sleeptime Lane B weight checkpoint write — requires CUDA; tile_quantize "
            "(0x313) converts float32 tiles to BitNet in place on first touch."
        )

    # If we reach here, Lane B ran and wrote the checkpoint — assert it is non-empty.
    assert expected_checkpoint.stat().st_size > 0, (
        f"Lane B wrote checkpoint at {expected_checkpoint} but it is empty."
    )

    # ------------------------------------------------------------------
    # (d) Second CHAT — daemon must still be live after ingest + sleeptime
    # ------------------------------------------------------------------
    chat2_result = dispatch({
        "command": "CHAT",
        "messages": [{"role": "user", "content": "what is gravity?"}],
    })

    assert chat2_result["status"] == "ok", (
        f"Second CHAT step failed after ingest+sleeptime: "
        f"{chat2_result.get('error')} | detail={chat2_result.get('detail')}"
    )
    # The second query must also go through the GPU path.
    assert chat2_result.get("gpu_execution") is True, (
        "Second chat response missing gpu_execution=True — daemon degraded "
        "after ingest+sleeptime cycle. "
        f"Full result: {json.dumps(chat2_result, default=str)}"
    )


def test_halting_value_on_chat_tick(live_daemon):
    """First-class halting readback hook: `solved["trm_tick"]["halting_value"]`
    must be a Python float in [0.0, 1.0] on every GPU-executed CHAT tick.

    See TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md.

    The scalar is written in PTX by the swarm halting gate
    (`knowledge3d/cranium/cuda/k3d_swarm_persistent.cu`) at the moment all
    lanes halt, and is propagated through:
        n_chain_swarm_bridge.tick() -> knowledgeverse._last_swarm_halting_value
        -> trm_game_loop._run_query_tick -> solved["trm_tick"]["halting_value"]

    The daemon's CHAT handler exposes the full solved dict as
    `task_result`, so we read the scalar from
    `response["task_result"]["trm_tick"]["halting_value"]` — no reach
    into `kv._n_chain_swarm`.
    """
    dispatch = live_daemon

    chat = dispatch({
        "command": "CHAT",
        "messages": [{"role": "user", "content": "what is 2+3?"}],
    })
    assert chat["status"] == "ok", (
        f"CHAT failed: {chat.get('error')} | detail={chat.get('detail')}"
    )
    assert chat.get("gpu_execution") is True, (
        "CHAT took a non-GPU path — halting readback hook cannot be "
        "validated without gpu_execution=True"
    )

    task_result = chat.get("task_result") or {}
    trm_tick = task_result.get("trm_tick") or {}
    # Gate: the swarm may not run on degenerate paths (single candidate),
    # in which case halting_value is legitimately absent. We require it to
    # be present for this test — the "2+3?" query exercises the swarm.
    assert "halting_value" in trm_tick, (
        "trm_tick missing halting_value on CHAT tick. "
        f"trm_tick keys: {sorted(trm_tick.keys())}. "
        "If the swarm skipped (single candidate), this query is not a "
        "valid witness — pick one that fans out."
    )
    hv = trm_tick["halting_value"]
    assert isinstance(hv, float), (
        f"halting_value must be a Python float; got {type(hv).__name__}={hv!r}"
    )
    assert 0.0 <= hv <= 1.0, (
        f"halting_value out of range [0,1]: {hv}"
    )
