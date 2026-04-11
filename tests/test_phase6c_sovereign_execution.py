from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from knowledge3d.bridge.headless_tablet import TabletIngest
from knowledge3d.cranium.bridges.trm_step_fused_bridge import ACTION_UPDATE_TABLET, TRMStepFusedBridge
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader
from knowledge3d.ingestion.star_crafter import build_default_star_crafter_program_table
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _ptr_value(value: object) -> int:
    return int(getattr(value, "value", value) or 0)


def _build_rpn_probe_ptx(tmp_path: Path) -> Path:
    source = tmp_path / "phase6c_rpn_probe.cu"
    header = (Path(__file__).resolve().parents[1] / "knowledge3d" / "cranium" / "cuda" / "rpn_execute_device.cuh").resolve()
    source.write_text(
        "\n".join(
            [
                "#include <cuda_runtime.h>",
                f'#include "{header}"',
                'extern "C" __global__ void phase6c_rpn_probe(',
                "    const unsigned char* program_table,",
                "    unsigned int program_offset,",
                "    unsigned int program_length,",
                "    int operand_0,",
                "    int operand_1,",
                "    int* out_ok,",
                "    int* out_result",
                ") {",
                "    if (blockIdx.x != 0 || threadIdx.x != 0) {",
                "        return;",
                "    }",
                "    int result = 0;",
                "    const int ok = rpn_execute_device(",
                "        program_table,",
                "        program_offset,",
                "        program_length,",
                "        operand_0,",
                "        operand_1,",
                "        &result",
                "    );",
                "    *out_ok = ok;",
                "    *out_result = result;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    target = tmp_path / "phase6c_rpn_probe.ptx"
    target.write_text(
        compile_cuda_file(
            source,
            arch="sm_86",
            use_fast_math=False,
            extra_nvcc_flags=["-I/usr/include", "-I/usr/include/x86_64-linux-gnu"],
        ),
        encoding="utf-8",
    )
    return target


def _run_rpn_program(
    *,
    ptx_path: Path,
    program_table_payload: bytes,
    program_offset: int,
    program_length: int,
    operand_0: int,
    operand_1: int,
) -> tuple[int, int]:
    module = loader.load_module_from_file(str(ptx_path))
    kernel = loader.get_function(module, "phase6c_rpn_probe")
    payload_type = ctypes.c_ubyte * len(program_table_payload)
    payload = payload_type.from_buffer_copy(program_table_payload)
    host_ok = ctypes.c_int32()
    host_result = ctypes.c_int32()
    d_program = loader.gpu_malloc(len(program_table_payload))
    d_ok = loader.gpu_malloc(ctypes.sizeof(host_ok))
    d_result = loader.gpu_malloc(ctypes.sizeof(host_result))
    try:
        loader.memcpy_htod(d_program, ctypes.cast(payload, ctypes.c_void_p), len(program_table_payload))
        loader.launch(
            kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                ctypes.c_uint64(_ptr_value(d_program)),
                ctypes.c_uint32(int(program_offset)),
                ctypes.c_uint32(int(program_length)),
                ctypes.c_int32(int(operand_0)),
                ctypes.c_int32(int(operand_1)),
                ctypes.c_uint64(_ptr_value(d_ok)),
                ctypes.c_uint64(_ptr_value(d_result)),
            ],
        )
        loader.synchronize()
        loader.memcpy_dtoh(ctypes.byref(host_ok), d_ok, ctypes.sizeof(host_ok))
        loader.memcpy_dtoh(ctypes.byref(host_result), d_result, ctypes.sizeof(host_result))
        return int(host_ok.value), int(host_result.value)
    finally:
        loader.gpu_free(d_program)
        loader.gpu_free(d_ok)
        loader.gpu_free(d_result)


def test_phase6c_math_task_packs_operands_into_tablet_reserved() -> None:
    envelope = TabletIngest.math_task(task_id="phase6c_pack", question="15/5?", expected_answer="3")
    action_buffer = envelope.to_action_buffer()

    reserved = [int(value) for value in action_buffer.buffer["tablet_reserved"][0]]

    assert reserved == [15, 5, 2, ord("/")]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("program_id", "operand_0", "operand_1", "expected"),
    [
        ("rpn_program_addition", 2, 3, 5),
        ("rpn_program_subtraction", 7, 4, 3),
        ("rpn_program_multiplication", 6, 8, 48),
        ("rpn_program_division", 15, 5, 3),
    ],
)
def test_phase6c_rpn_execute_device_runs_math_tier_programs(
    tmp_path: Path,
    program_id: str,
    operand_0: int,
    operand_1: int,
    expected: int,
) -> None:
    _ensure_cuda()
    layout = build_default_star_crafter_program_table()
    ptx_path = _build_rpn_probe_ptx(tmp_path)

    ok, result = _run_rpn_program(
        ptx_path=ptx_path,
        program_table_payload=layout.payload,
        program_offset=int(layout.offsets[program_id]) + 4,
        program_length=int(layout.lengths[program_id]),
        operand_0=operand_0,
        operand_1=operand_1,
    )

    assert ok == 1
    assert result == expected


@pytest.mark.gpu
def test_phase6c_tablet_sovereign_math_grid(tmp_path: Path) -> None:
    _ensure_cuda()
    kv = Knowledgeverse(storage_root=tmp_path / "kv_phase6c")
    tablet = TabletIngest
    boundary = None
    try:
        from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC

        boundary = HeadlessTabletMPC(knowledgeverse=kv, storage_root=tmp_path / "tablet_phase6c")
        for question, expected in (
            ("2+3?", 5),
            ("7-4?", 3),
            ("6*8?", 48),
            ("15/5?", 3),
        ):
            envelope = tablet.math_task(
                task_id=f"phase6c_{question}",
                question=question,
                expected_answer=str(expected),
            )
            result = boundary.submit(envelope)
            task_result = result["response"]["task_result"]
            words = list(task_result["action_buffer_words"])

            assert result["tablet_contract"]["sovereign_path"] == "tablet_bridge_ring"
            assert result["tablet_contract"]["output_action_type"] == "UPDATE_TABLET"
            assert task_result["answer_materialized"] is True
            assert result["emitted"]["answer_materialized"] is True
            assert result["emitted"]["numeric_answer"] == expected
            assert result["emitted"]["answer_text"] == str(expected)
            assert words[0] == ACTION_UPDATE_TABLET
            assert words[60] == 2
            assert words[61] == expected
            assert words[66] == 1
    finally:
        if boundary is not None:
            bridge = getattr(boundary, "_bridge", None)
            if bridge is not None and isinstance(bridge, TRMStepFusedBridge):
                bridge.stop_tick_loop(timeout=0.1)
