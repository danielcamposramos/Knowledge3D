from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletIngest
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.galaxy_vram_table import (
    EMBEDDING_DIMS,
    ROLE_ANSWER,
    ROLE_EXECUTOR,
    STAR_ANSWER_ELIGIBLE_OFFSET,
    STAR_EMBEDDING_OFFSET,
    STAR_EXECUTOR_REF_COUNT_OFFSET,
    STAR_EXECUTOR_REFS_OFFSET,
    STAR_FLAGS_OFFSET,
    STAR_GALAXY_ID_OFFSET,
    STAR_META_RULE_ADDR_OFFSET,
    STAR_POSITION_OFFSET,
    STAR_PROGRAM_FLAGS_OFFSET,
    STAR_PROGRAM_LENGTH_OFFSET,
    STAR_PROGRAM_OPCODE_COUNT_OFFSET,
    STAR_RECORD_BYTES,
    STAR_ROUTER_REF_COUNT_OFFSET,
    STAR_ROUTER_REFS_OFFSET,
    STAR_SELECTION_ROLE_OFFSET,
    STAR_STAR_HASH_OFFSET,
    STAR_TYPE_OFFSET,
    STAR_VALIDATOR_REF_COUNT_OFFSET,
    STAR_VALIDATOR_REFS_OFFSET,
)
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


PROBE_NAMES = ("sum_all", "count_value", "max_value", "min_value", "unique_count")
QUERY_TEXT = "2+3?"
REPORT_PATH = Path("TEMP/CODEX_PHASE6A_MATH_STAR_PROBE_2026-04-11.md")


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _ptr_value(value: Any) -> int:
    return int(getattr(value, "value", value) or 0)


def _ensure_probe_ptx() -> Path:
    source = Path("knowledge3d/cranium/cuda/galaxy_star_probe.cu")
    target = Path("knowledge3d/cranium/ptx/galaxy_star_probe.ptx")
    target_mtime = target.stat().st_mtime if target.exists() else 0.0
    if target_mtime < source.stat().st_mtime:
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


def _payload_keys(star: dict[str, Any]) -> list[str]:
    keys: list[str] = []

    def _scan(mapping: dict[str, Any], *, prefix: str = "") -> None:
        for key, value in mapping.items():
            lower = str(key).lower()
            if any(token in lower for token in ("rpn", "program", "meta_rule", "opcode")):
                keys.append(f"{prefix}{key}")
            if prefix == "" and key == "metadata" and isinstance(value, dict):
                _scan(value, prefix="metadata.")

    _scan(star)
    return sorted(set(keys))


def _query_embedding(kv: Knowledgeverse) -> list[float]:
    tablet = HeadlessTabletMPC(knowledgeverse=kv)
    envelope = TabletIngest.math_task(task_id="phase6a_probe", question=QUERY_TEXT, expected_answer="5")
    return [float(value) for value in tablet._query_embedding_for_envelope(envelope)]


def _select_probe_indices(stars: list[dict[str, Any]], query_embedding_512: list[float]) -> list[int]:
    index_by_id = {str(star.get("id") or ""): idx for idx, star in enumerate(stars)}
    selected: list[int] = []
    for star_id in PROBE_NAMES:
        idx = index_by_id.get(star_id)
        if idx is None:
            raise AssertionError(f"missing expected Math star: {star_id}")
        selected.append(int(idx))

    query_vector = [float(value) for value in list(query_embedding_512 or [])[:EMBEDDING_DIMS]]
    if len(query_vector) < EMBEDDING_DIMS:
        query_vector.extend([0.0] * (EMBEDDING_DIMS - len(query_vector)))
    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query)) or 1.0
    embeddings = np.asarray(
        [list(star.get("embedding") or [0.0] * EMBEDDING_DIMS)[:EMBEDDING_DIMS] for star in stars],
        dtype=np.float32,
    )
    norms = np.linalg.norm(embeddings, axis=1)
    denom = np.maximum(norms * query_norm, 1e-8)
    scores = embeddings @ query / denom
    for idx in np.argsort(scores)[::-1]:
        index = int(idx)
        if index not in selected:
            selected.append(index)
        if len(selected) >= 8:
            break
    return selected[:8]


def _probe_records(star_table_ptr: Any, star_indices: list[int]) -> list[bytes]:
    ptx_path = _ensure_probe_ptx()
    module = loader.load_module_from_file(str(ptx_path))
    kernel = loader.get_function(module, "galaxy_star_probe_records")

    index_type = ctypes.c_uint32 * len(star_indices)
    host_indices = index_type(*[int(index) for index in star_indices])
    out_bytes = len(star_indices) * STAR_RECORD_BYTES
    host_out = (ctypes.c_ubyte * out_bytes)()

    d_indices = loader.gpu_malloc(ctypes.sizeof(host_indices))
    d_out = loader.gpu_malloc(out_bytes)
    try:
        loader.memcpy_htod(d_indices, ctypes.cast(host_indices, ctypes.c_void_p), ctypes.sizeof(host_indices))
        loader.launch(
            kernel,
            grid=(len(star_indices), 1, 1),
            block=(128, 1, 1),
            params=[
                ctypes.c_uint64(_ptr_value(star_table_ptr)),
                ctypes.c_uint64(_ptr_value(d_indices)),
                ctypes.c_uint64(_ptr_value(d_out)),
                ctypes.c_uint32(len(star_indices)),
            ],
        )
        loader.synchronize()
        loader.memcpy_dtoh(ctypes.cast(host_out, ctypes.c_void_p), d_out, out_bytes)
    finally:
        loader.gpu_free(d_indices)
        loader.gpu_free(d_out)

    payload = bytes(host_out)
    return [
        payload[offset : offset + STAR_RECORD_BYTES]
        for offset in range(0, len(payload), STAR_RECORD_BYTES)
    ]


def _decode_record(record_bytes: bytes) -> dict[str, Any]:
    embedding = struct.unpack_from(f"<{EMBEDDING_DIMS}f", record_bytes, STAR_EMBEDDING_OFFSET)
    router_ref_count = struct.unpack_from("<I", record_bytes, STAR_ROUTER_REF_COUNT_OFFSET)[0]
    executor_ref_count = struct.unpack_from("<I", record_bytes, STAR_EXECUTOR_REF_COUNT_OFFSET)[0]
    validator_ref_count = struct.unpack_from("<I", record_bytes, STAR_VALIDATOR_REF_COUNT_OFFSET)[0]
    return {
        "embedding_norm": math.sqrt(sum(float(value) * float(value) for value in embedding)),
        "galaxy_id_u32": struct.unpack_from("<I", record_bytes, STAR_GALAXY_ID_OFFSET)[0],
        "star_type": struct.unpack_from("<I", record_bytes, STAR_TYPE_OFFSET)[0],
        "selection_role_id": struct.unpack_from("<I", record_bytes, STAR_SELECTION_ROLE_OFFSET)[0],
        "flags": struct.unpack_from("<I", record_bytes, STAR_FLAGS_OFFSET)[0],
        "answer_eligible": struct.unpack_from("<I", record_bytes, STAR_ANSWER_ELIGIBLE_OFFSET)[0],
        "star_hash": struct.unpack_from("<Q", record_bytes, STAR_STAR_HASH_OFFSET)[0],
        "router_ref_count": int(router_ref_count),
        "router_refs_raw": list(struct.unpack_from("<2I", record_bytes, STAR_ROUTER_REFS_OFFSET)),
        "executor_ref_count": int(executor_ref_count),
        "executor_refs_raw": list(struct.unpack_from("<2I", record_bytes, STAR_EXECUTOR_REFS_OFFSET)),
        "validator_ref_count": int(validator_ref_count),
        "validator_refs_raw": list(struct.unpack_from("<2I", record_bytes, STAR_VALIDATOR_REFS_OFFSET)),
        "position": list(struct.unpack_from("<3f", record_bytes, STAR_POSITION_OFFSET)),
        "meta_rule_addr": struct.unpack_from("<I", record_bytes, STAR_META_RULE_ADDR_OFFSET)[0],
        "program_flags": struct.unpack_from("<I", record_bytes, STAR_PROGRAM_FLAGS_OFFSET)[0],
        "program_length": struct.unpack_from("<I", record_bytes, STAR_PROGRAM_LENGTH_OFFSET)[0],
        "program_opcode_count": struct.unpack_from("<I", record_bytes, STAR_PROGRAM_OPCODE_COUNT_OFFSET)[0],
        "mid_120_256_nonzero": sum(1 for value in record_bytes[120:256] if value != 0),
        "raw_hex_head": record_bytes[:64].hex(),
        "raw_hex_tail": record_bytes[256:].hex(),
        "layout_exhaustive": (STAR_PROGRAM_OPCODE_COUNT_OFFSET + 4) == STAR_RECORD_BYTES,
    }


def _verdict(host_star: dict[str, Any], decoded: dict[str, Any]) -> str:
    payload_keys = _payload_keys(host_star)
    if int(decoded.get("meta_rule_addr") or 0) > 0:
        return "HAS_RPN_PAYLOAD"
    if payload_keys:
        return "NO_RPN_PAYLOAD"
    if decoded["selection_role_id"] in (ROLE_EXECUTOR, ROLE_ANSWER) and decoded["executor_ref_count"] > 0:
        return "NO_RPN_PAYLOAD"
    return "NO_RPN_PAYLOAD"


def _write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 6.A Math Star Probe",
        "",
        f"Query: `{QUERY_TEXT}`",
        f"Probe count: {len(rows)}",
        "",
    ]
    for row in rows:
        decoded = row["decoded"]
        lines.extend(
            [
                f"## {row['label']}",
                f"- Index: {row['index']}",
                f"- ID: `{row['id']}`",
                f"- Name: `{row['name']}`",
                f"- Galaxy: `{row['galaxy_id']}`",
                f"- Host payload keys: {row['host_payload_keys']}",
                f"- Verdict: `{row['verdict']}`",
                f"- selection_role_id: `{decoded['selection_role_id']}`",
                f"- answer_eligible: `{decoded['answer_eligible']}`",
                f"- flags: `{decoded['flags']}`",
                f"- star_hash: `{decoded['star_hash']}`",
                f"- router_ref_count: `{decoded['router_ref_count']}` raw={decoded['router_refs_raw']}",
                f"- executor_ref_count: `{decoded['executor_ref_count']}` raw={decoded['executor_refs_raw']}",
                f"- validator_ref_count: `{decoded['validator_ref_count']}` raw={decoded['validator_refs_raw']}",
                f"- position: `{decoded['position']}`",
                f"- embedding_norm: `{decoded['embedding_norm']:.6f}`",
                f"- nonzero bytes in [120,256): `{decoded['mid_120_256_nonzero']}`",
                f"- raw head hex: `{decoded['raw_hex_head']}`",
                f"- raw tail hex: `{decoded['raw_hex_tail']}`",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _collect_probe_rows() -> list[dict[str, Any]]:
    kv = Knowledgeverse()
    runtime = kv._get_sovereign_hot_path()
    stars = [dict(star) for star in getattr(runtime, "_host_stars", [])]
    assert len(stars) > 0
    assert getattr(runtime.star_table, "star_count", 0) > 0

    probe_indices = _select_probe_indices(stars, _query_embedding(kv))
    assert len(probe_indices) == 8

    raw_records = _probe_records(runtime.star_table.gpu_ptr, probe_indices)
    assert len(raw_records) == 8
    rows: list[dict[str, Any]] = []
    for order, (star_index, record_bytes) in enumerate(zip(probe_indices, raw_records)):
        host_star = stars[star_index]
        decoded = _decode_record(record_bytes)
        verdict = _verdict(host_star, decoded)
        label = "named" if order < len(PROBE_NAMES) else "top_cosine"
        host_payload_keys = _payload_keys(host_star)
        row = {
            "label": label,
            "index": int(star_index),
            "id": str(host_star.get("id") or ""),
            "name": str(host_star.get("name") or ""),
            "galaxy_id": host_star.get("galaxy_id"),
            "host_payload_keys": host_payload_keys,
            "verdict": verdict,
            "decoded": decoded,
        }
        rows.append(row)
        print(
            f"{label} idx={row['index']} id={row['id']} role={decoded['selection_role_id']} "
            f"answer_eligible={decoded['answer_eligible']} host_payload_keys={host_payload_keys} "
            f"verdict={verdict}"
        )
        print(f"  head={decoded['raw_hex_head']}")
        print(f"  tail={decoded['raw_hex_tail']}")

    _write_report(rows)
    return rows


def _run_probe_subprocess() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".") or "."
    env["K3D_PYTEST_PHASE6A_SUBPROCESS"] = "1"
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.gpu
def test_phase6a_math_star_probe() -> None:
    _ensure_cuda()
    proc = _run_probe_subprocess()
    if proc.returncode != 0:
        raise AssertionError(f"phase6a probe subprocess failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    summary_line = ""
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith("PHASE6A_JSON:"):
            summary_line = line
            break
    if not summary_line:
        raise AssertionError(f"phase6a probe summary missing\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    summary = json.loads(summary_line.split("PHASE6A_JSON:", 1)[1])

    rows = list(summary.get("rows") or [])
    assert len(rows) == 8
    assert all(row["verdict"] == "NO_RPN_PAYLOAD" for row in rows)
    assert rows[0]["id"] == "sum_all"
    assert rows[1]["id"] == "count_value"
    assert rows[2]["id"] == "max_value"
    assert rows[3]["id"] == "min_value"
    assert rows[4]["id"] == "unique_count"
    assert int(summary.get("star_count") or 0) > 0
    assert Path(str(summary.get("report_path"))).exists()


if __name__ == "__main__":
    _ensure_cuda()
    probe_rows = _collect_probe_rows()
    print(
        "PHASE6A_JSON:"
        + json.dumps(
            {
                "star_count": len(probe_rows),
                "report_path": str(REPORT_PATH),
                "rows": [
                    {
                        "index": row["index"],
                        "id": row["id"],
                        "name": row["name"],
                        "verdict": row["verdict"],
                        "host_payload_keys": row["host_payload_keys"],
                        "selection_role_id": row["decoded"]["selection_role_id"],
                        "answer_eligible": row["decoded"]["answer_eligible"],
                    }
                    for row in probe_rows
                ],
            },
            ensure_ascii=True,
        )
    )
