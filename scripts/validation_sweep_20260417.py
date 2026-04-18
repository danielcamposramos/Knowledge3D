#!/usr/bin/env python3
"""Run the 2026-04-17 live-engine validation sweep through the daemon ROUTE path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.math_competitions import math_answers_match
from benchmarks.math_competitions import UnifiedMathBenchmark
from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import MEANING_CLASS_INDEX


OUTPUT_ROOT = REPO_ROOT / "TEMP" / "validation_sweep_2026-04-17"
DAEMON_LOG = OUTPUT_ROOT / "daemon.log"
RING_TRACE_PATH = OUTPUT_ROOT / "ring_trace.jsonl"
JANET_QUERY = "Janet had 16 ducks and bought 2 more. How many ducks does Janet have now?"
MEANING_CLASSES = [
    "FACTUAL_RECALL",
    "DEFINITION_LOOKUP",
    "MULTI_HOP_INFERENCE",
    "NUMERIC_COMPUTE",
    "SPATIAL_TRANSFORM",
    "COMPARATIVE_CHOICE",
    "GROUNDED_DIALOG",
    "GENERATIVE_COMPOSITION",
]
FORBIDDEN_WIRE_KEYS = {"competition", "dataset", "source", "task_type", "surface_kind"}
SWEEP_ORDER = ("mmlu", "gsm8k", "math_competitions", "lhe", "arc_agi_1")
BENCHMARK_BUDGETS = {
    "mmlu": {"max_wall_ms": 45000, "ceiling_s": 15 * 60},
    "gsm8k": {"max_wall_ms": 45000, "ceiling_s": 15 * 60},
    "math_competitions": {"max_wall_ms": 60000, "ceiling_s": 20 * 60},
    "lhe": {"max_wall_ms": 90000, "ceiling_s": 30 * 60},
    "arc_agi_1": {"max_wall_ms": 60000, "ceiling_s": 20 * 60},
}
ROUND_B_MULTI_HOP_PROMPTS = [
    "Janet buys 3 notebooks at $4 each and then uses a $2 coupon. How much does she pay?",
    "A train travels 2 hours at 50 mph and then 1 hour at 30 mph. How far did it go altogether?",
    "If all blue boxes contain 4 marbles and there are 3 blue boxes, how many marbles are there?",
    "Mia reads 12 pages on Monday and twice that on Tuesday. How many pages did she read in total?",
    "A baker makes 18 rolls, sells 7, and packs the rest equally into 11 bags. How many go in each bag?",
]
ROUND_B_DIRECT_COMPUTE_PROMPTS = ["2 + 3 = ?", "sqrt(16)", "14 * 6", "100 / 4", "9 squared"]


class BenchmarkUnavailable(RuntimeError):
    """Raised when a benchmark dataset is not locally runnable."""


class SweepFailure(RuntimeError):
    """Raised when a fail-fast validation invariant is violated."""


class StdioDaemon:
    """JSON-line stdio client that tolerates noisy non-JSON boot logs."""

    def __init__(self, *, storage_root: Path, daemon_log_path: Path):
        env = os.environ.copy()
        env.pop("K3D_BYPASS_GAME_LOOP", None)
        env["K3D_RING_TRACE_PATH"] = str(RING_TRACE_PATH)
        python_path = env.get("PYTHONPATH", "")
        repo_str = str(REPO_ROOT)
        env["PYTHONPATH"] = f"{repo_str}:{python_path}" if python_path else repo_str
        self._log_handle = daemon_log_path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "k3d_daemon.py"),
                "--mode",
                "stdio",
                "--storage-root",
                str(storage_root),
                "--warm-gpu-runtime-on-boot",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("daemon_stdio_unavailable")

    def start(self, *, timeout_sec: float = 1800.0) -> None:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            packet = self._read_json_packet(deadline=deadline)
            if packet is None:
                break
            if str(packet.get("message") or "").strip() == "k3d_daemon_started":
                return
        raise RuntimeError("daemon_boot_timeout")

    def send(self, payload: dict[str, Any], *, timeout_sec: float = 240.0) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise RuntimeError(f"daemon_exited:{self._proc.returncode}")
        wire = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        self._proc.stdin.write(wire + "\n")
        self._proc.stdin.flush()
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            packet = self._read_json_packet(deadline=deadline)
            if packet is None:
                break
            if str(packet.get("message") or "").strip() == "k3d_daemon_started":
                continue
            return packet
        raise RuntimeError(f"daemon_response_timeout:{payload.get('command')}")

    def shutdown(self) -> None:
        try:
            if self._proc.poll() is None:
                try:
                    self.send({"command": "SHUTDOWN"}, timeout_sec=30.0)
                except Exception:
                    pass
        finally:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._log_handle.close()

    def _read_json_packet(self, *, deadline: float) -> dict[str, Any] | None:
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"daemon_exited:{self._proc.returncode}")
                time.sleep(0.05)
                continue
            self._log_handle.write(line)
            self._log_handle.flush()
            text = line.strip()
            if not text:
                continue
            try:
                packet = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(packet, dict):
                return packet
        return None


class TcpDaemonClient:
    """JSON-line TCP client for an already-running daemon instance."""

    def __init__(self, *, host: str, port: int):
        self.host = str(host)
        self.port = int(port)

    def start(self, *, timeout_sec: float = 1800.0) -> None:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            try:
                response = self.send({"command": "PING"}, timeout_sec=30.0)
            except Exception:
                time.sleep(1.0)
                continue
            if str(response.get("status") or "").lower() == "ok":
                return
            time.sleep(1.0)
        raise RuntimeError("daemon_tcp_attach_timeout")

    def send(self, payload: dict[str, Any], *, timeout_sec: float = 240.0) -> dict[str, Any]:
        wire = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.create_connection((self.host, self.port), timeout=float(timeout_sec)) as sock:
            sock.sendall(wire)
            sock.settimeout(float(timeout_sec))
            chunks: list[bytes] = []
            while True:
                block = sock.recv(65536)
                if not block:
                    break
                chunks.append(block)
                if b"\n" in block:
                    break
        if not chunks:
            raise RuntimeError("daemon_tcp_empty_response")
        raw = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace")
        packet = json.loads(raw)
        if not isinstance(packet, dict):
            raise RuntimeError("daemon_tcp_response_not_object")
        return packet

    def shutdown(self) -> None:
        return None


def _run_command(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _clean_old_daemons() -> None:
    subprocess.run(
        ["pkill", "-f", r"knowledge3d\.daemon\.main|scripts/k3d_daemon.py|k3d_daemon"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(1.0)


def _grep_token_in_set_count() -> int:
    try:
        output = _run_command(
            [
                "bash",
                "-lc",
                'grep -rnE "any\\(token in \\{|token in \\{" knowledge3d/knowledgeverse/navigator_specialist.py | wc -l',
            ]
        )
        return int(output or "0")
    except Exception:
        return -1


def _git_head() -> str:
    return _run_command(["git", "rev-parse", "HEAD"])


def _knowledgeverse_line_count() -> int:
    output = _run_command(["wc", "-l", "knowledge3d/knowledgeverse/knowledgeverse.py"])
    return int(output.split()[0])


def _median_ms(values: list[float]) -> int:
    if not values:
        return 0
    return int(round(statistics.median(values)))


def _p95_ms(values: list[float]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return int(round(ordered[0]))
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return int(round(ordered[index]))


def _sample_indexes(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    rng = random.Random(20260417 + total + count)
    return sorted(rng.sample(range(total), count))


def _normalize_scalar_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value).strip()


def _extract_raw_answer(response: dict[str, Any]) -> Any:
    outer = response.get("task_result")
    inner = outer.get("task_result") if isinstance(outer, dict) else None
    for bucket in (inner, outer, response):
        if not isinstance(bucket, dict):
            continue
        for key in ("predicted_answer", "response", "answer", "result"):
            if bucket.get(key) not in (None, ""):
                return bucket.get(key)
    return None


def _extract_arc_grid(response: dict[str, Any]) -> Any:
    outer = response.get("task_result")
    inner = outer.get("task_result") if isinstance(outer, dict) else None
    for bucket in (inner, outer, response):
        if not isinstance(bucket, dict):
            continue
        for key in ("output_grid", "predicted_grid", "predicted"):
            if bucket.get(key) not in (None, ""):
                return bucket.get(key)
        game_action = bucket.get("game_action")
        if isinstance(game_action, dict) and game_action.get("output_grid") is not None:
            return game_action.get("output_grid")
    return None


def _extract_meaning_class(response: dict[str, Any]) -> str:
    outer = response.get("task_result")
    inner = outer.get("task_result") if isinstance(outer, dict) else None
    for bucket in (inner, outer, response.get("route"), response):
        if not isinstance(bucket, dict):
            continue
        token = str(bucket.get("meaning_class") or "").strip().upper()
        if token:
            return token
    return "UNKNOWN"


def _extract_ring_sample(response: dict[str, Any], *, enqueue_ts: float, output_ts: float) -> dict[str, Any]:
    dispatched = response.get("task_result") if isinstance(response.get("task_result"), dict) else {}
    trm_io = dispatched.get("trm_io") if isinstance(dispatched.get("trm_io"), dict) else {}
    trm_tick = dispatched.get("trm_tick") if isinstance(dispatched.get("trm_tick"), dict) else {}
    return {
        "request_id": str(trm_io.get("request_id") or ""),
        "enqueue_ts": enqueue_ts,
        "wait_ticks": 1,
        "output_ts": output_ts,
        "tick": int(trm_io.get("tick") or trm_tick.get("tick") or 0),
        "mode": str(dispatched.get("mode") or ""),
    }


def _task_result_bucket(response: dict[str, Any]) -> dict[str, Any]:
    dispatched = response.get("task_result")
    if not isinstance(dispatched, dict):
        return {}
    nested = dispatched.get("task_result")
    if isinstance(nested, dict):
        return nested
    return dispatched


def _is_wall_timeout(response: dict[str, Any]) -> bool:
    bucket = _task_result_bucket(response)
    return str(bucket.get("failure_code") or "").strip().lower() == "wall_timeout"


def _assert_ring_used(response: dict[str, Any], *, context: str) -> None:
    dispatched = response.get("task_result")
    if not isinstance(dispatched, dict):
        raise SweepFailure(f"{context}:missing_dispatched_result")
    if str(dispatched.get("mode") or "") != "query_tick":
        raise SweepFailure(f"{context}:not_query_tick")
    if not isinstance(dispatched.get("trm_io"), dict):
        raise SweepFailure(f"{context}:missing_trm_io")
    if not isinstance(dispatched.get("trm_tick"), dict):
        raise SweepFailure(f"{context}:missing_trm_tick")
    if not isinstance(dispatched.get("action_buffers"), list):
        raise SweepFailure(f"{context}:missing_action_buffers")


def _assert_no_wire_leakage(payload: dict[str, Any], *, context: str) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    for token in FORBIDDEN_WIRE_KEYS:
        if f'"{token}"' in serialized:
            raise SweepFailure(f"{context}:forbidden_wire_key:{token}")


def _janet_payload() -> dict[str, Any]:
    return {
        "command": "ROUTE",
        "task": {
            "task_id": "janet_probe",
            "query": JANET_QUERY,
            "question": JANET_QUERY,
            "prompt": JANET_QUERY,
            "messages": [{"role": "user", "content": JANET_QUERY}],
            "expected_answer": "18",
        },
        "route_policy": "all_live_galaxies",
        "use_enriched": True,
    }


def _janet_pass(response: dict[str, Any]) -> bool:
    answer = _normalize_scalar_answer(_extract_raw_answer(response))
    return answer == "18"


def _load_mmlu_questions(limit: int) -> list[dict[str, Any]]:
    benchmark = MMLUBenchmark.__new__(MMLUBenchmark)
    benchmark.dataset_path = MMLUBenchmark._resolve_dataset_path(benchmark, None)
    benchmark.max_questions = limit
    benchmark.query_scope_galaxies = []
    benchmark.subjects = []
    benchmark.split = "test"
    benchmark.runtime_seed_knowledge = False
    benchmark.tablet_boundary = None
    benchmark.used_synthetic_fallback = False
    try:
        rows = MMLUBenchmark._load_questions(benchmark)
    except FileNotFoundError as exc:
        raise BenchmarkUnavailable(str(exc)) from exc
    if not rows:
        raise BenchmarkUnavailable("mmlu_dataset_empty")
    return rows


def _load_gsm8k_questions(limit: int) -> list[dict[str, Any]]:
    benchmark = UnifiedMathBenchmark.__new__(UnifiedMathBenchmark)
    benchmark.gsm8k_dataset_path = UnifiedMathBenchmark._resolve_math_dataset_path(benchmark, None)
    benchmark.max_math_questions = limit
    benchmark.dataset_sources = []
    try:
        rows = UnifiedMathBenchmark._load_word_math_problems(benchmark)
    except FileNotFoundError as exc:
        raise BenchmarkUnavailable(str(exc)) from exc
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "id": str(row.get("id") or ""),
                "question_text": str(row.get("problem_text") or ""),
                "correct_answer": str(row.get("answer") or ""),
            }
        )
    if not normalized:
        raise BenchmarkUnavailable("gsm8k_dataset_empty")
    return normalized


def _extract_word_math_answer(raw_answer: Any) -> str | None:
    text = str(raw_answer or "").strip()
    if not text:
        return None
    marker = "####"
    if marker in text:
        return text.split(marker, 1)[1].splitlines()[0].strip().replace(",", "")
    tail = text.splitlines()[-1].strip()
    return tail or None


def _extract_last_boxed(text: str) -> str | None:
    last_pos = -1
    marker = ""
    for candidate in (r"\boxed{", r"\fbox{"):
        pos = text.rfind(candidate)
        if pos > last_pos:
            last_pos = pos
            marker = candidate
    if last_pos < 0:
        return None
    start = last_pos + len(marker)
    depth = 1
    out: list[str] = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


def _extract_symbolic_math_answer(solution: Any) -> str | None:
    text = str(solution or "").strip()
    if not text:
        return None
    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed.strip()
    if "####" in text:
        return text.split("####", 1)[1].splitlines()[0].strip()
    tail = text.splitlines()[-1].strip().rstrip(".")
    if "=" in tail:
        return tail.rsplit("=", 1)[1].strip()
    return tail or None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_math_competition_questions(limit: int) -> list[dict[str, Any]]:
    benchmark = UnifiedMathBenchmark.__new__(UnifiedMathBenchmark)
    benchmark.dataset_mode = "present"
    benchmark.dataset_path = UnifiedMathBenchmark._resolve_dataset_path(benchmark, None)
    benchmark.max_problems = limit
    benchmark.dataset_sources = []
    try:
        rows = UnifiedMathBenchmark._load_competition_math_problems(benchmark)
    except FileNotFoundError as exc:
        raise BenchmarkUnavailable(str(exc)) from exc
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "id": str(row.get("id") or ""),
                "question_text": str(row.get("problem_text") or ""),
                "correct_answer": str(row.get("answer") or ""),
                "domain_hint": str(row.get("competition") or row.get("source_key") or "competition_math"),
            }
        )
    if not normalized:
        raise BenchmarkUnavailable("math_competitions_dataset_missing")
    return normalized


def _load_lhe_questions(limit: int) -> list[dict[str, Any]]:
    benchmark = LastHumanityExamBenchmark.__new__(LastHumanityExamBenchmark)
    benchmark.dataset_path = LastHumanityExamBenchmark._resolve_dataset_path(benchmark, None)
    benchmark.max_questions = limit
    benchmark.dataset_source = "unknown"
    benchmark.dataset_file = None
    benchmark.synthetic_fallback = False
    try:
        rows = LastHumanityExamBenchmark._load_questions(benchmark)
    except FileNotFoundError as exc:
        raise BenchmarkUnavailable(str(exc)) from exc
    if not rows:
        raise BenchmarkUnavailable("lhe_dataset_unavailable:gated_or_missing_payload")
    return rows


def _load_arc_questions(limit: int) -> list[dict[str, Any]]:
    benchmark = ARCAGI2Benchmark.__new__(ARCAGI2Benchmark)
    benchmark.dataset_version = "arc_agi_1"
    benchmark.dataset_path = ARCAGI2Benchmark._resolve_dataset_path(benchmark, None, "arc_agi_1")
    benchmark.max_tasks = limit
    try:
        rows = ARCAGI2Benchmark._load_tasks(benchmark)
    except FileNotFoundError as exc:
        raise BenchmarkUnavailable(str(exc)) from exc
    if not rows:
        raise BenchmarkUnavailable("arc_dataset_empty")
    return rows


def _build_route_payload(benchmark: str, row: dict[str, Any]) -> dict[str, Any]:
    if benchmark == "mmlu":
        query = str(row["question_text"])
        task = {
            "task_id": str(row["id"]),
            "query": query,
            "question": query,
            "prompt": query,
            "messages": [{"role": "user", "content": query}],
            "options": list(row["options"]),
            "choices": list(row["options"]),
            "subject": str(row["subject"]),
            "expected_answer": str(row["correct_answer"]),
        }
        return {
            "command": "ROUTE",
            "task": task,
            "domain_hint": str(row["subject"]),
            "route_policy": "all_live_galaxies",
            "use_enriched": True,
        }
    if benchmark == "gsm8k":
        query = str(row["question_text"])
        task = {
            "task_id": str(row["id"]),
            "query": query,
            "question": query,
            "prompt": query,
            "messages": [{"role": "user", "content": query}],
            "expected_answer": str(row["correct_answer"]),
        }
        return {
            "command": "ROUTE",
            "task": task,
            "domain_hint": "math",
            "route_policy": "all_live_galaxies",
            "use_enriched": True,
        }
    if benchmark == "math_competitions":
        query = str(row["question_text"])
        task = {
            "task_id": str(row["id"]),
            "query": query,
            "question": query,
            "prompt": query,
            "messages": [{"role": "user", "content": query}],
            "expected_answer": str(row["correct_answer"]),
        }
        return {
            "command": "ROUTE",
            "task": task,
            "domain_hint": str(row.get("domain_hint") or "competition_math"),
            "route_policy": "all_live_galaxies",
            "use_enriched": True,
        }
    if benchmark == "lhe":
        query = str(row["question_text"])
        task = {
            "task_id": str(row["id"]),
            "query": query,
            "question": query,
            "prompt": query,
            "messages": [{"role": "user", "content": query}],
            "expected_answer": str(row["correct_answer"]),
        }
        options = row.get("options")
        if isinstance(options, list) and options:
            task["options"] = list(options)
            task["choices"] = list(options)
        return {
            "command": "ROUTE",
            "task": task,
            "domain_hint": str(row.get("domain") or "multi"),
            "route_policy": "all_live_galaxies",
            "use_enriched": True,
        }
    if benchmark == "arc_agi_1":
        prompt = "Infer the transformation from the training examples and produce the output grid for the input grid."
        task = {
            "task_id": str(row["id"]),
            "query": prompt,
            "question": prompt,
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "training_examples": list(row["train"]),
            "input_grid": row["test"][0].get("input"),
            "expected_output": row["test"][0].get("output"),
        }
        return {
            "command": "ROUTE",
            "task": task,
            "domain_hint": "visual",
            "route_policy": "all_live_galaxies",
            "use_enriched": True,
        }
    raise KeyError(benchmark)


def _score_item(benchmark: str, row: dict[str, Any], response: dict[str, Any]) -> tuple[bool, Any]:
    if benchmark == "mmlu":
        raw_answer = _extract_raw_answer(response)
        predicted = MMLUBenchmark._normalize_option_prediction(raw_answer, list(row["options"]))
        return predicted == row["correct_answer"], predicted
    if benchmark == "gsm8k":
        predicted = _extract_raw_answer(response)
        return math_answers_match(predicted, row["correct_answer"]), predicted
    if benchmark == "math_competitions":
        predicted = _extract_raw_answer(response)
        return math_answers_match(predicted, row["correct_answer"]), predicted
    if benchmark == "lhe":
        raw = _normalize_scalar_answer(_extract_raw_answer(response))
        options = list(row.get("options") or [])
        helper = LastHumanityExamBenchmark.__new__(LastHumanityExamBenchmark)
        if options:
            predicted = helper._normalize_option_prediction(raw, options)
            return predicted == row["correct_answer"], predicted
        predicted = raw
        return helper._open_ended_match(predicted, str(row["correct_answer"])), predicted
    if benchmark == "arc_agi_1":
        predicted = _extract_arc_grid(response)
        expected = row["test"][0].get("output")
        return predicted == expected, predicted
    raise KeyError(benchmark)


def _benchmark_json_name(benchmark: str) -> str:
    return {
        "mmlu": "mmlu.json",
        "gsm8k": "gsm8k.json",
        "math_competitions": "math_competitions.json",
        "lhe": "lhe.json",
        "arc_agi_1": "arc_agi_1.json",
    }[benchmark]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _ring_trace_line_count() -> int:
    if not RING_TRACE_PATH.exists():
        return 0
    try:
        output = _run_command(["wc", "-l", str(RING_TRACE_PATH)])
        return int(output.split()[0])
    except Exception:
        return 0


def _render_summary(
    *,
    head: str,
    line_count: int,
    janet_t0: bool,
    janet_t_end: bool,
    sanity: dict[str, tuple[bool, str]],
    benchmark_payloads: dict[str, dict[str, Any]],
    ring_trace_lines: int,
    commentary: str,
) -> str:
    lines: list[str] = []
    lines.append(f"`git rev-parse HEAD`: `{head}`")
    lines.append(f"`wc -l knowledge3d/knowledgeverse/knowledgeverse.py`: `{line_count}`")
    lines.append("")
    lines.append("Accuracies:")
    for benchmark in SWEEP_ORDER:
        payload = benchmark_payloads.get(benchmark)
        if not payload:
            continue
        lines.append(
            f"- `{benchmark}`: `{payload.get('correct', 0)}/{payload.get('items', 0)}` "
            f"(`{payload.get('accuracy', 0.0):.3f}`)"
        )
    lines.append("")
    lines.append("Sanity Criteria:")
    for key, (ok, note) in sanity.items():
        lines.append(f"- {'✅' if ok else '❌'} `{key}`: {note}")
    lines.append("")
    lines.append("Stall Ledger:")
    for benchmark in SWEEP_ORDER:
        payload = benchmark_payloads.get(benchmark)
        if not payload:
            continue
        lines.append(
            f"- `{benchmark}`: stalled_at_item=`{payload.get('stalled_at_item')}`, "
            f"wall_timeouts=`{payload.get('wall_timeouts', 0)}`, "
            f"wall_ceiling_hit=`{bool(payload.get('wall_ceiling_hit', False))}`, "
            f"produced_outputs=`{payload.get('produced_outputs', 0)}`"
        )
    lines.append("")
    lines.append(f"Janet T0: {'PASS' if janet_t0 else 'FAIL'}")
    lines.append(f"Janet T_end: {'PASS' if janet_t_end else 'FAIL'}")
    lines.append(f"`wc -l TEMP/validation_sweep_2026-04-17/ring_trace.jsonl`: `{ring_trace_lines}`")
    lines.append("")
    lines.append(commentary.strip())
    return "\n".join(lines).strip() + "\n"


def _load_previous_payloads() -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for benchmark in SWEEP_ORDER:
        path = OUTPUT_ROOT / _benchmark_json_name(benchmark)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads[benchmark] = payload
    return payloads


def _aggregate_meaning_counts(payloads: dict[str, dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for payload in payloads.values():
        for name, count in dict(payload.get("meaning_class_argmax_counts") or {}).items():
            counts[str(name)] += int(count or 0)
    return counts


def _navigator_softmax_snapshot(storage_root: Path) -> tuple[list[dict[str, Any]], str]:
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
        include_runtime_artifacts=False,
        include_runtime_language_enrichment=False,
    )
    rows: list[dict[str, Any]] = []
    try:
        for label, prompts in (("multi_hop", ROUND_B_MULTI_HOP_PROMPTS), ("direct_compute", ROUND_B_DIRECT_COMPUTE_PROMPTS)):
            for prompt in prompts:
                task = {"query": prompt, "question": prompt, "prompt": prompt}
                _, meaning_dist, _, _ = kv._navigator_emission(
                    query_embedding=kv._embed_query_gpu(prompt, task=task),
                    task=task,
                    query_text=prompt,
                    options=None,
                    stars=None,
                )
                rows.append(
                    {
                        "bucket": label,
                        "prompt": prompt,
                        "multi_hop": float(meaning_dist[MEANING_CLASS_INDEX["MULTI_HOP_INFERENCE"]]),
                        "numeric_compute": float(meaning_dist[MEANING_CLASS_INDEX["NUMERIC_COMPUTE"]]),
                        "winner": MEANING_CLASSES[max(range(len(meaning_dist)), key=meaning_dist.__getitem__)],
                    }
                )
    finally:
        try:
            kv._trm_game_loop.stop()
        except Exception:
            pass
    multi_rows = [row for row in rows if row["bucket"] == "multi_hop"]
    wins = sum(1 for row in multi_rows if float(row["multi_hop"]) > float(row["numeric_compute"]))
    if rows and all(abs(float(row["multi_hop"]) - float(row["numeric_compute"])) <= 1.0e-9 for row in rows):
        note = "Navigator is flat: multi-hop and numeric softmax are tied across all 10 prompts."
    else:
        note = f"MULTI_HOP_INFERENCE beats NUMERIC_COMPUTE on {wins}/{len(multi_rows)} multi-hop prompts."
    return rows, note


def _round_b_arc_verdict(payload: dict[str, Any]) -> str:
    sampled = list(payload.get("sampled_outputs") or [])[:3]
    if not sampled:
        return "no ARC samples captured"
    executors = sorted(
        {
            str(
                ((row.get("task_result") or {}).get("task_result") or {}).get("executor_star")
                or (row.get("route") or {}).get("executor_star")
                or ""
            ).strip()
            for row in sampled
        }
        - {""}
    )
    validators = sorted(
        {
            str(
                ((row.get("task_result") or {}).get("task_result") or {}).get("validator_star")
                or (row.get("route") or {}).get("validator_star")
                or ""
            ).strip()
            for row in sampled
        }
        - {""}
    )
    swarm_roles = sorted(
        {
            tuple(((row.get("task_result") or {}).get("task_result") or {}).get("trace_roles") or (row.get("route") or {}).get("trace_roles") or [])
            for row in sampled
        }
    )
    return (
        f"executors={executors or ['none']}, validators={validators or ['none']}, "
        f"trace_roles={swarm_roles or [('none',)]}"
    )


def _render_round_b_delta(
    *,
    previous_payloads: dict[str, dict[str, Any]],
    current_payloads: dict[str, dict[str, Any]],
    softmax_rows: list[dict[str, Any]],
    softmax_note: str,
) -> str:
    previous_counts = _aggregate_meaning_counts(previous_payloads)
    current_counts = _aggregate_meaning_counts(current_payloads)
    previous_timeouts = sum(int(payload.get("wall_timeouts", 0) or 0) for payload in previous_payloads.values())
    current_timeouts = sum(int(payload.get("wall_timeouts", 0) or 0) for payload in current_payloads.values())
    lines = ["### Round B delta", ""]
    lines.append(f"- meaning counts before: `{dict(sorted(previous_counts.items()))}`")
    lines.append(f"- meaning counts after: `{dict(sorted(current_counts.items()))}`")
    lines.append(f"- wall_timeouts before: `{previous_timeouts}`")
    lines.append(f"- wall_timeouts after: `{current_timeouts}`")
    lines.append(f"- ARC executor verdict: `{_round_b_arc_verdict(current_payloads.get('arc_agi_1', {}))}`")
    lines.append("")
    lines.append("### Round B.3 — multi-hop separability")
    for row in softmax_rows:
        lines.append(
            f"- `{row['bucket']}` `{row['prompt']}` -> "
            f"MULTI_HOP_INFERENCE=`{row['multi_hop']:.6f}`, "
            f"NUMERIC_COMPUTE=`{row['numeric_compute']:.6f}`, "
            f"winner=`{row['winner']}`"
        )
    lines.append(f"- {softmax_note}")
    return "\n".join(lines).strip() + "\n"


def run_sweep(
    *,
    storage_root: Path,
    limit: int,
    daemon_host: str | None = None,
    daemon_port: int = 7777,
) -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    previous_payloads = _load_previous_payloads()
    managed_stdio = daemon_host is None
    if managed_stdio:
        _clean_old_daemons()
        try:
            RING_TRACE_PATH.unlink()
        except FileNotFoundError:
            pass

    head = _git_head()
    line_count = _knowledgeverse_line_count()
    token_count_t0 = _grep_token_in_set_count()
    if os.environ.get("K3D_BYPASS_GAME_LOOP"):
        raise SweepFailure("K3D_BYPASS_GAME_LOOP_must_be_unset")

    if managed_stdio:
        daemon: Any = StdioDaemon(storage_root=storage_root, daemon_log_path=DAEMON_LOG)
    else:
        daemon = TcpDaemonClient(host=str(daemon_host), port=int(daemon_port))
    sweep_started_wall = time.time()
    try:
        daemon.start()
        status0 = daemon.send({"command": "PING"}, timeout_sec=600.0)
        tick0 = daemon.send({"command": "TICK_STATUS"}, timeout_sec=60.0)
        time.sleep(1.0)
        tick1 = daemon.send({"command": "TICK_STATUS"}, timeout_sec=60.0)
        ticks0 = int(((tick0.get("tick_driver") or {}) if isinstance(tick0, dict) else {}).get("ticks_total", 0) or 0)
        ticks1 = int(((tick1.get("tick_driver") or {}) if isinstance(tick1, dict) else {}).get("ticks_total", 0) or 0)
        tick1_driver = (tick1.get("tick_driver") or {}) if isinstance(tick1, dict) else {}
        if not bool(tick1_driver.get("active", tick1_driver.get("running", False))):
            raise SweepFailure("tick_driver_inactive")
        if ticks1 <= ticks0:
            raise SweepFailure(f"tick_driver_not_advancing:{ticks0}->{ticks1}")

        janet_start_response = daemon.send(_janet_payload(), timeout_sec=600.0)
        janet_t0 = _janet_pass(janet_start_response)
        if not janet_t0:
            raise SweepFailure("janet_t0_failed")

        binding = status0.get("gpu_binding") if isinstance(status0.get("gpu_binding"), dict) else {}
        default_counts = status0.get("default_galaxy_counts") if isinstance(status0.get("default_galaxy_counts"), dict) else {}

        benchmark_payloads: dict[str, dict[str, Any]] = {}
        all_meaning_counts: Counter[str] = Counter()
        wire_samples_all: list[dict[str, Any]] = []
        ring_check_notes: list[str] = []

        loaders = {
            "mmlu": lambda: _load_mmlu_questions(limit),
            "gsm8k": lambda: _load_gsm8k_questions(limit),
            "math_competitions": lambda: _load_math_competition_questions(limit),
            "lhe": lambda: _load_lhe_questions(limit),
            "arc_agi_1": lambda: _load_arc_questions(limit),
        }

        for benchmark in SWEEP_ORDER:
            budget = BENCHMARK_BUDGETS[benchmark]
            try:
                rows = loaders[benchmark]()
            except BenchmarkUnavailable as exc:
                payload = {
                    "benchmark": benchmark,
                    "items": limit,
                    "correct": 0,
                    "incorrect": 0,
                    "errors": limit,
                    "accuracy": 0.0,
                    "median_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "stalled_at_item": None,
                    "wall_ceiling_hit": False,
                    "wall_timeouts": 0,
                    "produced_outputs": 0,
                    "meaning_class_argmax_counts": {name: 0 for name in MEANING_CLASSES},
                    "sampled_envelopes": [],
                    "sampled_outputs": [],
                    "ring_samples": [],
                    "tick_stats_at_end": daemon.send({"command": "TICK_STATUS"}, timeout_sec=60.0).get("tick_driver", {}),
                    "error": str(exc),
                }
                benchmark_payloads[benchmark] = payload
                _write_json(OUTPUT_ROOT / _benchmark_json_name(benchmark), payload)
                continue

            latencies_ms: list[float] = []
            responses: list[dict[str, Any]] = []
            envelopes: list[dict[str, Any]] = []
            ring_samples: list[dict[str, Any]] = []
            errors = 0
            correct = 0
            wall_timeouts = 0
            produced_outputs = 0
            wall_ceiling_hit = False
            stalled_at_item: int | None = None
            benchmark_started = time.monotonic()

            for index, row in enumerate(rows):
                if (time.monotonic() - benchmark_started) >= float(budget["ceiling_s"]):
                    wall_ceiling_hit = True
                    stalled_at_item = index
                    break
                payload = _build_route_payload(benchmark, row)
                payload["max_wall_ms"] = int(budget["max_wall_ms"])
                _assert_no_wire_leakage(payload, context=f"{benchmark}:{index}")
                if len(wire_samples_all) < 5:
                    wire_samples_all.append({"benchmark": benchmark, "index": index, "payload": payload})
                t0 = time.time()
                try:
                    response = daemon.send(
                        payload,
                        timeout_sec=max(240.0, float(budget["max_wall_ms"]) / 1000.0 + 30.0),
                    )
                except Exception as exc:
                    errors += 1
                    if stalled_at_item is None:
                        stalled_at_item = index
                    response = {"status": "error", "error": str(exc)}
                    responses.append(response)
                    envelopes.append(payload)
                    continue
                t1 = time.time()
                _assert_ring_used(response, context=f"{benchmark}:{index}")
                ring_check_notes.append(f"{benchmark}:{index}")
                ok, _predicted = _score_item(benchmark, row, response)
                is_wall_timeout = _is_wall_timeout(response)
                if is_wall_timeout:
                    wall_timeouts += 1
                    if stalled_at_item is None:
                        stalled_at_item = index
                correct += int(ok)
                produced_outputs += 1
                latencies_ms.append((t1 - t0) * 1000.0)
                responses.append(response)
                envelopes.append(payload)
                meaning_class = _extract_meaning_class(response)
                all_meaning_counts[meaning_class] += 1
                if len(ring_samples) < 3:
                    ring_samples.append(_extract_ring_sample(response, enqueue_ts=t0, output_ts=t1))

            sampled_indexes = _sample_indexes(len(responses), 3)
            sampled_envelopes = [envelopes[index] for index in sampled_indexes]
            sampled_outputs = [responses[index] for index in sampled_indexes]
            completed_items = len(responses)
            incorrect = max(0, completed_items - correct - errors)
            bench_meaning = Counter(_extract_meaning_class(response) for response in responses if response.get("status") == "ok")
            tick_stats = daemon.send({"command": "TICK_STATUS"}, timeout_sec=60.0).get("tick_driver", {})
            payload = {
                "benchmark": benchmark,
                "items": completed_items,
                "correct": correct,
                "incorrect": incorrect,
                "errors": errors,
                "accuracy": (correct / completed_items) if completed_items else 0.0,
                "median_latency_ms": _median_ms(latencies_ms),
                "p95_latency_ms": _p95_ms(latencies_ms),
                "stalled_at_item": stalled_at_item,
                "wall_ceiling_hit": wall_ceiling_hit,
                "wall_timeouts": wall_timeouts,
                "produced_outputs": produced_outputs,
                "meaning_class_argmax_counts": {
                    name: int(bench_meaning.get(name, 0))
                    for name in MEANING_CLASSES
                } | {
                    name: int(count)
                    for name, count in sorted(bench_meaning.items())
                    if name not in MEANING_CLASSES
                },
                "sampled_envelopes": sampled_envelopes,
                "sampled_outputs": sampled_outputs,
                "ring_samples": ring_samples,
                "tick_stats_at_end": tick_stats,
            }
            benchmark_payloads[benchmark] = payload
            _write_json(OUTPUT_ROOT / _benchmark_json_name(benchmark), payload)

        janet_end_response = daemon.send(_janet_payload(), timeout_sec=600.0)
        janet_t_end = _janet_pass(janet_end_response)
        tick_end = daemon.send({"command": "TICK_STATUS"}, timeout_sec=60.0)
        token_count_t_end = _grep_token_in_set_count()

        wall_seconds = max(1.0, time.time() - sweep_started_wall)
        tick_stats = tick_end.get("tick_driver") if isinstance(tick_end.get("tick_driver"), dict) else {}
        ticks_total = int(tick_stats.get("ticks_total", 0) or 0)
        distinct_meanings = {name for name, count in all_meaning_counts.items() if count > 0 and name != "UNKNOWN"}
        wire_ok = True
        wire_note = "5 sampled payloads contained no forbidden keys."
        for item in wire_samples_all:
            try:
                _assert_no_wire_leakage(item["payload"], context=f"wire_sample:{item['benchmark']}:{item['index']}")
            except SweepFailure as exc:
                wire_ok = False
                wire_note = str(exc)
                break

        ring_ok = all(
            benchmark_payloads.get(name, {}).get("errors", 0) == 0
            or "error" in benchmark_payloads.get(name, {})
            for name in SWEEP_ORDER
        ) and len(ring_check_notes) >= sum(
            min(3, int(benchmark_payloads.get(name, {}).get("items", 0) or 0))
            for name in SWEEP_ORDER
            if "error" not in benchmark_payloads.get(name, {})
        )
        ring_note = (
            "All live benchmark items returned query_tick/trm_io/action_buffers."
            if ring_ok
            else "At least one live item missed the ring evidence."
        )

        produced_total = sum(
            int(benchmark_payloads.get(name, {}).get("produced_outputs", 0) or 0)
            for name in SWEEP_ORDER
        ) + 2
        tick_bounds_ok = (
            int(tick_stats.get("error_ticks", 0) or 0) == 0
            and int(ticks_total) >= int(produced_total)
            and int(ticks_total) <= int(max(1.0, wall_seconds) * 50.0)
        )

        sanity = {
            "ring_used": (ring_ok, ring_note),
            "meaning_class_spread": (
                len(distinct_meanings) >= 4,
                f"{len(distinct_meanings)} distinct argmax classes across {sum(all_meaning_counts.values())} routed items.",
            ),
            "wire_leakage": (wire_ok, wire_note),
            "tickdriver_bounds": (
                tick_bounds_ok,
                f"ticks_total={ticks_total}, produced_total={produced_total}, error_ticks={tick_stats.get('error_ticks', 0)}, wall_seconds={wall_seconds:.1f}.",
            ),
            "janet_integrity": (
                janet_t0 and janet_t_end,
                f"T0={'PASS' if janet_t0 else 'FAIL'}, T_end={'PASS' if janet_t_end else 'FAIL'}.",
            ),
            "token_in_set_count": (
                token_count_t0 == 1 and token_count_t_end == 1,
                f"pre={token_count_t0}, post={token_count_t_end}.",
            ),
        }

        ring_trace_lines = _ring_trace_line_count()
        commentary = (
            f"Daemon booted with default galaxies loaded={bool(default_counts)} and gpu_binding_total="
            f"{binding.get('total', 0) if isinstance(binding, dict) else 0}. "
            f"The sweep preserved the live ring path where datasets were locally runnable, "
            f"and the ring trace recorded `{ring_trace_lines}` events. "
            f"HLE remains blocked in this checkout if its payload stayed unavailable, because the local `hle-src` repo "
            f"contains only the evaluation scaffold and the gated `cais/hle` split cannot be fetched anonymously."
        )

        summary_text = _render_summary(
            head=head,
            line_count=line_count,
            janet_t0=janet_t0,
            janet_t_end=janet_t_end,
            sanity=sanity,
            benchmark_payloads=benchmark_payloads,
            ring_trace_lines=ring_trace_lines,
            commentary=commentary,
        )
        softmax_rows, softmax_note = _navigator_softmax_snapshot(storage_root)
        summary_text += "\n" + _render_round_b_delta(
            previous_payloads=previous_payloads,
            current_payloads=benchmark_payloads,
            softmax_rows=softmax_rows,
            softmax_note=softmax_note,
        )
        (OUTPUT_ROOT / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    finally:
        daemon.shutdown()

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=Path("/K3D/Knowledge3D.local"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--daemon-host", type=str, default=None)
    parser.add_argument("--daemon-port", type=int, default=7777)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_sweep(
        storage_root=args.storage_root,
        limit=max(1, int(args.limit)),
        daemon_host=args.daemon_host,
        daemon_port=int(args.daemon_port),
    )


if __name__ == "__main__":
    raise SystemExit(main())
