"""Persistent K3D daemon entrypoint (game-style runtime).

The daemon keeps one Knowledgeverse + TRM instance alive and serves JSON
commands over stdio or TCP line protocol. This avoids one-shot script
orchestration and enforces a single-world process lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.specialists.math_specialist import MathSpecialist
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge
from knowledge3d.cranium.bridges.procedural_material_bridge import ProceduralMaterialBridge

try:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
except Exception:  # pragma: no cover
    ModularRPNEngine = None  # type: ignore[assignment]

try:
    from knowledge3d.cranium.sovereign.loader import get_vram_usage
except Exception:  # pragma: no cover
    get_vram_usage = None  # type: ignore[assignment]

try:
    from knowledge3d.gpu.perf_counters import gpu_utilisation
except Exception:  # pragma: no cover
    gpu_utilisation = None  # type: ignore[assignment]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_env_path(var_name: str, path_value: str) -> None:
    current = os.environ.get(var_name, "").strip()
    if not current:
        os.environ[var_name] = path_value
        return
    items = [item for item in current.split(":") if item]
    if path_value in items:
        return
    os.environ[var_name] = f"{current}:{path_value}"


def _configure_cuda_include_paths() -> dict[str, Any]:
    """
    Ensure NVRTC can resolve core CUDA headers (e.g., cuda_fp16.h).

    This is a daemon-level safeguard for sovereign PTX query/runtime paths:
    we do not enable fallbacks; we only make required CUDA include paths explicit.
    """
    include_candidates: list[Path] = [
        Path("/usr/local/cuda/include"),
        Path("/usr/include"),
    ]
    include_candidates.extend(sorted(Path("/usr/local").glob("cuda*/include")))

    selected: Path | None = None
    for inc in include_candidates:
        if not inc.exists():
            continue
        if (inc / "cuda_fp16.h").exists():
            selected = inc
            break

    configured = {"applied": False, "include_path": None, "cuda_path": None}
    if selected is None:
        return configured

    include_str = str(selected)
    _append_env_path("CPATH", include_str)
    _append_env_path("CPLUS_INCLUDE_PATH", include_str)

    # Derive CUDA_PATH from include parent when possible.
    cuda_root = selected.parent if selected.name == "include" else selected
    if cuda_root.exists() and not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = str(cuda_root)

    configured["applied"] = True
    configured["include_path"] = include_str
    configured["cuda_path"] = os.environ.get("CUDA_PATH")
    return configured


@dataclass
class DaemonConfig:
    storage_root: Path
    require_ptx_query: bool = True
    eager_load_default_galaxies: bool = True
    host: str = "127.0.0.1"
    port: int = 7777


class K3DDaemon:
    """Long-lived command server for K3D runtime orchestration."""

    def __init__(
        self,
        config: DaemonConfig,
        *,
        knowledgeverse: Knowledgeverse | None = None,
        math_specialist: MathSpecialist | None = None,
    ):
        self.config = config
        self.started_at = _now_iso()
        self._shutdown_requested = False
        self._command_count = 0
        self._gpu_calls_total = 0
        self._cuda_env = _configure_cuda_include_paths()
        self._repo_root = Path(__file__).resolve().parents[2]
        self._boot_status_paths = [
            config.storage_root / "runtime" / "runtime_boot.json",
            self._repo_root / "viewer" / "public" / "runtime_boot.json",
        ]
        self._drawing_bridge: ProceduralDrawingBridge | None = None
        self._geometry_bridge: ProceduralGeometryBridge | None = None
        self._material_bridge: ProceduralMaterialBridge | None = None
        self._drawing_warmup: dict[str, Any] = {}
        self._geometry_warmup: dict[str, Any] = {}
        self._material_warmup: dict[str, Any] = {}
        self._write_boot_status(stage="daemon_boot", progress=0.05, state="starting")

        os.environ["K3D_REQUIRE_PTX_QUERY"] = "true" if config.require_ptx_query else "false"

        self._write_boot_status(stage="knowledgeverse_load", progress=0.2, state="loading")
        self.kv = knowledgeverse or Knowledgeverse(
            storage_root=config.storage_root,
            eager_load_default_galaxies=config.eager_load_default_galaxies,
        )
        self.trm = self.kv.trm_navigator
        self.math_specialist = math_specialist or MathSpecialist(knowledgeverse=self.kv, parent=self.trm)
        self._default_counts = self.kv.ensure_default_galaxies_loaded()
        self._write_boot_status(
            stage="knowledgeverse_ready",
            progress=0.55,
            state="loading",
            extra={"default_galaxy_counts": dict(self._default_counts)},
        )
        self._warmup_boot_runtime()
        self._write_boot_status(
            stage="ready",
            progress=1.0,
            state="ready",
            extra={
                "drawing_warmup": dict(self._drawing_warmup),
                "geometry_warmup": dict(self._geometry_warmup),
                "material_warmup": dict(self._material_warmup),
            },
        )

    def _write_boot_status(
        self,
        *,
        stage: str,
        progress: float,
        state: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": "ok",
            "state": state,
            "stage": stage,
            "progress": max(0.0, min(1.0, float(progress))),
            "timestamp": _now_iso(),
            "pid": int(os.getpid()),
        }
        if extra:
            payload.update(extra)
        for path in self._boot_status_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            except Exception:
                continue

    def _warmup_boot_runtime(self) -> None:
        if os.environ.get("K3D_WARMUP_DRAWING", "1") != "1":
            self._drawing_warmup = {"status": "skipped", "reason": "K3D_WARMUP_DRAWING=0"}
        else:
            self._write_boot_status(stage="drawing_runtime_warmup", progress=0.72, state="warming")
            try:
                self._drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=64)
                self._drawing_warmup = self._drawing_bridge.warmup_runtime()
                self._write_boot_status(
                    stage="drawing_runtime_warm",
                    progress=0.84,
                    state="warming",
                    extra={"drawing_warmup": dict(self._drawing_warmup)},
                )
            except Exception as exc:
                self._drawing_warmup = {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                self._write_boot_status(
                    stage="drawing_runtime_warmup_failed",
                    progress=0.84,
                    state="warning",
                    extra={"drawing_warmup": dict(self._drawing_warmup)},
                )

        if os.environ.get("K3D_WARMUP_GEOMETRY", "1") != "1":
            self._geometry_warmup = {"status": "skipped", "reason": "K3D_WARMUP_GEOMETRY=0"}
        else:
            self._write_boot_status(
                stage="geometry_runtime_warmup",
                progress=0.9,
                state="warming",
                extra={"drawing_warmup": dict(self._drawing_warmup)},
            )
            try:
                self._geometry_bridge = ProceduralGeometryBridge()
                self._geometry_warmup = self._geometry_bridge.warmup_runtime()
                self._write_boot_status(
                    stage="geometry_runtime_warm",
                    progress=0.96,
                    state="warming",
                    extra={
                        "drawing_warmup": dict(self._drawing_warmup),
                        "geometry_warmup": dict(self._geometry_warmup),
                    },
                )
            except Exception as exc:
                self._geometry_warmup = {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                self._write_boot_status(
                    stage="geometry_runtime_warmup_failed",
                    progress=0.96,
                    state="warning",
                    extra={
                        "drawing_warmup": dict(self._drawing_warmup),
                        "geometry_warmup": dict(self._geometry_warmup),
                    },
                )

        if os.environ.get("K3D_WARMUP_MATERIAL", "1") != "1":
            self._material_warmup = {"status": "skipped", "reason": "K3D_WARMUP_MATERIAL=0"}
            return

        self._write_boot_status(
            stage="material_runtime_warmup",
            progress=0.985,
            state="warming",
            extra={
                "drawing_warmup": dict(self._drawing_warmup),
                "geometry_warmup": dict(self._geometry_warmup),
            },
        )
        try:
            self._material_bridge = ProceduralMaterialBridge()
            self._material_warmup = self._material_bridge.warmup_runtime()
            self._write_boot_status(
                stage="material_runtime_warm",
                progress=0.995,
                state="warming",
                extra={
                    "drawing_warmup": dict(self._drawing_warmup),
                    "geometry_warmup": dict(self._geometry_warmup),
                    "material_warmup": dict(self._material_warmup),
                },
            )
        except Exception as exc:
            self._material_warmup = {
                "status": "error",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
            self._write_boot_status(
                stage="material_runtime_warmup_failed",
                progress=0.995,
                state="warning",
                extra={
                    "drawing_warmup": dict(self._drawing_warmup),
                    "geometry_warmup": dict(self._geometry_warmup),
                    "material_warmup": dict(self._material_warmup),
                },
            )

    def _gpu_snapshot(self) -> dict[str, Any]:
        used = 0
        total = 0
        util = 0.0
        if get_vram_usage is not None:
            try:
                used, total = get_vram_usage()
            except Exception:
                used, total = 0, 0
        if gpu_utilisation is not None:
            try:
                util = float(gpu_utilisation(default=0.0))
            except Exception:
                util = 0.0
        return {
            "vram_used_bytes": int(used),
            "vram_total_bytes": int(total),
            "gpu_utilization": float(util),
        }

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_requested

    def status_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "timestamp": _now_iso(),
            "daemon_started_at": self.started_at,
            "pid": int(os.getpid()),
            "require_ptx_query": bool(self.config.require_ptx_query),
            "manifest_version": str(self.kv.manifest_version),
            "default_galaxy_counts": dict(self._default_counts),
            "command_count": int(self._command_count),
            "gpu_calls_total": int(self._gpu_calls_total),
            "cuda_env": dict(self._cuda_env),
            "drawing_warmup": dict(self._drawing_warmup),
            "geometry_warmup": dict(self._geometry_warmup),
            "material_warmup": dict(self._material_warmup),
            "boot_status_paths": [str(path) for path in self._boot_status_paths],
        }

    def _gpu_call_snapshot(self) -> int:
        if ModularRPNEngine is None:
            return 0
        try:
            return int(ModularRPNEngine.get_global_gpu_call_count())
        except Exception:
            return 0

    def _dispatch_task(self, *, route: dict[str, Any], task: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        specialist = str(route.get("specialist", "grammar")).lower()
        task_type = str(task.get("type", "")).upper()

        if specialist == "visual":
            if task_type != "ARC_TASK":
                return {"status": "not_implemented", "reason": "visual_specialist_expected_arc_task"}
            training_examples = task.get("training_examples")
            input_grid = task.get("input_grid")
            if not isinstance(training_examples, list) or input_grid is None:
                return {"status": "error", "error": "arc_task_missing_training_or_input"}
            program = self.trm.compose(
                task_examples=training_examples,
                specialist=route.get("specialist", "visual"),
                use_enriched=use_enriched,
            )
            output_grid = self.trm.execute(program, input_data=input_grid)
            return {
                "status": "ok",
                "task_type": "ARC_TASK",
                "task_id": task.get("task_id"),
                "program_type": program.get("program_type"),
                "output_grid": output_grid,
            }

        if specialist == "math":
            question = str(task.get("question", "") or task.get("query", "")).strip()
            if not question:
                return {"status": "error", "error": "math_task_missing_question"}
            solved = self.math_specialist.process(task, use_enriched=use_enriched)
            return {
                "status": "ok" if solved.get("status") == "success" else "error",
                "task_type": task_type or "MATH_TASK",
                "task_id": task.get("task_id"),
                **solved,
            }

        if specialist in {"chat", "grammar", "any"}:
            messages = task.get("messages")
            if not isinstance(messages, list):
                prompt = str(task.get("prompt", "") or task.get("query", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "chat_task_missing_prompt"}
                messages = [{"role": "user", "content": prompt}]
            response = self.trm.process_chat(messages, use_enriched=use_enriched)
            return {
                "status": "ok",
                "task_type": task_type or "CHAT_TASK",
                "task_id": task.get("task_id"),
                "response": response,
            }

        return {
            "status": "not_implemented",
            "reason": f"specialist_dispatch_not_implemented:{specialist}",
            "task_type": task_type,
        }

    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._command_count += 1
        cmd = str(payload.get("command", "")).strip().upper()
        if not cmd:
            return {"status": "error", "error": "missing_command"}

        if cmd in {"PING", "STATUS"}:
            return self.status_payload()

        if cmd == "SHUTDOWN":
            self._shutdown_requested = True
            return {"status": "ok", "message": "shutdown_requested", "timestamp": _now_iso()}

        if cmd == "ROUTE":
            task = payload.get("task")
            if task is not None and not isinstance(task, dict):
                return {"status": "error", "error": "task_must_be_object"}
            task_obj = task if isinstance(task, dict) else None
            query = str(
                payload.get("query", "")
                or (task_obj or {}).get("query", "")
                or (task_obj or {}).get("question", "")
                or (task_obj or {}).get("prompt", "")
                or (task_obj or {}).get("type", "")
            ).strip()
            if not query:
                return {"status": "error", "error": "missing_query_or_task"}
            use_enriched = bool(payload.get("use_enriched", True))
            route = self.trm.route(
                query=query,
                specialist=str(payload.get("specialist", "auto")),
                domain_hint=payload.get("domain_hint") or (task_obj or {}).get("domain_hint"),
                galaxy_names=payload.get("galaxies") or (task_obj or {}).get("galaxies"),
            )
            response: dict[str, Any] = {"status": "ok", "route": route}
            if task_obj is not None:
                response["task_result"] = self._dispatch_task(
                    route=route,
                    task=task_obj,
                    use_enriched=use_enriched,
                )
            return response

        if cmd == "QUERY":
            query = str(payload.get("query", "")).strip()
            if not query:
                return {"status": "error", "error": "missing_query"}
            top_k = int(payload.get("top_k", 10))
            rows = self.trm.query(
                query=query,
                galaxy_names=payload.get("galaxies"),
                top_k=max(1, top_k),
                specialist=str(payload.get("specialist", "auto")),
                domain_hint=payload.get("domain_hint"),
            )
            return {
                "status": "ok",
                "count": len(rows),
                "results": rows,
            }

        if cmd == "SOLVE_MATH":
            question = str(payload.get("question", "") or payload.get("query", "")).strip()
            if not question:
                return {"status": "error", "error": "missing_question"}
            use_enriched = bool(payload.get("use_enriched", True))
            solved = self.math_specialist.process({"question": question}, use_enriched=use_enriched)
            if solved.get("status") != "success":
                return {
                    "status": "error",
                    "error": "math_specialist_failed",
                    "detail": solved,
                }
            return {
                "status": "ok",
                "result": solved.get("result"),
                "rpn_program": solved.get("rpn_program"),
                "coefficients": solved.get("coefficients"),
                "pattern_id": solved.get("pattern_id"),
                "template_id": solved.get("template_id"),
            }

        if cmd == "CHAT":
            messages = payload.get("messages")
            if not isinstance(messages, list):
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "missing_messages_or_prompt"}
                messages = [{"role": "user", "content": prompt}]
            response = self.trm.process_chat(messages, use_enriched=bool(payload.get("use_enriched", True)))
            return {"status": "ok", "response": response}

        return {"status": "error", "error": "unknown_command", "command": cmd}

    def _handle_line(self, raw_line: str) -> str:
        cmd_started = time.perf_counter()
        gpu_before = self._gpu_snapshot()
        gpu_calls_before = self._gpu_call_snapshot()
        line = raw_line.strip()
        if not line:
            response = {"status": "error", "error": "empty_command"}
            return json.dumps(response, separators=(",", ":"))
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "status": "error",
                "error": "invalid_json",
                "detail": str(exc),
            }
            return json.dumps(
                {"status": "error", "error": "invalid_json", "detail": str(exc)},
                separators=(",", ":"),
            )
        if not isinstance(payload, dict):
            return json.dumps({"status": "error", "error": "command_must_be_object"}, separators=(",", ":"))
        try:
            result = self.handle_command(payload)
        except Exception as exc:
            result = {
                "status": "error",
                "error": "command_execution_failed",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
        gpu_after = self._gpu_snapshot()
        gpu_calls_after = self._gpu_call_snapshot()
        gpu_calls_this_command = max(0, int(gpu_calls_after - gpu_calls_before))
        self._gpu_calls_total += gpu_calls_this_command
        elapsed_ms = (time.perf_counter() - cmd_started) * 1000.0
        result["telemetry"] = {
            "elapsed_ms": float(elapsed_ms),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "daemon_command_count": int(self._command_count),
            "gpu_call_counter_before": int(gpu_calls_before),
            "gpu_call_counter_after": int(gpu_calls_after),
            "gpu_calls_this_command": int(gpu_calls_this_command),
            "gpu_calls_total": int(self._gpu_calls_total),
            "fallback_triggered": False,
        }
        return json.dumps(result, separators=(",", ":"), sort_keys=True)

    def serve_stdio(self) -> int:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "message": "k3d_daemon_started",
                    "mode": "stdio",
                    "timestamp": _now_iso(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        for line in sys.stdin:
            response = self._handle_line(line)
            print(response, flush=True)
            if self._shutdown_requested:
                break
        return 0

    def serve_tcp(self) -> int:
        daemon = self

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:  # type: ignore[override]
                raw = self.rfile.readline().decode("utf-8", errors="replace")
                if not raw:
                    return
                out = daemon._handle_line(raw) + "\n"
                self.wfile.write(out.encode("utf-8"))

        with ReusableTCPServer((self.config.host, self.config.port), Handler) as server:
            server.timeout = 0.2
            while not self._shutdown_requested:
                server.handle_request()
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run persistent K3D daemon command loop.")
    parser.add_argument("--storage-root", default="../Knowledge3D.local", help="Knowledgeverse storage root.")
    parser.add_argument(
        "--mode",
        choices=("stdio", "tcp"),
        default="stdio",
        help="Command transport mode.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="TCP host when --mode=tcp.")
    parser.add_argument("--port", type=int, default=7777, help="TCP port when --mode=tcp.")
    parser.add_argument(
        "--allow-nonsovereign-query",
        action="store_true",
        help="Allow CPU query path for diagnostics (default is strict PTX query required).",
    )
    parser.add_argument(
        "--no-eager-load-default-galaxies",
        action="store_true",
        help="Disable eager default galaxy load at startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = DaemonConfig(
        storage_root=Path(args.storage_root),
        require_ptx_query=not bool(args.allow_nonsovereign_query),
        eager_load_default_galaxies=not bool(args.no_eager_load_default_galaxies),
        host=str(args.host),
        port=int(args.port),
    )
    daemon = K3DDaemon(config=config)
    if args.mode == "tcp":
        return daemon.serve_tcp()
    return daemon.serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
