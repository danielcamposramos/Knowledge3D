"""Small nvcc-backed PTX compiler helper for kernel-focused tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _resolve_nvcc() -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("nvcc")


def _resolve_host_compiler() -> str | None:
    explicit = os.environ.get("CUDAHOSTCXX", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    for candidate in ("/usr/bin/g++-13", "/usr/bin/g++-12"):
        if Path(candidate).is_file():
            return candidate
    return None


def compile_cuda_file(
    file_path: str | Path,
    *,
    arch: str = "sm_86",
    use_fast_math: bool = True,
    extra_nvcc_flags: list[str] | None = None,
) -> str:
    """Compile a CUDA source file to PTX text.

    Host-control `.cu` sources without device entrypoints are returned as source
    text because they are consumed through Python/driver wrappers rather than PTX
    kernel launch.
    """

    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    if "__global__" not in source:
        return source

    nvcc = _resolve_nvcc()
    if not nvcc:
        raise RuntimeError("nvcc not found on PATH; PTX compilation requires a CUDA toolchain")

    include_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="k3d-ptx-") as tmpdir:
        out_path = Path(tmpdir) / f"{path.stem}.ptx"
        host_compiler = _resolve_host_compiler()
        cmd = [
            nvcc,
            "--ptx",
            f"-arch={arch}",
            "-O3",
            "-allow-unsupported-compiler",
        ]
        if use_fast_math:
            cmd.append("--use_fast_math")
        if extra_nvcc_flags:
            cmd.extend(extra_nvcc_flags)
        if host_compiler:
            cmd.extend(["-ccbin", host_compiler])
        cmd.extend([
            "-I",
            str(include_dir),
            str(path),
            "-o",
            str(out_path),
        ])
        env = os.environ.copy()
        if host_compiler:
            env["CUDAHOSTCXX"] = host_compiler
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"nvcc failed for {path}: {proc.stderr.strip() or proc.stdout.strip() or 'unknown error'}"
            )
        return out_path.read_text(encoding="utf-8")


__all__ = ["compile_cuda_file"]
