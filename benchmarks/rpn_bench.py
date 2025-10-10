from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from knowledge3d.core.legacy_rpn_python import LegacyPythonRPN


def synthesize_pairs(n: int, d: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, d)).astype(np.float64)
    b = rng.normal(size=(n, d)).astype(np.float64)
    return a, b


def cosine_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # vectorized cosine similarity
    dot = (a * b).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    denom[denom == 0] = 1
    return dot / denom


def cosine_rpn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = LegacyPythonRPN()
    out = np.zeros(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        out[i] = r.cosine(a[i].tolist(), b[i].tolist())
    return out


@dataclass
class Result:
    n: int
    d: int
    direct_ms: float
    rpn_ms: float
    max_diff: float


def run_case(n: int, d: int) -> Result:
    a, b = synthesize_pairs(n, d)
    t0 = time.perf_counter(); cd = cosine_direct(a, b); t1 = time.perf_counter()
    cr = cosine_rpn(a, b); t2 = time.perf_counter()
    max_diff = float(np.max(np.abs(cd - cr)))
    return Result(n, d, (t1 - t0) * 1000, (t2 - t1) * 1000, max_diff)


def main():
    out_dir = "../Knowledge3D.local/benchmarks"
    cases = [(1000, 64), (2000, 64), (1000, 256)]
    print("n d direct_ms rpn_ms max_diff")
    rows = []
    for n, d in cases:
        r = run_case(n, d)
        rows.append(r)
        print(r.n, r.d, f"{r.direct_ms:.1f}", f"{r.rpn_ms:.1f}", f"{r.max_diff:.2e}")

    # optional: write md
    try:
        import os
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        md = Path(out_dir) / "rpn_results.md"
        with md.open("w", encoding="utf-8") as f:
            f.write("# RPN vs Direct Cosine Benchmark\n\n")
            f.write("| n | d | direct (ms) | rpn (ms) | max diff |\n")
            f.write("|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r.n} | {r.d} | {r.direct_ms:.1f} | {r.rpn_ms:.1f} | {r.max_diff:.2e} |\n")
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
