from __future__ import annotations

"""
Compare two offline benchmark JSON reports and write a summary (JSON + Markdown).

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.compare_benchmarks \
    --a docs/reports/status/chat_benchmark_offline_default.json \
    --b docs/reports/status/chat_benchmark_offline_rlwhf.json \
    --out-json docs/reports/status/rlwhf_comparison.json \
    --out-md docs/reports/status/rlwhf_comparison.md
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(d: dict, mode: str) -> tuple[float|None, float|None, float|None]:
    m = d.get("modes", {}).get(mode, {})
    return m.get("latency_ms_avg"), m.get("latency_ms_p50"), m.get("context_sim_avg")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Compare benchmark JSON reports")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()
    A = load(Path(args.a))
    B = load(Path(args.b))
    modes = ["llm", "llm_rag", "k3d"]
    rows = {}
    for m in modes:
        a_la, a_lp, a_cs = fmt(A, m)
        b_la, b_lp, b_cs = fmt(B, m)
        rows[m] = {
            "a": {"latency_ms_avg": a_la, "latency_ms_p50": a_lp, "context_sim_avg": a_cs},
            "b": {"latency_ms_avg": b_la, "latency_ms_p50": b_lp, "context_sim_avg": b_cs},
            "delta": {
                "latency_ms_avg": (b_la - a_la) if (a_la is not None and b_la is not None) else None,
                "latency_ms_p50": (b_lp - a_lp) if (a_lp is not None and b_lp is not None) else None,
                "context_sim_avg": (b_cs - a_cs) if (a_cs is not None and b_cs is not None) else None,
            }
        }
    out = {
        "a": str(args.a),
        "b": str(args.b),
        "rows": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    def line(m):
        r = rows[m]
        def f(x):
            return "n/a" if x is None else (f"{x:.1f}" if isinstance(x, (int, float)) else str(x))
        return (
            f"- {m}: latency_avg {f(r['a']['latency_ms_avg'])} → {f(r['b']['latency_ms_avg'])} (Δ {f(r['delta']['latency_ms_avg'])}); "
            f"p50 {f(r['a']['latency_ms_p50'])} → {f(r['b']['latency_ms_p50'])} (Δ {f(r['delta']['latency_ms_p50'])}); "
            f"sim {f(r['a']['context_sim_avg'])} → {f(r['b']['context_sim_avg'])} (Δ {f(r['delta']['context_sim_avg'])})"
        )
    md = (
        "# RLWHF Comparison (Offline Benchmarks)\n\n"
        f"Default: {args.a}\n\n"
        f"RLWHF: {args.b}\n\n"
        + "\n".join(line(m) for m in ["llm", "llm_rag", "k3d"]) + "\n"
    )
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

