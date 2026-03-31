"""Knowledge3D benchmark suite package."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ArcAgi2Adapter",
    "ARCAGI2Benchmark",
    "GSM8KBenchmark",
    "IMOBenchmark",
    "MathCompetitionBenchmark",
    "UnifiedMathBenchmark",
    "LastHumanityExamBenchmark",
    "MMLUBenchmark",
]

_EXPORTS = {
    "ArcAgi2Adapter": ("benchmarks.arc_agi_2_adapter", "ArcAgi2Adapter"),
    "ARCAGI2Benchmark": ("benchmarks.arc_agi_2", "ARCAGI2Benchmark"),
    "GSM8KBenchmark": ("benchmarks.gsm8k", "GSM8KBenchmark"),
    "IMOBenchmark": ("benchmarks.imo_bench", "IMOBenchmark"),
    "MathCompetitionBenchmark": ("benchmarks.math_competitions", "MathCompetitionBenchmark"),
    "UnifiedMathBenchmark": ("benchmarks.math_competitions", "UnifiedMathBenchmark"),
    "LastHumanityExamBenchmark": ("benchmarks.last_humanity_exam", "LastHumanityExamBenchmark"),
    "MMLUBenchmark": ("benchmarks.mmlu", "MMLUBenchmark"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
