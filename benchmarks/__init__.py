"""Knowledge3D benchmark suite package."""

from .arc_agi_2_adapter import ArcAgi2Adapter
from .arc_agi_2 import ARCAGI2Benchmark
from .last_humanity_exam import LastHumanityExamBenchmark
from .math_competitions import MathCompetitionBenchmark

__all__ = [
    "ArcAgi2Adapter",
    "ARCAGI2Benchmark",
    "MathCompetitionBenchmark",
    "LastHumanityExamBenchmark",
]
