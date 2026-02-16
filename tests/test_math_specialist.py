from __future__ import annotations

from typing import Any

from knowledge3d.knowledgeverse.specialists.math_specialist import MathSpecialist


class _MiniGalaxy:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []


class _MiniGalaxyManager:
    def __init__(self) -> None:
        self._galaxies: dict[str, _MiniGalaxy] = {}

    def get_galaxy(self, name: str) -> _MiniGalaxy:
        if name not in self._galaxies:
            self._galaxies[name] = _MiniGalaxy()
        return self._galaxies[name]

    def add_entry(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        self.get_galaxy(galaxy_name).entries.append(dict(entry))

    def query(
        self,
        query_text: str,
        specialist: str = "math",
        top_k: int = 10,
        galaxies=None,
        preferred_pattern_type: str | None = None,
    ):
        target = list(galaxies or self._galaxies.keys())
        rows: list[dict[str, Any]] = []
        for name in target:
            for entry in self.get_galaxy(name).entries:
                rows.append({"entry": entry, "score": 1.0, "galaxy": name})
        return rows[: max(1, int(top_k))]


class _MiniKV:
    def __init__(self) -> None:
        self.galaxy_manager = _MiniGalaxyManager()
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        self.events.append((event_type, dict(event_data)))


def test_math_specialist_linear_equation_composes_rpn() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "11 3 - 2 /":
            return 4.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "If 2x + 3 = 11, what is x?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 4.0
    assert out["rpn_program"] == "11 3 - 2 /"
    assert out["coefficients"] == {"a": 2.0, "b": 3.0, "c": 11.0}


def test_math_specialist_backward_equation_composes_rpn() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        normalized = " ".join(program.strip().split())
        if normalized in {"11 3 - 2 /", "3 11 - -2 /"}:
            return 4.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "If 11 = 2x + 3, what is x?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 4.0
    assert out["pattern_type"] == "linear_equation"


def test_math_specialist_arithmetic_addition_template() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "7 5 +":
            return 12.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "What is 7 + 5?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 12.0
    assert out["rpn_program"] == "7 5 +"
    assert out["pattern_type"] == "arithmetic_add"


def test_math_specialist_ratio_template() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "8 2 /":
            return 4.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "What is the ratio 8:2?"}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 4.0
    assert out["rpn_program"] == "8 2 /"
    assert out["pattern_type"] == "ratio"


def test_math_specialist_proportion_template() -> None:
    kv = _MiniKV()

    def _fake_eval(program: str) -> float | None:
        if program.strip() == "3 4 * 2 /":
            return 6.0
        return None

    specialist = MathSpecialist(knowledgeverse=kv, evaluator=_fake_eval)
    out = specialist.process({"question": "Solve 2/3 = 4/x for x."}, use_enriched=True)

    assert out["status"] == "success"
    assert out["result"] == 6.0
    assert out["rpn_program"] == "3 4 * 2 /"
    assert out["pattern_type"] == "proportion"
