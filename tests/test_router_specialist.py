from __future__ import annotations

from types import SimpleNamespace

import pytest

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.router_specialist import RoutingDecision, RouterSpecialistTrainer


class _FakeAdapter:
    def __init__(self) -> None:
        self.A = HostTensorF32.from_array_like([[1.0, 0.0], [0.0, 1.0]], rows=2, cols=2)
        self.B = HostTensorF32.from_array_like([[0.5, 0.0], [0.0, 0.5]], rows=2, cols=2)
        self.A_shadow = self.A.copy()
        self.B_shadow = self.B.copy()
        self.rank = 2
        self.alpha = 1.0


class _FakeBase:
    def __init__(self) -> None:
        self.specialists = {
            "router": {"adapter": _FakeAdapter(), "dims": 2},
            "math": {},
            "chat": {},
        }

    def get_base_at_dim(self, dims: int):
        return HostTensorF32.from_array_like([[1.0, 0.0], [0.0, 1.0]], rows=dims, cols=dims)


class _FakeSwarm:
    def __init__(self) -> None:
        self.base = _FakeBase()
        self.config = SimpleNamespace(specialist_learning_rate=0.05)
        self._router_calls = 0

    def register_specialist(self, name, required_dims, rank):
        self.base.specialists[name] = {"adapter": _FakeAdapter(), "dims": required_dims, "rank": rank}

    def compute_with_specialist(self, input_data, specialist_name):
        self._router_calls += 1
        if specialist_name != "router":
            return [0.0, 0.0]
        return [0.1, 0.9]


def test_prepare_router_arrays_uses_host_tensors():
    trainer = RouterSpecialistTrainer(_FakeSwarm())
    decisions = [
        RoutingDecision(
            input_data=[1.0, 2.0],
            task_description="route math",
            specialist_weights={"math": 0.8, "chat": 0.2},
            outcome_performance=0.9,
            timestamp="2026-03-24T00:00:00",
        ),
        RoutingDecision(
            input_data=[3.0],
            task_description="route chat",
            specialist_weights={"chat": 0.7, "math": 0.3},
            outcome_performance=0.6,
            timestamp="2026-03-24T00:00:01",
        ),
    ]

    inputs, targets = trainer._prepare_router_arrays(decisions, 4, ["math", "chat"])

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert inputs.to_nested_list()[0] == [1.0, 2.0, 1.0, 2.0]
    assert targets.to_nested_list()[0][0] == pytest.approx(0.9)
    assert targets.to_nested_list()[1][1] == pytest.approx(0.6)


def test_train_from_history_splits_and_delegates(monkeypatch):
    swarm = _FakeSwarm()
    trainer = RouterSpecialistTrainer(swarm)

    history = [
        RoutingDecision([1.0, 0.0], "math", {"math": 0.9}, 0.9, "t0"),
        RoutingDecision([0.0, 1.0], "chat", {"chat": 0.8}, 0.8, "t1"),
        RoutingDecision([1.0, 1.0], "math", {"math": 0.7}, 0.7, "t2"),
        RoutingDecision([2.0, 2.0], "chat", {"chat": 0.6}, 0.6, "t3"),
    ]

    captured = {}

    def _fake_train(self, train_decisions, val_decisions, epochs, learning_rate, log_prefix):
        captured["train"] = len(train_decisions)
        captured["val"] = len(val_decisions)
        captured["epochs"] = epochs
        captured["learning_rate"] = learning_rate
        captured["log_prefix"] = log_prefix
        return {
            "success": True,
            "baseline_metrics": {"accuracy": 0.5, "avg_correct_weight": 0.5},
            "candidate_metrics": {"accuracy": 1.0, "avg_correct_weight": 0.9},
            "improvement": 0.4,
            "epochs": epochs,
            "train_samples": len(train_decisions),
            "val_samples": len(val_decisions),
        }

    monkeypatch.setattr(RouterSpecialistTrainer, "_train_router_gpu", _fake_train)

    result = trainer.train_from_history(history, epochs=3, filter_threshold=0.5)

    assert captured == {
        "train": 2,
        "val": 2,
        "epochs": 3,
        "learning_rate": 0.05,
        "log_prefix": "[Router]",
    }
    assert result["specialists"] == ["math", "chat"]
    assert result["success"] is True
