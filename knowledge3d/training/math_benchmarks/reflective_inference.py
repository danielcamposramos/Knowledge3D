"""
Reflective inference utilities for Phase 5.1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import contextlib
import io
import json
from pathlib import Path

from knowledge3d.cranium.sovereign_trm import BOS_ID, PAD_ID, RULE_OFFSET, SovereignTRM
from knowledge3d.training.math_benchmarks.recursive_solver import RecursiveSolver


class ReflectiveSolver:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        recursive_solver: Optional[RecursiveSolver] = None,
        max_steps: int = 64,
        confident_threshold: float = 0.9,
        verify_threshold: float = 0.5,
        quiet: bool = False,
    ) -> None:
        self._checkpoint_path = str(checkpoint_path)
        self._max_steps = int(max_steps)
        self._confident_threshold = float(confident_threshold)
        self._verify_threshold = float(verify_threshold)
        self._quiet = bool(quiet)
        self._recursive_solver = recursive_solver or RecursiveSolver(verbose=not self._quiet)

        meta = self._load_metadata(Path(self._checkpoint_path))
        self._embedding_dim = int(meta.get("embedding_dim", 0))
        self._hidden_dim = int(meta.get("hidden_dim", 0))
        self._vocab_size = int(meta.get("vocab_size", 0))
        self._base_vocab_size = int(meta.get("base_vocab_size", self._vocab_size))
        self._rule_registry = list(meta.get("rule_registry") or [])

        if not self._vocab_size or not self._hidden_dim:
            raise ValueError("Missing model dimensions in metadata.json")

        self._trm = SovereignTRM(
            vocab_size=self._vocab_size,
            embedding_dim=self._embedding_dim or self._vocab_size,
            hidden_dim=self._hidden_dim,
        )
        checkpoint_dir = Path(self._checkpoint_path)
        if not checkpoint_dir.is_dir():
            raise ValueError(
                "Sovereign TRM requires a converted checkpoint directory (run convert_v7_to_sovereign.py)."
            )
        self._trm.load_weights(str(checkpoint_dir))

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        if path.is_dir():
            meta_path = path / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata not found: {meta_path}")
            return json.loads(meta_path.read_text(encoding="utf-8"))

        if path.suffix == ".pt":
            try:
                import torch  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("PyTorch is required to read .pt metadata") from exc
            ckpt = torch.load(str(path), map_location="cpu")
            return {
                "embedding_dim": int(ckpt.get("embedding_dim", 0)),
                "hidden_dim": int(ckpt.get("hidden_dim", 0)),
                "vocab_size": int(ckpt.get("vocab_size", 0)),
                "base_vocab_size": int(ckpt.get("base_vocab_size", ckpt.get("vocab_size", 0))),
                "rule_registry": list(ckpt.get("rule_registry") or []),
                "control_tokens": bool(ckpt.get("control_tokens", False)),
            }

        raise ValueError(f"Unsupported checkpoint path: {path}")

    def _predict_rules(self, problem_text: str) -> Tuple[List[str], List[float], List[str]]:
        tokens = self._tokenize_problem(problem_text)
        rule_ids, confidences = self._trm.infer(tokens, max_rules=self._max_steps)

        rules: List[str] = []
        tags: List[str] = []
        for idx, conf_value in zip(rule_ids, confidences):
            rule = self._rule_registry[idx] if idx < len(self._rule_registry) else f"unknown_{idx}"
            rules.append(rule)
            if conf_value >= self._confident_threshold:
                tags.append("<CONFIDENT>")
            elif conf_value >= self._verify_threshold:
                tags.append("<UNCERTAIN>")
            else:
                tags.append("<VERIFY>")

        return rules, confidences, tags

    def _tokenize_problem(self, problem_text: str) -> List[int]:
        if not problem_text:
            return [BOS_ID]
        max_tokens = 256
        tokens = [BOS_ID]
        for byte in problem_text.encode("utf-8", errors="ignore")[:max_tokens]:
            token = int(byte)
            if token >= self._vocab_size:
                token = token % self._vocab_size
            tokens.append(token)
        return tokens

    @staticmethod
    def _compare_rules(predicted: List[str], actual: List[str]) -> List[int]:
        if not predicted or not actual:
            return [0] * len(predicted)
        correctness: List[int] = []
        for idx, rule in enumerate(predicted):
            if idx >= len(actual):
                correctness.append(0)
                continue
            if rule == actual[idx]:
                correctness.append(1)
            else:
                correctness.append(0)
                correctness.extend([0] * (len(predicted) - len(correctness)))
                break
        return correctness

    def solve(self, problem_text: str) -> Tuple[Optional[float], Dict[str, Any], Dict[str, Any]]:
        rules, confidences, tags = self._predict_rules(problem_text)
        verification_requested = any(tag == "<VERIFY>" for tag in tags)

        if self._quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = self._recursive_solver.solve(problem_text)
                trace = self._recursive_solver.get_last_trace() if result is not None else {}
        else:
            result = self._recursive_solver.solve(problem_text)
            trace = self._recursive_solver.get_last_trace() if result is not None else {}
        step_sequence = trace.get("step_sequence") or []
        actual_rules = [step.get("rule") for step in step_sequence if step.get("rule")]
        verification_labels = self._compare_rules(rules, actual_rules) if verification_requested else []

        meta = {
            "predicted_rules": rules,
            "confidence_scores": confidences,
            "confidence_tags": tags,
            "verification_requested": verification_requested,
            "verification_labels": verification_labels,
            "confidence_thresholds": {
                "confident": self._confident_threshold,
                "verify": self._verify_threshold,
            },
        }
        return result, trace, meta


__all__ = ["ReflectiveSolver"]
