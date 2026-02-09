"""Persistent ternary quality memory for pattern/routing outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _ternary(value: float, low: float = -0.20, high: float = 0.20) -> int:
    if value <= low:
        return -1
    if value >= high:
        return 1
    return 0


def _ternary_index(value: int) -> int:
    if value <= -1:
        return 0
    if value >= 1:
        return 2
    return 1


def _pool_id(correctness_t: int, honesty_t: int, transfer_t: int) -> str:
    i0 = _ternary_index(correctness_t)
    i1 = _ternary_index(honesty_t)
    i2 = _ternary_index(transfer_t)
    ordinal = i0 * 9 + i1 * 3 + i2
    return f"pool_{i0}{i1}{i2}_{ordinal:02d}"


@dataclass
class QualityPrior:
    """EMA quality prior and compact ternary decomposition."""

    pattern_id: str
    prior: float
    count: int
    correctness_t: int
    honesty_t: int
    transfer_t: int
    pool_id: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "prior": self.prior,
            "count": self.count,
            "correctness_t": self.correctness_t,
            "honesty_t": self.honesty_t,
            "transfer_t": self.transfer_t,
            "pool_id": self.pool_id,
            "updated_at": self.updated_at,
        }


class TernaryQualityMemory:
    """
    Compact persistent quality memory.

    - Keeps per-pattern EMA prior in [-1, +1]
    - Decomposes each update into ternary axes
    - Emits optional Galaxy entries for auditability
    """

    def __init__(
        self,
        *,
        state_path: str | Path,
        alpha: float = 0.10,
        emit_galaxy_entries: bool = True,
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.alpha = max(0.01, min(0.8, float(alpha)))
        self.emit_galaxy_entries = bool(emit_galaxy_entries)
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            self._state = {}
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            self._state = {}
            return
        if not isinstance(payload, dict):
            self._state = {}
            return
        self._state = {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get_prior(self, pattern_id: str) -> QualityPrior | None:
        key = str(pattern_id or "").strip()
        if not key:
            return None
        payload = self._state.get(key)
        if payload is None:
            return None
        return QualityPrior(
            pattern_id=key,
            prior=float(payload.get("prior", 0.0)),
            count=int(payload.get("count", 0)),
            correctness_t=int(payload.get("correctness_t", 0)),
            honesty_t=int(payload.get("honesty_t", 0)),
            transfer_t=int(payload.get("transfer_t", 0)),
            pool_id=str(payload.get("pool_id", "pool_111_13")),
            updated_at=str(payload.get("updated_at", "")),
        )

    def update(
        self,
        *,
        pattern_id: str,
        outcome: int,
        confidence: float,
        transfer_signal: float | None = None,
        knowledgeverse: Any | None = None,
        specialist: str = "visual",
        galaxy: str = "Grammar",
        source: str = "runtime",
    ) -> QualityPrior | None:
        """
        Update pattern quality with ternary-aware decomposition.

        `outcome` should be in {-1, 0, +1} where:
        - +1 success
        - 0 uncertain
        - -1 failure
        """
        key = str(pattern_id or "").strip()
        if not key:
            return None

        outcome_clamped = int(max(-1, min(1, int(outcome))))
        conf = max(0.0, min(1.0, float(confidence)))
        prev = self._state.get(key, {"prior": 0.0, "count": 0})
        prev_prior = float(prev.get("prior", 0.0))
        count = int(prev.get("count", 0)) + 1

        # EMA update in [-1, +1].
        prior = _clamp((1.0 - self.alpha) * prev_prior + self.alpha * float(outcome_clamped))

        # Ternary decomposition (non-binary feedback):
        # - correctness from outcome
        # - honesty from confidence calibration vs outcome
        # - transfer from optional external signal, otherwise use prior drift
        honesty_raw = (conf - 0.5) if outcome_clamped >= 0 else (0.5 - conf)
        transfer_raw = float(transfer_signal) if transfer_signal is not None else (prior - prev_prior)
        correctness_t = outcome_clamped
        honesty_t = _ternary(honesty_raw)
        transfer_t = _ternary(transfer_raw)
        pool_id = _pool_id(correctness_t, honesty_t, transfer_t)
        updated_at = datetime.now(tz=timezone.utc).isoformat()

        self._state[key] = {
            "prior": prior,
            "count": count,
            "correctness_t": correctness_t,
            "honesty_t": honesty_t,
            "transfer_t": transfer_t,
            "pool_id": pool_id,
            "updated_at": updated_at,
            "source": source,
        }
        self._save()

        record = QualityPrior(
            pattern_id=key,
            prior=prior,
            count=count,
            correctness_t=correctness_t,
            honesty_t=honesty_t,
            transfer_t=transfer_t,
            pool_id=pool_id,
            updated_at=updated_at,
        )

        if self.emit_galaxy_entries and knowledgeverse is not None:
            self._emit_galaxy_record(
                knowledgeverse=knowledgeverse,
                record=record,
                specialist=specialist,
                galaxy=galaxy,
                source=source,
            )
        return record

    def _emit_galaxy_record(
        self,
        *,
        knowledgeverse: Any,
        record: QualityPrior,
        specialist: str,
        galaxy: str,
        source: str,
    ) -> None:
        try:
            knowledgeverse.log_event(
                event_type="ternary_quality_update",
                event_data={
                    "pattern_id": record.pattern_id,
                    "quality_prior": record.prior,
                    "quality_count": record.count,
                    "correctness_t": record.correctness_t,
                    "honesty_t": record.honesty_t,
                    "transfer_t": record.transfer_t,
                    "pool_id": record.pool_id,
                    "specialist": specialist,
                    "galaxy": galaxy,
                    "query": f"quality update {record.pattern_id}",
                    "confidence": max(0.0, min(1.0, (record.prior + 1.0) / 2.0)),
                    "verification": "ternary_quality_memory",
                },
            )
        except Exception:
            pass
        try:
            knowledgeverse.galaxy_manager.add_entry(
                "Grammar",
                {
                    "id": f"quality_prior_{record.pattern_id}_{record.count:05d}",
                    "name": f"Quality Prior {record.pattern_id}",
                    "domain": "grammar",
                    "category": "quality_prior",
                    "rpn_program": "OUTCOME EMA_UPDATE TERNARY_POOL_ENCODE",
                    "metadata": {
                        **record.as_dict(),
                        "source": source,
                        "generated": False,
                    },
                },
            )
        except Exception:
            pass

