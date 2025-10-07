from __future__ import annotations

"""
Action routing bridge for the fused head output layer.

The router keeps the CUDA hot path on device while exposing a Python façade
that integrates navigation, dialogue, memory consolidation and tablet logging.
It mirrors the workflow captured in the Step7.2 chain – confidence adjustment
occurs on the GPU, actions are recorded to a zero-copy mmap buffer, and memory
writes raise enriched consolidation tickets.
"""

from dataclasses import dataclass, field
import json
import mmap
import time
from pathlib import Path
from typing import Dict, Optional

from knowledge3d.cranium.actions import ActionBuffer, ActionResult, ActionType
from knowledge3d.cranium.actions.confidence_propagation import ConfidencePropagator

try:  # Optional – enhanced multi-modal confidence manager
    from knowledge3d.cranium.actions.enhanced_multi_modal_confidence_propagation import (
        EnhancedMultiModalConfidencePropagator,
    )

    _HAS_MULTI_MODAL = True
except Exception:  # pragma: no cover
    EnhancedMultiModalConfidencePropagator = None  # type: ignore
    _HAS_MULTI_MODAL = False

try:
    from knowledge3d.spatial.semantic_navigator import SemanticNavigator

    _HAS_NAVIGATOR = True
except Exception:  # pragma: no cover
    SemanticNavigator = None  # type: ignore
    _HAS_NAVIGATOR = False

try:
    from knowledge3d.cranium.phase10.sleep_time_compute import SleepTimeCompute

    _HAS_SLEEP_PIPELINE = True
except Exception:  # pragma: no cover
    SleepTimeCompute = None  # type: ignore
    _HAS_SLEEP_PIPELINE = False


def _timestamp_us() -> int:
    return int(time.time() * 1_000_000)


@dataclass
class ConsolidationTicket:
    handle: Optional[int]
    node_count: int
    timestamp: float
    action_type: Optional[str] = None
    alpha_used: Optional[float] = None
    confidence: Optional[float] = None
    curiosity: Optional[float] = None


class ActionRouter:
    """
    Translate decoded GPU actions into system side-effects.

    Parameters
    ----------
    galaxy_path: path to the volatile galaxy GLB
    house_path: path to the persistent house GLB
    tablet_config: legacy hook for tablet metadata (unused)
    sleep_check_interval: seconds between background sleep triggers
    mmap_tablet_log_path: path to the zero-copy tablet log buffer
    mmap_tablet_log_size: buffer size in bytes
    """

    def __init__(
        self,
        *,
        galaxy_path: Optional[str | Path] = None,
        house_path: Optional[str | Path] = None,
        tablet_config: Optional[dict] = None,
        sleep_check_interval: float = 30.0,
        mmap_tablet_log_path: str | Path = "tablet_log.mmap",
        mmap_tablet_log_size: int = 4 * 1024 * 1024,
    ) -> None:
        self.galaxy_path = Path(galaxy_path) if galaxy_path else None
        self.house_path = Path(house_path) if house_path else None
        self.tablet_config = tablet_config or {}
        self.sleep_check_interval = float(sleep_check_interval)
        self._last_sleep_check = time.time()

        self.confidence_propagator = ConfidencePropagator()
        self.multi_modal = (
            EnhancedMultiModalConfidencePropagator() if _HAS_MULTI_MODAL else None
        )

        self._navigator: Optional[SemanticNavigator] = None

        self.replay_log_path = Path("replay_actions.jsonl")
        self._mmap_path = Path(mmap_tablet_log_path)
        self._mmap_size = int(mmap_tablet_log_size)
        self._mmap_file: Optional[mmap.mmap] = None
        self._mmap_offset = 0
        self._init_tablet_mmap()

        self.stats: Dict[str, int] = {
            "nav_actions": 0,
            "dialogue_actions": 0,
            "memory_writes": 0,
            "tablet_updates": 0,
            "no_actions": 0,
        }

    # ------------------------------------------------------------------
    def _init_tablet_mmap(self) -> None:
        self._mmap_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._mmap_path.exists():
            with self._mmap_path.open("wb") as f:
                f.truncate(self._mmap_size)
        self._mmap_handle = self._mmap_path.open("r+b")
        self._mmap_file = mmap.mmap(self._mmap_handle.fileno(), self._mmap_size)
        self._mmap_offset = 0

    def _write_to_mmap(self, payload: Dict[str, object]) -> None:
        if self._mmap_file is None:
            return
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        if self._mmap_offset + len(line) > self._mmap_size:
            self._mmap_offset = 0
        self._mmap_file.seek(self._mmap_offset)
        self._mmap_file.write(line)
        self._mmap_offset += len(line)

    # ------------------------------------------------------------------
    def _ensure_navigator(self) -> Optional[SemanticNavigator]:
        if not _HAS_NAVIGATOR or self.house_path is None:
            return None
        if self._navigator is None:
            try:
                self._navigator = SemanticNavigator()
                self._navigator.load_house(str(self.house_path))
            except Exception:
                self._navigator = None
        return self._navigator

    # ------------------------------------------------------------------
    def dispatch(self, action_buffer: ActionBuffer) -> ActionResult:
        action_type = action_buffer.get_action_type()
        base_confidence = action_buffer.get_confidence()
        curiosity = action_buffer.get_curiosity()

        adjusted_confidence = self.confidence_propagator.propagate_confidence(
            [base_confidence], [curiosity], base_confidence
        )
        final_confidence = float(adjusted_confidence[0]) if adjusted_confidence.size else base_confidence
        alpha_used = self.confidence_propagator._compute_alpha(None)  # deliberate use of internal helper

        self._log_action_for_replay(action_type, base_confidence, final_confidence, curiosity, alpha_used)

        if action_type == ActionType.NAV_MOVE:
            metadata = self._dispatch_nav_move(action_buffer, final_confidence, curiosity)
            self.stats["nav_actions"] += 1
            metadata["alpha"] = alpha_used
            return ActionResult(action_type, final_confidence, curiosity, metadata.get("success", False), metadata)

        if action_type == ActionType.DIALOGUE:
            metadata = self._dispatch_dialogue(action_buffer, final_confidence, curiosity)
            metadata["alpha"] = alpha_used
            self.stats["dialogue_actions"] += 1
            return ActionResult(action_type, final_confidence, curiosity, True, metadata)

        if action_type == ActionType.WRITE_MEM:
            success, metadata = self._dispatch_memory(action_buffer, final_confidence, curiosity, alpha_used)
            metadata["alpha"] = alpha_used
            self.stats["memory_writes"] += 1
            return ActionResult(action_type, final_confidence, curiosity, success, metadata)

        if action_type == ActionType.UPDATE_TABLET:
            metadata = self._dispatch_tablet_update(action_buffer, final_confidence, curiosity)
            metadata["alpha"] = alpha_used
            self.stats["tablet_updates"] += 1
            return ActionResult(action_type, final_confidence, curiosity, True, metadata)

        self.stats["no_actions"] += 1
        return ActionResult(ActionType.NO_ACTION, final_confidence, curiosity, False, {})

    # ------------------------------------------------------------------
    def _dispatch_nav_move(
        self,
        action_buffer: ActionBuffer,
        confidence: float,
        curiosity: float,
    ) -> Dict[str, object]:
        navigator = self._ensure_navigator()
        position, nav_confidence = action_buffer.extract_nav_move()
        metadata = {
            "position": position.tolist(),
            "confidence": nav_confidence,
            "adjusted_confidence": confidence,
            "curiosity": curiosity,
            "success": navigator is not None,
        }
        if navigator is not None:
            metadata["message"] = "Navigator dispatch queued"
        else:
            metadata["message"] = "Navigator unavailable"
        return metadata

    def _dispatch_dialogue(
        self,
        action_buffer: ActionBuffer,
        confidence: float,
        curiosity: float,
    ) -> Dict[str, object]:
        tokens, thinking_score = action_buffer.extract_dialogue_tokens()
        payload = {
            "timestamp": _timestamp_us(),
            "type": "dialogue",
            "confidence": confidence,
            "curiosity": curiosity,
            "thinking_score": thinking_score,
            "tokens": tokens.tolist(),
        }
        self._write_to_mmap(payload)
        return payload

    def _dispatch_memory(
        self,
        action_buffer: ActionBuffer,
        confidence: float,
        curiosity: float,
        alpha_used: float,
    ) -> tuple[bool, Dict[str, object]]:
        zone, embedding, mem_conf = action_buffer.extract_memory_write()
        if zone != 3:
            return False, {"reason": "invalid_zone", "zone_id": zone}

        ticket = ConsolidationTicket(
            handle=self._galaxy_handle(),
            node_count=0,
            timestamp=time.time(),
            action_type=ActionType.WRITE_MEM.name,
            alpha_used=alpha_used,
            confidence=confidence,
            curiosity=curiosity,
        )

        success = self._trigger_sleep(ticket)
        payload = {
            "timestamp": _timestamp_us(),
            "type": "memory_write",
            "zone": zone,
            "confidence": confidence,
            "curiosity": curiosity,
            "embedding": embedding.tolist(),
            "ticket": {
                "alpha_used": alpha_used,
                "success": success,
            },
        }
        self._write_to_mmap(payload)
        return success, payload

    def _dispatch_tablet_update(
        self,
        action_buffer: ActionBuffer,
        confidence: float,
        curiosity: float,
    ) -> Dict[str, object]:
        mutation_type, payload = action_buffer.extract_tablet_mutation()
        record = {
            "timestamp": _timestamp_us(),
            "type": "tablet_update",
            "confidence": confidence,
            "curiosity": curiosity,
            "mutation_type": mutation_type,
            "payload": payload.tolist(),
        }
        self._write_to_mmap(record)
        return record

    # ------------------------------------------------------------------
    def _trigger_sleep(self, ticket: ConsolidationTicket) -> bool:
        now = time.time()
        if now - self._last_sleep_check < self.sleep_check_interval:
            return False
        self._last_sleep_check = now
        if not _HAS_SLEEP_PIPELINE or self.house_path is None or self.galaxy_path is None:
            return False

        try:
            stc = SleepTimeCompute(str(self.house_path), str(self.galaxy_path))
            stc.load_house()
            stc.load_galaxy()
            stc.materialize_chat_history({"chat_history": []})  # warm paths
            stc.save_house()
            return True
        except Exception:
            return False

    def _galaxy_handle(self) -> Optional[int]:
        # Placeholder: we cannot easily materialise a handle without the live GPU house.
        return None

    def _log_action_for_replay(
        self,
        action_type: ActionType,
        raw_confidence: float,
        final_confidence: float,
        curiosity: float,
        alpha_used: float,
    ) -> None:
        entry = {
            "timestamp": _timestamp_us(),
            "action_type": action_type.name,
            "raw_confidence": raw_confidence,
            "final_confidence": final_confidence,
            "curiosity": curiosity,
            "adaptive_alpha_used": alpha_used,
        }
        self.replay_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.replay_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
