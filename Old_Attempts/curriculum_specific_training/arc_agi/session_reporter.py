"""
Training session reporter — structured logs for every run.

Outputs to: knowledge3d/training/arc_agi/logs/session_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List


class SessionReporter:
    """
    Generates structured training session reports.
    """

    def __init__(self, session_name: Optional[str] = None):
        self.start_time = datetime.now()
        self.session_id = session_name or self.start_time.strftime("%Y%m%d_%H%M%S")

        self.report: Dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "config": {},
            "datasets": {
                "arc_agi_1": {"path": None, "task_count": 0},
                "arc_agi_2": {"path": None, "task_count": 0},
                "arc_agi_3": {"path": None, "task_count": 0},
                "total_tasks": 0,
            },
            "math_benchmarks": {
                "datasets": [],
                "total_problems": 0,
                "by_source": {},
            },
            "epochs": [],
            "summary": {
                "total_epochs": 0,
                "best_epoch": None,
                "best_accuracy": 0.0,
                "final_accuracy": 0.0,
                "discoveries": {"proposed": 0, "preserved": 0, "promoted": 0, "canonical": 0},
                "grammar_rules": {"initial": 0, "final": 0, "discovered": 0},
            },
            "errors": [],
            "warnings": [],
        }

        self.logs_dir = Path(__file__).parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def set_config(self, config: Dict[str, Any]) -> None:
        self.report["config"] = config

    def set_datasets(
        self,
        arc1_path: str,
        arc1_count: int,
        arc2_path: str,
        arc2_count: int,
        arc3_path: Optional[str] = None,
        arc3_count: int = 0,
    ) -> None:
        self.report["datasets"] = {
            "arc_agi_1": {"path": arc1_path, "task_count": arc1_count},
            "arc_agi_2": {"path": arc2_path, "task_count": arc2_count},
            "arc_agi_3": {"path": arc3_path, "task_count": arc3_count},
            "total_tasks": arc1_count + arc2_count + arc3_count,
        }

    def log_epoch(self, epoch: int, metrics: Dict[str, Any], math_metrics: Optional[Dict[str, Any]] = None) -> None:
        epoch_data = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        if math_metrics:
            epoch_data["math_accuracy"] = math_metrics.get("overall", {}).get("accuracy", 0.0)
            epoch_data["math_by_source"] = math_metrics.get("by_source", {})
        self.report["epochs"].append(epoch_data)

        accuracy = metrics.get("accuracy", 0.0)
        if accuracy > self.report["summary"]["best_accuracy"]:
            self.report["summary"]["best_accuracy"] = accuracy
            self.report["summary"]["best_epoch"] = epoch

        self.report["summary"]["total_epochs"] = epoch
        self.report["summary"]["final_accuracy"] = accuracy

    def log_discovery_stats(self, proposed: int, preserved: int, promoted: int, canonical: int) -> None:
        self.report["summary"]["discoveries"] = {
            "proposed": proposed,
            "preserved": preserved,
            "promoted": promoted,
            "canonical": canonical,
        }

    def log_grammar_stats(self, initial: int, final: int, discovered: int) -> None:
        self.report["summary"]["grammar_rules"] = {"initial": initial, "final": final, "discovered": discovered}

    def set_math_benchmarks(self, datasets: List[str], total_problems: int, by_source: Dict[str, int]) -> None:
        self.report["math_benchmarks"] = {
            "datasets": datasets,
            "total_problems": total_problems,
            "by_source": by_source,
        }

    def log_error(self, error: str, context: Optional[str] = None) -> None:
        self.report["errors"].append({"timestamp": datetime.now().isoformat(), "error": error, "context": context})

    def log_warning(self, warning: str) -> None:
        self.report["warnings"].append({"timestamp": datetime.now().isoformat(), "warning": warning})

    def finalize(self) -> str:
        end_time = datetime.now()
        self.report["end_time"] = end_time.isoformat()
        self.report["duration_seconds"] = (end_time - self.start_time).total_seconds()

        report_path = self.logs_dir / f"session_{self.session_id}.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)

        print(f"[SESSION REPORT] Saved to {report_path}")
        return str(report_path)

    def print_summary(self) -> None:
        s = self.report["summary"]
        d = s["discoveries"]
        g = s["grammar_rules"]

        print("\n" + "=" * 60)
        print(f"SESSION REPORT: {self.session_id}")
        print("=" * 60)
        print(f"Duration: {self.report['duration_seconds']:.1f}s")
        print(f"Total epochs: {s['total_epochs']}")
        print(f"Best accuracy: {s['best_accuracy']:.2%} (epoch {s['best_epoch']})")
        print(f"Final accuracy: {s['final_accuracy']:.2%}")
        print("\nDiscoveries:")
        print(f"  Proposed: {d['proposed']}")
        print(f"  Preserved: {d['preserved']}")
        print(f"  Promoted: {d['promoted']}")
        print(f"  Canonical: {d['canonical']}")
        print("\nGrammar Rules:")
        print(f"  Initial: {g['initial']}")
        print(f"  Final: {g['final']} (+{g['discovered']} discovered)")
        print(f"\nErrors: {len(self.report['errors'])}")
        print(f"Warnings: {len(self.report['warnings'])}")
        print("=" * 60 + "\n")


__all__ = ["SessionReporter"]
