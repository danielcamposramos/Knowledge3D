"""Latency Profiler - Claude's Enhancement #2

Stage-level latency profiling with adaptive budget allocation for <35µs target.
"""
import ctypes
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class LatencyProfiler:
    """Stage-level latency profiler with adaptive budget allocation."""

    STAGE_NAMES = [
        "sparsity_calc", "query", "cross_modal", "weight_assembly",
        "rpn_exec", "crystallize", "confidence"
    ]

    def __init__(self, total_budget_us=35.0):
        self.total_budget_us = total_budget_us

        # Initial budget allocation (in microseconds)
        self.stage_budgets = {
            "sparsity_calc": 5.0,
            "query": 10.0,
            "cross_modal": 3.0,
            "weight_assembly": 5.0,
            "rpn_exec": 8.0,
            "crystallize": 2.0,
            "confidence": 2.0
        }

        self.stage_times = {name: [] for name in self.STAGE_NAMES}
        self.current_stage = None
        self.stage_start_time = None
        self.adaptive_enabled = True

        # Telemetry
        self.total_inferences = 0
        self.budget_breaches = 0

    def start_stage(self, stage_name: str):
        """Start timing a stage"""
        if stage_name not in self.STAGE_NAMES:
            logger.warning(f"Unknown stage: {stage_name}")
            return

        self.current_stage = stage_name
        self.stage_start_time = time.perf_counter()

    def end_stage(self, stage_name: str):
        """End timing a stage and update statistics"""
        if stage_name != self.current_stage:
            logger.warning(f"Stage mismatch: expected {self.current_stage}, got {stage_name}")
            return

        if self.stage_start_time is None:
            return

        elapsed = time.perf_counter() - self.stage_start_time
        self.stage_times[stage_name].append(elapsed)

        # Keep only last 100 measurements
        if len(self.stage_times[stage_name]) > 100:
            self.stage_times[stage_name].pop(0)

        self.current_stage = None
        self.stage_start_time = None

        # Check if stage exceeded budget
        if elapsed > self.stage_budgets[stage_name] * 1e-6:
            logger.debug(f"Stage {stage_name} exceeded budget: {elapsed*1e6:.2f}µs > {self.stage_budgets[stage_name]:.2f}µs")

    def get_stage_stats(self, stage_name: str) -> dict:
        """Get statistics for a stage"""
        times = self.stage_times.get(stage_name, [])
        if not times:
            return {
                "avg_us": 0.0,
                "max_us": 0.0,
                "min_us": 0.0,
                "budget_us": self.stage_budgets.get(stage_name, 0.0)
            }

        return {
            "avg_us": np.mean(times) * 1e6,
            "max_us": np.max(times) * 1e6,
            "min_us": np.min(times) * 1e6,
            "budget_us": self.stage_budgets.get(stage_name, 0.0)
        }

    def get_full_report(self) -> dict:
        """Get comprehensive profiling report"""
        report = {
            "total_budget_us": self.total_budget_us,
            "total_inferences": self.total_inferences,
            "budget_breaches": self.budget_breaches,
            "stages": {}
        }

        total_actual = 0.0
        for stage_name in self.STAGE_NAMES:
            stats = self.get_stage_stats(stage_name)
            report["stages"][stage_name] = stats
            total_actual += stats["avg_us"]

        report["total_actual_us"] = total_actual
        report["budget_utilization"] = total_actual / self.total_budget_us if self.total_budget_us > 0 else 0.0

        return report

    def adapt_budgets(self):
        """Adaptive budget reallocation based on actual performance"""
        if not self.adaptive_enabled:
            return

        # Calculate actual average times
        actual_times = {}
        for stage_name in self.STAGE_NAMES:
            times = self.stage_times.get(stage_name, [])
            if times:
                actual_times[stage_name] = np.mean(times) * 1e6  # Convert to µs

        if not actual_times:
            return

        # Find stages that are consistently under/over budget
        total_actual = sum(actual_times.values())

        if total_actual > self.total_budget_us * 1.1:
            # Over budget - reduce all proportionally
            scale_factor = (self.total_budget_us * 0.95) / total_actual
            for stage_name in self.STAGE_NAMES:
                if stage_name in actual_times:
                    self.stage_budgets[stage_name] = actual_times[stage_name] * scale_factor
        elif total_actual < self.total_budget_us * 0.8:
            # Under budget - reallocate slack to bottleneck stages
            slack = self.total_budget_us - total_actual

            # Find slowest stage
            slowest_stage = max(actual_times.items(), key=lambda x: x[1])

            # Give slack to slowest stage
            self.stage_budgets[slowest_stage[0]] += slack * 0.5

            logger.debug(f"Adapted budgets: gave {slack*0.5:.2f}µs to {slowest_stage[0]}")

    def record_inference_complete(self, total_latency_us: float):
        """Record completed inference"""
        self.total_inferences += 1

        if total_latency_us > self.total_budget_us:
            self.budget_breaches += 1

        # Adapt budgets every 100 inferences
        if self.total_inferences % 100 == 0:
            self.adapt_budgets()

    def get_latency_breakdown(self) -> dict:
        """Get current latency breakdown"""
        breakdown = {}
        for stage_name in self.STAGE_NAMES:
            stats = self.get_stage_stats(stage_name)
            breakdown[stage_name] = stats["avg_us"] / 1e6  # Convert to seconds
        return breakdown
