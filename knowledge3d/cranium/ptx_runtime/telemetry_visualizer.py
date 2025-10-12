"""Telemetry Visualizer - Claude's Enhancement #6

Memory-efficient visualization with streaming telemetry and Prometheus metrics.
"""
import os
import json
import time
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)


class TelemetryVisualizer:
    """Memory-efficient visualization with streaming telemetry"""

    def __init__(self, buffer_size=64):
        self.buffer_size = buffer_size
        self.inference_buffer = deque(maxlen=buffer_size)
        self.latency_buffer = deque(maxlen=buffer_size)
        self.error_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        self.output_dir = os.getenv("K3D_TELEMETRY_OUTPUT_DIR", "./telemetry_output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.output_dir, "thinking_tags.prom")

        # Start background telemetry thread
        self.running = True
        self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self.telemetry_thread.start()

        logger.info(f"Telemetry visualizer initialized (output: {self.output_dir})")

    def record_inference(self, input_embedding, output_tags, latency_breakdown, mode, error=None):
        """Record inference data"""
        with self.buffer_lock:
            input_hash = hash(input_embedding.tobytes())

            entry = {
                "timestamp": time.time(),
                "input_hash": input_hash,
                "output_tags": [(t[0], float(t[1])) for t in output_tags] if output_tags else [],
                "latency_breakdown": {k: float(v) for k, v in latency_breakdown.items()},
                "mode": int(mode),
                "error": str(error) if error else None
            }

            self.inference_buffer.append(entry)

            self.latency_buffer.append({
                "timestamp": time.time(),
                "total_latency": sum(latency_breakdown.values()),
                "breakdown": latency_breakdown
            })

            if error:
                self.error_buffer.append({
                    "timestamp": time.time(),
                    "error": str(error),
                    "mode": mode
                })

    def _telemetry_loop(self):
        """Background telemetry processing"""
        while self.running:
            try:
                self._export_prometheus_metrics()

                # Export detailed telemetry every 30 seconds
                if int(time.time()) % 30 == 0:
                    self._export_detailed_telemetry()

                time.sleep(5)
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                time.sleep(5)

    def _export_prometheus_metrics(self):
        """Export Prometheus metrics"""
        with self.buffer_lock:
            if not self.latency_buffer:
                return

            recent_latencies = [entry["total_latency"] for entry in list(self.latency_buffer)[-10:]]
            avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
            max_latency = max(recent_latencies) if recent_latencies else 0

            current_time = time.time()
            recent_errors = [e for e in self.error_buffer if current_time - e["timestamp"] < 60]
            error_rate = len(recent_errors) / 60.0

            metrics = [
                f"# HELP thinking_tag_latency_us Average thinking tag inference latency in microseconds",
                f"# TYPE thinking_tag_latency_us gauge",
                f"thinking_tag_latency_us {avg_latency * 1e6}",
                f"# HELP thinking_tag_max_latency_us Maximum thinking tag inference latency in microseconds",
                f"# TYPE thinking_tag_max_latency_us gauge",
                f"thinking_tag_max_latency_us {max_latency * 1e6}",
                f"# HELP thinking_tag_error_rate Error rate per second",
                f"# TYPE thinking_tag_error_rate gauge",
                f"thinking_tag_error_rate {error_rate}",
                f"# HELP thinking_tag_buffer_utilization Buffer utilization (0-1)",
                f"# TYPE thinking_tag_buffer_utilization gauge",
                f"thinking_tag_buffer_utilization {len(self.inference_buffer) / self.buffer_size}",
                f"# HELP thinking_tag_total_inferences Total inferences recorded",
                f"# TYPE thinking_tag_total_inferences counter",
                f"thinking_tag_total_inferences {len(self.inference_buffer)}"
            ]

            try:
                with open(self.metrics_file, 'w') as f:
                    for metric in metrics:
                        f.write(metric + '\n')
            except Exception as e:
                logger.error(f"Failed to write Prometheus metrics: {e}")

    def _export_detailed_telemetry(self):
        """Export detailed telemetry"""
        with self.buffer_lock:
            if not self.inference_buffer:
                return

            telemetry = {
                "timestamp": time.time(),
                "buffer_size": self.buffer_size,
                "current_utilization": len(self.inference_buffer) / self.buffer_size,
                "recent_inferences": list(self.inference_buffer)[-10:],
                "recent_errors": list(self.error_buffer)[-5:],
                "latency_stats": self._calculate_latency_stats()
            }

            telemetry_file = os.path.join(self.output_dir, f"telemetry_{int(time.time())}.json")
            try:
                with open(telemetry_file, 'w') as f:
                    json.dump(telemetry, f, indent=2)

                logger.info(f"Exported telemetry to {telemetry_file}")
            except Exception as e:
                logger.error(f"Failed to export telemetry: {e}")

    def _calculate_latency_stats(self):
        """Calculate latency statistics"""
        if not self.latency_buffer:
            return {}

        latencies = [entry["total_latency"] for entry in self.latency_buffer]

        stage_stats = {}
        if self.latency_buffer:
            for stage in self.latency_buffer[0]["breakdown"].keys():
                stage_latencies = [entry["breakdown"].get(stage, 0.0) for entry in self.latency_buffer]
                stage_stats[stage] = {
                    "avg_us": sum(stage_latencies) / len(stage_latencies) * 1e6,
                    "max_us": max(stage_latencies) * 1e6,
                    "min_us": min(stage_latencies) * 1e6
                }

        return {
            "total": {
                "avg_us": sum(latencies) / len(latencies) * 1e6,
                "max_us": max(latencies) * 1e6,
                "min_us": min(latencies) * 1e6
            },
            "stages": stage_stats
        }

    def get_stats(self) -> dict:
        """Get current statistics"""
        with self.buffer_lock:
            return {
                "buffer_size": self.buffer_size,
                "inferences_recorded": len(self.inference_buffer),
                "errors_recorded": len(self.error_buffer),
                "utilization": len(self.inference_buffer) / self.buffer_size
            }

    def shutdown(self):
        """Shutdown telemetry thread"""
        self.running = False
        self.telemetry_thread.join(timeout=2.0)
        logger.info("Telemetry visualizer shutdown")
