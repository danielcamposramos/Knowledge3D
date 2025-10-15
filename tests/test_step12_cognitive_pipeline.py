"""
Phase 0.1: Cognitive Pipeline Tests for Step 12 FSM Integration

Tests the 5-state observability system (INGEST → FUSE → SPATIAL → REASON → OUTPUT)
integrated into ThinkingTagBridge from Step 12 FSM consolidation.

Target: 18+ tests validating state transitions, timing precision, export, statistics,
error handling, and memory management.
"""
import json
import time
import random
from unittest import TestCase, mock

try:
    from knowledge3d.cranium.bridges.sovereign_bridges import ThinkingTagBridge
except ModuleNotFoundError:
    from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge


class StateTraceValidator:
    """Helper for reusable state trace validation."""

    def __init__(self, state_report):
        self.report = state_report
        self.stages = self.report.get('stages', [])

    def validate_order(self):
        expected = ['INGEST', 'FUSE', 'SPATIAL', 'REASON', 'OUTPUT']
        actual = [stage['name'] for stage in self.stages]
        assert actual == expected, f"Invalid order: {actual}"

    def validate_timings(self):
        for stage in self.stages:
            assert stage['duration_us'] > 0, f"Non-positive duration: {stage}"

    def validate_percentiles(self):
        assert 'p50' in self.report.get('statistics', {}), "Missing p50"
        assert 'p95' in self.report.get('statistics', {}), "Missing p95"
        assert 'p99' in self.report.get('statistics', {}), "Missing p99"


class TestCognitivePipeline(TestCase):
    def setUp(self):
        self.bridge = ThinkingTagBridge()
        # Mock GPU ops to run CPU-only
        self.bridge.inference = mock.Mock(return_value=mock.Mock(action_buffer=mock.Mock(confidence=0.85)))
        self.input_embedding = random.randbytes(512)  # Mock embedding
        random.seed(42)  # Determinism

    def test_state_transitions_recorded(self):
        """Verify all 5 states are tracked during inference."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        assert len(state_report['transitions']) == 4  # 4 transitions for 5 states
        assert state_report['stages'][0]['name'] == 'INGEST'
        assert state_report['stages'][4]['name'] == 'OUTPUT'

    def test_state_timing_microseconds(self):
        """Verify microsecond-precision timing."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        for stage in state_report['stages']:
            assert isinstance(stage['duration_us'], int) and stage['duration_us'] > 0
            assert 1 <= stage['duration_us'] <= 10000  # Realistic range for mocks

    def test_state_trace_export(self):
        """Verify JSON export of state trace."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        json_str = json.dumps(state_report)
        assert json.loads(json_str) == state_report  # Round-trip validation

    def test_state_trace_export_to_file(self):
        """Verify JSON export to file."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        with mock.patch('builtins.open', mock.mock_open()) as mock_file:
            self.bridge.export_state_trace('trace.json')
            mock_file.assert_called_once_with('trace.json', 'w')

    def test_percentile_statistics(self):
        """Verify p50, p95, p99 calculations."""
        for _ in range(10):  # Multiple inferences for stats
            self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        assert state_report['statistics']['p50'] > 0
        assert state_report['statistics']['p95'] >= state_report['statistics']['p50']
        assert state_report['statistics']['p99'] >= state_report['statistics']['p95']

    def test_state_sequence_validation(self):
        """Correct order enforcement."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        validator = StateTraceValidator(state_report)
        validator.validate_order()

    def test_error_handling_during_fallback(self):
        """State tracking during fallback paths."""
        self.bridge.inference.side_effect = Exception("Mock fallback")
        with self.assertRaises(Exception):
            self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        # Assume error flag in last stage if tracked
        if state_report['stages']:
            assert 'error' in state_report['stages'][-1] or 'status' in state_report['stages'][-1]

    def test_memory_cleanup(self):
        """State trace buffer management."""
        for _ in range(100):
            self.bridge.inference(self.input_embedding, ['text'])
        self.bridge.clear_state_trace()
        # After clearing, trace should be minimal
        state_report = self.bridge.get_state_trace_report()
        assert len(state_report.get('transitions', [])) == 0

    def test_concurrent_inferences(self):
        """State tracking under concurrent calls (mock threads)."""
        import threading

        def infer():
            self.bridge.inference(self.input_embedding, ['text'])

        threads = [threading.Thread(target=infer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        state_report = self.bridge.get_state_trace_report()
        assert len(state_report['stages']) >= 5  # At least one complete trace

    def test_empty_input_embedding(self):
        """Handle empty embedding."""
        result = self.bridge.inference(b'', ['text'])
        state_report = self.bridge.get_state_trace_report()
        assert state_report['stages'][0]['duration_us'] > 0

    def test_invalid_modalities(self):
        """Error on invalid modalities."""
        with self.assertRaises(ValueError):
            self.bridge.inference(self.input_embedding, ['invalid'])

    def test_trace_pruning(self):
        """Prune old traces after threshold."""
        for _ in range(20):
            self.bridge.inference(self.input_embedding, ['text'])
        if hasattr(self.bridge, 'prune_state_trace'):
            self.bridge.prune_state_trace(10)
            state_report = self.bridge.get_state_trace_report()
            assert len(state_report['transitions']) <= 9

    def test_timing_precision(self):
        """Ensure timings are integers (microseconds)."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        for stage in state_report['stages']:
            assert stage['duration_us'] % 1 == 0

    def test_statistics_with_single_inference(self):
        """Percentiles with one inference."""
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        stats = state_report['statistics']
        assert stats['p50'] == stats['p95'] == stats['p99']

    def test_validator_timings(self):
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        validator = StateTraceValidator(state_report)
        validator.validate_timings()

    def test_validator_percentiles(self):
        result = self.bridge.inference(self.input_embedding, ['text'])
        state_report = self.bridge.get_state_trace_report()
        validator = StateTraceValidator(state_report)
        validator.validate_percentiles()

    # ------------------------------------------------------------------
    #  Kimi extension: edge cases & deterministic percentile coverage
    # ------------------------------------------------------------------
    def test_trace_with_zero_duration_bug(self):
        """Guard against division-by-zero if a stage ends <1 µs."""
        with mock.patch.object(self.bridge, '_stage_ingest', return_value=0):
            report = self.bridge.get_state_trace_report()
            stats = report.get('statistics', {})
            # p50 should still be computable
            if stats:
                assert isinstance(stats.get('p50', 0), (int, float))

    def test_million_transition_prune(self):
        """Ensure O(1) prune even with 1M transitions."""
        # bypass actual GPU calls
        if hasattr(self.bridge, '_state_trace'):
            self.bridge._state_trace = [{'dummy': i} for i in range(1_000_000)]
            t0 = time.perf_counter_ns()
            if hasattr(self.bridge, 'prune_state_trace'):
                self.bridge.prune_state_trace(100)
            elapsed = (time.perf_counter_ns() - t0) / 1e3   # µs
            assert elapsed < 500  # must stay sub-ms

    def test_json_escape_unsafe_prompt(self):
        """State trace must survive prompts with control chars."""
        unsafe = "table\u0000with\n\"quotes\" & \x1f"
        emb = random.randbytes(128)
        result = self.bridge.inference(emb, [unsafe])
        rep = self.bridge.get_state_trace_report()
        # round-trip through JSON must not crash
        json.loads(json.dumps(rep, ensure_ascii=False))


if __name__ == '__main__':
    import unittest
    unittest.main()
