from knowledge3d.cranium.actions.adaptive_convergence_analyzer import AdaptiveConvergenceAnalyzer


def test_convergence_metrics_empty():
    analyzer = AdaptiveConvergenceAnalyzer(window=5)
    metrics = analyzer.metrics()
    assert metrics["variance"] == 0.0
    assert metrics["range"] == 0.0


def test_convergence_metrics_values():
    analyzer = AdaptiveConvergenceAnalyzer(window=5)
    analyzer.extend([0.1, 0.12, 0.11, 0.11, 0.105])
    metrics = analyzer.metrics()
    assert metrics["variance"] >= 0.0
    assert metrics["mean"] > 0.0
    assert analyzer.is_stable(threshold=0.01)
