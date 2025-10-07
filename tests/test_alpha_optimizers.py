from knowledge3d.cranium.actions.alpha_rl_optimizer import AlphaRLOptimizer, AlphaRange
from knowledge3d.cranium.actions.advanced_alpha_rl_optimizer import AdvancedAlphaRLOptimizer


def test_basic_optimizer_bounds():
    ranges = {"NAV_MOVE": AlphaRange(base=0.1, minimum=0.05, maximum=0.2)}
    opt = AlphaRLOptimizer(ranges)
    alpha = opt.update("NAV_MOVE", reward=1.0)
    assert ranges["NAV_MOVE"].minimum <= alpha <= ranges["NAV_MOVE"].maximum


def test_advanced_optimizer_momentum():
    ranges = {"DIALOGUE": AlphaRange(base=0.08, minimum=0.04, maximum=0.18)}
    opt = AdvancedAlphaRLOptimizer(ranges)
    initial = opt.get_alpha("DIALOGUE")
    opt.update_with_context("DIALOGUE", reward=0.5, context_score=2.0)
    updated = opt.get_alpha("DIALOGUE")
    assert updated >= initial
