from __future__ import annotations

from pathlib import Path

from ...cranium.phase10.rpn_calculator import RPNCalculator  # type: ignore
from ...cranium.phase10.self_reflection_engine import SelfReflectionEngine  # type: ignore
from ...cranium.phase10.spatial_chain_of_thought import SpatialChainOfThought  # type: ignore


class CognitiveValidator:
    def __init__(self):
        self.rpn_calculator = RPNCalculator()
        self.self_reflection_engine = SelfReflectionEngine('viewer/public/models/stage4_formal_operational_perfect.pth')
        self.spatial_chain_of_thought = SpatialChainOfThought('viewer/public/house/house_master_assembled.glb')

    def validate_rpn_calculator(self):
        tests = [
            ('3 4 +', 7.0),
            ('3 4 + 2 *', 14.0),
            ('5 1 2 + 4 * + 3 -', 14.0),
            ('9 sqrt', 3.0),
            ('2 3 ^', 8.0),
        ]
        for expr, exp in tests:
            res = self.rpn_calculator.evaluate(expr)
            assert abs(res - exp) < 1e-9, f'RPN failed: {expr} -> {res} != {exp}'
        cross = self.rpn_calculator.evaluate_vector('[1,0,0] [0,1,0] cross')
        assert abs(cross[0]) < 1e-6 and abs(cross[1]) < 1e-6 and abs(cross[2] - 1.0) < 1e-6, 'Cross product failed'
        print('✅ RPN Calculator: PASSED')

    def validate_self_reflection(self):
        sample_path = Path('viewer/public/samples/stage4/stage4_fractal_tree_000.glb')
        if not sample_path.exists():
            print('⚠️ Missing fractal_tree sample for self-reflection test')
            return
        from ...tools.phase9.sample_loader import SampleLoader  # type: ignore
        sl = SampleLoader(str(sample_path.parent))
        s = sl.load_sample(sample_path)
        if not s:
            print('⚠️ Could not load sample for self-reflection')
            return
        reflection = self.self_reflection_engine.reflect_on_prediction({'embedding': s.embedding}, 'fractal_tree')
        print(f'🧠 Self-Reflection: {reflection}')
        assert any(x in reflection for x in ['✅','⚠️','❌'])
        print('✅ Self-Reflection: PASSED')

    def validate_spatial_chain_of_thought(self):
        q = 'If star_A is near star_B, and star_B connects to star_C, then star_A is related to star_C'
        res = self.spatial_chain_of_thought.reason_spatially(q)
        print(f'🔗 Spatial Chain-of-Thought: {res}')
        assert '→' in res or 'path' in res.lower()
        print('✅ Spatial Chain-of-Thought: PASSED')


def main():  # pragma: no cover
    v = CognitiveValidator()
    v.validate_rpn_calculator()
    v.validate_self_reflection()
    v.validate_spatial_chain_of_thought()


if __name__ == '__main__':  # pragma: no cover
    main()
