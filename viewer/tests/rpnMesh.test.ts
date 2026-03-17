import { createVisualRpnEngine } from '../src/rpn';

describe('RPN mesh generation', () => {
  it('generates cube geometry', () => {
    const engine = createVisualRpnEngine();
    engine.execute('1.0 GEN_CUBE');
    const geometry = engine.context.mesh!.toGeometry();
    expect(geometry.getAttribute('position').count).toBeGreaterThan(0);
  });

  it('executes the memory tablet visual program', () => {
    const engine = createVisualRpnEngine();
    engine.execute(
      '1.0 GEN_CUBE 0.40 0.28 0.02 MAT4_SCALE MAT4_APPLY ' +
      '1.0 GEN_CUBE 0.36 0.24 0.01 MAT4_SCALE MAT4_APPLY ' +
      '0.0 0.0 0.011 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT'
    );
    const geometry = engine.context.mesh!.toGeometry();
    expect(geometry.getAttribute('position').count).toBeGreaterThan(0);
  });

  it('executes lathe and extrude house programs', () => {
    const engine = createVisualRpnEngine();
    engine.execute(
      '0.0 0.0 MOVE 0.16 0.0 LINE 0.08 0.14 LINE CLOSE 0.16 EXTRUDE'
    );
    expect(engine.context.mesh!.toGeometry().getAttribute('position').count).toBeGreaterThan(0);

    engine.reset();
    engine.execute(
      '0.8 0.0 MOVE 0.92 0.16 LINE 0.92 0.52 LINE 0.56 0.7 0.22 0.6 QUAD 0.0 0.6 LINE CLOSE 20 LATHE'
    );
    expect(engine.context.mesh!.toGeometry().getAttribute('position').count).toBeGreaterThan(0);
  });
});
