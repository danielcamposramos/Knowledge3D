import { createVisualRpnEngine } from '../src/rpn';

describe('RpnEngine', () => {
  it('evaluates arithmetic operations', () => {
    const engine = createVisualRpnEngine();
    expect(engine.execute('3 4 ADD')).toEqual([7]);
    engine.reset();
    expect(engine.execute('10 3 SUB')).toEqual([7]);
    engine.reset();
    expect(engine.execute('2 3 POW')).toEqual([8]);
  });

  it('creates scale matrices', () => {
    const engine = createVisualRpnEngine();
    const result = engine.execute('2 3 4 MAT4_SCALE');
    expect(result[0]).toBeInstanceOf(Float32Array);
    const matrix = result[0] as Float32Array;
    expect(matrix[0]).toBeCloseTo(2);
    expect(matrix[5]).toBeCloseTo(3);
    expect(matrix[10]).toBeCloseTo(4);
  });
});
