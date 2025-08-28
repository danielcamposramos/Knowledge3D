import { RPN } from '../src/rpn';

describe('RPN', () => {
  it('evaluates basic arithmetic', () => {
    const r = new RPN({ maxDepth: 8, keepTop: 5 });
    expect(r.eval([2, 3, '+'])).toBe(5);
    expect(r.eval([10, 2, '/'])).toBeCloseTo(5);
    expect(r.eval([5, 2, '*'])).toBe(10);
  });

  it('computes cosine similarity', () => {
    const r = new RPN();
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    const c = [1, 0, 0];
    expect(r.cosine(a, b)).toBeCloseTo(0, 6);
    expect(r.cosine(a, c)).toBeCloseTo(1, 6);
  });
});

