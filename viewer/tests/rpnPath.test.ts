import { createVisualRpnEngine, rpnToSVG } from '../src/rpn';

describe('RPN path rendering', () => {
  it('builds SVG path strings from path ops', () => {
    const engine = createVisualRpnEngine();
    engine.execute('0 0 MOVE 1 0 LINE 0.5 1 LINE CLOSE');
    const svg = engine.context.path!.toSVGPath();
    expect(svg).toContain('M');
    expect(svg).toContain('L');
    expect(svg).toContain('Z');
  });

  it('creates an SVG element', () => {
    const svg = rpnToSVG('0 0 MOVE 1 0 LINE 0.5 1 LINE CLOSE');
    expect(svg.tagName.toLowerCase()).toBe('svg');
    expect(svg.querySelector('path')?.getAttribute('d')).toContain('M');
  });
});
