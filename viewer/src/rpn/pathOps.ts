import type { RpnOpHandler } from './engine';
import { popNumber } from './engine';

export const pathOps: Record<string, RpnOpHandler> = {
  MOVE: (stack, ctx) => {
    const y = popNumber(stack);
    const x = popNumber(stack);
    ctx.path?.move(x, y);
  },
  LINE: (stack, ctx) => {
    const y = popNumber(stack);
    const x = popNumber(stack);
    ctx.path?.line(x, y);
  },
  QUAD: (stack, ctx) => {
    const y = popNumber(stack);
    const x = popNumber(stack);
    const cy = popNumber(stack);
    const cx = popNumber(stack);
    ctx.path?.quad(cx, cy, x, y);
  },
  CUBIC: (stack, ctx) => {
    const y = popNumber(stack);
    const x = popNumber(stack);
    const c2y = popNumber(stack);
    const c2x = popNumber(stack);
    const c1y = popNumber(stack);
    const c1x = popNumber(stack);
    ctx.path?.cubic(c1x, c1y, c2x, c2y, x, y);
  },
  ARC: (stack, ctx) => {
    const endAngle = popNumber(stack);
    const startAngle = popNumber(stack);
    const radius = popNumber(stack);
    const cy = popNumber(stack);
    const cx = popNumber(stack);
    ctx.path?.arc(cx, cy, radius, startAngle, endAngle);
  },
  CLOSE: (_stack, ctx) => {
    ctx.path?.close();
  },
};
