import type { PathBuilder } from './pathBuilder';
import type { MeshBuilder } from './meshBuilder';

export type RpnValue = number | Float32Array | object;

export interface RpnOpHandler {
  (stack: RpnValue[], context: RpnContext): void;
}

export interface RpnContext {
  path?: PathBuilder;
  mesh?: MeshBuilder;
  matrixStack: Float32Array[];
}

function isNumericToken(token: string): boolean {
  if (!token) return false;
  const value = Number(token);
  return Number.isFinite(value);
}

function popValue(stack: RpnValue[]): RpnValue {
  const value = stack.pop();
  if (value === undefined) {
    throw new Error('RPN underflow');
  }
  return value;
}

export function popNumber(stack: RpnValue[]): number {
  const value = popValue(stack);
  if (typeof value !== 'number') {
    throw new Error(`Expected number, received ${typeof value}`);
  }
  return value;
}

export function popMatrix(stack: RpnValue[]): Float32Array {
  const value = popValue(stack);
  if (!(value instanceof Float32Array) || value.length !== 16) {
    throw new Error('Expected 4x4 matrix');
  }
  return value;
}

export class RpnEngine {
  private ops: Map<string, RpnOpHandler> = new Map();
  private stack: RpnValue[] = [];
  public context: RpnContext;

  constructor() {
    this.context = { matrixStack: [] };
    this.registerCoreOps();
  }

  registerOp(name: string, handler: RpnOpHandler): void {
    this.ops.set(String(name).trim().toUpperCase(), handler);
  }

  registerModule(module: Record<string, RpnOpHandler>): void {
    for (const [name, handler] of Object.entries(module)) {
      this.registerOp(name, handler);
    }
  }

  execute(program: string): RpnValue[] {
    const tokens = String(program || '').split(/\s+/).filter(Boolean);
    for (const token of tokens) {
      if (isNumericToken(token)) {
        this.stack.push(Number(token));
        continue;
      }
      const handler = this.ops.get(token.toUpperCase());
      if (!handler) {
        throw new Error(`Unknown RPN op: ${token}`);
      }
      handler(this.stack, this.context);
    }
    return [...this.stack];
  }

  reset(): void {
    this.stack.length = 0;
    this.context.matrixStack = [];
    if (this.context.path) this.context.path.reset();
    if (this.context.mesh) this.context.mesh.reset();
  }

  private registerCoreOps(): void {
    const unary = (fn: (value: number) => number): RpnOpHandler => (stack) => {
      stack.push(fn(popNumber(stack)));
    };
    const binary = (fn: (a: number, b: number) => number): RpnOpHandler => (stack) => {
      const b = popNumber(stack);
      const a = popNumber(stack);
      stack.push(fn(a, b));
    };
    const registerAliases = (names: string[], handler: RpnOpHandler) => {
      for (const name of names) this.registerOp(name, handler);
    };

    registerAliases(['ADD', '+'], binary((a, b) => a + b));
    registerAliases(['SUB', '-'], binary((a, b) => a - b));
    registerAliases(['MUL', '*'], binary((a, b) => a * b));
    registerAliases(['DIV', '/'], binary((a, b) => (b === 0 ? 0 : a / b)));
    registerAliases(['POW', 'pow'], binary((a, b) => Math.pow(a, b)));
    registerAliases(['MIN', 'min'], binary((a, b) => Math.min(a, b)));
    registerAliases(['MAX', 'max'], binary((a, b) => Math.max(a, b)));
    registerAliases(['GT'], binary((a, b) => (a > b ? 1 : 0)));
    registerAliases(['LT'], binary((a, b) => (a < b ? 1 : 0)));
    registerAliases(['EQ'], binary((a, b) => (a === b ? 1 : 0)));
    registerAliases(['ATAN2', 'atan2'], binary((a, b) => Math.atan2(a, b)));

    registerAliases(['NEGATE'], unary((value) => -value));
    registerAliases(['ABS', 'abs'], unary((value) => Math.abs(value)));
    registerAliases(['SQRT', 'sqrt'], unary((value) => Math.sqrt(Math.max(0, value))));
    registerAliases(['SIN', 'sin'], unary((value) => Math.sin(value)));
    registerAliases(['COS', 'cos'], unary((value) => Math.cos(value)));
    registerAliases(['TAN', 'tan'], unary((value) => Math.tan(value)));
    registerAliases(['ASIN'], unary((value) => Math.asin(value)));
    registerAliases(['ACOS'], unary((value) => Math.acos(value)));

    registerAliases(['DUP', 'dup'], (stack) => {
      stack.push(popValue(stack));
      stack.push(stack[stack.length - 1]);
    });
    registerAliases(['SWAP', 'swap'], (stack) => {
      const a = popValue(stack);
      const b = popValue(stack);
      stack.push(a);
      stack.push(b);
    });
    registerAliases(['DROP', 'drop'], (stack) => {
      popValue(stack);
    });

    registerAliases(['PI'], (stack) => stack.push(Math.PI));
    registerAliases(['TAU'], (stack) => stack.push(Math.PI * 2));
    registerAliases(['E'], (stack) => stack.push(Math.E));
  }
}
