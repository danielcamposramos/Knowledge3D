export type RPNToken = number | string;

export interface RPNOptions {
  maxDepth?: number; // total capacity
  keepTop?: number;  // keep at least this many elements after auto-clean
}

export class RPN {
  private stack: number[] = [];
  private maxDepth: number;
  private keepTop: number;

  constructor(opts: RPNOptions = {}) {
    this.maxDepth = opts.maxDepth ?? 64;
    this.keepTop = Math.max(5, opts.keepTop ?? 5); // ensure at least 5 levels of memory
  }

  reset() {
    this.stack.length = 0;
  }

  push(x: number) {
    this.stack.push(x);
    this.autoClean();
  }

  pop(): number {
    const v = this.stack.pop();
    if (v === undefined) throw new Error('RPN underflow');
    return v;
  }

  peek(n = 0): number {
    const idx = this.stack.length - 1 - n;
    if (idx < 0) throw new Error('RPN underflow');
    return this.stack[idx];
  }

  size() {
    return this.stack.length;
  }

  private autoClean() {
    if (this.stack.length > this.maxDepth) {
      // Drop oldest but keep at least keepTop values
      const keep = Math.max(this.keepTop, 1);
      this.stack.splice(0, this.stack.length - keep);
    }
  }

  eval(tokens: RPNToken[]): number {
    for (const t of tokens) {
      if (typeof t === 'number') {
        this.push(t);
        continue;
      }
      switch (t) {
        case '+': this.push(this.pop() + this.pop()); break;
        case '-': {
          const b = this.pop(); const a = this.pop(); this.push(a - b); break;
        }
        case '*': this.push(this.pop() * this.pop()); break;
        case '/': {
          const b = this.pop(); const a = this.pop(); this.push(a / b || 0); break;
        }
        case 'sqrt': this.push(Math.sqrt(this.pop())); break;
        case 'abs': this.push(Math.abs(this.pop())); break;
        case 'dup': this.push(this.peek()); break;
        case 'drop': this.pop(); break;
        case 'swap': { const a = this.pop(), b = this.pop(); this.push(a); this.push(b); break; }
        default:
          throw new Error(`RPN unknown op: ${t}`);
      }
    }
    return this.peek(0);
  }

  dot(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc = this.eval([acc, a[i], b[i], '*', '+']);
    }
    return acc;
  }

  norm(a: number[]): number {
    let acc = 0;
    for (let i = 0; i < a.length; i++) {
      acc = this.eval([acc, a[i], a[i], '*', '+']);
    }
    return Math.sqrt(acc);
  }

  cosine(a: number[], b: number[]): number {
    const d = this.dot(a, b);
    const na = this.norm(a);
    const nb = this.norm(b);
    if (na === 0 || nb === 0) return 0;
    return this.eval([d, na, '/', nb, '/']);
  }
}

