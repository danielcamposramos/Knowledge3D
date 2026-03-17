# Phase H11: DOM Projection — K3D Speaks the Web

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H9 (Browser RPN Interpreter) COMPLETE, Phase H10 (House Interaction) COMPLETE
**Sovereignty:** I/O path (browser rendering, flexible).
**Build:** Use `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` — NOT npm/vite from viewer/.
**Origin:** Christoph Dorn direction (2026-03-17): "build a projection runtime that writes to the DOM directly"

---

## Context

The RPN engine (H9) currently projects to three output targets:

| Target | Builder | Output |
|--------|---------|--------|
| Three.js Mesh | `MeshBuilder` | `THREE.BufferGeometry` → 3D scene |
| SVG | `PathBuilder.toSVGPath()` | `SVGSVGElement` → vector graphics |
| Canvas 2D | `PathBuilder.toCanvas2D()` | Canvas rendering context calls |

All three produce **visual geometry** from the same procedural programs. But K3D's Dual-Client Contract says: same data, two clients, same reality. The web's native reality is the **DOM**. If K3D can project into DOM elements, any website becomes an incremental migration target — K3D-generated content replacing static HTML, no 3D viewer required.

Christoph's directive: start with `<p>`, then expand. The goal is for K3D to "speak the current DOM web paradigm" as an immediate path for adoption.

**What this phase adds:**

1. A `DomBuilder` that accumulates DOM elements on the RPN stack — analogous to how `MeshBuilder` accumulates geometry
2. A `domOps` module registering DOM-specific RPN operations — analogous to `meshOps`
3. A `createDomRpnEngine()` factory — analogous to `createVisualRpnEngine()`
4. A convenience `rpnToDOM()` function — analogous to `rpnToMesh()` and `rpnToSVG()`
5. Integration into the tablet's ContentApp overlay — proving the pipeline works

---

## Architectural Principle: DOM as Projection, Not Template

This is NOT a template engine. K3D doesn't "fill in" HTML templates with data. It **projects** meaning-layer knowledge into DOM structure, the same way it projects visual_rpn into geometry.

The pipeline:

```
Galaxy Entry (meaning + references)
    ↓
RPN Program (procedural composition)
    ↓
DomBuilder (accumulates elements)
    ↓
DOM Tree (HTMLElement)
```

The same entry that produces a 3D mesh via `MeshBuilder` can produce a DOM subtree via `DomBuilder`. Different projection, same source of truth. This is the Dual-Client Contract extended to the web itself.

---

## Deliverables

### Track A: DomBuilder + domOps Module
### Track B: DOM RPN Engine Factory + rpnToDOM
### Track C: ContentApp DOM Projection Integration
### Track D: Character/Word DOM Rendering

---

## Track A: DomBuilder + domOps Module

### A1. Create `viewer/src/rpn/domBuilder.ts`

The DomBuilder maintains a stack of DOM elements, analogous to MeshBuilder's geometry stack. Elements are created, configured, and composed via RPN operations.

```typescript
export class DomBuilder {
  private elements: HTMLElement[] = [];
  private root: HTMLElement;

  constructor() {
    this.root = document.createElement('div');
    this.root.className = 'k3d-dom-root';
  }

  reset(): void {
    this.elements = [];
    this.root = document.createElement('div');
    this.root.className = 'k3d-dom-root';
  }

  /** Push a new element onto the stack. */
  pushElement(tag: string): void {
    const el = document.createElement(tag);
    this.elements.push(el);
  }

  /** Pop the top element from the stack. */
  popElement(): HTMLElement | undefined {
    return this.elements.pop();
  }

  /** Peek at the top element without removing it. */
  peekElement(): HTMLElement | undefined {
    return this.elements[this.elements.length - 1];
  }

  /** Pop the top element and append it as a child of the new top. */
  appendChild(): void {
    if (this.elements.length < 2) return;
    const child = this.elements.pop()!;
    this.elements[this.elements.length - 1].appendChild(child);
  }

  /** Pop the top element and append it to root. */
  appendToRoot(): void {
    const el = this.elements.pop();
    if (el) this.root.appendChild(el);
  }

  /** Set text content on top element. */
  setText(text: string): void {
    const el = this.peekElement();
    if (el) el.textContent = text;
  }

  /** Set an attribute on top element. */
  setAttribute(name: string, value: string): void {
    const el = this.peekElement();
    if (el) el.setAttribute(name, value);
  }

  /** Set a CSS class on top element. */
  setClass(className: string): void {
    const el = this.peekElement();
    if (el) el.className = className;
  }

  /** Get the composed DOM tree. */
  toDOM(): HTMLElement {
    // Flush remaining stack elements to root
    while (this.elements.length) {
      this.root.appendChild(this.elements.shift()!);
    }
    return this.root;
  }

  hasElements(): boolean {
    return this.elements.length > 0 || this.root.childElementCount > 0;
  }
}
```

### A2. Create `viewer/src/rpn/domOps.ts`

DOM operations for the RPN engine. These follow the same pattern as `meshOps` and `pathOps` — registered via `engine.registerModule(domOps)`.

**Design choice:** String values need to get onto the RPN stack. The stack type `RpnValue = number | Float32Array | object` already supports `object`. We use `{ str: string }` wrapper objects for string values on the stack. This avoids changing the core engine type system.

```typescript
import type { RpnOpHandler } from './engine';
import { popNumber } from './engine';

// String wrapper for RPN stack
export interface RpnString { str: string }
export function popString(stack: any[]): string {
  const val = stack.pop();
  if (val === undefined) throw new Error('RPN underflow');
  if (typeof val === 'object' && val !== null && 'str' in val) return val.str;
  if (typeof val === 'number') return String(val);
  throw new Error(`Expected string, received ${typeof val}`);
}

export const domOps: Record<string, RpnOpHandler> = {
  // --- String operations ---
  /** Push a string literal. Usage: STR_<text> (underscore-delimited) */
  // NOTE: String literals handled specially in engine execute loop (see A3)

  // --- Element creation ---
  /** Create element by tag. Usage: STR_p DOM_CREATE → pushes <p> */
  DOM_CREATE: (stack, ctx) => {
    const tag = popString(stack);
    ctx.dom?.pushElement(tag);
  },

  // --- Convenience: common elements ---
  /** Create <p>. */
  DOM_P: (_stack, ctx) => { ctx.dom?.pushElement('p'); },
  /** Create <span>. */
  DOM_SPAN: (_stack, ctx) => { ctx.dom?.pushElement('span'); },
  /** Create <div>. */
  DOM_DIV: (_stack, ctx) => { ctx.dom?.pushElement('div'); },
  /** Create <h1>. */
  DOM_H1: (_stack, ctx) => { ctx.dom?.pushElement('h1'); },
  /** Create <h2>. */
  DOM_H2: (_stack, ctx) => { ctx.dom?.pushElement('h2'); },
  /** Create <h3>. */
  DOM_H3: (_stack, ctx) => { ctx.dom?.pushElement('h3'); },
  /** Create <ul>. */
  DOM_UL: (_stack, ctx) => { ctx.dom?.pushElement('ul'); },
  /** Create <li>. */
  DOM_LI: (_stack, ctx) => { ctx.dom?.pushElement('li'); },
  /** Create <section>. */
  DOM_SECTION: (_stack, ctx) => { ctx.dom?.pushElement('section'); },
  /** Create <article>. */
  DOM_ARTICLE: (_stack, ctx) => { ctx.dom?.pushElement('article'); },

  // --- Text content ---
  /** Set text on top element. Usage: STR_Hello DOM_P DOM_TEXT → <p>Hello</p> */
  DOM_TEXT: (stack, ctx) => {
    const text = popString(stack);
    ctx.dom?.setText(text);
  },

  // --- Attributes ---
  /** Set attribute. Usage: STR_value STR_name DOM_ATTR */
  DOM_ATTR: (stack, ctx) => {
    const value = popString(stack);
    const name = popString(stack);
    ctx.dom?.setAttribute(name, value);
  },

  /** Set class. Usage: STR_highlight DOM_CLASS */
  DOM_CLASS: (stack, ctx) => {
    const className = popString(stack);
    ctx.dom?.setClass(className);
  },

  // --- Composition ---
  /** Pop top element, append as child of new top. */
  DOM_APPEND: (_stack, ctx) => { ctx.dom?.appendChild(); },

  /** Pop top element, append to root container. */
  DOM_EMIT: (_stack, ctx) => { ctx.dom?.appendToRoot(); },
};
```

### A3. Extend RpnContext for DOM

Add `dom?: DomBuilder` to `RpnContext` interface:

```typescript
// In engine.ts — extend the interface:
export interface RpnContext {
  path?: PathBuilder;
  mesh?: MeshBuilder;
  dom?: DomBuilder;    // NEW
  matrixStack: Float32Array[];
}
```

### A4. String literal support in RPN engine

The RPN engine currently handles numeric literals by detecting `Number.isFinite()`. For DOM projection, we need string literals on the stack. Add a `STR_` prefix convention:

In `RpnEngine.execute()`, add this check after the numeric literal check:

```typescript
// After numeric check, before op lookup:
if (token.toUpperCase().startsWith('STR_')) {
  // Push string value (everything after STR_ prefix, underscores become spaces)
  this.stack.push({ str: token.slice(4).replace(/_/g, ' ') });
  continue;
}
```

This allows: `STR_Hello_World DOM_P DOM_TEXT` → `<p>Hello World</p>`

**Why underscore-to-space?** RPN programs are whitespace-delimited. Underscores are the natural escape for spaces in token-based languages. This keeps the engine simple.

---

## Track B: DOM RPN Engine Factory + rpnToDOM

### B1. Add `createDomRpnEngine()` and `rpnToDOM()` to `viewer/src/rpn/index.ts`

```typescript
import { DomBuilder } from './domBuilder';
import { domOps } from './domOps';

export function createDomRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.context.dom = new DomBuilder();
  engine.registerModule(domOps);
  return engine;
}

export function rpnToDOM(program: string): HTMLElement {
  const engine = createDomRpnEngine();
  engine.execute(program);
  return engine.context.dom!.toDOM();
}
```

Also export from the barrel:
```typescript
export { DomBuilder, rpnToDOM, domOps };
```

### B2. Combined engine (visual + DOM)

Some programs may need both visual AND DOM output (e.g., an object that renders as 3D mesh in the House AND as HTML on the tablet). Add a combined factory:

```typescript
export function createFullRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.context.path = new PathBuilder();
  engine.context.mesh = new MeshBuilder();
  engine.context.dom = new DomBuilder();
  engine.registerModule(mat4Ops);
  engine.registerModule(pathOps);
  engine.registerModule(meshOps);
  engine.registerModule(domOps);
  return engine;
}
```

---

## Track C: ContentApp DOM Projection Integration

### C1. Extend ContentApp overlay to use DOM projection

Currently, `ContentApp.openOverlay()` manually builds DOM elements from `ContentPage` data. Enhance it to also render any `dom_rpn` program associated with a node:

When a House node has Galaxy references, the ContentApp should be able to render Galaxy entry content as DOM via `rpnToDOM()`. This proves the pipeline works end-to-end: click object → behavior_rpn dispatches → tablet shows content → content includes DOM-projected Galaxy data.

In `apps.ts`, update `ContentApp.openOverlay()`:

```typescript
// After rendering existing sections, if any section has an RPN program,
// try DOM projection:
if (section.heading === 'Content Galaxy' && section.lines[0]) {
  try {
    const ref = section.lines[0].replace('Load: ', '');
    // Render a DOM preview card showing what this Galaxy entry would produce
    const preview = document.createElement('div');
    preview.className = 'k3d-dom-preview';
    preview.style.border = '1px dashed #555';
    preview.style.padding = '8px';
    preview.style.marginTop = '8px';
    preview.style.background = '#1a1f25';
    const label = document.createElement('div');
    label.style.color = '#666';
    label.style.fontSize = '11px';
    label.textContent = 'DOM Projection Preview';
    preview.appendChild(label);
    // Render the node's identity as DOM via rpnToDOM
    const domProgram = buildNodeDomProgram(this.page);
    if (domProgram) {
      const rendered = rpnToDOM(domProgram);
      rendered.style.marginTop = '6px';
      preview.appendChild(rendered);
    }
    card.appendChild(preview);
  } catch { /* DOM projection is best-effort */ }
}
```

### C2. Create `viewer/src/behavior/domProjection.ts`

A helper that converts House node data into DOM RPN programs. This is the bridge between the content layer (H10) and DOM projection (H11).

```typescript
import type { ContentPage } from './contentRenderer';

/**
 * Builds a DOM RPN program string from a ContentPage.
 * This demonstrates K3D projecting structured knowledge into web-native DOM.
 */
export function buildNodeDomProgram(page: ContentPage | null): string | null {
  if (!page) return null;
  const tokens: string[] = [];

  // Title as <h2>
  const safeTitle = page.title.replace(/\s+/g, '_');
  tokens.push(`STR_${safeTitle}`, 'DOM_H2', 'DOM_TEXT', 'DOM_EMIT');

  // Each section as <section> with <h3> heading and <p> lines
  for (const section of page.sections) {
    tokens.push('DOM_SECTION');

    // Heading
    const safeHeading = section.heading.replace(/\s+/g, '_');
    tokens.push(`STR_${safeHeading}`, 'DOM_H3', 'DOM_TEXT', 'DOM_APPEND');

    // Lines as <p> elements
    for (const line of section.lines) {
      const safeLine = line.replace(/\s+/g, '_');
      tokens.push(`STR_${safeLine}`, 'DOM_P', 'DOM_TEXT', 'DOM_APPEND');
    }

    tokens.push('DOM_EMIT');
  }

  return tokens.join(' ');
}
```

---

## Track D: Character/Word DOM Rendering

### D1. Character-to-DOM projection

Character Galaxy entries have `char_refs` arrays. When projecting to DOM, each character reference can become a `<span>` with semantic data attributes preserving the K3D identity:

```typescript
// In domOps.ts, add:

/** Render a K3D character reference as a <span>.
 *  Usage: STR_char_u0041 DOM_CHAR → <span data-k3d-char="char_u0041">A</span>
 */
DOM_CHAR: (stack, ctx) => {
  const charRef = popString(stack);
  ctx.dom?.pushElement('span');
  ctx.dom?.setAttribute('data-k3d-ref', charRef);
  // Extract codepoint from char_u{hex} pattern
  const match = charRef.match(/^char_u([0-9a-fA-F]{4,6})$/);
  if (match) {
    const codepoint = parseInt(match[1], 16);
    ctx.dom?.setText(String.fromCodePoint(codepoint));
  }
  ctx.dom?.setClass('k3d-char');
},

/** Render a K3D word as a sequence of character <span>s inside a <span>.
 *  Usage: STR_word_hello DOM_WORD → <span class="k3d-word" data-k3d-ref="word_hello">
 *           <span class="k3d-char" data-k3d-ref="char_u0068">h</span>
 *           <span class="k3d-char" data-k3d-ref="char_u0065">e</span>
 *           ...
 *         </span>
 *  NOTE: This is a simplified version. Full char_refs resolution requires Galaxy lookup.
 *  For H11, we decode characters from the word text directly.
 */
DOM_WORD: (stack, ctx) => {
  const wordRef = popString(stack);
  ctx.dom?.pushElement('span');
  ctx.dom?.setClass('k3d-word');
  ctx.dom?.setAttribute('data-k3d-ref', wordRef);
  // Extract word text from word_{text} pattern
  const wordText = wordRef.replace(/^word_/, '').replace(/_/g, ' ');
  for (const ch of wordText) {
    const hex = ch.codePointAt(0)!.toString(16).padStart(4, '0');
    ctx.dom?.pushElement('span');
    ctx.dom?.setClass('k3d-char');
    ctx.dom?.setAttribute('data-k3d-ref', `char_u${hex}`);
    ctx.dom?.setText(ch);
    ctx.dom?.appendChild(); // append char span to word span
  }
},
```

### D2. Semantic data attributes

Every DOM element projected by K3D carries `data-k3d-*` attributes preserving its identity in the knowledge graph. This means:

- A `<p>` produced by K3D can be traced back to its Galaxy source
- A `<span class="k3d-char">` can be traced to its Character Galaxy entry
- CSS can style K3D-projected content differently from static HTML
- JavaScript can query K3D content via `document.querySelectorAll('[data-k3d-ref]')`

This is the DOM-level expression of the Dual-Client Contract: the human sees styled text, but the semantic identity is preserved in data attributes for any system that wants to read it.

---

## Tips for Codex

**Tip 1 — RpnContext extension is minimal.** Just add `dom?: DomBuilder` to the interface in `engine.ts`. The `?` means existing code that doesn't use DOM is unaffected. Same pattern as `path?` and `mesh?`.

**Tip 2 — String literal prefix.** `STR_Hello_World` becomes `{ str: "Hello World" }` on the stack. Add the `STR_` check in `execute()` AFTER the numeric check, BEFORE the op lookup. This is a 4-line change to the engine.

**Tip 3 — DomBuilder parallels MeshBuilder.** If you understand how MeshBuilder works (geometry stack, push/pop/merge), DomBuilder is the same pattern with DOM elements instead of BufferGeometry.

**Tip 4 — DOM_APPEND vs DOM_EMIT.** `DOM_APPEND` pops the top element and makes it a child of the new top (nesting). `DOM_EMIT` pops the top element and appends it to the root container (finalization). This is the DOM equivalent of `CSG_UNION` (combine) vs "flush to output".

**Tip 5 — domOps does NOT import Three.js.** Unlike meshOps, domOps has zero Three.js dependencies. It only uses browser DOM APIs (`document.createElement`, etc.). This is important for future use outside the 3D viewer.

**Tip 6 — Test with jsdom.** Jest + jsdom provides `document.createElement()` in Node.js. The DOM tests can run without a browser, same as all existing viewer tests.

**Tip 7 — Keep string escaping simple.** `STR_` prefix with underscore-to-space is deliberately minimal. Complex text formatting (markdown, rich text) is a future concern. H11 proves the pipeline with simple text.

**Tip 8 — data-k3d-ref attributes.** Every projected DOM element should carry `data-k3d-ref` when a Galaxy reference is known. This is the semantic thread connecting DOM output back to the knowledge graph. Don't skip it.

---

## Tests

### `viewer/tests/domBuilder.test.ts`

```typescript
import { DomBuilder } from '../src/rpn/domBuilder';

describe('DomBuilder', () => {
  it('creates and emits a paragraph', () => {
    const builder = new DomBuilder();
    builder.pushElement('p');
    builder.setText('Hello World');
    const root = builder.toDOM();
    expect(root.querySelector('p')?.textContent).toBe('Hello World');
  });

  it('nests elements via appendChild', () => {
    const builder = new DomBuilder();
    builder.pushElement('ul');
    builder.pushElement('li');
    builder.setText('Item 1');
    builder.appendChild();
    const root = builder.toDOM();
    const ul = root.querySelector('ul');
    expect(ul?.children.length).toBe(1);
    expect(ul?.querySelector('li')?.textContent).toBe('Item 1');
  });

  it('sets attributes on elements', () => {
    const builder = new DomBuilder();
    builder.pushElement('span');
    builder.setAttribute('data-k3d-ref', 'char_u0041');
    builder.setText('A');
    const root = builder.toDOM();
    const span = root.querySelector('span');
    expect(span?.getAttribute('data-k3d-ref')).toBe('char_u0041');
    expect(span?.textContent).toBe('A');
  });

  it('resets cleanly', () => {
    const builder = new DomBuilder();
    builder.pushElement('p');
    builder.setText('First');
    builder.toDOM();
    builder.reset();
    const root = builder.toDOM();
    expect(root.children.length).toBe(0);
  });
});
```

### `viewer/tests/domOps.test.ts`

```typescript
import { rpnToDOM } from '../src/rpn';

describe('DOM RPN operations', () => {
  it('creates a paragraph with text', () => {
    const root = rpnToDOM('STR_Hello_World DOM_P DOM_TEXT DOM_EMIT');
    const p = root.querySelector('p');
    expect(p?.textContent).toBe('Hello World');
  });

  it('creates nested list', () => {
    const root = rpnToDOM(
      'DOM_UL ' +
      'STR_Item_one DOM_LI DOM_TEXT DOM_APPEND ' +
      'STR_Item_two DOM_LI DOM_TEXT DOM_APPEND ' +
      'DOM_EMIT'
    );
    const ul = root.querySelector('ul');
    expect(ul?.children.length).toBe(2);
    expect(ul?.children[0].textContent).toBe('Item one');
  });

  it('creates heading with class', () => {
    const root = rpnToDOM('STR_k3d-title DOM_H1 DOM_CLASS STR_Welcome DOM_TEXT DOM_EMIT');
    const h1 = root.querySelector('h1');
    expect(h1?.className).toBe('k3d-title');
    expect(h1?.textContent).toBe('Welcome');
  });

  it('renders character reference', () => {
    const root = rpnToDOM('STR_char_u0041 DOM_CHAR DOM_EMIT');
    const span = root.querySelector('span.k3d-char');
    expect(span?.textContent).toBe('A');
    expect(span?.getAttribute('data-k3d-ref')).toBe('char_u0041');
  });

  it('renders word as character spans', () => {
    const root = rpnToDOM('STR_word_cat DOM_WORD DOM_EMIT');
    const word = root.querySelector('span.k3d-word');
    expect(word?.getAttribute('data-k3d-ref')).toBe('word_cat');
    const chars = word?.querySelectorAll('span.k3d-char');
    expect(chars?.length).toBe(3);
    expect(chars?.[0].textContent).toBe('c');
    expect(chars?.[1].textContent).toBe('a');
    expect(chars?.[2].textContent).toBe('t');
  });

  it('string literals convert underscores to spaces', () => {
    const root = rpnToDOM('STR_Hello_beautiful_World DOM_P DOM_TEXT DOM_EMIT');
    expect(root.querySelector('p')?.textContent).toBe('Hello beautiful World');
  });
});
```

### `viewer/tests/domProjection.test.ts`

```typescript
import { buildNodeDomProgram } from '../src/behavior/domProjection';
import { rpnToDOM } from '../src/rpn';

describe('DOM projection from ContentPage', () => {
  it('projects a content page to DOM', () => {
    const page = {
      title: 'Mathematics Primer',
      sections: [
        { heading: 'Identity', lines: ['Star ID: book_math', 'Class: book'] },
        { heading: 'References', lines: ['concept_mathematics'] },
      ],
    };
    const program = buildNodeDomProgram(page);
    expect(program).toBeTruthy();
    const root = rpnToDOM(program!);
    expect(root.querySelector('h2')?.textContent).toBe('Mathematics Primer');
    const sections = root.querySelectorAll('section');
    expect(sections.length).toBe(2);
    expect(sections[0].querySelector('h3')?.textContent).toBe('Identity');
  });

  it('returns null for null page', () => {
    expect(buildNodeDomProgram(null)).toBeNull();
  });
});
```

### Non-regression

All existing tests must pass. Build via `build.sh`.

---

## Success Criteria

1. `DomBuilder` creates, nests, and emits DOM elements
2. `domOps` module registers all specified operations (DOM_P, DOM_SPAN, DOM_DIV, DOM_H1-H3, DOM_UL, DOM_LI, DOM_SECTION, DOM_ARTICLE, DOM_TEXT, DOM_ATTR, DOM_CLASS, DOM_APPEND, DOM_EMIT, DOM_CREATE, DOM_CHAR, DOM_WORD)
3. `STR_` prefix pushes string literals onto RPN stack (underscores → spaces)
4. `rpnToDOM('STR_Hello DOM_P DOM_TEXT DOM_EMIT')` produces `<div class="k3d-dom-root"><p>Hello</p></div>`
5. `DOM_CHAR` renders character references with `data-k3d-ref` attributes
6. `DOM_WORD` renders word references as nested character `<span>`s
7. ContentApp overlay shows DOM projection preview for Galaxy content entries
8. All existing tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/rpn/domBuilder.ts` | **NEW** — DOM element stack builder |
| `viewer/src/rpn/domOps.ts` | **NEW** — DOM RPN operations module |
| `viewer/src/rpn/engine.ts` | Add `dom?: DomBuilder` to RpnContext, add `STR_` literal support |
| `viewer/src/rpn/index.ts` | Add `createDomRpnEngine()`, `createFullRpnEngine()`, `rpnToDOM()`, exports |
| `viewer/src/behavior/domProjection.ts` | **NEW** — ContentPage → DOM RPN program builder |
| `viewer/src/apps.ts` | Extend ContentApp overlay with DOM projection preview |
| `viewer/tests/domBuilder.test.ts` | **NEW** |
| `viewer/tests/domOps.test.ts` | **NEW** |
| `viewer/tests/domProjection.test.ts` | **NEW** |

---

## Architectural Note: Why This Matters

This phase makes K3D **web-native**. Every website is now a potential K3D projection surface. The same procedural programs that build 3D Houses, render SVG glyphs, and draw on Canvas 2D can now compose HTML documents.

The `data-k3d-ref` attributes on projected DOM elements create a **semantic web** in the original Tim Berners-Lee sense — HTML elements that carry machine-readable identity linking them back to the Galaxy knowledge graph. But unlike RDF/OWL (which bolted semantics onto existing HTML), K3D's DOM projection is **generative** — the semantics don't annotate existing content, they PRODUCE the content.

This is the first step toward Christoph's vision: K3D as the engine behind web content, with the current DOM paradigm as an incremental migration target. Today, `<p>`. Tomorrow, full semantic HTML applications. The 3D House remains the canonical reality; DOM projection is one of many views into it.

**Pipeline summary:**
```
Galaxy Entry → RPN Program → { MeshBuilder  → Three.js scene (3D)
                              { PathBuilder  → SVG element (2D vector)
                              { PathBuilder  → Canvas2D calls (2D raster)
                              { DomBuilder   → HTML elements (web)    ← NEW
```

Four projections. One source of truth. The Dual-Client Contract, now speaking the web's language.
