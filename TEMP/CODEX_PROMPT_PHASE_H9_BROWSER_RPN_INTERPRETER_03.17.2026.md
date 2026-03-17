# Phase H9: Browser-Side RPN Interpreter (Tier 2 Projection)

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H8 (Viewer House Integration) COMPLETE
**Sovereignty:** I/O path (browser rendering, flexible).
**Build:** Use `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` — NOT npm/vite from viewer/.

---

## Context

The House GLB carries `visual_rpn` metadata on every node (54 objects). The viewer loads this metadata via `loadHouseScene.ts` but never executes it — all geometry comes pre-baked in the GLTF mesh. This is **Tier 1**: server executes RPN → GLTF geometry → browser renders via Three.js.

**Tier 2** adds a browser-side RPN interpreter: the server sends RPN strings, the browser executes them locally and renders the result. This enables:
- Live procedural editing (change RPN, see result instantly)
- Reduced server dependency (browser generates its own geometry)
- Foundation for Christoph's JS component architecture
- Path toward Tier 3 (WebGPU sovereign rendering)

The existing `RpnApp` in the tablet handles arithmetic only (`+`, `-`, `*`, `/`, `sin`, `cos`, etc.). This phase adds **visual RPN execution**: 2D drawing paths and 3D mesh generation.

---

## Deliverables

### Track A: RPN Stack Machine (Core Engine)
### Track B: 2D Path Renderer (Canvas 2D / SVG)
### Track C: 3D Mesh Generator (Three.js)
### Track D: Integration with House Viewer

---

## Track A: RPN Stack Machine

### A1. Create `viewer/src/rpn/engine.ts`

A generic stack-based RPN evaluator. This is the core — modular, extensible, renderer-agnostic.

```typescript
export type RpnValue = number | Float32Array | object;

export interface RpnOpHandler {
  (stack: RpnValue[], context: RpnContext): void;
}

export interface RpnContext {
  /** 2D path builder — populated by Track B */
  path?: PathBuilder;
  /** 3D mesh builder — populated by Track C */
  mesh?: MeshBuilder;
  /** Mat4 stack for transform accumulation */
  matrixStack: Float32Array[];  // each is 16 floats
}

export class RpnEngine {
  private ops: Map<string, RpnOpHandler> = new Map();
  private stack: RpnValue[] = [];
  private context: RpnContext;

  constructor() {
    this.context = { matrixStack: [] };
    this.registerCoreOps();
  }

  /** Register an operation handler by name. */
  registerOp(name: string, handler: RpnOpHandler): void;

  /** Register a batch of operations from a module. */
  registerModule(module: Record<string, RpnOpHandler>): void;

  /** Execute an RPN program string. */
  execute(program: string): RpnValue[];

  /** Reset stack and context for a new program. */
  reset(): void;
}
```

### A2. Core operations (arithmetic + stack)

Register the basic operations that already work in `RpnApp` but as proper typed handlers:

- **Arithmetic**: `ADD`, `SUB`, `MUL`, `DIV`, `NEGATE`, `ABS`, `SQRT`, `POW`
- **Trig**: `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN2`
- **Stack**: `DUP`, `SWAP`, `DROP`
- **Comparison**: `GT`, `LT`, `EQ`, `MIN`, `MAX`
- **Constants**: `PI`, `TAU`, `E`

Numbers are parsed as literals and pushed onto the stack automatically.

### A3. Matrix4 operations

These are critical — every House object uses MAT4 transforms:

```typescript
// viewer/src/rpn/mat4Ops.ts
export const mat4Ops: Record<string, RpnOpHandler> = {
  MAT4_IDENTITY: (stack) => { /* push 4x4 identity as Float32Array(16) */ },
  MAT4_TRANSLATE: (stack) => { /* pop z,y,x → push translation matrix */ },
  MAT4_SCALE: (stack) => { /* pop z,y,x → push scale matrix */ },
  MAT4_ROTATE_X: (stack) => { /* pop angle → push rotation matrix */ },
  MAT4_ROTATE_Y: (stack) => { /* pop angle → push rotation matrix */ },
  MAT4_ROTATE_Z: (stack) => { /* pop angle → push rotation matrix */ },
  MAT4_MUL: (stack) => { /* pop two mat4s → push product */ },
  MAT4_APPLY: (stack, ctx) => { /* pop mat4 → apply to current mesh/path */ },
};
```

**TIP:** Use `THREE.Matrix4` internally for the math — it already handles all 4x4 operations. Convert to/from Float32Array for the stack.

---

## Track B: 2D Path Renderer

### B1. Create `viewer/src/rpn/pathBuilder.ts`

Builds a 2D path from RPN path operations, then renders to Canvas 2D or SVG.

```typescript
export interface PathPoint {
  type: 'move' | 'line' | 'quad' | 'cubic' | 'arc' | 'close';
  coords: number[];  // varies by type
}

export class PathBuilder {
  private segments: PathPoint[] = [];

  move(x: number, y: number): void;
  line(x: number, y: number): void;
  quad(cx: number, cy: number, x: number, y: number): void;
  cubic(c1x: number, c1y: number, c2x: number, c2y: number, x: number, y: number): void;
  close(): void;

  /** Render to Canvas 2D context. */
  toCanvas2D(ctx: CanvasRenderingContext2D): void;

  /** Render to SVG path string (d attribute). */
  toSVGPath(): string;

  /** Get bounding box of all points. */
  bounds(): { minX: number; minY: number; maxX: number; maxY: number };
}
```

### B2. Path RPN operations

```typescript
// viewer/src/rpn/pathOps.ts
export const pathOps: Record<string, RpnOpHandler> = {
  MOVE: (stack, ctx) => { /* pop y,x → ctx.path.move(x,y) */ },
  LINE: (stack, ctx) => { /* pop y,x → ctx.path.line(x,y) */ },
  QUAD: (stack, ctx) => { /* pop y,x,cy,cx → ctx.path.quad(cx,cy,x,y) */ },
  CUBIC: (stack, ctx) => { /* pop y,x,c2y,c2x,c1y,c1x → ctx.path.cubic(...) */ },
  CLOSE: (stack, ctx) => { /* ctx.path.close() */ },
};
```

### B3. SVG output mode

Add a function that takes a `visual_rpn` string and produces an SVG element:

```typescript
export function rpnToSVG(program: string): SVGElement;
```

This is the Tier 2 projection primitive — server sends RPN, browser produces SVG. Christoph can build his component architecture around this function.

---

## Track C: 3D Mesh Generator

### C1. Create `viewer/src/rpn/meshBuilder.ts`

Builds Three.js geometry from RPN mesh operations.

```typescript
export class MeshBuilder {
  private geometries: THREE.BufferGeometry[] = [];
  private transforms: THREE.Matrix4[] = [];

  /** Add a primitive geometry. */
  addPrimitive(geom: THREE.BufferGeometry): void;

  /** Apply a transform to the most recent geometry. */
  applyTransform(matrix: THREE.Matrix4): void;

  /** CSG union of top two geometries on the stack. */
  csgUnion(): void;

  /** CSG subtract (second from first). */
  csgSubtract(): void;

  /** Get the final merged Three.js geometry. */
  toGeometry(): THREE.BufferGeometry;

  /** Get as Three.js Mesh with default material. */
  toMesh(material?: THREE.Material): THREE.Mesh;
}
```

### C2. Primitive generation operations

```typescript
// viewer/src/rpn/meshOps.ts
export const meshOps: Record<string, RpnOpHandler> = {
  GEN_CUBE: (stack, ctx) => {
    const size = stack.pop() as number;
    const geom = new THREE.BoxGeometry(size, size, size);
    ctx.mesh!.addPrimitive(geom);
  },
  GEN_CYLINDER: (stack, ctx) => {
    // pop closed,segments,height,radius
    const closed = stack.pop() as number;
    const segments = stack.pop() as number;
    const height = stack.pop() as number;
    const radius = stack.pop() as number;
    const geom = new THREE.CylinderGeometry(radius, radius, height, segments, 1, !closed);
    ctx.mesh!.addPrimitive(geom);
  },
  GEN_CONE: (stack, ctx) => {
    // pop segments,height,radius
    const segments = stack.pop() as number;
    const height = stack.pop() as number;
    const radius = stack.pop() as number;
    const geom = new THREE.ConeGeometry(radius, height, segments);
    ctx.mesh!.addPrimitive(geom);
  },
  GEN_TORUS: (stack, ctx) => {
    // pop tubularSegments,radialSegments,tube,radius
    const tubularSegments = stack.pop() as number;
    const radialSegments = stack.pop() as number;
    const tube = stack.pop() as number;
    const radius = stack.pop() as number;
    const geom = new THREE.TorusGeometry(radius, tube, radialSegments, tubularSegments);
    ctx.mesh!.addPrimitive(geom);
  },
  GEN_UV_SPHERE: (stack, ctx) => {
    // pop widthSegments,heightSegments,radius
    const widthSegments = stack.pop() as number;
    const heightSegments = stack.pop() as number;
    const radius = stack.pop() as number;
    const geom = new THREE.SphereGeometry(radius, widthSegments, heightSegments);
    ctx.mesh!.addPrimitive(geom);
  },
  GEN_PLANE: (stack, ctx) => {
    // pop segmentsY,segmentsX,height,width
    const segmentsY = stack.pop() as number;
    const segmentsX = stack.pop() as number;
    const height = stack.pop() as number;
    const width = stack.pop() as number;
    const geom = new THREE.PlaneGeometry(width, height, segmentsX, segmentsY);
    ctx.mesh!.addPrimitive(geom);
  },
};
```

### C3. CSG operations

For CSG (union, subtract, intersect), use Three.js CSG via `three-bvh-csg` or a lightweight approach:

**Option A (recommended for now):** Use `three-bvh-csg` npm package — it's well-maintained and handles boolean mesh operations.

```bash
cd /K3D/Knowledge3D.local/envs/viewer-build
npm install three-bvh-csg --save
cp package.json "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/viewer/package.json"
```

**Option B (if keeping deps minimal):** Approximate CSG by simply merging geometries for UNION and skipping SUBTRACT. This produces visually acceptable results for the House objects (the server-baked GLTF is still available as fallback).

```typescript
CSG_UNION: (stack, ctx) => { ctx.mesh!.csgUnion(); },
CSG_SUBTRACT: (stack, ctx) => { ctx.mesh!.csgSubtract(); },
```

### C4. EXTRUDE and LATHE

These are used by the Bathtub (LATHE) and Prism (EXTRUDE):

```typescript
EXTRUDE: (stack, ctx) => {
  // pop depth → extrude current 2D path into 3D shape
  const depth = stack.pop() as number;
  const shape = pathToThreeShape(ctx.path!);  // Convert PathBuilder → THREE.Shape
  const geom = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false });
  ctx.mesh!.addPrimitive(geom);
},
LATHE: (stack, ctx) => {
  // pop segments → revolve current 2D path around Y axis
  const segments = stack.pop() as number;
  const points = pathToLathePoints(ctx.path!);  // Convert PathBuilder → THREE.Vector2[]
  const geom = new THREE.LatheGeometry(points, segments);
  ctx.mesh!.addPrimitive(geom);
},
```

**TIP:** `THREE.ExtrudeGeometry` and `THREE.LatheGeometry` already exist in Three.js. The only work is converting `PathBuilder` segments into `THREE.Shape` (for extrude) or `THREE.Vector2[]` (for lathe).

---

## Track D: Integration with House Viewer

### D1. Create `viewer/src/rpn/index.ts`

Module barrel that exports the complete RPN engine with all operation modules registered:

```typescript
import { RpnEngine } from './engine';
import { mat4Ops } from './mat4Ops';
import { pathOps } from './pathOps';
import { meshOps } from './meshOps';

export function createVisualRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.registerModule(mat4Ops);
  engine.registerModule(pathOps);
  engine.registerModule(meshOps);
  return engine;
}

export { RpnEngine, PathBuilder, MeshBuilder };
export { rpnToSVG } from './pathBuilder';
export { rpnToMesh } from './meshBuilder';
```

### D2. Live RPN preview in tablet

Enhance the existing `RpnApp` in `apps.ts` to support visual RPN:
- When user types a `visual_rpn` program, show a live preview
- Render 2D programs as SVG on the tablet canvas
- Render 3D programs as a small Three.js viewport or wireframe sketch

### D3. House object RPN inspection

When hovering over a House object (tooltip from H8), show the object's `visual_rpn` program. Clicking opens it in the enhanced RpnApp where the user can edit and see the result update live.

### D4. Regression test: server vs browser geometry

Add a validation that compares server-generated geometry (from GLTF) with browser-generated geometry (from RPN interpreter) for a subset of House objects. This ensures the Tier 2 projection produces equivalent results to Tier 1.

---

## Tips for Codex

**Tip 1 — Match Python argument order.** The Python mesh_opcodes use this stack convention:
- `GEN_CUBE`: `size GEN_CUBE` (1 arg)
- `GEN_CYLINDER`: `radius height segments closed GEN_CYLINDER` (4 args)
- `GEN_CONE`: `radius height segments GEN_CONE` (3 args)
- `GEN_TORUS`: `radius tube radialSegments tubularSegments GEN_TORUS` (4 args)
- `MAT4_SCALE`: `pop 3 values from stack (z, y, x)` then `x y z MAT4_SCALE`

Check `knowledge3d/cranium/ptx_runtime/mesh_opcodes.py` for exact argument order. The browser must match.

**Tip 2 — Three.js geometry is enough.** Don't write custom mesh generation. Three.js has `BoxGeometry`, `CylinderGeometry`, `ConeGeometry`, `TorusGeometry`, `SphereGeometry`, `PlaneGeometry`, `ExtrudeGeometry`, `LatheGeometry` — all parameterized. Map RPN ops directly to these.

**Tip 3 — CSG can be deferred.** If `three-bvh-csg` proves complex to integrate, start with a merge-only approach (UNION = merge buffers, SUBTRACT = skip). The GLTF fallback still shows the correct geometry. CSG is a Tier 2.5 enhancement.

**Tip 4 — Test with actual House visual_rpn programs.** The 54 House objects provide real test cases. Examples:
- Simple: `"1.0 GEN_CUBE 0.40 0.28 0.02 MAT4_SCALE MAT4_APPLY"` (Memory Tablet body)
- Compound: `"0.025 0.22 8 1 GEN_CYLINDER ... CSG_UNION"` (Hammer tool)
- Profile-based: `"0.0 0.0 MOVE 0.16 0.0 LINE 0.08 0.14 LINE CLOSE 0.16 EXTRUDE"` (Prism)
- Lathe: `"0.8 0.0 MOVE 0.92 0.16 LINE ... CLOSE 20 LATHE"` (Bathtub)

**Tip 5 — Build from SSD.** `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` — see `TEMP/CODEX_DIRECTIVE_VIEWER_BUILD_ENV_03.17.2026.md`.

**Tip 6 — Module structure.** Create a `viewer/src/rpn/` directory for the interpreter modules. Keep each concern in its own file: `engine.ts`, `mat4Ops.ts`, `pathOps.ts`, `meshOps.ts`, `pathBuilder.ts`, `meshBuilder.ts`, `index.ts`.

---

## Tests

### `viewer/tests/rpnEngine.test.ts`

```typescript
test('arithmetic operations', () => {
  const engine = createVisualRpnEngine();
  expect(engine.execute('3 4 ADD')).toEqual([7]);
  expect(engine.execute('10 3 SUB')).toEqual([7]);
});

test('MAT4_SCALE creates scale matrix', () => {
  const engine = createVisualRpnEngine();
  const result = engine.execute('2.0 3.0 4.0 MAT4_SCALE');
  // result[0] should be a Float32Array(16) with scale 2,3,4
  expect(result[0]).toBeInstanceOf(Float32Array);
});
```

### `viewer/tests/rpnPath.test.ts`

```typescript
test('path operations build SVG path', () => {
  const engine = createVisualRpnEngine();
  engine.execute('0.0 0.0 MOVE 1.0 0.0 LINE 0.5 1.0 LINE CLOSE');
  const svg = engine.context.path!.toSVGPath();
  expect(svg).toContain('M');
  expect(svg).toContain('L');
  expect(svg).toContain('Z');
});
```

### `viewer/tests/rpnMesh.test.ts`

```typescript
test('GEN_CUBE produces geometry with vertices', () => {
  const engine = createVisualRpnEngine();
  engine.execute('1.0 GEN_CUBE');
  const geom = engine.context.mesh!.toGeometry();
  expect(geom.attributes.position.count).toBeGreaterThan(0);
});

test('Memory Tablet visual_rpn produces geometry', () => {
  const engine = createVisualRpnEngine();
  engine.execute(
    '1.0 GEN_CUBE 0.40 0.28 0.02 MAT4_SCALE MAT4_APPLY ' +
    '1.0 GEN_CUBE 0.36 0.24 0.01 MAT4_SCALE MAT4_APPLY ' +
    '0.0 0.0 0.011 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT'
  );
  const geom = engine.context.mesh!.toGeometry();
  expect(geom.attributes.position.count).toBeGreaterThan(0);
});
```

### `tests/test_rpn_parity.py` (Python-side)

```python
def test_house_visual_rpn_programs_are_parseable():
    """All House visual_rpn programs should be valid for the browser interpreter."""
    from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS
    from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
    from knowledge3d.knowledgeverse.house_doors import HOUSE_DOORS
    # ... all collections
    known_ops = {
        'GEN_CUBE', 'GEN_CYLINDER', 'GEN_CONE', 'GEN_TORUS', 'GEN_UV_SPHERE',
        'GEN_PLANE', 'GEN_ICOSPHERE',
        'MAT4_SCALE', 'MAT4_TRANSLATE', 'MAT4_ROTATE_X', 'MAT4_ROTATE_Y', 'MAT4_ROTATE_Z',
        'MAT4_APPLY', 'MAT4_MUL',
        'CSG_UNION', 'CSG_SUBTRACT', 'CSG_INTERSECT',
        'MOVE', 'LINE', 'QUAD', 'CUBIC', 'ARC', 'CLOSE',
        'EXTRUDE', 'LATHE',
    }
    for star in ALL_HOUSE_STARS:
        if not star.visual_rpn:
            continue
        tokens = star.visual_rpn.split()
        for token in tokens:
            try:
                float(token)  # numeric literal
            except ValueError:
                assert token in known_ops, f"Unknown op '{token}' in {star.star_id}"
```

---

## Success Criteria

1. `RpnEngine` executes arithmetic + MAT4 operations correctly
2. `PathBuilder` produces valid SVG path strings from MOVE/LINE/QUAD/CLOSE programs
3. `MeshBuilder` produces Three.js geometry from GEN_CUBE/GEN_CYLINDER/etc. programs
4. At least 3 House visual_rpn programs execute successfully in the browser (e.g., Memory Tablet, a door frame, a tool)
5. Enhanced RpnApp shows live visual preview for RPN programs
6. All existing tests pass, TypeScript clean
7. Build succeeds via `build.sh`

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/rpn/engine.ts` | **NEW** — Core RPN stack machine |
| `viewer/src/rpn/mat4Ops.ts` | **NEW** — Matrix4 operation handlers |
| `viewer/src/rpn/pathBuilder.ts` | **NEW** — 2D path builder with SVG/Canvas output |
| `viewer/src/rpn/pathOps.ts` | **NEW** — Path RPN operation handlers |
| `viewer/src/rpn/meshBuilder.ts` | **NEW** — 3D mesh builder with Three.js output |
| `viewer/src/rpn/meshOps.ts` | **NEW** — Mesh generation RPN operation handlers |
| `viewer/src/rpn/index.ts` | **NEW** — Module barrel + factory function |
| `viewer/src/apps.ts` | Update RpnApp with visual preview |
| `viewer/src/main.ts` | Wire RPN inspection on House object click |
| `viewer/tests/rpnEngine.test.ts` | **NEW** |
| `viewer/tests/rpnPath.test.ts` | **NEW** |
| `viewer/tests/rpnMesh.test.ts` | **NEW** |
| `tests/test_rpn_parity.py` | **NEW** — Validate all House RPN programs are browser-compatible |
| `viewer/package.json` | Add `three-bvh-csg` if using CSG (optional) |

---

## Architectural Note

This is the **Dual Client Contract** expressed as code. The same `visual_rpn` programs that execute on the GPU (sovereign Cranium path, PTX kernels) now also execute in the browser (Tier 2 projection, TypeScript + Three.js). Same program, two execution environments, equivalent output.

The `RpnEngine` is intentionally modular — new operation modules can be registered without changing the core. This is the extension point for Christoph's JS component architecture: he can add rendering backends (SVG, WebGL, WebGPU) as modules that plug into the same engine.

**Tier progression:**
- Tier 1 (done): Server → GLTF → Three.js GLTFLoader (pre-baked geometry)
- **Tier 2 (this phase):** Server → RPN string → Browser RPN interpreter → Three.js/SVG (live generation)
- Tier 3 (future): Server → RPN string → Browser WebGPU kernel → sovereign rendering
