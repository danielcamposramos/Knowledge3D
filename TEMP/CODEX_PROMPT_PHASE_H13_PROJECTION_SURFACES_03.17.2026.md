# Phase H13: Projection Surfaces — HoloDesk and Galaxy Pod

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H11b (Living Room + HoloDesk) COMPLETE, Phase H12 (static content) COMPLETE
**Sovereignty:** I/O path (viewer rendering, flexible).
**Build:** Viewer only: `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh`

---

## Context

The original K3D house design features two key projection surfaces:

1. **HoloDesk** (Living Room) — A flat table projecting holographic 3D content upward. In the original design: "floating prominently in the center of the room are large, translucent holographic screens displaying complex schematics." This is the augmented collaboration surface — shared 3D model viewing, AR/VR interaction.

2. **Galaxy Pod** (Bathtub Observatory) — A spherical projection surface rendering the Galaxy Universe as a stellarium-like dome. In the original design: "a massive, dark spherical pod... Inside, a vibrant, high-resolution projection of a spiral galaxy glows in shades of pink, purple, and gold." This is the introspective surface — the avatar's internal brain made visible.

Both are **3D projection targets within the House**: objects that don't just exist as furniture but actively render dynamic content. They bridge the House (external reality) with the Galaxy (internal brain) — making the invisible visible.

**Current state:** `furniture_holodesk` exists with behavior `HOLODESK ACTIVATE PROJECT_3D` which maps to `browse_galaxy` (opens Galaxy app). `furniture_bathtub` exists with behavior `PORTAL REST REFLECT`. But neither actually renders projected 3D content in the scene.

**Goal:** Make these objects into live projection surfaces: the HoloDesk shows floating 3D content above its table surface, the Galaxy Pod shows a spherical star field inside the bathtub's lathe geometry.

---

## Deliverables

### Track A: Projection Surface Abstraction
### Track B: HoloDesk Projector (Flat / Upward)
### Track C: Galaxy Pod Projector (Spherical / Inward)
### Track D: Integration + Visual Polish

---

## Track A: Projection Surface Abstraction

### A1. Create `viewer/src/projection/surface.ts`

A `ProjectionSurface` is a Three.js group attached to a House node that renders dynamic content at a designated position and orientation.

```typescript
import * as THREE from 'three';

export interface ProjectionConfig {
  /** Anchor position relative to the parent node (local coords) */
  anchor: THREE.Vector3;
  /** Maximum bounding size for projected content */
  bounds: THREE.Vector3;
  /** Visual style: 'holographic' (flat upward) or 'stellarium' (spherical inward) */
  mode: 'holographic' | 'stellarium';
}

export class ProjectionSurface {
  readonly group: THREE.Group;
  readonly config: ProjectionConfig;
  private content: THREE.Object3D | null = null;
  private _visible: boolean = false;

  constructor(config: ProjectionConfig) { ... }

  /** Replace current projected content */
  setContent(object: THREE.Object3D): void { ... }

  /** Clear projected content */
  clear(): void { ... }

  /** Toggle projection visibility (with fade) */
  show(): void { ... }
  hide(): void { ... }

  /** Per-frame update (rotation, glow pulse, etc.) */
  update(delta: number): void { ... }
}
```

Key design decisions:
- The `group` is a child of the House node's `Object3D`. Positioned at `anchor` relative to the furniture.
- Content is auto-scaled to fit within `bounds`.
- `update(delta)` enables slow rotation for holographic mode and starfield animation for stellarium mode.

### A2. Create `viewer/src/projection/index.ts`

Barrel export for the projection module.

---

## Track B: HoloDesk Projector

### B1. Create `viewer/src/projection/holodeskProjector.ts`

The HoloDesk projects content **upward** from its table surface. Content floats above the desk with a holographic visual treatment.

```typescript
import * as THREE from 'three';
import { ProjectionSurface } from './surface';
import type { HouseNode } from '../loadHouseScene';

export class HolodeskProjector {
  readonly surface: ProjectionSurface;
  private rotationSpeed: number = 0.15; // radians/sec

  constructor(holodeskNode: HouseNode) {
    this.surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 0.6, 0),  // above table top (0.45 + margin)
      bounds: new THREE.Vector3(1.4, 1.0, 0.8), // projection volume
      mode: 'holographic',
    });
    holodeskNode.object.add(this.surface.group);
  }

  /** Project a Three.js mesh (from RPN visual_rpn execution) */
  projectMesh(mesh: THREE.Object3D): void {
    // Apply holographic material: wireframe, semi-transparent, emissive cyan
    mesh.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshBasicMaterial({
          color: 0x00ddff,
          wireframe: true,
          transparent: true,
          opacity: 0.7,
        });
      }
    });
    this.surface.setContent(mesh);
    this.surface.show();
  }

  /** Project content from a HouseNode's visual_rpn */
  projectNodeVisual(node: HouseNode, rpnToMesh: (program: string) => THREE.Mesh): void {
    if (!node.visualRpn) return;
    const mesh = rpnToMesh(node.visualRpn);
    this.projectMesh(mesh);
  }

  /** Clear the projection */
  dismiss(): void {
    this.surface.clear();
    this.surface.hide();
  }

  update(delta: number): void {
    this.surface.update(delta);
    // Slow rotation for holographic effect
    if (this.surface.group.children.length > 0) {
      this.surface.group.rotation.y += this.rotationSpeed * delta;
    }
  }
}
```

Visual treatment for holographic mode:
- **Wireframe** mesh with semi-transparent cyan material (`0x00ddff`)
- Slow Y-axis rotation (0.15 rad/s — one full rotation every ~42 seconds)
- Content auto-scaled to fit within the projection volume
- Emissive glow effect (no scene lighting dependency)

### B2. HoloDesk activation wiring

When `HOLODESK` behavior fires in the activator:
1. If no content is projected → show the HoloDesk's own visual (a default rotating mesh, e.g., the Knowledge Tree or the House itself)
2. If a specific node was recently inspected → project that node's visual_rpn mesh as a hologram
3. Toggle behavior: clicking HoloDesk again dismisses the projection

Update `activator.ts` to handle this:

```typescript
// In HouseActivator, add holodeskProjector reference
private holodesk: HolodeskProjector | null = null;

// When browse_galaxy fires from HoloDesk click:
private handleBrowseGalaxy(): void {
  if (this.holodesk) {
    // Toggle: if visible, dismiss; if hidden, project default
    if (this.holodesk.surface.visible) {
      this.holodesk.dismiss();
    } else {
      // Project the knowledge tree as default hologram
      const tree = this.scene.nodesByStarId.get('furniture_knowledge_tree');
      if (tree) this.holodesk.projectNodeVisual(tree, rpnToMesh);
    }
  }
  this.tablet.showFocus();
  this.tablet.dispatch({ type: 'open_app', payload: { id: 'galaxy' } });
}
```

---

## Track C: Galaxy Pod Projector (Spherical)

### C1. Create `viewer/src/projection/galaxyPodProjector.ts`

The Bathtub is the introspection room. Its Galaxy Pod projects a spherical star field **inward** — the user looks up from inside the bathtub vessel and sees the Galaxy Universe rendered as a stellarium dome.

```typescript
import * as THREE from 'three';
import { ProjectionSurface } from './surface';
import type { HouseNode } from '../loadHouseScene';

export class GalaxyPodProjector {
  readonly surface: ProjectionSurface;
  private stars: THREE.Points | null = null;
  private rotationSpeed: number = 0.02; // very slow celestial rotation

  constructor(bathtubNode: HouseNode) {
    this.surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 1.5, 0), // above the bathtub rim
      bounds: new THREE.Vector3(3, 3, 3),     // sphere radius ~1.5
      mode: 'stellarium',
    });
    bathtubNode.object.add(this.surface.group);
  }

  /** Generate a point cloud representing Galaxy entries */
  projectGalaxy(entries: Array<{ star_id: string; domain: string }>): void {
    const count = Math.max(entries.length, 200); // pad with random stars if few entries
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    // Domain → color mapping
    const domainColors: Record<string, THREE.Color> = {
      'Mathematics': new THREE.Color(0x4488ff),
      'Language': new THREE.Color(0x44ff88),
      'Physics': new THREE.Color(0xff8844),
      'Biology': new THREE.Color(0x88ff44),
      'Tools': new THREE.Color(0xff44ff),
      'default': new THREE.Color(0xaaaaff),
    };

    for (let i = 0; i < count; i++) {
      // Fibonacci sphere distribution
      const phi = Math.acos(1 - 2 * (i + 0.5) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = 1.4 + Math.random() * 0.2; // slight depth variation
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);

      // Color by domain
      const entry = entries[i % entries.length];
      const domainKey = entry?.domain?.split('/').pop() || 'default';
      const color = domainColors[domainKey] || domainColors['default'];
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    this.stars = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      sizeAttenuation: true,
    }));

    this.surface.setContent(this.stars);
    this.surface.show();
  }

  /** Project using seed concepts from house-content.json */
  projectFromContent(concepts: Record<string, { star_id: string; domain: string }>): void {
    this.projectGalaxy(Object.values(concepts));
  }

  dismiss(): void {
    this.surface.clear();
    this.surface.hide();
    this.stars = null;
  }

  update(delta: number): void {
    this.surface.update(delta);
    if (this.stars) {
      this.stars.rotation.y += this.rotationSpeed * delta;
      this.stars.rotation.x += this.rotationSpeed * 0.3 * delta;
    }
  }
}
```

Visual treatment for stellarium mode:
- **Point cloud** with Fibonacci sphere distribution
- Domain-colored stars (Mathematics=blue, Language=green, Physics=orange, etc.)
- Very slow dual-axis rotation (celestial drift effect)
- Minimum 200 points even if fewer Galaxy entries exist (padding with random stars)
- Positioned above the bathtub rim, inside the room's shell

### C2. Galaxy Pod activation wiring

When the user enters the Bathtub room (room_enter), or clicks the bathtub furniture:
1. Galaxy Pod auto-activates — the stellarium appears above the bathtub
2. Populated from `house-content.json` concepts + book entries
3. Clicking the bathtub again toggles the stellarium off/on

Update the room context listener:

```typescript
// When entering Bathtub room, auto-activate Galaxy Pod
roomContext.onEnter((room) => {
  if (room.starId === 'room_bathtub' && galaxyPod) {
    const content = getLoadedContent();
    if (content) galaxyPod.projectFromContent(content.concepts);
  }
});
```

---

## Track D: Integration + Visual Polish

### D1. Create projection surfaces on House load

In `main.ts`, after loading the House scene, create projectors for known furniture:

```typescript
import { HolodeskProjector } from './projection/holodeskProjector';
import { GalaxyPodProjector } from './projection/galaxyPodProjector';

// After loadHouseScene:
const holodeskNode = scene.nodesByStarId.get('furniture_holodesk');
const holodesk = holodeskNode ? new HolodeskProjector(holodeskNode) : null;

const bathtubNode = scene.nodesByStarId.get('furniture_bathtub');
const galaxyPod = bathtubNode ? new GalaxyPodProjector(bathtubNode) : null;

// In render loop:
function animate(delta: number) {
  holodesk?.update(delta);
  galaxyPod?.update(delta);
}
```

### D2. Pass projectors to HouseActivator

Extend `HouseActivator` constructor to accept optional projectors:

```typescript
constructor(
  scene: LoadedHouseScene,
  roomCamera: RoomCamera,
  tablet: Tablet3D,
  roomContext: RoomContext,
  options?: {
    holodesk?: HolodeskProjector;
    galaxyPod?: GalaxyPodProjector;
  },
)
```

### D3. HoloDesk glow rim (optional polish)

Add a subtle emissive rim to the HoloDesk's recessed frame when projection is active. This is purely visual — a thin line of light around the projection surface edge. Can be a simple `THREE.LineLoop` with emissive material at the desk's top edge.

### D4. Dimming integration

When projections are active, the room's non-projection objects could dim slightly (0.6 opacity vs the current 0.3 for off-room objects). This focuses attention on the projected content. Wire through the existing `RoomContext` opacity system.

---

## Tips for Codex

**Tip 1 — Projection = child of furniture Object3D.** The projector's `group` is added as a child of the furniture node's Three.js object. This means it inherits the furniture's world transform automatically. Position the anchor in LOCAL coordinates relative to the furniture.

**Tip 2 — Auto-scale content to bounds.** When `setContent()` receives a mesh, compute its bounding box, then scale it to fit within `config.bounds`. Center it at `config.anchor`. This ensures any visual_rpn mesh fits the projection volume regardless of its original size.

**Tip 3 — Holographic material = simple and cheap.** Use `MeshBasicMaterial` with wireframe + transparency. Don't use `ShaderMaterial` or custom shaders. The holographic look comes from wireframe + cyan color + slow rotation + transparency. Keep it lightweight.

**Tip 4 — Fibonacci sphere for star distribution.** The formula `phi = acos(1 - 2*(i+0.5)/N)`, `theta = pi*(1+sqrt(5))*i` distributes N points nearly-uniformly on a sphere. Much better than random distribution for a stellarium look.

**Tip 5 — Galaxy Pod uses content data.** Call `getLoadedContent()` from `contentLoader.ts` to get seed concepts. Each concept has a `domain` field — use it for color coding. If content isn't loaded, fall back to a generic 200-star field with white/blue colors.

**Tip 6 — Toggle pattern.** Both projectors use a toggle: click to show, click again to dismiss. Track visibility state in the `ProjectionSurface`. Don't create new geometry on every click — cache and show/hide.

**Tip 7 — Delta-based animation.** The `update(delta)` methods receive elapsed seconds since last frame. Use `delta` for rotation speed, not frame count. This ensures consistent animation regardless of frame rate.

**Tip 8 — Build via SSD.** `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` only. No npm/jest/vite from viewer/ directly.

---

## Tests

### Viewer: `viewer/tests/projectionSurface.test.ts`

```typescript
describe('ProjectionSurface', () => {
  it('adds content as child of group', () => {
    // Create surface, set content, verify group.children.length
  });

  it('clears content on dismiss', () => {
    // Create surface, set content, clear, verify group.children.length === 0
  });

  it('auto-scales content to bounds', () => {
    // Create large mesh, set content, verify it fits within bounds
  });
});
```

### Viewer: `viewer/tests/holodeskProjector.test.ts`

```typescript
describe('HolodeskProjector', () => {
  it('creates holographic wireframe material', () => {
    // Project a mesh, verify material is wireframe + transparent + cyan
  });

  it('toggles visibility on repeated activation', () => {
    // Show, verify visible; show again, verify hidden
  });

  it('applies slow rotation on update', () => {
    // Call update(1.0), verify rotation.y changed
  });
});
```

### Viewer: `viewer/tests/galaxyPodProjector.test.ts`

```typescript
describe('GalaxyPodProjector', () => {
  it('creates point cloud from concept entries', () => {
    // projectGalaxy with mock entries, verify Points object exists
  });

  it('colors stars by domain', () => {
    // Verify color buffer has domain-appropriate values
  });

  it('pads to minimum 200 points', () => {
    // projectGalaxy with 5 entries, verify 200 points
  });
});
```

### Non-regression

All 14 existing viewer suites + all Python tests must continue to pass.

---

## Success Criteria

1. `HolodeskProjector` renders a holographic wireframe mesh floating above the HoloDesk table
2. Clicking the HoloDesk toggles the projection on/off
3. `GalaxyPodProjector` renders a colored point-cloud stellarium sphere above the bathtub
4. Entering the Bathtub room auto-activates the Galaxy Pod with content-sourced entries
5. Both projectors animate (slow rotation) via `update(delta)`
6. Content auto-scales to fit projection bounds
7. All existing tests pass, new projection tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/projection/surface.ts` | **NEW** — ProjectionSurface abstraction |
| `viewer/src/projection/holodeskProjector.ts` | **NEW** — HoloDesk holographic projector |
| `viewer/src/projection/galaxyPodProjector.ts` | **NEW** — Galaxy Pod stellarium projector |
| `viewer/src/projection/index.ts` | **NEW** — Barrel export |
| `viewer/src/behavior/activator.ts` | Wire projectors into activation |
| `viewer/src/main.ts` | Create projectors, add to render loop |
| `viewer/tests/projectionSurface.test.ts` | **NEW** |
| `viewer/tests/holodeskProjector.test.ts` | **NEW** |
| `viewer/tests/galaxyPodProjector.test.ts` | **NEW** |

---

## Architectural Note

These projection surfaces embody the core K3D vision: the House is not a static museum but a **living space** where knowledge is literally visible. The HoloDesk makes the Galaxy's content tangible for collaboration — "share your 3D models." The Galaxy Pod makes the avatar's internal brain visible — the stellarium IS the Galaxy Universe rendered as a spatial experience.

The pattern is composable:
- **Any furniture can become a projection surface** — just attach a `ProjectionSurface` to its Object3D
- **Any RPN program can be projected** — visual_rpn → rpnToMesh → projectMesh
- **Any data can become a stellarium** — entries with positions/colors → point cloud

Future surfaces: the Gallery displays could project animated drawings, the Workshop tools could project construction previews, the Observatory telescope could project a zoomed Galaxy neighborhood. The abstraction supports all of these.

This is the HoloDesk from the original vision — finally alive.
