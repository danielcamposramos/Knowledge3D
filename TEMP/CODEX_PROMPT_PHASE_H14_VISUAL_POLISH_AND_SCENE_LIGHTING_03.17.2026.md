# Phase H14: Visual Polish — Materials, Lighting, and Scene Atmosphere

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H13 (projection surfaces) COMPLETE
**Sovereignty:** I/O path (viewer rendering, flexible).
**Build:** Viewer only: `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh`

---

## Context

The House has 6 rooms, 60+ nodes, projection surfaces, navigation, content rendering, and DOM projection. But visually it's **wireframe NormalMaterial on every mesh**. The viewer works but doesn't look like the original vision — warm wood floors, cool cyan labs, glowing holographics, atmospheric lighting.

This phase adds the **visual layer** that makes the House demo-ready: per-domain material palettes, scene lighting, ambient occlusion hints, and atmosphere. No geometry changes — just how existing geometry is perceived.

**Goal:** A first-time viewer opens the browser and sees something that evokes the original cross-section render: warm Living Room, scholarly Library, lush Garden, industrial Workshop, art-lit Gallery, moody Bathtub Observatory.

---

## Deliverables

### Track A: Material System (per-domain palettes)
### Track B: Scene Lighting
### Track C: Room Atmosphere + Skybox
### Track D: Integration

---

## Track A: Material System

### A1. Create `viewer/src/materials/palette.ts`

A material palette maps `meaningClass` + `domain` combinations to Three.js materials.

```typescript
import * as THREE from 'three';

export interface MaterialPalette {
  room: THREE.Material;
  furniture: THREE.Material;
  door: THREE.Material;
  book: THREE.Material;
  tool: THREE.Material;
  display: THREE.Material;
  instrument: THREE.Material;
  tree: THREE.Material;
  tablet: THREE.Material;
  default: THREE.Material;
}

/** Base shared settings for all K3D materials */
const BASE_PARAMS = { flatShading: true, side: THREE.DoubleSide };

export function createDomainPalette(domain: string): MaterialPalette {
  // Each room domain has a distinct warm/cool feel
  const palettes: Record<string, () => MaterialPalette> = {
    'House/LivingRoom': livingRoomPalette,
    'House/Library': libraryPalette,
    'House/Garden': gardenPalette,
    'House/Workshop': workshopPalette,
    'House/Gallery': galleryPalette,
    'House/Bathtub': bathtubPalette,
  };
  const factory = palettes[domain] || defaultPalette;
  return factory();
}
```

Domain-specific palettes (example):

| Domain | Room Color | Furniture Color | Mood |
|--------|-----------|----------------|------|
| Living Room | Warm cream `#f5e6d3` | Soft grey `#b8b0a8` | Welcoming, collaborative |
| Library | Rich walnut `#5c3d2e` | Dark wood `#8b6914` | Scholarly, warm |
| Garden | Earthy green `#4a7c59` | Natural brown `#8b7355` | Organic, lush |
| Workshop | Industrial grey `#6b6b6b` | Metallic `#a0a0a0` | Functional, precise |
| Gallery | Pure white `#f0f0f0` | Accent red `#c04040` | Clean, artistic |
| Bathtub | Deep indigo `#1a1a3e` | Dark ceramic `#2a2a4a` | Introspective, moody |

Materials use `MeshStandardMaterial` (PBR) with `flatShading: true` for the procedural geometry aesthetic. Roughness varies by meaning_class: rooms are matte (0.9), furniture is semi-gloss (0.5), tools are metallic (0.3, metalness 0.6).

### A2. Create `viewer/src/materials/index.ts`

Barrel export.

### A3. Apply materials on House load

After `loadHouseScene()`, traverse all nodes and replace materials based on their `meaningClass` and `domain`:

```typescript
function applyHouseMaterials(scene: LoadedHouseScene): void {
  const domainPalettes = new Map<string, MaterialPalette>();
  scene.nodesByStarId.forEach((node) => {
    const domain = node.houseRoom || node.domain;
    if (!domainPalettes.has(domain)) {
      domainPalettes.set(domain, createDomainPalette(domain));
    }
    const palette = domainPalettes.get(domain)!;
    const material = palette[node.meaningClass as keyof MaterialPalette] || palette.default;
    node.object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = material;
      }
    });
  });
}
```

---

## Track B: Scene Lighting

### B1. Create `viewer/src/materials/lighting.ts`

Lighting setup for the House scene:

```typescript
export function createHouseLighting(scene: THREE.Scene): THREE.Group {
  const lights = new THREE.Group();
  lights.name = 'k3d-house-lights';

  // Soft ambient fill — prevents pitch-black areas
  const ambient = new THREE.AmbientLight(0xffffff, 0.3);
  lights.add(ambient);

  // Warm hemisphere — sky warm, ground cool
  const hemisphere = new THREE.HemisphereLight(0xffeedd, 0x223344, 0.4);
  lights.add(hemisphere);

  // Primary directional — sun-like, soft shadows
  const sun = new THREE.DirectionalLight(0xfff8e8, 0.6);
  sun.position.set(20, 40, 10);
  sun.castShadow = false; // enable later if perf allows
  lights.add(sun);

  // Cool fill from opposite side — prevents one-sided flatness
  const fill = new THREE.DirectionalLight(0xd0e0ff, 0.2);
  fill.position.set(-15, 20, -10);
  lights.add(fill);

  scene.add(lights);
  return lights;
}
```

Lighting is deliberately simple — no shadow maps (perf), no point lights per room (complexity). Just 4 lights that make PBR materials look good globally.

### B2. Remove NormalMaterial default

The current `meshBuilder.toMesh()` defaults to `MeshNormalMaterial({ wireframe: true })`. After materials are applied, the scene should NOT have any remaining NormalMaterial meshes. Any missed nodes get the `default` palette material.

---

## Track C: Room Atmosphere + Skybox

### C1. Create `viewer/src/materials/atmosphere.ts`

A simple gradient skybox using `THREE.Color` on the scene background:

```typescript
export function setHouseAtmosphere(scene: THREE.Scene): void {
  // Soft gradient background — warm top, cool bottom
  scene.background = new THREE.Color(0x1a1a2e);

  // Fog for depth cues — rooms far away fade gently
  scene.fog = new THREE.Fog(0x1a1a2e, 60, 150);
}
```

Fog range `60-150` corresponds to ~1 room visible clearly, adjacent rooms slightly faded, far rooms ghostly. This naturally focuses attention on the current room.

### C2. Ground plane (optional)

A large subtle ground plane below the rooms to anchor the scene spatially. Thin, dark, non-distracting:

```typescript
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(300, 300),
  new THREE.MeshStandardMaterial({
    color: 0x111122,
    roughness: 1.0,
    metalness: 0.0,
  }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.5;
ground.receiveShadow = false;
scene.add(ground);
```

---

## Track D: Integration

### D1. Wire material application in main.ts

After `loadHouseScene()` and before the first render:

```typescript
import { applyHouseMaterials } from './materials';
import { createHouseLighting } from './materials/lighting';
import { setHouseAtmosphere } from './materials/atmosphere';

// After loadHouseScene:
applyHouseMaterials(loadedHouseScene);
createHouseLighting(scene);
setHouseAtmosphere(scene);
```

### D2. Ensure projection materials are preserved

The HoloDesk projector applies its own cyan wireframe material. The Galaxy Pod applies its own PointsMaterial. These must NOT be overwritten by the palette system. Solution: `applyHouseMaterials()` only affects nodes in the `nodesByStarId` map traversal, NOT child groups added dynamically by projectors.

### D3. Room dimming integration

The existing opacity dimming system (0.3 for off-room objects) already works via material traversal. Verify it still works with `MeshStandardMaterial` — the `transparent` and `opacity` fields are the same API. No changes expected.

---

## Tips for Codex

**Tip 1 — MeshStandardMaterial, not ShaderMaterial.** Use Three.js PBR materials. They respond to lighting naturally. `flatShading: true` gives the procedural geometry a clean, low-poly aesthetic without smoothing artifacts.

**Tip 2 — DoubleSide for room shells.** Room geometry is CSG-subtracted boxes. The viewer can see both inner and outer faces. Use `side: THREE.DoubleSide` on room materials so the interior walls are visible.

**Tip 3 — Material instances, not clones.** Create ONE material per palette slot. Share it across all nodes of that type in that domain. Don't clone per node — that wastes GPU memory and breaks batching.

**Tip 4 — Don't touch projection materials.** The HoloDesk and Galaxy Pod apply their own materials during projection. The palette system only applies to House nodes on initial load. Projector content is ephemeral and managed by the projector classes.

**Tip 5 — Fog matches background.** Set `scene.fog` color to the same value as `scene.background`. This creates a seamless fade to "infinity" instead of an abrupt color boundary.

**Tip 6 — Roughness/metalness by meaning_class.** Rooms are matte walls (roughness 0.9). Furniture is semi-gloss (0.5-0.7). Tools are metallic (roughness 0.3, metalness 0.6). Books are matte (0.8). This gives visual variety from the same lighting.

**Tip 7 — Build via SSD.** `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` only.

---

## Tests

### Viewer: `viewer/tests/materialPalette.test.ts`

```typescript
describe('material palette', () => {
  it('creates distinct palettes for each domain', () => {
    // Verify Living Room palette differs from Workshop palette
  });

  it('returns MeshStandardMaterial for all slots', () => {
    // Verify all palette entries are MeshStandardMaterial
  });

  it('uses shared material instances (not clones)', () => {
    // Create palette twice for same domain, verify same object references
  });
});
```

### Viewer: `viewer/tests/lighting.test.ts`

```typescript
describe('house lighting', () => {
  it('creates 4 lights in a group', () => {
    // Verify AmbientLight + HemisphereLight + 2 DirectionalLight
  });

  it('adds lights to the scene', () => {
    // Verify scene.children includes the lights group
  });
});
```

### Non-regression

All 17 existing viewer suites + all Python tests must continue to pass. Projection tests must still work (materials not overwritten).

---

## Success Criteria

1. Each room domain has a visually distinct material palette (warm Living Room, scholarly Library, green Garden, grey Workshop, white Gallery, dark Bathtub)
2. Scene has hemisphere + directional lighting that makes PBR materials visible
3. Background + fog create depth atmosphere
4. No remaining `MeshNormalMaterial` on any House node
5. Projection surfaces (HoloDesk, Galaxy Pod) retain their own materials
6. Room dimming (30% opacity) still works with new materials
7. All existing tests pass, new material tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/materials/palette.ts` | **NEW** — Domain-specific material palettes |
| `viewer/src/materials/lighting.ts` | **NEW** — Scene lighting setup |
| `viewer/src/materials/atmosphere.ts` | **NEW** — Background + fog |
| `viewer/src/materials/index.ts` | **NEW** — Barrel export |
| `viewer/src/main.ts` | Apply materials, lighting, atmosphere on House load |
| `viewer/tests/materialPalette.test.ts` | **NEW** |
| `viewer/tests/lighting.test.ts` | **NEW** |

---

## Architectural Note

This phase transitions the viewer from "functional prototype" to "visual experience." The House was always meant to be a place you want to inhabit — warm, atmospheric, distinct per domain. The material system is composable: as new rooms are added, they just need a palette factory function. The lighting is global and cheap — no per-room light management needed.

This is the last piece before the MVP is truly demo-ready: navigate between visually distinct rooms, click objects, see content on the tablet, watch holographic projections, explore the Galaxy Pod — all in an atmospheric 3D environment running in any browser.
