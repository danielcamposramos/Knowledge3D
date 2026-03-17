# Phase H15: Navigation UX — Keyboard Controls, Minimap, and Room Labels

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H14 (visual polish) COMPLETE
**Sovereignty:** I/O path (viewer rendering, flexible).
**Build:** Viewer only: `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh`

---

## Context

The House is visually atmospheric and interactive — click objects, traverse doors, see content. But navigation is limited to:
1. Click a door to move to the next room
2. OrbitControls for camera (mouse drag)

For a demo-ready MVP, users need:
- **Keyboard shortcuts** to move between rooms without clicking doors
- **A minimap** showing the room layout and current position
- **Room labels** floating above each room to orient the visitor
- **A welcome overlay** explaining basic controls on first load

**Goal:** A first-time user opens the viewer and immediately knows how to navigate. Arrow keys move between rooms. A minimap shows where they are. Labels identify each room. The tablet shows room context on every transition.

---

## Deliverables

### Track A: Keyboard Navigation
### Track B: Room Minimap (2D HTML overlay)
### Track C: Room Labels (3D floating text)
### Track D: Welcome Overlay

---

## Track A: Keyboard Navigation

### A1. Create `viewer/src/navigation/keyboardNav.ts`

Keyboard-driven room traversal using the nav graph.

```typescript
import type { LoadedHouseScene, HouseNode } from '../loadHouseScene';

export interface KeyboardNavCallbacks {
  onRoomChange: (room: HouseNode) => void;
}

export class KeyboardNav {
  private scene: LoadedHouseScene;
  private callbacks: KeyboardNavCallbacks;
  private handler: (event: KeyboardEvent) => void;

  constructor(scene: LoadedHouseScene, callbacks: KeyboardNavCallbacks) {
    this.scene = scene;
    this.callbacks = callbacks;
    this.handler = this.onKeyDown.bind(this);
  }

  attach(): void {
    window.addEventListener('keydown', this.handler);
  }

  detach(): void {
    window.removeEventListener('keydown', this.handler);
  }

  private onKeyDown(event: KeyboardEvent): void {
    // Ignore if typing in an input field
    const tag = (event.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    switch (event.key) {
      case 'ArrowRight':
      case 'd':
        this.moveToNeighbor(1);  // next neighbor
        event.preventDefault();
        break;
      case 'ArrowLeft':
      case 'a':
        this.moveToNeighbor(-1); // previous neighbor
        event.preventDefault();
        break;
      case '1': case '2': case '3': case '4': case '5': case '6':
        this.moveToRoomIndex(parseInt(event.key, 10) - 1);
        event.preventDefault();
        break;
      case 'h':
        this.moveToRoom('room_living'); // home
        event.preventDefault();
        break;
    }
  }

  private moveToNeighbor(direction: number): void {
    const currentId = this.scene.currentRoom;
    const edges = this.scene.navGraph.edges.filter(e => e.from === currentId);
    if (!edges.length) return;
    // Sort edges by target room position for consistent left/right ordering
    const sorted = edges
      .map(e => this.scene.nodesByStarId.get(e.to))
      .filter((n): n is HouseNode => !!n)
      .sort((a, b) => a.housePosition[0] - b.housePosition[0]);
    const currentNode = this.scene.nodesByStarId.get(currentId);
    if (!currentNode) return;
    const currentX = currentNode.housePosition[0];
    const target = direction > 0
      ? sorted.find(n => n.housePosition[0] > currentX)
      : sorted.reverse().find(n => n.housePosition[0] < currentX);
    if (target) this.callbacks.onRoomChange(target);
  }

  private moveToRoomIndex(index: number): void {
    const rooms = this.scene.rooms.sort((a, b) => a.housePosition[0] - b.housePosition[0]);
    const room = rooms[index];
    if (room) this.callbacks.onRoomChange(room);
  }

  private moveToRoom(starId: string): void {
    const room = this.scene.nodesByStarId.get(starId);
    if (room && room.meaningClass === 'room') this.callbacks.onRoomChange(room);
  }
}
```

Key bindings:
| Key | Action |
|-----|--------|
| `ArrowRight` / `d` | Move to next room (by X position) |
| `ArrowLeft` / `a` | Move to previous room (by X position) |
| `1`-`6` | Jump to room by index (left-to-right order) |
| `h` | Go home (Living Room) |

### A2. Create `viewer/src/navigation/index.ts`

Barrel export.

---

## Track B: Room Minimap (2D HTML Overlay)

### B1. Create `viewer/src/navigation/minimap.ts`

A small HTML-based minimap in the bottom-left corner showing room positions and the current room indicator.

```typescript
import type { LoadedHouseScene, HouseNode } from '../loadHouseScene';

export class Minimap {
  readonly element: HTMLDivElement;
  private dots: Map<string, HTMLDivElement> = new Map();
  private currentDot: HTMLDivElement | null = null;

  constructor(scene: LoadedHouseScene) {
    this.element = document.createElement('div');
    this.element.className = 'k3d-minimap';
    // Position: bottom-left, small, semi-transparent
    Object.assign(this.element.style, {
      position: 'fixed',
      bottom: '16px',
      left: '16px',
      width: '200px',
      height: '40px',
      background: 'rgba(0,0,0,0.5)',
      borderRadius: '8px',
      padding: '6px 10px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      zIndex: '100',
      pointerEvents: 'auto',
    });

    // Create dots for each room, ordered by X position
    const rooms = scene.rooms.sort((a, b) => a.housePosition[0] - b.housePosition[0]);
    for (const room of rooms) {
      const dot = document.createElement('div');
      dot.title = room.surfaceForms.en?.word_ref || room.starId;
      Object.assign(dot.style, {
        width: '12px',
        height: '12px',
        borderRadius: '50%',
        background: '#666',
        cursor: 'pointer',
        transition: 'all 0.3s',
      });
      dot.dataset.starId = room.starId;
      this.element.appendChild(dot);
      this.dots.set(room.starId, dot);
    }

    document.body.appendChild(this.element);
  }

  setCurrentRoom(starId: string): void {
    this.dots.forEach((dot, id) => {
      const active = id === starId;
      dot.style.background = active ? '#00ddff' : '#666';
      dot.style.transform = active ? 'scale(1.4)' : 'scale(1)';
      dot.style.boxShadow = active ? '0 0 8px #00ddff' : 'none';
    });
  }

  onClick(callback: (starId: string) => void): void {
    this.dots.forEach((dot, starId) => {
      dot.addEventListener('click', () => callback(starId));
    });
  }

  destroy(): void {
    this.element.remove();
  }
}
```

Visual: a thin horizontal bar at bottom-left with one dot per room. Current room glows cyan. Click a dot to jump there.

---

## Track C: Room Labels (3D Floating Text)

### C1. Create `viewer/src/navigation/roomLabels.ts`

CSS2D labels floating above each room, using Three.js CSS2DRenderer if available, or simple HTML overlays projected from 3D positions.

Use a simple approach: HTML div elements positioned via Three.js `Vector3.project()` in the render loop.

```typescript
import * as THREE from 'three';
import type { LoadedHouseScene, HouseNode } from '../loadHouseScene';

export class RoomLabels {
  private labels: Array<{ node: HouseNode; element: HTMLDivElement }> = [];
  private container: HTMLDivElement;

  constructor(scene: LoadedHouseScene) {
    this.container = document.createElement('div');
    this.container.className = 'k3d-room-labels';
    Object.assign(this.container.style, {
      position: 'fixed',
      top: '0',
      left: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: '50',
    });

    for (const room of scene.rooms) {
      const label = document.createElement('div');
      label.textContent = room.surfaceForms.en?.word_ref || room.starId;
      Object.assign(label.style, {
        position: 'absolute',
        color: 'white',
        fontSize: '13px',
        fontFamily: 'monospace',
        textShadow: '0 1px 4px rgba(0,0,0,0.8)',
        opacity: '0.7',
        whiteSpace: 'nowrap',
        transform: 'translate(-50%, -100%)',
        transition: 'opacity 0.4s',
      });
      this.container.appendChild(label);
      this.labels.push({ node: room, element: label });
    }

    document.body.appendChild(this.container);
  }

  /** Call in render loop to reproject labels */
  update(camera: THREE.Camera): void {
    const vec = new THREE.Vector3();
    for (const { node, element } of this.labels) {
      node.object.getWorldPosition(vec);
      vec.y += 6; // float above room
      vec.project(camera);
      const x = (vec.x * 0.5 + 0.5) * window.innerWidth;
      const y = (-vec.y * 0.5 + 0.5) * window.innerHeight;
      const behind = vec.z > 1;
      element.style.left = `${x}px`;
      element.style.top = `${y}px`;
      element.style.display = behind ? 'none' : 'block';
    }
  }

  /** Highlight current room label */
  setCurrentRoom(starId: string): void {
    for (const { node, element } of this.labels) {
      const active = node.starId === starId;
      element.style.opacity = active ? '1' : '0.5';
      element.style.fontSize = active ? '15px' : '13px';
      element.style.color = active ? '#00ddff' : 'white';
    }
  }

  destroy(): void {
    this.container.remove();
  }
}
```

---

## Track D: Welcome Overlay

### D1. Create `viewer/src/navigation/welcome.ts`

A one-time overlay shown on first House load, explaining controls:

```typescript
export function showWelcomeOverlay(): HTMLDivElement {
  const overlay = document.createElement('div');
  overlay.className = 'k3d-welcome';
  Object.assign(overlay.style, {
    position: 'fixed',
    top: '0', left: '0', width: '100%', height: '100%',
    background: 'rgba(0,0,0,0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: '200',
    cursor: 'pointer',
  });

  overlay.innerHTML = `
    <div style="background: #1a1a2e; border: 1px solid #00ddff40; border-radius: 12px;
                padding: 32px 40px; max-width: 420px; color: white; font-family: monospace;
                text-align: center;">
      <h2 style="color: #00ddff; margin: 0 0 16px;">Knowledge3D House</h2>
      <p style="line-height: 1.6; margin: 0 0 20px; font-size: 14px; opacity: 0.8;">
        Navigate between rooms using <b>Arrow Keys</b> or <b>A/D</b>.<br>
        Jump to a room with <b>1-6</b> or press <b>H</b> for home.<br>
        <b>Click</b> objects to inspect them on the tablet.<br>
        <b>Drag</b> to orbit the camera.
      </p>
      <p style="font-size: 12px; opacity: 0.5;">Click anywhere to begin</p>
    </div>
  `;

  overlay.addEventListener('click', () => overlay.remove(), { once: true });
  document.body.appendChild(overlay);
  return overlay;
}
```

Show once per session (check `sessionStorage`). Dismiss on click or any keypress.

---

## Tips for Codex

**Tip 1 — Navigation uses existing nav graph.** Don't reinvent pathfinding. The `navGraph.edges` already know which rooms connect. Arrow keys walk the edge list filtered by current room. Sort by X position for consistent left/right.

**Tip 2 — Minimap dots = one per room, ordered by X.** The rooms are arranged linearly along X. The minimap is a simple horizontal strip. Current room glows cyan.

**Tip 3 — Room labels use `Vector3.project()`.** No need for CSS2DRenderer dependency. Project the room's world position to screen coords in the render loop. Labels are plain HTML divs with `position: absolute`.

**Tip 4 — Welcome overlay is session-only.** Check `sessionStorage.getItem('k3d-welcome-shown')`. Set it after first display. Don't persist across browser sessions — new visitors should always see it.

**Tip 5 — Room change callback is central.** Both keyboard nav AND minimap clicks AND door clicks all funnel through the same `onRoomChange(room)` callback. This callback calls `roomCamera.goToRoom()`, `roomContext.setRoom()`, updates minimap, updates labels. Single source of truth.

**Tip 6 — Build via SSD.** `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` only.

---

## Tests

### Viewer: `viewer/tests/keyboardNav.test.ts`

```typescript
describe('KeyboardNav', () => {
  it('moves to next room on ArrowRight', () => {
    // Mock scene with 3 rooms, verify callback fires with next room
  });

  it('moves to previous room on ArrowLeft', () => {
    // Mock scene with 3 rooms, verify callback fires with previous room
  });

  it('jumps to room by number key', () => {
    // Press '2', verify second room (by X position) is selected
  });

  it('ignores keys when focused on input', () => {
    // Set event.target to INPUT, verify no callback
  });
});
```

### Viewer: `viewer/tests/minimap.test.ts`

```typescript
describe('Minimap', () => {
  it('creates one dot per room', () => {
    // Verify element.children.length === rooms.length
  });

  it('highlights current room dot', () => {
    // setCurrentRoom, verify active dot has cyan background
  });
});
```

### Non-regression

All 19 existing viewer suites + all Python tests must continue to pass.

---

## Success Criteria

1. Arrow keys / A/D move between adjacent rooms following nav graph edges
2. Number keys 1-6 jump directly to rooms (ordered by X position)
3. H key returns to Living Room
4. Minimap shows room dots with current room glowing cyan
5. Clicking minimap dot navigates to that room
6. Room labels float above rooms in 3D, highlighting the current room
7. Welcome overlay shows on first load, dismisses on click/keypress
8. All existing tests pass, new navigation tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/navigation/keyboardNav.ts` | **NEW** — Keyboard room traversal |
| `viewer/src/navigation/minimap.ts` | **NEW** — 2D room minimap overlay |
| `viewer/src/navigation/roomLabels.ts` | **NEW** — 3D floating room labels |
| `viewer/src/navigation/welcome.ts` | **NEW** — First-load welcome overlay |
| `viewer/src/navigation/index.ts` | **NEW** — Barrel export |
| `viewer/src/main.ts` | Wire navigation, minimap, labels, welcome |
| `viewer/tests/keyboardNav.test.ts` | **NEW** |
| `viewer/tests/minimap.test.ts` | **NEW** |

---

## Architectural Note

This phase completes the viewer MVP UX loop. After H15, a first-time visitor can:
1. Open the viewer in a browser
2. See the welcome overlay explaining controls
3. Navigate rooms via keyboard, minimap clicks, or door clicks
4. See where they are via minimap and floating labels
5. Click objects to inspect them on the tablet
6. Watch holographic projections on the HoloDesk
7. Experience the Galaxy Pod stellarium in the Bathtub

That's a self-contained, demo-ready 3D knowledge house running in any modern browser. No server required. Just static files.
