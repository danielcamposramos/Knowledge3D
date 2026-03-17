# Phase H10: House Interaction System — Activating behavior_rpn

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H9 (Browser RPN Interpreter) COMPLETE
**Sovereignty:** I/O path (viewer interaction, flexible).
**Build:** Use `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` — NOT npm/vite from viewer/.

---

## Context

The House is physically complete (54 nodes, 278KB, 4167 vertices), viewable in browser (H8), and procedurally regenerable client-side (H9). Every House object carries `behavior_rpn` describing what happens when activated:

- Rooms: `ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN LIBRARY ACTIVATE_SHELVES`
- Doors: `DOOR_TRAVERSE CONNECT House/Library House/Garden`
- Books: `OPEN_BOOK LOAD_GALAXY Book/MathematicsPrimer`
- Tools: `TOOL_OBJECT INSPECT MAP_TO_TOOL_GALAXY`
- Displays: `DISPLAY CURATE REFERENCE_GALAXY`
- Instruments: `OBSERVE INTROSPECT RECORD_GALAXY_VIEW`
- Tablet: `TABLET ACTIVATE BROWSE_GALAXY QUERY_KNOWLEDGE INSPECT_PROGRAMS`

But none of these programs actually DO anything yet. The viewer loads them as metadata but never executes them. This phase adds a **behavior interpreter** that processes `behavior_rpn` on user interaction, enabling:

- Click a door → camera transitions to connected room (via nav_graph)
- Click a book → tablet shows the book's Galaxy content entries
- Click a tool → tablet shows linked Tool Galaxy entries
- Click a display → tablet shows referenced knowledge entries
- Click the tablet → opens Galaxy browser mode
- Enter a room → room's knowledge domain loads as context

---

## Deliverables

### Track A: Behavior RPN Interpreter
### Track B: Object Activation Pipeline
### Track C: Tablet Content Renderer
### Track D: Room Context System

---

## Track A: Behavior RPN Interpreter

### A1. Create `viewer/src/behavior/interpreter.ts`

A specialized interpreter for `behavior_rpn` programs. Unlike `visual_rpn` (which produces geometry), `behavior_rpn` produces **actions** — side effects in the viewer.

```typescript
export type BehaviorAction =
  | { type: 'room_enter'; room: string; domain: string }
  | { type: 'door_traverse'; roomA: string; roomB: string }
  | { type: 'load_galaxy'; galaxyRef: string }
  | { type: 'inspect_object'; starId: string }
  | { type: 'activate_display'; taxonomyRefs: string[] }
  | { type: 'browse_galaxy' }
  | { type: 'noop' };

export function interpretBehavior(
  behaviorRpn: string,
  node: HouseNode,
): BehaviorAction {
  const tokens = behaviorRpn.trim().split(/\s+/);
  const command = tokens[0];
  switch (command) {
    case 'ROOM_ENTER':
      return { type: 'room_enter', room: node.houseRoom, domain: node.domain };
    case 'DOOR_TRAVERSE': {
      // tokens: DOOR_TRAVERSE CONNECT House/Library House/Garden
      return { type: 'door_traverse', roomA: tokens[2] || '', roomB: tokens[3] || '' };
    }
    case 'OPEN_BOOK':
      return { type: 'load_galaxy', galaxyRef: node.galaxyRef || tokens[2] || '' };
    case 'TOOL_OBJECT':
      return { type: 'inspect_object', starId: node.starId };
    case 'DISPLAY':
      return { type: 'activate_display', taxonomyRefs: node.taxonomyRefs };
    case 'OBSERVE':
      return { type: 'inspect_object', starId: node.starId };
    case 'TABLET':
      return { type: 'browse_galaxy' };
    case 'PORTAL':
      return { type: 'inspect_object', starId: node.starId };
    default:
      return { type: 'noop' };
  }
}
```

This is intentionally simple — a token-based dispatch, not a full RPN stack machine. Behavior programs are declarative labels, not computation.

---

## Track B: Object Activation Pipeline

### B1. Create `viewer/src/behavior/activator.ts`

Wires behavior interpretation to viewer actions. This is the main event handler.

```typescript
export class HouseActivator {
  private scene: LoadedHouseScene;
  private roomCamera: RoomCamera;
  private tablet: Tablet3D;

  constructor(scene: LoadedHouseScene, roomCamera: RoomCamera, tablet: Tablet3D) {
    this.scene = scene;
    this.roomCamera = roomCamera;
    this.tablet = tablet;
  }

  /** Called when user clicks a House object. */
  activate(node: HouseNode): void {
    const action = interpretBehavior(node.behaviorRpn, node);
    switch (action.type) {
      case 'door_traverse':
        this.handleDoorTraverse(action.roomA, action.roomB);
        break;
      case 'load_galaxy':
        this.handleLoadGalaxy(action.galaxyRef, node);
        break;
      case 'inspect_object':
        this.handleInspect(node);
        break;
      case 'activate_display':
        this.handleDisplay(action.taxonomyRefs, node);
        break;
      case 'browse_galaxy':
        this.handleBrowseGalaxy();
        break;
      case 'room_enter':
        this.handleRoomEnter(action.room, action.domain);
        break;
      case 'noop':
        break;
    }
  }

  private handleDoorTraverse(roomA: string, roomB: string): void {
    // Determine which room is "the other one" relative to current room
    const currentRoomNode = this.scene.nodesByStarId.get(this.scene.currentRoom);
    const currentHouseRoom = currentRoomNode?.houseRoom || '';
    const targetHouseRoom = currentHouseRoom === roomA ? roomB : roomA;
    // Find room star_id from house_room string
    const targetRoom = this.scene.rooms.find(r => r.houseRoom === targetHouseRoom);
    if (targetRoom) {
      this.roomCamera.goToRoom(targetRoom.starId);
      this.scene.currentRoom = targetRoom.starId;
    }
  }

  private handleLoadGalaxy(galaxyRef: string, node: HouseNode): void {
    // Show book content on tablet
    this.tablet.publish({
      type: 'showContent',
      payload: {
        title: node.surfaceForms.en?.word_ref || node.starId,
        galaxyRef,
        taxonomyRefs: node.taxonomyRefs,
        componentRefs: node.componentRefs,
      },
    });
  }

  private handleInspect(node: HouseNode): void {
    // Show object details on tablet
    this.tablet.publish({
      type: 'showContent',
      payload: {
        title: node.surfaceForms.en?.word_ref || node.starId,
        meaningClass: node.meaningClass,
        domain: node.domain,
        taxonomyRefs: node.taxonomyRefs,
        visualRpn: node.visualRpn,
        behaviorRpn: node.behaviorRpn,
      },
    });
  }

  private handleDisplay(taxonomyRefs: string[], node: HouseNode): void {
    this.tablet.publish({
      type: 'showContent',
      payload: {
        title: node.surfaceForms.en?.word_ref || node.starId,
        display: true,
        taxonomyRefs,
      },
    });
  }

  private handleBrowseGalaxy(): void {
    this.tablet.publish({ type: 'openApp', payload: { app: 'Galaxy' } });
  }

  private handleRoomEnter(room: string, domain: string): void {
    // Update room context indicator
    this.tablet.publish({
      type: 'roomContext',
      payload: { room, domain },
    });
  }
}
```

### B2. Wire raycasting to activation

In `main.ts`, on mouse click (not hover — hover shows tooltip), raycast into the House scene. If a House node is hit, call `activator.activate(node)`.

```typescript
// In main.ts click handler:
const intersects = raycaster.intersectObjects(houseScene.root.children, true);
for (const hit of intersects) {
  const k3d = hit.object.userData?.k3d;
  if (k3d?.star_id) {
    const node = houseScene.nodesByStarId.get(k3d.star_id);
    if (node) {
      activator.activate(node);
      break;
    }
  }
}
```

---

## Track C: Tablet Content Renderer

### C1. Create `viewer/src/behavior/contentRenderer.ts`

When a book, tool, display, or object is activated, the tablet needs to show its content. This renderer formats House node data into tablet-displayable content.

```typescript
export interface ContentPage {
  title: string;
  sections: ContentSection[];
}

export interface ContentSection {
  heading: string;
  lines: string[];
}

export function renderNodeContent(node: HouseNode): ContentPage {
  const sections: ContentSection[] = [];

  // Identity
  sections.push({
    heading: 'Identity',
    lines: [
      `Star ID: ${node.starId}`,
      `Class: ${node.meaningClass}`,
      `Domain: ${node.domain}`,
      `Room: ${node.houseRoom}`,
    ],
  });

  // Surface Forms (multilingual)
  const formLines: string[] = [];
  for (const [lang, form] of Object.entries(node.surfaceForms)) {
    formLines.push(`${lang}: ${form.word_ref}`);
  }
  if (formLines.length) {
    sections.push({ heading: 'Names', lines: formLines });
  }

  // References
  if (node.taxonomyRefs.length) {
    sections.push({
      heading: 'References',
      lines: node.taxonomyRefs.map(ref => `→ ${ref}`),
    });
  }

  // Galaxy (for books)
  if (node.galaxyRef) {
    sections.push({
      heading: 'Content Galaxy',
      lines: [`Load: ${node.galaxyRef}`],
    });
  }

  // Programs
  if (node.visualRpn) {
    sections.push({
      heading: 'Visual Program',
      lines: [node.visualRpn],
    });
  }
  sections.push({
    heading: 'Behavior Program',
    lines: [node.behaviorRpn],
  });

  return {
    title: node.surfaceForms.en?.word_ref || node.starId,
    sections,
  };
}
```

### C2. Add ContentApp to tablet

Create a new tablet app (or extend the existing Console app) that displays `ContentPage` data:

```typescript
// In apps.ts, add a ContentApp that listens for 'showContent' events
class ContentApp extends TabletApp {
  // Renders ContentPage sections as scrollable text on the tablet canvas
  // Each section has a heading and indented lines
  // Galaxy refs are shown as clickable links
  // visual_rpn is shown with a mini SVG preview (using rpnToSVG from H9)
}
```

---

## Track D: Room Context System

### D1. Create `viewer/src/behavior/roomContext.ts`

Tracks which room the user is currently in and provides context-aware information.

```typescript
export class RoomContext {
  private currentRoom: HouseNode | null = null;
  private onRoomChange: ((room: HouseNode) => void)[] = [];

  setRoom(room: HouseNode): void {
    if (this.currentRoom?.starId === room.starId) return;
    this.currentRoom = room;
    for (const callback of this.onRoomChange) callback(room);
  }

  onEnter(callback: (room: HouseNode) => void): void {
    this.onRoomChange.push(callback);
  }

  get current(): HouseNode | null {
    return this.currentRoom;
  }
}
```

### D2. Room context triggers

When the camera transitions to a new room (via door or direct navigation):
1. Update `RoomContext` with the new room
2. Fire `roomContext` event to tablet (shows current room name + domain)
3. Highlight the current room's objects (full opacity vs dimmed for other rooms)
4. Update DoorsApp to show only doors reachable from the current room

### D3. Room-aware object dimming

Objects not in the current room render at reduced opacity (30%). This gives spatial context without hiding content.

```typescript
// When room changes:
houseScene.nodesByStarId.forEach((node) => {
  if (node.meaningClass === 'room') return;  // rooms always visible
  const inCurrentRoom = node.houseRoom === currentRoom.houseRoom;
  const material = (node.object as THREE.Mesh).material as THREE.MeshStandardMaterial;
  if (material) {
    material.transparent = !inCurrentRoom;
    material.opacity = inCurrentRoom ? 1.0 : 0.3;
  }
});
```

---

## Tips for Codex

**Tip 1 — Click vs hover distinction.** H8 added hover tooltips (surface form + truncated RPN). H10 adds click activation (full behavior execution). These are separate event paths — don't conflate them.

**Tip 2 — Tablet event system.** The existing `Tablet3D.publish()` already handles events. The new events (`showContent`, `roomContext`, `openApp`) should follow the same pattern as existing events (`openDoor`, `applyLayers`, etc.).

**Tip 3 — behavior_rpn is NOT computed.** Unlike `visual_rpn` (which is an RPN stack program), `behavior_rpn` is a declarative label string. Don't run it through the `RpnEngine` — just tokenize and dispatch.

**Tip 4 — Galaxy content is metadata, not geometry.** When a book is "opened" (clicked), the tablet shows the book's content entries as text (titles, taxonomy refs, chapters). The actual Galaxy data lives server-side. The viewer shows what's available, not the full Galaxy.

**Tip 5 — Room dimming must handle materials.** GLTF-loaded meshes use whatever material Three.js assigns. Access the material via `(mesh as THREE.Mesh).material` and set `transparent`/`opacity`. If the material is shared, you may need to clone it first.

**Tip 6 — ContentApp can be simple.** Canvas 2D text rendering on the tablet surface is already proven (all 18 existing apps use it). The ContentApp just draws section headings and text lines. No HTML needed.

---

## Tests

### `viewer/tests/behaviorInterpreter.test.ts`

```typescript
test('door behavior produces door_traverse action', () => {
  const action = interpretBehavior('DOOR_TRAVERSE CONNECT House/Library House/Garden', mockNode);
  expect(action.type).toBe('door_traverse');
  expect(action.roomA).toBe('House/Library');
});

test('book behavior produces load_galaxy action', () => {
  const node = { ...mockNode, galaxyRef: 'Book/MathematicsPrimer' };
  const action = interpretBehavior('OPEN_BOOK LOAD_GALAXY Book/MathematicsPrimer', node);
  expect(action.type).toBe('load_galaxy');
  expect(action.galaxyRef).toBe('Book/MathematicsPrimer');
});

test('tablet behavior produces browse_galaxy action', () => {
  const action = interpretBehavior('TABLET ACTIVATE BROWSE_GALAXY', mockNode);
  expect(action.type).toBe('browse_galaxy');
});

test('unknown behavior produces noop', () => {
  const action = interpretBehavior('UNKNOWN_COMMAND', mockNode);
  expect(action.type).toBe('noop');
});
```

### `viewer/tests/contentRenderer.test.ts`

```typescript
test('renders node content with all sections', () => {
  const page = renderNodeContent(mockBookNode);
  expect(page.title).toBeTruthy();
  expect(page.sections.find(s => s.heading === 'Identity')).toBeTruthy();
  expect(page.sections.find(s => s.heading === 'Content Galaxy')).toBeTruthy();
});
```

### Non-regression

All existing tests must pass. Build via `build.sh`.

---

## Success Criteria

1. Clicking a door transitions camera to the connected room
2. Clicking a book shows its Galaxy content entries on the tablet
3. Clicking a tool/display/instrument shows its details and references on the tablet
4. Clicking the Memory Tablet opens Galaxy browser mode
5. Room context updates when camera transitions between rooms
6. Objects outside the current room are visually dimmed
7. All existing tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/behavior/interpreter.ts` | **NEW** — behavior_rpn → BehaviorAction |
| `viewer/src/behavior/activator.ts` | **NEW** — Action → viewer side effects |
| `viewer/src/behavior/contentRenderer.ts` | **NEW** — Node data → tablet content |
| `viewer/src/behavior/roomContext.ts` | **NEW** — Room tracking + change events |
| `viewer/src/behavior/index.ts` | **NEW** — Module barrel |
| `viewer/src/apps.ts` | Add ContentApp |
| `viewer/src/main.ts` | Wire click → activator, room context |
| `viewer/tests/behaviorInterpreter.test.ts` | **NEW** |
| `viewer/tests/contentRenderer.test.ts` | **NEW** |

---

## Architectural Note

This phase makes the House **alive**. Every `behavior_rpn` program now has a concrete effect. The pattern is: user interacts with 3D object → behavior_rpn is interpreted → viewer action fires → tablet/camera/context updates. This is the game loop in embryonic form — the same pattern that TRM will eventually drive autonomously on GPU (Phase D).

The interaction model is intentionally simple: click → dispatch → effect. No complex state machines. This keeps the viewer lightweight while proving the behavior_rpn vocabulary works in practice. When TRM takes over, it'll use the same BehaviorAction types, just dispatched from GPU instead of mouse clicks.
