import * as THREE from 'three';

import type { HouseNode, LoadedHouseScene } from '../src/loadHouseScene';
import { Minimap } from '../src/navigation';

function makeRoom(starId: string, x: number): HouseNode {
  return {
    starId,
    meaningClass: 'room',
    domain: `House/${starId}`,
    houseRoom: `House/${starId}`,
    housePosition: [x, 0, 0],
    surfaceForms: { en: { word_ref: starId, char_refs: [] } },
    behaviorRpn: '',
    taxonomyRefs: [],
    componentRefs: [],
    object: new THREE.Object3D(),
  };
}

function makeScene(): LoadedHouseScene {
  const rooms = [makeRoom('room_living', -10), makeRoom('room_library', 0), makeRoom('room_garden', 18)];
  return {
    root: new THREE.Group(),
    nodesByStarId: new Map(rooms.map((room) => [room.starId, room])),
    rooms,
    doors: [],
    navGraph: { nodes: rooms.map((room) => room.starId), edges: [] },
    currentRoom: 'room_living',
  };
}

describe('Minimap', () => {
  it('creates one dot per room', () => {
    const minimap = new Minimap(makeScene());
    expect(minimap.element.children.length).toBe(3);
    minimap.destroy();
  });

  it('highlights current room dot', () => {
    const minimap = new Minimap(makeScene());
    minimap.setCurrentRoom('room_library');
    const dot = minimap.element.querySelector('[data-star-id="room_library"]') as HTMLDivElement | null;
    expect(dot).toBeTruthy();
    expect(dot?.style.boxShadow).toContain('#00ddff');
    expect(dot?.style.transform).toBe('scale(1.4)');
    minimap.destroy();
  });
});
