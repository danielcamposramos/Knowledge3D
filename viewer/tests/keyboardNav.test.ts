import * as THREE from 'three';

import type { HouseNode, LoadedHouseScene } from '../src/loadHouseScene';
import { KeyboardNav } from '../src/navigation';

function makeRoom(starId: string, x: number): HouseNode {
  const object = new THREE.Object3D();
  object.position.set(x, 0, 0);
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
    object,
  };
}

function makeScene(): LoadedHouseScene {
  const roomLiving = makeRoom('room_living', -10);
  const roomLibrary = makeRoom('room_library', 0);
  const roomGarden = makeRoom('room_garden', 18);
  const rooms = [roomLiving, roomLibrary, roomGarden];
  return {
    root: new THREE.Group(),
    nodesByStarId: new Map(rooms.map((room) => [room.starId, room])),
    rooms,
    doors: [],
    navGraph: {
      nodes: rooms.map((room) => room.starId),
      edges: [
        { door: 'door_living_library', from: 'room_living', to: 'room_library', cost: 1 },
        { door: 'door_living_library', from: 'room_library', to: 'room_living', cost: 1 },
        { door: 'door_library_garden', from: 'room_library', to: 'room_garden', cost: 1 },
        { door: 'door_library_garden', from: 'room_garden', to: 'room_library', cost: 1 },
      ],
    },
    currentRoom: 'room_library',
  };
}

describe('KeyboardNav', () => {
  it('moves to next room on ArrowRight', () => {
    const scene = makeScene();
    const onRoomChange = jest.fn((room: HouseNode) => { scene.currentRoom = room.starId; });
    const nav = new KeyboardNav(scene, { onRoomChange });
    nav.attach();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    nav.detach();
    expect(onRoomChange).toHaveBeenCalledWith(expect.objectContaining({ starId: 'room_garden' }));
  });

  it('moves to previous room on ArrowLeft', () => {
    const scene = makeScene();
    const onRoomChange = jest.fn((room: HouseNode) => { scene.currentRoom = room.starId; });
    const nav = new KeyboardNav(scene, { onRoomChange });
    nav.attach();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    nav.detach();
    expect(onRoomChange).toHaveBeenCalledWith(expect.objectContaining({ starId: 'room_living' }));
  });

  it('jumps to room by number key', () => {
    const scene = makeScene();
    const onRoomChange = jest.fn();
    const nav = new KeyboardNav(scene, { onRoomChange });
    nav.attach();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '2', bubbles: true }));
    nav.detach();
    expect(onRoomChange).toHaveBeenCalledWith(expect.objectContaining({ starId: 'room_library' }));
  });

  it('ignores keys when focused on input', () => {
    const scene = makeScene();
    const onRoomChange = jest.fn();
    const nav = new KeyboardNav(scene, { onRoomChange });
    nav.attach();
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    input.remove();
    nav.detach();
    expect(onRoomChange).not.toHaveBeenCalled();
  });
});
