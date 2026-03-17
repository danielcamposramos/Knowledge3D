import * as THREE from 'three';

import type { HouseNode } from '../src/loadHouseScene';
import { RoomCamera } from '../src/roomCamera';

function makeRoom(starId: string, x: number): HouseNode {
  const object = new THREE.Object3D();
  object.position.set(x, 0, 0);
  return {
    starId,
    meaningClass: 'room',
    domain: `House/${starId}`,
    houseRoom: `House/${starId}`,
    housePosition: [x, 0, 0],
    surfaceForms: {},
    behaviorRpn: '',
    taxonomyRefs: [],
    componentRefs: [],
    object,
  };
}

describe('RoomCamera', () => {
  it('moves camera target toward destination room', () => {
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    const controls = {
      target: new THREE.Vector3(),
      update: jest.fn(),
    };
    const library = makeRoom('room_library', 0);
    const garden = makeRoom('room_garden', 18);
    const roomCamera = new RoomCamera(camera, controls, [library, garden], 'room_library');

    roomCamera.goToRoom('room_garden');
    roomCamera.update(0.5);

    expect(roomCamera.currentRoom).toBe('room_garden');
    expect(camera.position.x).toBeGreaterThan(0);
    expect(controls.target.x).toBeGreaterThan(0);
  });
});
