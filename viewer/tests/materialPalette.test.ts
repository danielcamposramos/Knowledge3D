import * as THREE from 'three';

import type { HouseNode, LoadedHouseScene } from '../src/loadHouseScene';
import { applyHouseMaterials, createDomainPalette } from '../src/materials';

function makeNode(
  starId: string,
  meaningClass: string,
  houseRoom: string,
  parent?: THREE.Object3D,
): HouseNode {
  const object = new THREE.Group();
  object.userData = { k3d: { star_id: starId } };
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshNormalMaterial());
  object.add(mesh);
  if (parent) parent.add(object);
  return {
    starId,
    meaningClass,
    domain: houseRoom,
    houseRoom,
    housePosition: [0, 0, 0],
    surfaceForms: {},
    behaviorRpn: '',
    taxonomyRefs: [],
    componentRefs: [],
    object,
  };
}

describe('material palette', () => {
  it('creates distinct palettes for each domain', () => {
    const living = createDomainPalette('House/LivingRoom');
    const workshop = createDomainPalette('House/Workshop');
    expect(living.room.color.getHex()).not.toBe(workshop.room.color.getHex());
  });

  it('returns MeshStandardMaterial for all slots', () => {
    const palette = createDomainPalette('House/Library');
    for (const material of Object.values(palette)) {
      expect(material).toBeInstanceOf(THREE.MeshStandardMaterial);
    }
  });

  it('uses shared material instances for the same domain', () => {
    const first = createDomainPalette('House/Garden');
    const second = createDomainPalette('House/Garden');
    expect(first).toBe(second);
    expect(first.room).toBe(second.room);
    expect(first.furniture).toBe(second.furniture);
  });

  it('applies materials to each node without leaving MeshNormalMaterial', () => {
    const room = makeNode('room_library', 'room', 'House/Library');
    const book = makeNode('book_mathematics_primer', 'book', 'House/Library', room.object);
    const loadedScene: LoadedHouseScene = {
      root: new THREE.Group(),
      nodesByStarId: new Map([
        [room.starId, room],
        [book.starId, book],
      ]),
      rooms: [room],
      doors: [],
      navGraph: { nodes: [], edges: [] },
      currentRoom: room.starId,
    };
    applyHouseMaterials(loadedScene);

    const roomMesh = room.object.children[0] as THREE.Mesh;
    const bookMesh = book.object.children[0] as THREE.Mesh;
    expect(roomMesh.material).toBeInstanceOf(THREE.MeshStandardMaterial);
    expect(bookMesh.material).toBeInstanceOf(THREE.MeshStandardMaterial);
    expect(roomMesh.material).not.toBeInstanceOf(THREE.MeshNormalMaterial);
    expect(bookMesh.material).not.toBeInstanceOf(THREE.MeshNormalMaterial);
  });
});
