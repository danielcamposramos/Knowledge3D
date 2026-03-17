import * as THREE from 'three';

import { loadHouseScene } from '../src/loadHouseScene';

jest.mock('three/examples/jsm/loaders/GLTFLoader.js', () => {
  class GLTFLoader {
    async loadAsync() {
      const scene = new THREE.Group();
      const houseRoot = new THREE.Group();
      houseRoot.name = 'House';
      houseRoot.userData = {
        k3d: {
          star_id: 'house_root',
          meaning_class: 'house',
          nav_graph: {
            nodes: ['room_library', 'room_garden'],
            edges: [{ door: 'door_library_garden', from: 'room_library', to: 'room_garden', cost: 1 }],
          },
        },
      };
      const library = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
      library.name = 'room_library';
      library.userData = {
        k3d: {
          star_id: 'room_library',
          meaning_class: 'room',
          domain: 'House/Library',
          house_room: 'House/Library',
          house_position: [0, 0, 0],
          surface_forms: { en: { word_ref: 'library', char_refs: ['char_l'] } },
          behavior_rpn: 'ROOM_ENTER',
          taxonomy_refs: [],
          component_refs: [],
        },
      };
      const door = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
      door.name = 'door_library_garden';
      door.userData = {
        k3d: {
          star_id: 'door_library_garden',
          meaning_class: 'door',
          domain: 'House/Connectivity',
          house_room: 'House/Library',
          house_position: [9, 0, 0],
          surface_forms: { en: { word_ref: 'library_garden_door', char_refs: ['char_d'] } },
          behavior_rpn: 'DOOR_TRAVERSE CONNECT House/Library House/Garden',
          taxonomy_refs: ['House/Library', 'House/Garden'],
          component_refs: [],
        },
      };
      houseRoot.add(library);
      houseRoot.add(door);
      scene.add(houseRoot);
      return { scene };
    }
  }
  return { GLTFLoader };
});

describe('loadHouseScene', () => {
  it('indexes house nodes and nav graph from GLTF extras', async () => {
    const loaded = await loadHouseScene('/house.glb');
    expect(loaded.currentRoom).toBe('room_library');
    expect(loaded.rooms.map((room) => room.starId)).toEqual(['room_library']);
    expect(loaded.doors.map((door) => door.starId)).toEqual(['door_library_garden']);
    expect(loaded.nodesByStarId.get('room_library')?.houseRoom).toBe('House/Library');
    expect(loaded.navGraph.nodes).toEqual(['room_library', 'room_garden']);
    expect(loaded.navGraph.edges[0]?.door).toBe('door_library_garden');
  });
});
