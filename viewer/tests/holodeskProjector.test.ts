import * as THREE from 'three';

import type { HouseNode } from '../src/loadHouseScene';
import { HolodeskProjector } from '../src/projection';
import { rpnToMesh } from '../src/rpn';

function makeNode(): HouseNode {
  return {
    starId: 'furniture_holodesk',
    meaningClass: 'furniture',
    domain: 'House/Living',
    houseRoom: 'House/Living',
    housePosition: [0, 0, 0],
    surfaceForms: { en: { word_ref: 'HoloDesk', char_refs: [] } },
    behaviorRpn: 'HOLODESK ACTIVATE PROJECT_3D',
    taxonomyRefs: [],
    componentRefs: [],
    visualRpn: '1.0 GEN_CUBE',
    object: new THREE.Group(),
  };
}

describe('HolodeskProjector', () => {
  it('creates holographic wireframe material', () => {
    const projector = new HolodeskProjector(makeNode());
    projector.projectMesh(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshNormalMaterial()));
    const mesh = projector.surface.group.getObjectByProperty('isMesh', true) as THREE.Mesh | undefined;
    expect(mesh).toBeTruthy();
    const material = mesh?.material as THREE.MeshBasicMaterial;
    expect(material.wireframe).toBe(true);
    expect(material.transparent).toBe(true);
    expect(material.color.getHex()).toBe(0x00ddff);
  });

  it('toggles visibility on repeated activation of the same node', () => {
    const node = makeNode();
    const projector = new HolodeskProjector(node);
    expect(projector.toggleNodeVisual(node, rpnToMesh)).toBe(true);
    expect(projector.visible).toBe(true);
    expect(projector.toggleNodeVisual(node, rpnToMesh)).toBe(false);
    expect(projector.visible).toBe(false);
  });

  it('applies slow rotation on update', () => {
    const node = makeNode();
    const projector = new HolodeskProjector(node);
    projector.projectNodeVisual(node, rpnToMesh);
    const start = projector.surface.group.rotation.y;
    projector.update(1.0);
    expect(projector.surface.group.rotation.y).toBeGreaterThan(start);
  });
});
