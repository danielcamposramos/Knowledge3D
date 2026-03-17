import * as THREE from 'three';

import type { HouseNode } from '../src/loadHouseScene';
import { GalaxyPodProjector } from '../src/projection';

function makeBathtubNode(): HouseNode {
  return {
    starId: 'furniture_bathtub',
    meaningClass: 'furniture',
    domain: 'House/Bathtub',
    houseRoom: 'House/Bathtub',
    housePosition: [0, 0, 0],
    surfaceForms: { en: { word_ref: 'Bathtub', char_refs: [] } },
    behaviorRpn: 'PORTAL REST REFLECT',
    taxonomyRefs: [],
    componentRefs: [],
    object: new THREE.Group(),
  };
}

function findPoints(object: THREE.Object3D): THREE.Points | null {
  let points: THREE.Points | null = null;
  object.traverse((child) => {
    if (!points && child instanceof THREE.Points) {
      points = child;
    }
  });
  return points;
}

describe('GalaxyPodProjector', () => {
  it('creates point cloud from concept entries', () => {
    const projector = new GalaxyPodProjector(makeBathtubNode());
    projector.projectGalaxy([{ star_id: 'concept_mathematics', domain: 'Mathematics' }]);
    const points = findPoints(projector.surface.group);
    expect(points).toBeTruthy();
  });

  it('colors stars by domain', () => {
    const projector = new GalaxyPodProjector(makeBathtubNode());
    projector.projectGalaxy([{ star_id: 'concept_mathematics', domain: 'Mathematics' }]);
    const points = findPoints(projector.surface.group);
    const colors = points?.geometry.getAttribute('color');
    const expected = new THREE.Color(0x4488ff);
    expect(colors).toBeTruthy();
    expect(colors?.getX(0)).toBeCloseTo(expected.r, 3);
    expect(colors?.getY(0)).toBeCloseTo(expected.g, 3);
    expect(colors?.getZ(0)).toBeCloseTo(expected.b, 3);
  });

  it('pads to a minimum 200 points', () => {
    const projector = new GalaxyPodProjector(makeBathtubNode());
    projector.projectGalaxy([{ star_id: 'concept_mathematics', domain: 'Mathematics' }]);
    const points = findPoints(projector.surface.group);
    const positions = points?.geometry.getAttribute('position');
    expect(positions?.count).toBe(200);
  });
});
