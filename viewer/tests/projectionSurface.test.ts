import * as THREE from 'three';

import { ProjectionSurface } from '../src/projection';

function contentChildCount(surface: ProjectionSurface): number {
  return ((surface as any).contentRoot as THREE.Group).children.length;
}

describe('ProjectionSurface', () => {
  it('adds content as child of the projection group', () => {
    const surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 0, 0),
      bounds: new THREE.Vector3(1, 1, 1),
      mode: 'stellarium',
    });
    surface.setContent(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial()));
    expect(contentChildCount(surface)).toBe(1);
  });

  it('clears content on demand', () => {
    const surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 0, 0),
      bounds: new THREE.Vector3(1, 1, 1),
      mode: 'stellarium',
    });
    surface.setContent(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial()));
    surface.clear();
    expect(contentChildCount(surface)).toBe(0);
  });

  it('auto-scales content to fit inside bounds', () => {
    const surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 0, 0),
      bounds: new THREE.Vector3(2, 1, 1),
      mode: 'stellarium',
    });
    surface.setContent(new THREE.Mesh(new THREE.BoxGeometry(10, 8, 6), new THREE.MeshBasicMaterial()));
    const bounds = new THREE.Box3().setFromObject(surface.group);
    const size = bounds.getSize(new THREE.Vector3());
    expect(size.x).toBeLessThanOrEqual(2.01);
    expect(size.y).toBeLessThanOrEqual(1.01);
    expect(size.z).toBeLessThanOrEqual(1.01);
  });
});
