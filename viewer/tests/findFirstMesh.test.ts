import * as THREE from 'three';
import { findFirstMesh } from '../src/findFirstMesh';

describe('findFirstMesh', () => {
  it('finds the first mesh in a nested scene graph', () => {
    const root = new THREE.Group();
    const group = new THREE.Group();
    root.add(group);
    const mesh = new THREE.Mesh(new THREE.BufferGeometry());
    group.add(mesh);

    expect(findFirstMesh(root)).toBe(mesh);
  });

  it('returns undefined when no mesh is present', () => {
    const root = new THREE.Group();
    root.add(new THREE.Object3D());

    expect(findFirstMesh(root)).toBeUndefined();
  });
});
