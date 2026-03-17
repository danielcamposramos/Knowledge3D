import * as THREE from 'three';

import { createHouseLighting, setHouseAtmosphere } from '../src/materials';

describe('house lighting', () => {
  it('creates 4 lights in a group', () => {
    const scene = new THREE.Scene();
    const lights = createHouseLighting(scene);
    expect(lights.children).toHaveLength(4);
    expect(lights.children[0]).toBeInstanceOf(THREE.AmbientLight);
    expect(lights.children[1]).toBeInstanceOf(THREE.HemisphereLight);
    expect(lights.children[2]).toBeInstanceOf(THREE.DirectionalLight);
    expect(lights.children[3]).toBeInstanceOf(THREE.DirectionalLight);
  });

  it('adds lights to the scene', () => {
    const scene = new THREE.Scene();
    const lights = createHouseLighting(scene);
    expect(scene.children.includes(lights)).toBe(true);
  });

  it('sets background and fog atmosphere', () => {
    const scene = new THREE.Scene();
    const group = setHouseAtmosphere(scene);
    expect(scene.children.includes(group)).toBe(true);
    expect(scene.background).toBeInstanceOf(THREE.Color);
    expect(scene.fog).toBeInstanceOf(THREE.Fog);
  });
});
