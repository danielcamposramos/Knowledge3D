import * as THREE from 'three';

const LIGHT_GROUP_NAME = 'k3d-house-lights';

export function createHouseLighting(scene: THREE.Scene): THREE.Group {
  const existing = scene.getObjectByName(LIGHT_GROUP_NAME);
  if (existing) {
    scene.remove(existing);
  }

  const lights = new THREE.Group();
  lights.name = LIGHT_GROUP_NAME;

  const ambient = new THREE.AmbientLight(0xffffff, 0.3);
  const hemisphere = new THREE.HemisphereLight(0xffeedd, 0x223344, 0.4);
  const sun = new THREE.DirectionalLight(0xfff8e8, 0.6);
  sun.position.set(20, 40, 10);
  const fill = new THREE.DirectionalLight(0xd0e0ff, 0.2);
  fill.position.set(-15, 20, -10);

  lights.add(ambient, hemisphere, sun, fill);
  scene.add(lights);
  return lights;
}
