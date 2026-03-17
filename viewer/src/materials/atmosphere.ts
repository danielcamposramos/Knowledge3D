import * as THREE from 'three';

const ATMOSPHERE_GROUP_NAME = 'k3d-house-atmosphere';
const ATMOSPHERE_COLOR = 0x1a1a2e;

export function setHouseAtmosphere(scene: THREE.Scene): THREE.Group {
  const existing = scene.getObjectByName(ATMOSPHERE_GROUP_NAME);
  if (existing) {
    scene.remove(existing);
  }

  scene.background = new THREE.Color(ATMOSPHERE_COLOR);
  scene.fog = new THREE.Fog(ATMOSPHERE_COLOR, 60, 150);

  const group = new THREE.Group();
  group.name = ATMOSPHERE_GROUP_NAME;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(300, 300),
    new THREE.MeshStandardMaterial({
      color: 0x111122,
      roughness: 1.0,
      metalness: 0.0,
      side: THREE.DoubleSide,
    }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.5;
  group.add(ground);
  scene.add(group);
  return group;
}

export function clearHouseAtmosphere(scene: THREE.Scene): void {
  const existing = scene.getObjectByName(ATMOSPHERE_GROUP_NAME);
  if (existing) {
    scene.remove(existing);
  }
  scene.background = new THREE.Color(0x111111);
  scene.fog = null;
}
