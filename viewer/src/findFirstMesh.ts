import * as THREE from 'three';

/**
 * Traverses the scene to find the first THREE.Mesh instance.
 * @param root - Root object to begin traversal from.
 * @returns The first mesh found, or undefined if none exists.
 */
export function findFirstMesh(root: THREE.Object3D): THREE.Mesh | undefined {
  let found: THREE.Mesh | undefined;
  root.traverse((obj) => {
    if (!found && (obj as THREE.Mesh).isMesh) {
      found = obj as THREE.Mesh;
    }
  });
  return found;
}
