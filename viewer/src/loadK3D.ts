import * as THREE from 'three';

export interface K3DRecord {
  id: string;
  vector: [number, number, number];
  metadata: Record<string, unknown>;
}

export async function fetchK3D(url: string): Promise<K3DRecord[]> {
  const res = await fetch(url);
  return (await res.json()) as K3DRecord[];
}

export function buildPoints(records: K3DRecord[]): THREE.Points {
  const positions = new Float32Array(records.length * 3);
  records.forEach((r, i) => {
    positions.set(r.vector, i * 3);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({ size: 0.05 });
  return new THREE.Points(geometry, material);
}
