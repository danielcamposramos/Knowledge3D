import * as THREE from 'three';
import type { K3DRecord } from './loadK3D';

export class K3DAgent {
  public object: THREE.Object3D;
  public followCamera = false;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private records: K3DRecord[] = [];
  private recordMap: Map<string, K3DRecord> = new Map();
  private target: THREE.Vector3 | null = null;
  private path: THREE.Vector3[] = [];
  private speed = 1.5; // units per second

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.scene = scene;
    this.camera = camera;
    const geom = new THREE.SphereGeometry(0.12, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffcc00 });
    this.object = new THREE.Mesh(geom, mat);
    this.object.position.set(0, 0, 0);
    this.scene.add(this.object);
  }

  setRecords(records: K3DRecord[]) {
    this.records = records || [];
    this.recordMap = new Map(this.records.map(r => [r.id, r]));
    if (this.records.length > 0) {
      const v = this.records[0].vector;
      this.object.position.set(v[0], v[1], v[2]);
      this.target = null;
      this.path = [];
    }
  }

  goToLabel(query: string) {
    if (!this.records.length || !query) return;
    const q = query.toLowerCase();
    const found = this.records.find(r => (r.metadata?.label as string)?.toLowerCase().includes(q) || r.id.toLowerCase().includes(q));
    if (!found) return;
    // Try neighbor-based path; fallback to direct
    const start = this.findClosestRecord(this.object.position);
    const pathIds = this.shortestPath(start?.id, found.id);
    if (pathIds && pathIds.length > 1) {
      this.path = pathIds.map(id => {
        const r = this.recordMap.get(id)!;
        return new THREE.Vector3(r.vector[0], r.vector[1], r.vector[2]);
      });
      this.target = this.path.shift() || null;
    } else {
      const [x, y, z] = found.vector;
      this.target = new THREE.Vector3(x, y, z);
      this.path = [];
    }
  }

  private findClosestRecord(pos: THREE.Vector3): K3DRecord | undefined {
    let best: K3DRecord | undefined;
    let bestD = Number.POSITIVE_INFINITY;
    for (const r of this.records) {
      const dx = r.vector[0] - pos.x;
      const dy = r.vector[1] - pos.y;
      const dz = r.vector[2] - pos.z;
      const d2 = dx*dx + dy*dy + dz*dz;
      if (d2 < bestD) { bestD = d2; best = r; }
    }
    return best;
  }

  private shortestPath(fromId: string | undefined, toId: string): string[] | null {
    if (!fromId) return null;
    if (fromId === toId) return [fromId];
    const visited = new Set<string>([fromId]);
    const queue: string[][] = [[fromId]];
    while (queue.length) {
      const path = queue.shift()!;
      const last = path[path.length - 1];
      const node = this.recordMap.get(last);
      const neigh = (node?.neighbors as string[] | undefined) || [];
      for (const nid of neigh) {
        if (visited.has(nid)) continue;
        visited.add(nid);
        const next = path.concat(nid);
        if (nid === toId) return next;
        queue.push(next);
      }
    }
    return null;
  }

  update(dt: number) {
    if (!this.target) return;
    const current = this.object.position;
    const dir = new THREE.Vector3().subVectors(this.target, current);
    const dist = dir.length();
    if (dist < 1e-3) {
      if (this.path.length > 0) {
        this.target = this.path.shift() || null;
      } else {
        this.target = null;
      }
      return;
    }
    dir.normalize();
    const step = Math.min(dist, this.speed * dt);
    current.addScaledVector(dir, step);

    if (this.followCamera) {
      const offset = new THREE.Vector3(0, 0.6, 1.6);
      const camPos = new THREE.Vector3().copy(current).add(offset);
      this.camera.position.lerp(camPos, 0.2);
      this.camera.lookAt(current);
    }
  }
}
