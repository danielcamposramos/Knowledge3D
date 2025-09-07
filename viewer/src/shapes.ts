import * as THREE from 'three';
import type { K3DRecord } from './loadK3D';

// Modality bit flags
const F_TEXT = 1 << 0;
const F_IMAGE = 1 << 1;
const F_AUDIO = 1 << 2;
const F_VIDEO = 1 << 3;

export function modalityMask(r: K3DRecord): number {
  let m = 0;
  const md: any = r.metadata || {};
  const t = String(md.type || '').toLowerCase();
  if (t === 'text') m |= F_TEXT;
  if (t === 'image') m |= F_IMAGE;
  if (t === 'audio') m |= F_AUDIO;
  if (t === 'video') m |= F_VIDEO;
  // Heuristics: if fields exist
  if (md.text) m |= F_TEXT;
  if (md.image) m |= F_IMAGE;
  if (md.video) m |= F_VIDEO;
  return m;
}

function shapeGeometry(mask: number): THREE.BufferGeometry {
  // Pick a geometry based on modalities present.
  // Single modality shapes: text= tetrahedron, image=box, audio=octahedron, video=icosahedron
  // Mixed -> dodecahedron
  const bits = [F_TEXT, F_IMAGE, F_AUDIO, F_VIDEO].filter(b => (mask & b) !== 0).length;
  if (bits >= 2) return new THREE.DodecahedronGeometry(0.5, 0);
  if (mask & F_TEXT) return new THREE.TetrahedronGeometry(0.6, 0);
  if (mask & F_IMAGE) return new THREE.BoxGeometry(0.8, 0.8, 0.8);
  if (mask & F_AUDIO) return new THREE.OctahedronGeometry(0.7, 0);
  if (mask & F_VIDEO) return new THREE.IcosahedronGeometry(0.6, 0);
  return new THREE.SphereGeometry(0.5, 8, 6);
}

const RAY_COLORS: Record<string, number> = {
  text: 0x88ccff,
  image: 0xffcc66,
  audio: 0x66ff88,
  video: 0xff6699,
};

function rayDirections(mask: number): Array<{ dir: THREE.Vector3; color: number }>{
  const dirs: Array<{dir: THREE.Vector3; color: number}> = [];
  if (mask & F_TEXT) dirs.push({ dir: new THREE.Vector3(1, 0.2, 0).normalize(), color: RAY_COLORS.text });
  if (mask & F_IMAGE) dirs.push({ dir: new THREE.Vector3(-1, 0.1, -0.1).normalize(), color: RAY_COLORS.image });
  if (mask & F_AUDIO) dirs.push({ dir: new THREE.Vector3(0.2, 1, 0.2).normalize(), color: RAY_COLORS.audio });
  if (mask & F_VIDEO) dirs.push({ dir: new THREE.Vector3(-0.2, -0.3, 1).normalize(), color: RAY_COLORS.video });
  return dirs;
}

export function buildInstancedStars(
  records: K3DRecord[],
  positions: Float32Array,
  scene: THREE.Scene,
  maxNodes = 2000
): THREE.Group {
  // Builds a near-field group with instanced star shapes and modality rays.
  const group = new THREE.Group();
  const n = Math.min(records.length, maxNodes);

  // Partition by geometry type to reduce draw calls
  const buckets = new Map<string, { geom: THREE.BufferGeometry; indices: number[] }>();
  const masks: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const m = modalityMask(records[i]);
    masks[i] = m;
    const geom = shapeGeometry(m);
    const key = `${geom.type}:${geom.uuid}`; // approximate key; geometries differ by class
    if (!buckets.has(key)) buckets.set(key, { geom, indices: [] });
    buckets.get(key)!.indices.push(i);
  }

  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.1, roughness: 0.6 });
  for (const { geom, indices } of buckets.values()) {
    const inst = new THREE.InstancedMesh(geom, mat, indices.length);
    const m = new THREE.Matrix4();
    const s = new THREE.Vector3(1, 1, 1);
    const q = new THREE.Quaternion();
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const x = positions[i * 3 + 0];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      // Minimal spacing-based scale based on local density later (placeholder 1)
      s.setScalar(0.18);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), (i * 0.17) % (Math.PI * 2));
      m.compose(new THREE.Vector3(x, y, z), q, s);
      inst.setMatrixAt(k, m);
    }
    inst.instanceMatrix.needsUpdate = true;
    group.add(inst);
  }

  // Rays: build per-modality cylinder instancing for thickness
  const rayGeom = new THREE.CylinderGeometry(0.03, 0.03, 1.0, 6, 1, true);
  const rayMats: Record<string, THREE.MeshBasicMaterial> = {
    text: new THREE.MeshBasicMaterial({ color: RAY_COLORS.text }),
    image: new THREE.MeshBasicMaterial({ color: RAY_COLORS.image }),
    audio: new THREE.MeshBasicMaterial({ color: RAY_COLORS.audio }),
    video: new THREE.MeshBasicMaterial({ color: RAY_COLORS.video }),
  };
  const rayBuckets: Record<string, Array<number>> = { text: [], image: [], audio: [], video: [] };
  for (let i = 0; i < n; i++) {
    const m = masks[i];
    if (m & F_TEXT) rayBuckets.text.push(i);
    if (m & F_IMAGE) rayBuckets.image.push(i);
    if (m & F_AUDIO) rayBuckets.audio.push(i);
    if (m & F_VIDEO) rayBuckets.video.push(i);
  }
  const tmp = new THREE.Matrix4();
  const up = new THREE.Vector3(0, 1, 0);
  function makeRays(kind: 'text'|'image'|'audio'|'video', indices: number[]) {
    if (!indices.length) return;
    const inst = new THREE.InstancedMesh(rayGeom, rayMats[kind], indices.length);
    const dirs = indices.map(i => {
      const mask = masks[i];
      const arr = rayDirections(mask);
      // pick matching dir
      const target = kind === 'text' ? RAY_COLORS.text : kind === 'image' ? RAY_COLORS.image : kind === 'audio' ? RAY_COLORS.audio : RAY_COLORS.video;
      const found = arr.find(a => a.color === target);
      return (found ? found.dir.clone() : new THREE.Vector3(1,0,0));
    });
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const x = positions[i*3+0]; const y = positions[i*3+1]; const z = positions[i*3+2];
      const dir = dirs[k];
      // length based on simple spacing heuristic: short rays to minimize overlap
      const len = 0.8; // actual length scaled per local spacing later
      const pos = new THREE.Vector3(x, y, z).add(dir.clone().multiplyScalar(len*0.5));
      const quat = new THREE.Quaternion();
      quat.setFromUnitVectors(up, dir);
      const scl = new THREE.Vector3(1, len, 1);
      tmp.compose(pos, quat, scl);
      inst.setMatrixAt(k, tmp);
    }
    inst.instanceMatrix.needsUpdate = true;
    group.add(inst);
  }
  makeRays('text', rayBuckets.text);
  makeRays('image', rayBuckets.image);
  makeRays('audio', rayBuckets.audio);
  makeRays('video', rayBuckets.video);

  scene.add(group);
  return group;
}

