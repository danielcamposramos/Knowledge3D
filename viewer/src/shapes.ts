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

// Shared geometry cache (to enable real instancing)
const GEO = {
  tetra: new THREE.TetrahedronGeometry(0.6, 0),
  box: new THREE.BoxGeometry(0.8, 0.8, 0.8),
  octa: new THREE.OctahedronGeometry(0.7, 0),
  icosa: new THREE.IcosahedronGeometry(0.6, 0),
  dodeca: new THREE.DodecahedronGeometry(0.5, 0),
  sphere: new THREE.SphereGeometry(0.5, 8, 6),
};

function shapeKey(mask: number): keyof typeof GEO {
  const bits = [F_TEXT, F_IMAGE, F_AUDIO, F_VIDEO].filter(b => (mask & b) !== 0).length;
  if (bits >= 2) return 'dodeca';
  if (mask & F_TEXT) return 'tetra';
  if (mask & F_IMAGE) return 'box';
  if (mask & F_AUDIO) return 'octa';
  if (mask & F_VIDEO) return 'icosa';
  return 'sphere';
}

function shapeGeometry(mask: number): THREE.BufferGeometry {
  return GEO[shapeKey(mask)];
}

function baseRadius(mask: number): number {
  // Approximate unscaled radius per geometry for spacing/ray calculations
  switch (shapeKey(mask)) {
    case 'tetra': return 0.6;
    case 'box': return 0.8;
    case 'octa': return 0.7;
    case 'icosa': return 0.6;
    case 'dodeca': return 0.5;
    default: return 0.5;
  }
}

const RAY_COLORS: Record<string, number> = {
  text: 0x88ccff,
  image: 0xffcc66,
  audio: 0x66ff88,
  video: 0xff6699,
};

const RAY_THICKNESS: Record<'text'|'image'|'audio'|'video', number> = {
  // Thickness by format (x,z scale of cylinder)
  text: 0.018,
  image: 0.030,
  audio: 0.024,
  video: 0.034,
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

  // Precompute local spacing: nearest-neighbor distance per node
  const idToIndex = new Map<string, number>();
  for (let i = 0; i < n; i++) idToIndex.set(records[i].id, i);
  const nnDist: number[] = new Array(n).fill(Number.POSITIVE_INFINITY);
  for (let i = 0; i < n; i++) {
    const r = records[i];
    const hasN = Array.isArray(r.neighbors) && r.neighbors!.length > 0;
    if (hasN) {
      for (const nid of r.neighbors!) {
        const j = idToIndex.get(nid);
        if (j === undefined || j === i) continue;
        const dx = positions[i*3+0]-positions[j*3+0];
        const dy = positions[i*3+1]-positions[j*3+1];
        const dz = positions[i*3+2]-positions[j*3+2];
        const d = Math.hypot(dx, dy, dz);
        if (d > 0 && d < nnDist[i]) nnDist[i] = d;
      }
    }
  }
  // Fallback: for any remaining INF, do a light O(n^2) search (n <= 2000)
  const needFallback = nnDist.some(v => !isFinite(v));
  if (needFallback) {
    for (let i = 0; i < n; i++) {
      if (isFinite(nnDist[i])) continue;
      let best = Number.POSITIVE_INFINITY;
      const ix = positions[i*3+0], iy = positions[i*3+1], iz = positions[i*3+2];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = ix-positions[j*3+0];
        const dy = iy-positions[j*3+1];
        const dz = iz-positions[j*3+2];
        const d = Math.hypot(dx, dy, dz);
        if (d > 0 && d < best) best = d;
      }
      nnDist[i] = isFinite(best) ? best : 1.0;
    }
  }
  // Global median for normalization
  const sorted = [...nnDist].sort((a,b) => a-b);
  const median = sorted.length ? sorted[Math.floor(sorted.length*0.5)] : 1.0;
  const safeMedian = (median > 1e-6) ? median : 1.0;

  // Partition by geometry type to reduce draw calls
  const buckets = new Map<string, { geom: THREE.BufferGeometry; indices: number[] }>();
  const masks: number[] = new Array(n);
  const scales: number[] = new Array(n);
  const bitsCount = (m: number) => [F_TEXT, F_IMAGE, F_AUDIO, F_VIDEO].reduce((s, b) => s + ((m & b) ? 1 : 0), 0);
  for (let i = 0; i < n; i++) {
    const m = modalityMask(records[i]);
    masks[i] = m;
    // Density-aware star scale: smaller in dense zones, larger in sparse
    const rel = Math.max(0.5, Math.min(1.5, nnDist[i] / safeMedian));
    scales[i] = 0.18 * rel; // base 0.18 scaled by density
    const geom = shapeGeometry(m);
    const key = shapeKey(m);
    if (!buckets.has(key)) buckets.set(key, { geom, indices: [] });
    buckets.get(key)!.indices.push(i);
  }

  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.35, metalness: 0.1, roughness: 0.6 });
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
      s.setScalar(scales[i]);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), (i * 0.17) % (Math.PI * 2));
      m.compose(new THREE.Vector3(x, y, z), q, s);
      inst.setMatrixAt(k, m);
    }
    inst.instanceMatrix.needsUpdate = true;
    group.add(inst);
  }

  // Rays: build per-modality cylinder instancing for thickness
  const rayGeom = new THREE.CylinderGeometry(0.03, 0.03, 1.0, 10, 1, true);
  // Subtle vertical gradient via vertex colors (brighter at base, slightly dim at tip)
  {
    const pos = rayGeom.getAttribute('position') as THREE.BufferAttribute;
    const count = pos.count;
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < count; i++) {
      const y = pos.getY(i);
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const colors = new Float32Array(count * 3);
    const span = Math.max(1e-6, maxY - minY);
    for (let i = 0; i < count; i++) {
      const y = pos.getY(i);
      const t = (y - minY) / span; // 0=base, 1=tip
      // factor: base 1.0 → tip 0.65
      const f = 1.0 - 0.35 * t;
      colors[i*3+0] = f;
      colors[i*3+1] = f;
      colors[i*3+2] = f;
    }
    rayGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  const rayMats: Record<string, THREE.MeshBasicMaterial> = {
    text: new THREE.MeshBasicMaterial({ color: RAY_COLORS.text, vertexColors: true }),
    image: new THREE.MeshBasicMaterial({ color: RAY_COLORS.image, vertexColors: true }),
    audio: new THREE.MeshBasicMaterial({ color: RAY_COLORS.audio, vertexColors: true }),
    video: new THREE.MeshBasicMaterial({ color: RAY_COLORS.video, vertexColors: true }),
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
  const rayLenSum: number[] = new Array(n).fill(0);
  const rayLenCnt: number[] = new Array(n).fill(0);
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
      // Spacing-aware length: respect nearest spacing minus star radii margin
      const margin = 2.0 * (scales[i] * baseRadius(masks[i])) + 0.05;
      const candidate = Math.max(0.12, 0.45 * Math.max(0.0, nnDist[i] - margin));
      const len = Math.min(0.8, candidate);
      const pos = new THREE.Vector3(x, y, z).add(dir.clone().multiplyScalar(len*0.5));
      const quat = new THREE.Quaternion();
      quat.setFromUnitVectors(up, dir);
      const thick = RAY_THICKNESS[kind];
      const scl = new THREE.Vector3(thick / 0.03, len, thick / 0.03);
      tmp.compose(pos, quat, scl);
      inst.setMatrixAt(k, tmp);
      rayLenSum[i] += len; rayLenCnt[i] += 1;
    }
    inst.instanceMatrix.needsUpdate = true;
    group.add(inst);
  }
  makeRays('text', rayBuckets.text);
  makeRays('image', rayBuckets.image);
  makeRays('audio', rayBuckets.audio);
  makeRays('video', rayBuckets.video);

  // Inject a lightweight shape embedding for AI-side enrichment (dual rendering)
  for (let i = 0; i < n; i++) {
    try {
      const m = masks[i];
      const t = (m & F_TEXT) ? 1 : 0;
      const im = (m & F_IMAGE) ? 1 : 0;
      const au = (m & F_AUDIO) ? 1 : 0;
      const v = (m & F_VIDEO) ? 1 : 0;
      const mix = bitsCount(m) / 4.0;
      const nn = Math.max(0, Math.min(1, nnDist[i] / (safeMedian * 2.0)));
      const scl = Math.max(0, Math.min(1, scales[i] / (0.18 * 1.5)));
      const rAvg = rayLenCnt[i] > 0 ? (rayLenSum[i] / rayLenCnt[i]) : 0;
      const rN = Math.max(0, Math.min(1, rAvg / 0.8));
      const shapeEmb = [t, im, au, v, nn, scl, rN, mix];
      const md: any = records[i].metadata || {};
      md.shape_embedding = shapeEmb;
      md.shape_bits = m;
      md.nn_dist = nnDist[i];
      md.ray_len_avg = rAvg;
      (records[i] as any).metadata = md;
    } catch {}
  }

  scene.add(group);
  return group;
}

// Build instanced garden branches/leaves for near view from explicit edges
export function buildInstancedBranches(
  positions: Float32Array,
  edges: Array<[number, number]>,
  scene: THREE.Scene,
  maxEdges = 5000
): THREE.Group {
  const group = new THREE.Group();
  // Cylinder oriented along edge vector for branches
  const cyl = new THREE.CylinderGeometry(0.02, 0.02, 1.0, 8, 1, true);
  const mat = new THREE.MeshStandardMaterial({ color: 0x7b5e2a, metalness: 0.05, roughness: 0.9 });
  const instCount = Math.min(edges.length, maxEdges);
  const inst = new THREE.InstancedMesh(cyl, mat, instCount);
  const tmp = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  for (let k = 0; k < instCount; k++) {
    const [ia, ib] = edges[k];
    const ax = positions[ia*3+0], ay = positions[ia*3+1], az = positions[ia*3+2];
    const bx = positions[ib*3+0], by = positions[ib*3+1], bz = positions[ib*3+2];
    const a = new THREE.Vector3(ax, ay, az);
    const b = new THREE.Vector3(bx, by, bz);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const dir = b.clone().sub(a);
    const len = Math.max(0.02, dir.length());
    dir.normalize();
    quat.setFromUnitVectors(up, dir);
    const scl = new THREE.Vector3(1, len, 1);
    tmp.compose(mid, quat, scl);
    inst.setMatrixAt(k, tmp);
  }
  inst.instanceMatrix.needsUpdate = true;
  group.add(inst);
  // Leaves: small spheres near branch ends
  const leafGeom = new THREE.IcosahedronGeometry(0.05, 0);
  const leafMat = new THREE.MeshPhongMaterial({ color: 0x4caf50, emissive: 0x163d1a });
  const leafCount = Math.min(instCount * 2, 16000);
  const leaf = new THREE.InstancedMesh(leafGeom, leafMat, leafCount);
  const lm = new THREE.Matrix4();
  let li = 0;
  for (let k = 0; k < instCount && li < leafCount; k++) {
    const [ia, ib] = edges[k];
    const ax = positions[ia*3+0], ay = positions[ia*3+1], az = positions[ia*3+2];
    const bx = positions[ib*3+0], by = positions[ib*3+1], bz = positions[ib*3+2];
    const a = new THREE.Vector3(ax, ay, az);
    const b = new THREE.Vector3(bx, by, bz);
    const jitter = () => (Math.random()-0.5) * 0.06;
    const sa = new THREE.Vector3().copy(a).add(new THREE.Vector3(jitter(), jitter(), jitter()));
    const sb = new THREE.Vector3().copy(b).add(new THREE.Vector3(jitter(), jitter(), jitter()));
    lm.makeTranslation(sa.x, sa.y, sa.z); leaf.setMatrixAt(li++, lm);
    if (li < leafCount) { lm.makeTranslation(sb.x, sb.y, sb.z); leaf.setMatrixAt(li++, lm); }
  }
  leaf.instanceMatrix.needsUpdate = true;
  group.add(leaf);
  scene.add(group);
  return group;
}
