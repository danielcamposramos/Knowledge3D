import * as THREE from 'three';

import { RpnEngine } from './engine';
import { mat4Ops } from './mat4Ops';
import { meshOps } from './meshOps';
import { pathOps } from './pathOps';
import { PathBuilder } from './pathBuilder';

function geometryBounds(geometry: THREE.BufferGeometry): THREE.Box3 {
  const clone = geometry.clone();
  clone.computeBoundingBox();
  return clone.boundingBox?.clone() || new THREE.Box3();
}

function mergeGeometryList(geometries: THREE.BufferGeometry[]): THREE.BufferGeometry {
  if (!geometries.length) {
    return new THREE.BufferGeometry();
  }
  if (geometries.length === 1) {
    return geometries[0].clone();
  }
  const expanded = geometries.map((geometry) => {
    const clone = geometry.clone();
    const normalized = clone.index ? clone.toNonIndexed() || clone : clone;
    if (!normalized.getAttribute('normal')) normalized.computeVertexNormals();
    return normalized;
  });
  const totalVertices = expanded.reduce((sum, geometry) => {
    const position = geometry.getAttribute('position');
    return sum + (position ? position.count : 0);
  }, 0);
  const positions = new Float32Array(totalVertices * 3);
  const normals = new Float32Array(totalVertices * 3);
  const uvs = new Float32Array(totalVertices * 2);
  let positionOffset = 0;
  let uvOffset = 0;
  for (const geometry of expanded) {
    const position = geometry.getAttribute('position');
    if (!position) continue;
    positions.set(position.array as ArrayLike<number>, positionOffset);
    const normal = geometry.getAttribute('normal');
    if (normal) normals.set(normal.array as ArrayLike<number>, positionOffset);
    const uv = geometry.getAttribute('uv');
    if (uv) uvs.set(uv.array as ArrayLike<number>, uvOffset);
    positionOffset += position.count * 3;
    uvOffset += position.count * 2;
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  merged.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  merged.computeBoundingBox();
  merged.computeBoundingSphere();
  return merged;
}

function geometryFromBox(bounds: THREE.Box3): THREE.BufferGeometry {
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  bounds.getSize(size);
  bounds.getCenter(center);
  const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
  geometry.applyMatrix4(new THREE.Matrix4().makeTranslation(center.x, center.y, center.z));
  return geometry;
}

function mergePair(a: THREE.BufferGeometry, b: THREE.BufferGeometry): THREE.BufferGeometry {
  return mergeGeometryList([a, b]);
}

export class MeshBuilder {
  private geometries: THREE.BufferGeometry[] = [];

  reset(): void {
    this.geometries.forEach((geometry) => geometry.dispose());
    this.geometries = [];
  }

  hasGeometry(): boolean {
    return this.geometries.length > 0;
  }

  addPrimitive(geom: THREE.BufferGeometry): void {
    this.geometries.push(geom.clone());
  }

  applyTransform(matrix: THREE.Matrix4): void {
    const current = this.geometries[this.geometries.length - 1];
    if (!current) return;
    current.applyMatrix4(matrix);
    current.computeBoundingBox();
    current.computeBoundingSphere();
  }

  csgUnion(): void {
    if (this.geometries.length < 2) return;
    const b = this.geometries.pop()!;
    const a = this.geometries.pop()!;
    this.geometries.push(mergePair(a, b));
  }

  csgSubtract(): void {
    if (this.geometries.length < 2) return;
    const subtractor = this.geometries.pop()!;
    const base = this.geometries.pop()!;
    const outer = geometryBounds(base);
    const inner = geometryBounds(subtractor);
    const overlap = outer.clone().intersect(inner);
    if (overlap.isEmpty()) {
      this.geometries.push(base);
      return;
    }
    const slabs: THREE.BufferGeometry[] = [];
    const min = outer.min.clone();
    const max = outer.max.clone();
    const iMin = overlap.min.clone();
    const iMax = overlap.max.clone();
    const boxes = [
      new THREE.Box3(new THREE.Vector3(min.x, min.y, min.z), new THREE.Vector3(iMin.x, max.y, max.z)),
      new THREE.Box3(new THREE.Vector3(iMax.x, min.y, min.z), new THREE.Vector3(max.x, max.y, max.z)),
      new THREE.Box3(new THREE.Vector3(iMin.x, min.y, min.z), new THREE.Vector3(iMax.x, iMin.y, max.z)),
      new THREE.Box3(new THREE.Vector3(iMin.x, iMax.y, min.z), new THREE.Vector3(iMax.x, max.y, max.z)),
      new THREE.Box3(new THREE.Vector3(iMin.x, iMin.y, min.z), new THREE.Vector3(iMax.x, iMax.y, iMin.z)),
      new THREE.Box3(new THREE.Vector3(iMin.x, iMin.y, iMax.z), new THREE.Vector3(iMax.x, iMax.y, max.z)),
    ].filter((box) => {
      const size = new THREE.Vector3();
      box.getSize(size);
      return size.x > 1e-6 && size.y > 1e-6 && size.z > 1e-6;
    });
    for (const box of boxes) slabs.push(geometryFromBox(box));
    if (!slabs.length) return;
    this.geometries.push(mergeGeometryList(slabs));
  }

  csgIntersect(): void {
    if (this.geometries.length < 2) return;
    const b = this.geometries.pop()!;
    const a = this.geometries.pop()!;
    const overlap = geometryBounds(a).intersect(geometryBounds(b));
    if (overlap.isEmpty()) return;
    this.geometries.push(geometryFromBox(overlap));
  }

  toGeometry(): THREE.BufferGeometry {
    if (!this.geometries.length) {
      return new THREE.BufferGeometry();
    }
    if (this.geometries.length === 1) {
      return this.geometries[0].clone();
    }
    return mergeGeometryList(this.geometries);
  }

  toMesh(material?: THREE.Material): THREE.Mesh {
    const mesh = new THREE.Mesh(
      this.toGeometry(),
      material || new THREE.MeshNormalMaterial({ wireframe: true }),
    );
    mesh.geometry.computeBoundingBox();
    mesh.geometry.computeBoundingSphere();
    return mesh;
  }
}

export function rpnToMesh(program: string): THREE.Mesh {
  const engine = new RpnEngine();
  engine.context.path = new PathBuilder();
  engine.context.mesh = new MeshBuilder();
  engine.registerModule(mat4Ops);
  engine.registerModule(pathOps);
  engine.registerModule(meshOps);
  engine.execute(program);
  return engine.context.mesh!.toMesh();
}
