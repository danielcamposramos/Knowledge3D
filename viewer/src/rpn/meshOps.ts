import * as THREE from 'three';

import type { RpnOpHandler } from './engine';
import { popNumber } from './engine';

export const meshOps: Record<string, RpnOpHandler> = {
  GEN_CUBE: (stack, ctx) => {
    const size = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.BoxGeometry(size, size, size));
  },
  GEN_CYLINDER: (stack, ctx) => {
    const closed = popNumber(stack);
    const segments = Math.max(3, Math.round(popNumber(stack)));
    const height = popNumber(stack);
    const radius = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.CylinderGeometry(radius, radius, height, segments, 1, closed <= 0));
  },
  GEN_CONE: (stack, ctx) => {
    const segments = Math.max(3, Math.round(popNumber(stack)));
    const height = popNumber(stack);
    const radius = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.ConeGeometry(radius, height, segments));
  },
  GEN_TORUS: (stack, ctx) => {
    const tubularSegments = Math.max(3, Math.round(popNumber(stack)));
    const radialSegments = Math.max(3, Math.round(popNumber(stack)));
    const tube = popNumber(stack);
    const radius = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.TorusGeometry(radius, tube, radialSegments, tubularSegments));
  },
  GEN_UV_SPHERE: (stack, ctx) => {
    const widthSegments = Math.max(3, Math.round(popNumber(stack)));
    const heightSegments = Math.max(2, Math.round(popNumber(stack)));
    const radius = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.SphereGeometry(radius, widthSegments, heightSegments));
  },
  GEN_PLANE: (stack, ctx) => {
    const segmentsY = Math.max(1, Math.round(popNumber(stack)));
    const segmentsX = Math.max(1, Math.round(popNumber(stack)));
    const height = popNumber(stack);
    const width = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.PlaneGeometry(width, height, segmentsX, segmentsY));
  },
  GEN_ICOSPHERE: (stack, ctx) => {
    const subdivisions = Math.max(0, Math.round(popNumber(stack)));
    const radius = popNumber(stack);
    ctx.mesh?.addPrimitive(new THREE.IcosahedronGeometry(radius, subdivisions));
  },
  CSG_UNION: (_stack, ctx) => {
    ctx.mesh?.csgUnion();
  },
  CSG_SUBTRACT: (_stack, ctx) => {
    ctx.mesh?.csgSubtract();
  },
  CSG_INTERSECT: (_stack, ctx) => {
    ctx.mesh?.csgIntersect();
  },
  EXTRUDE: (stack, ctx) => {
    const depth = popNumber(stack);
    const shape = ctx.path?.toShape();
    if (!shape) return;
    ctx.mesh?.addPrimitive(new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false }));
  },
  LATHE: (stack, ctx) => {
    const segments = Math.max(3, Math.round(popNumber(stack)));
    const points = ctx.path?.toLathePoints();
    if (!points || points.length < 2) return;
    ctx.mesh?.addPrimitive(new THREE.LatheGeometry(points, segments));
  },
};
