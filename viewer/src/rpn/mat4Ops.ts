import * as THREE from 'three';

import type { RpnOpHandler } from './engine';
import { popMatrix, popNumber } from './engine';

function matrixToArray(matrix: THREE.Matrix4): Float32Array {
  return new Float32Array(matrix.elements);
}

function arrayToMatrix(array: Float32Array): THREE.Matrix4 {
  return new THREE.Matrix4().fromArray(Array.from(array));
}

function pushMatrix(stack: (number | Float32Array | object)[], matrix: THREE.Matrix4) {
  stack.push(matrixToArray(matrix));
}

export const mat4Ops: Record<string, RpnOpHandler> = {
  MAT4_IDENTITY: (stack) => {
    pushMatrix(stack, new THREE.Matrix4().identity());
  },
  MAT4_TRANSLATE: (stack) => {
    const z = popNumber(stack);
    const y = popNumber(stack);
    const x = popNumber(stack);
    pushMatrix(stack, new THREE.Matrix4().makeTranslation(x, y, z));
  },
  MAT4_SCALE: (stack) => {
    const z = popNumber(stack);
    const y = popNumber(stack);
    const x = popNumber(stack);
    pushMatrix(stack, new THREE.Matrix4().makeScale(x, y, z));
  },
  MAT4_ROTATE_X: (stack) => {
    pushMatrix(stack, new THREE.Matrix4().makeRotationX(popNumber(stack)));
  },
  MAT4_ROTATE_Y: (stack) => {
    pushMatrix(stack, new THREE.Matrix4().makeRotationY(popNumber(stack)));
  },
  MAT4_ROTATE_Z: (stack) => {
    pushMatrix(stack, new THREE.Matrix4().makeRotationZ(popNumber(stack)));
  },
  MAT4_MUL: (stack) => {
    const b = arrayToMatrix(popMatrix(stack));
    const a = arrayToMatrix(popMatrix(stack));
    pushMatrix(stack, a.multiply(b));
  },
  MAT4_APPLY: (stack, ctx) => {
    const matrix = popMatrix(stack);
    const threeMatrix = arrayToMatrix(matrix);
    ctx.matrixStack.push(matrix);
    if (ctx.mesh?.hasGeometry()) {
      ctx.mesh.applyTransform(threeMatrix);
      return;
    }
    ctx.path?.applyMatrix(threeMatrix);
  },
};
