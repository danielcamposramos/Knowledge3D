import { readFileSync } from 'fs';
import { join } from 'path';
import * as THREE from 'three';
import { buildPoints, K3DRecord } from '../src/loadK3D';

test('buildPoints converts k3d vectors to Points', () => {
  const text = readFileSync(join(__dirname, '../../examples/sample_output.k3d'), 'utf-8');
  const records = JSON.parse(text) as K3DRecord[];
  const points = buildPoints(records);
  const pos = points.geometry.getAttribute('position') as THREE.BufferAttribute;
  expect(pos.count).toBe(records.length);
  expect(pos.getX(0)).toBeCloseTo(records[0].vector[0]);
});
