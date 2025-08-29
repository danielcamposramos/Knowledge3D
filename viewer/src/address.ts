export type Vec3 = [number, number, number];

export function spatialAddress(vec: Vec3, cellSize = 1.0, port = 0, label?: string): string {
  const [x, y, z] = vec;
  const rx = Math.floor(x / cellSize);
  const ry = Math.floor(y / cellSize);
  const rz = Math.floor(z / cellSize);
  const base = `k3d://${rx},${ry},${rz}:${port}@${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`;
  return label ? `${base}?label=${encodeURIComponent(label)}` : base;
}

// Expose a safe global accessor so runtime can build addresses from other modules
// without coupling import trees.
(window as any).k3dSpatialAddress = spatialAddress;
