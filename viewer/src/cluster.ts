export type ClusterResult = { labels: number[]; centers: number[][] };

function randInt(n: number) { return Math.floor(Math.random() * n); }

export function kmeans(data: number[][], k: number, maxIter = 50): ClusterResult {
  if (data.length === 0) return { labels: [], centers: [] };
  const dims = data[0].length;
  // init centers by sampling
  const centers = Array.from({ length: k }, () => data[randInt(data.length)].slice());
  const labels = new Array<number>(data.length).fill(0);

  function dist2(a: number[], b: number[]) {
    let s = 0; for (let i = 0; i < dims; i++) { const d = a[i] - b[i]; s += d * d; } return s;
  }

  for (let it = 0; it < maxIter; it++) {
    let changed = 0;
    // assign
    for (let i = 0; i < data.length; i++) {
      let best = 0; let bestd = dist2(data[i], centers[0]);
      for (let c = 1; c < k; c++) { const d = dist2(data[i], centers[c]); if (d < bestd) { bestd = d; best = c; } }
      if (labels[i] !== best) { labels[i] = best; changed++; }
    }
    // update
    const acc = Array.from({ length: k }, () => new Array<number>(dims).fill(0));
    const count = new Array<number>(k).fill(0);
    for (let i = 0; i < data.length; i++) { const c = labels[i]; count[c]++; const v = data[i]; for (let d = 0; d < dims; d++) acc[c][d] += v[d]; }
    for (let c = 0; c < k; c++) { const inv = count[c] > 0 ? 1 / count[c] : 0; for (let d = 0; d < dims; d++) centers[c][d] = acc[c][d] * inv; }
    if (changed === 0) break;
  }
  return { labels, centers };
}

export function palette(n: number): string[] {
  const colors: string[] = [];
  for (let i = 0; i < n; i++) {
    const h = (i * 360 / n) | 0;
    colors.push(`hsl(${h} 70% 50%)`);
  }
  return colors;
}

