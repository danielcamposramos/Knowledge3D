export interface K3DRecord {
  id: string;
  vector: [number, number, number];
  embedding: number[];
  metadata: Record<string, unknown>;
}

export async function fetchK3D(url: string): Promise<K3DRecord[]> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
  }
  return (await res.json()) as K3DRecord[];
}
