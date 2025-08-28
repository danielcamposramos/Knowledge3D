export interface HouseInfo {
  uri: string;
  expert: string;
  description?: string;
}

export interface CondoConfig {
  houses: HouseInfo[];
}

export async function fetchCondoConfig(url: string): Promise<CondoConfig> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
  }
  return (await res.json()) as CondoConfig;
}

export interface K3DRecord {
  id: string;
  vector: [number, number, number];
  embedding: number[];
  metadata: Record<string, unknown>;
  neighbors?: string[];
}

function composeRecordsFromEmbedded(
  ids: string[],
  vectors: number[][],
  embeddings: number[][],
  metadata: any[] = [],
  neighbors?: string[][]
): K3DRecord[] {
  const records: K3DRecord[] = [];
  for (let i = 0; i < ids.length; i++) {
    records.push({
      id: ids[i],
      vector: vectors[i] as [number, number, number],
      embedding: embeddings[i] ?? [],
      metadata: metadata[i] ?? {},
      neighbors: neighbors ? neighbors[i] : undefined,
    });
  }
  return records;
}

export async function fetchK3D(url: string): Promise<K3DRecord[]> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
  }

  // Embedded variant in glTF: parse primitive.extras.k3d
  const gltf = await res.json();
  if (!gltf || !gltf.meshes || !Array.isArray(gltf.meshes) || gltf.meshes.length === 0) {
    throw new Error('Invalid GLTF: missing meshes');
  }
  const mesh0 = gltf.meshes[0];
  const prims = mesh0.primitives || [];
  let embedded: any | undefined;
  for (const p of prims) {
    if (p && p.extras && p.extras.k3d) {
      embedded = p.extras.k3d;
      break;
    }
  }
  if (!embedded) {
    throw new Error('No embedded K3D payload found in GLTF (primitive.extras.k3d)');
  }
  const ids: string[] = embedded.ids || [];
  const vectors: number[][] = embedded.vectors || [];
  const embeddings: number[][] = embedded.embeddings || [];
  const metadata: any[] = embedded.metadata || [];
  const neighbors: string[][] | undefined = embedded.neighbors || undefined;
  return composeRecordsFromEmbedded(ids, vectors, embeddings, metadata, neighbors);
}
