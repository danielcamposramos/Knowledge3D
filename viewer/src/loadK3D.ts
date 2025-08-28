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

export async function loadK3DFromGLTF(url: string): Promise<K3DRecord[]> {
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
  const loader = new GLTFLoader();
  const gltf: any = await loader.loadAsync(url);
  const json = gltf.parser.json;
  if (!json || !json.meshes || json.meshes.length === 0) {
    throw new Error('Invalid GLTF: missing meshes');
  }
  const mesh0 = json.meshes[0];
  const prims = mesh0.primitives || [];
  let embedded: any | undefined;
  for (const p of prims) {
    if (p && p.extras && p.extras.k3d) { embedded = p.extras.k3d; break; }
  }
  if (!embedded) throw new Error('No embedded K3D payload found (primitive.extras.k3d)');

  const ids: string[] = embedded.ids || [];
  const metadata: any[] = embedded.metadata || [];
  const neighbors: string[][] | undefined = embedded.neighbors || undefined;

  // vectors
  let vectors: number[][] = [];
  if (embedded.vectorsView !== undefined) {
    const viewIdx = embedded.vectorsView as number;
    const buf = await gltf.parser.getDependency('bufferView', viewIdx);
    const arr = new Float32Array(buf);
    vectors = [];
    for (let i = 0; i < arr.length; i += 3) {
      vectors.push([arr[i], arr[i+1], arr[i+2]]);
    }
  } else if (embedded.vectors) {
    vectors = embedded.vectors as number[][];
  }

  // embeddings
  let embeddings: number[][] = [];
  if (embedded.embeddingsView !== undefined) {
    const viewIdx = embedded.embeddingsView as number;
    const dims = embedded.embeddingDims as number;
    const buf = await gltf.parser.getDependency('bufferView', viewIdx);
    const arr = new Float32Array(buf);
    embeddings = [];
    for (let i = 0; i < arr.length; i += dims) {
      embeddings.push(Array.from(arr.slice(i, i + dims)));
    }
  } else if (embedded.embeddings) {
    embeddings = embedded.embeddings as number[][];
  } else {
    embeddings = ids.map(() => []);
  }

  return composeRecordsFromEmbedded(ids, vectors, embeddings, metadata, neighbors);
}
