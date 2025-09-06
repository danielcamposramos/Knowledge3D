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

export interface K3DInfo {
  precision: string;
  dims: number;
  count: number;
  byteLengthVectors?: number;
  byteLengthEmbeddings?: number;
  temporal?: {
    alpha?: number;
    alphaMask?: number[];
  };
  ai?: {
    protocol?: string;
    flags?: {
      is_active?: boolean;
      is_traversable?: boolean;
      has_new_information?: boolean;
    };
    mask?: {
      has_new_information?: boolean[];
    };
  };
}

export interface LoadedK3D {
  data: K3DRecord[];
  info: K3DInfo;
  edges?: [string, string][];
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

export async function loadK3DFromGLTF(url: string): Promise<LoadedK3D> {
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
  const edges: [string, string][] | undefined = embedded.edges || undefined;
  const precGlobal: string = embedded.embeddingPrecision || 'f32';
  const dimsGlobal: number = embedded.embeddingDims || (embedded.embeddings?.[0]?.length ?? 0);
  const temporal: any = embedded.temporal || {};

  // AI-native extras on primitive
  const aiProtocol: string | undefined = embedded.ai_interaction_protocol;
  const aiFlags: any = embedded.ai_state_flags || undefined;
  const aiMask: any = embedded.ai_state_flags_mask || undefined;
  if (aiProtocol) {
    if (aiProtocol === 'direct_vector_manipulation') {
      console.log('[AI] protocol=direct_vector_manipulation: ready to process embeddings');
    } else {
      console.log(`[AI] protocol=${aiProtocol}`);
    }
  }
  if (aiFlags) {
    console.log('[AI] state flags:', aiFlags);
  }

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
    if (precGlobal === 'f16') {
      const u16 = new Uint16Array(buf);
      embeddings = [];
      for (let i = 0; i < u16.length; i += dims) {
        const row: number[] = [];
        for (let j = 0; j < dims; j++) row.push(halfToFloat(u16[i + j]));
        embeddings.push(row);
      }
    } else {
      const arr = new Float32Array(buf);
      embeddings = [];
      for (let i = 0; i < arr.length; i += dims) {
        embeddings.push(Array.from(arr.slice(i, i + dims)));
      }
    }
  } else if (embedded.embeddings) {
    embeddings = embedded.embeddings as number[][];
  } else {
    embeddings = ids.map(() => []);
  }

  const info: K3DInfo = {
    precision: precGlobal,
    dims: dimsGlobal,
    count: ids.length,
    byteLengthVectors: embedded.vectorsView !== undefined ? (json.bufferViews?.[embedded.vectorsView]?.byteLength ?? undefined) : undefined,
    byteLengthEmbeddings: embedded.embeddingsView !== undefined ? (json.bufferViews?.[embedded.embeddingsView]?.byteLength ?? undefined) : undefined,
    temporal: {
      alpha: typeof temporal.alpha === 'number' ? Math.max(0, Math.min(1, temporal.alpha)) : undefined,
      alphaMask: Array.isArray(temporal.alphaMask) ? temporal.alphaMask.map((x: any) => Math.max(0, Math.min(1, Number(x)))) : undefined,
    },
    ai: {
      protocol: aiProtocol,
      flags: aiFlags,
      mask: aiMask,
    },
  };

  return { data: composeRecordsFromEmbedded(ids, vectors, embeddings, metadata, neighbors), info, edges };
}

// IEEE754 half to float32
function halfToFloat(h: number): number {
  const s = (h >> 15) & 1;
  let e = (h >> 10) & 0x1f;
  let f = h & 0x3ff;
  if (e === 0) {
    if (f === 0) return s ? -0 : 0;
    while ((f & 0x400) === 0) { f <<= 1; e -= 1; }
    e += 1; f &= ~0x400;
  } else if (e === 31) {
    if (f === 0) return s ? -Infinity : Infinity;
    return NaN;
  }
  e = e + (127 - 15);
  const bits = (s << 31) | (e << 23) | (f << 13);
  const buf = new ArrayBuffer(4);
  new DataView(buf).setUint32(0, bits);
  return new DataView(buf).getFloat32(0);
}
