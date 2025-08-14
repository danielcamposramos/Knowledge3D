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

export async function fetchK3D(url: string): Promise<K3DRecord[]> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.statusText}`);
  }
  return (await res.json()) as K3DRecord[];
}
