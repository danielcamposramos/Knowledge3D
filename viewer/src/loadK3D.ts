export interface K3DNode {
  id: string;
  sourceVector?: number[];
  metadata: {
    label: string;
    [key: string]: unknown;
  };
  neighbors?: {
    nodeId: string;
    distance?: number;
  }[];
}

export interface K3DFile {
  asset: {
    version: string;
    generator?: string;
  };
  nodes: K3DNode[];
}

export async function fetchK3D(
  url: string
): Promise<Map<string, K3DNode>> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch K3D file from ${url}: ${res.statusText}`);
  }
  const k3dFile = (await res.json()) as K3DFile;

  const nodeMap = new Map<string, K3DNode>();
  for (const node of k3dFile.nodes) {
    nodeMap.set(node.id, node);
  }
  return nodeMap;
}
