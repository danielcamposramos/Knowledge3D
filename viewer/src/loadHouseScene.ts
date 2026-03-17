import * as THREE from 'three';

export interface HouseSurfaceForm {
  word_ref: string;
  char_refs: string[];
}

export interface HouseNode {
  starId: string;
  meaningClass: string;
  domain: string;
  houseRoom: string;
  housePosition: [number, number, number];
  surfaceForms: Record<string, HouseSurfaceForm>;
  behaviorRpn: string;
  taxonomyRefs: string[];
  componentRefs: string[];
  visualRpn?: string;
  galaxyRef?: string;
  object: THREE.Object3D;
}

export interface HouseNavGraph {
  nodes: string[];
  edges: Array<{ door: string; from: string; to: string; cost: number }>;
}

export interface LoadedHouseScene {
  root: THREE.Group;
  nodesByStarId: Map<string, HouseNode>;
  rooms: HouseNode[];
  doors: HouseNode[];
  navGraph: HouseNavGraph;
  currentRoom: string;
}

function normalizeSurfaceForms(payload: unknown): Record<string, HouseSurfaceForm> {
  const result: Record<string, HouseSurfaceForm> = {};
  if (!payload || typeof payload !== 'object') {
    return result;
  }
  for (const [language, raw] of Object.entries(payload as Record<string, unknown>)) {
    if (!raw || typeof raw !== 'object') {
      continue;
    }
    const wordRef = String((raw as Record<string, unknown>).word_ref || '').trim();
    const charRefs = Array.isArray((raw as Record<string, unknown>).char_refs)
      ? ((raw as Record<string, unknown>).char_refs as unknown[])
          .map((value) => String(value).trim())
          .filter(Boolean)
      : [];
    result[String(language).trim().toLowerCase()] = {
      word_ref: wordRef,
      char_refs: charRefs,
    };
  }
  return result;
}

function normalizeStringList(payload: unknown): string[] {
  return Array.isArray(payload)
    ? payload.map((value) => String(value).trim()).filter(Boolean)
    : [];
}

function normalizePosition(payload: unknown): [number, number, number] {
  const raw = Array.isArray(payload) ? payload : [];
  return [
    Number(raw[0] ?? 0),
    Number(raw[1] ?? 0),
    Number(raw[2] ?? 0),
  ];
}

function normalizeNavGraph(payload: unknown): HouseNavGraph {
  if (!payload || typeof payload !== 'object') {
    return { nodes: [], edges: [] };
  }
  const raw = payload as Record<string, unknown>;
  return {
    nodes: normalizeStringList(raw.nodes),
    edges: Array.isArray(raw.edges)
      ? raw.edges
          .filter((edge) => edge && typeof edge === 'object')
          .map((edge) => {
            const value = edge as Record<string, unknown>;
            return {
              door: String(value.door || '').trim(),
              from: String(value.from || '').trim(),
              to: String(value.to || '').trim(),
              cost: Number(value.cost ?? 1),
            };
          })
          .filter((edge) => edge.door && edge.from && edge.to)
      : [],
  };
}

export async function loadHouseScene(url: string): Promise<LoadedHouseScene> {
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);
  const root = gltf.scene;
  const nodesByStarId = new Map<string, HouseNode>();
  let navGraph: HouseNavGraph = { nodes: [], edges: [] };

  root.traverse((object) => {
    const k3d = (object.userData as Record<string, unknown> | undefined)?.k3d as
      | Record<string, unknown>
      | undefined;
    if (!k3d) {
      return;
    }
    if (k3d.nav_graph) {
      navGraph = normalizeNavGraph(k3d.nav_graph);
    }
    const starId = String(k3d.star_id || '').trim();
    if (!starId) {
      return;
    }
    nodesByStarId.set(starId, {
      starId,
      meaningClass: String(k3d.meaning_class || '').trim(),
      domain: String(k3d.domain || '').trim(),
      houseRoom: String(k3d.house_room || '').trim(),
      housePosition: normalizePosition(k3d.house_position),
      surfaceForms: normalizeSurfaceForms(k3d.surface_forms),
      behaviorRpn: String(k3d.behavior_rpn || '').trim(),
      taxonomyRefs: normalizeStringList(k3d.taxonomy_refs),
      componentRefs: normalizeStringList(k3d.component_refs),
      visualRpn: k3d.visual_rpn ? String(k3d.visual_rpn) : undefined,
      galaxyRef: k3d.galaxy_ref ? String(k3d.galaxy_ref) : undefined,
      object,
    });
  });

  const rooms = Array.from(nodesByStarId.values()).filter((node) => node.meaningClass === 'room');
  const doors = Array.from(nodesByStarId.values()).filter((node) => node.meaningClass === 'door');
  const currentRoom = nodesByStarId.has('room_library')
    ? 'room_library'
    : (rooms[0]?.starId || '');

  return {
    root,
    nodesByStarId,
    rooms,
    doors,
    navGraph,
    currentRoom,
  };
}
