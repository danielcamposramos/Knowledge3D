import * as THREE from 'three';

import type { HouseNode, LoadedHouseScene } from '../loadHouseScene';

export interface MaterialPalette {
  room: THREE.MeshStandardMaterial;
  furniture: THREE.MeshStandardMaterial;
  door: THREE.MeshStandardMaterial;
  book: THREE.MeshStandardMaterial;
  tool: THREE.MeshStandardMaterial;
  display: THREE.MeshStandardMaterial;
  instrument: THREE.MeshStandardMaterial;
  tree: THREE.MeshStandardMaterial;
  tablet: THREE.MeshStandardMaterial;
  default: THREE.MeshStandardMaterial;
}

type PaletteSpec = {
  room: number;
  furniture: number;
  door: number;
  book: number;
  tool: number;
  display: number;
  instrument: number;
  tree: number;
  tablet: number;
  default: number;
};

const BASE_PARAMS = {
  flatShading: true,
  side: THREE.DoubleSide,
} satisfies Partial<THREE.MeshStandardMaterialParameters>;

const paletteCache = new Map<string, MaterialPalette>();

function makeMaterial(
  color: number,
  extra: Partial<THREE.MeshStandardMaterialParameters> = {},
): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.75,
    metalness: 0.05,
    ...BASE_PARAMS,
    ...extra,
  });
}

function buildPalette(spec: PaletteSpec): MaterialPalette {
  return {
    room: makeMaterial(spec.room, { roughness: 0.9, metalness: 0.0 }),
    furniture: makeMaterial(spec.furniture, { roughness: 0.58, metalness: 0.08 }),
    door: makeMaterial(spec.door, { roughness: 0.62, metalness: 0.06 }),
    book: makeMaterial(spec.book, { roughness: 0.84, metalness: 0.02 }),
    tool: makeMaterial(spec.tool, { roughness: 0.3, metalness: 0.6 }),
    display: makeMaterial(spec.display, { roughness: 0.42, metalness: 0.1 }),
    instrument: makeMaterial(spec.instrument, { roughness: 0.34, metalness: 0.45 }),
    tree: makeMaterial(spec.tree, { roughness: 0.88, metalness: 0.0 }),
    tablet: makeMaterial(spec.tablet, { roughness: 0.2, metalness: 0.35 }),
    default: makeMaterial(spec.default, { roughness: 0.7, metalness: 0.05 }),
  };
}

function livingRoomPalette(): MaterialPalette {
  return buildPalette({
    room: 0xf5e6d3,
    furniture: 0xb8b0a8,
    door: 0xc7a27a,
    book: 0xd0b18a,
    tool: 0x8ea7b8,
    display: 0xd7c4b6,
    instrument: 0xb6b9c7,
    tree: 0x7a9a71,
    tablet: 0x6e727f,
    default: 0xc9b8a7,
  });
}

function libraryPalette(): MaterialPalette {
  return buildPalette({
    room: 0x5c3d2e,
    furniture: 0x7a5239,
    door: 0x8b6914,
    book: 0xb48b5a,
    tool: 0x9c8f7a,
    display: 0x9b7257,
    instrument: 0x9b8c63,
    tree: 0x5f7a47,
    tablet: 0x545d67,
    default: 0x71503e,
  });
}

function gardenPalette(): MaterialPalette {
  return buildPalette({
    room: 0x4a7c59,
    furniture: 0x8b7355,
    door: 0x7d6247,
    book: 0xb4c28a,
    tool: 0x7c8b6a,
    display: 0x6d8e63,
    instrument: 0x97a97c,
    tree: 0x5aa05a,
    tablet: 0x5b6572,
    default: 0x68886d,
  });
}

function workshopPalette(): MaterialPalette {
  return buildPalette({
    room: 0x6b6b6b,
    furniture: 0x8a8a8a,
    door: 0x7b6c58,
    book: 0x9d8c6c,
    tool: 0xa0a0a0,
    display: 0x8d8d96,
    instrument: 0x9ab0b9,
    tree: 0x77826d,
    tablet: 0x6e7380,
    default: 0x83838b,
  });
}

function galleryPalette(): MaterialPalette {
  return buildPalette({
    room: 0xf0f0f0,
    furniture: 0xc04040,
    door: 0xb8b0a4,
    book: 0xd7bfaa,
    tool: 0xa9a9b6,
    display: 0xffffff,
    instrument: 0xc6c0d6,
    tree: 0x8eb27f,
    tablet: 0x7a818c,
    default: 0xdbdbdb,
  });
}

function bathtubPalette(): MaterialPalette {
  return buildPalette({
    room: 0x1a1a3e,
    furniture: 0x2a2a4a,
    door: 0x5c4c72,
    book: 0x8d7fa4,
    tool: 0x7d88a5,
    display: 0x615b88,
    instrument: 0x7a88b6,
    tree: 0x546a8b,
    tablet: 0x56607a,
    default: 0x35355d,
  });
}

function defaultPalette(): MaterialPalette {
  return buildPalette({
    room: 0xb9bcc6,
    furniture: 0x9c9fa9,
    door: 0x8c8274,
    book: 0xbba48a,
    tool: 0x8e98a8,
    display: 0xcfd2d8,
    instrument: 0x9ab4c0,
    tree: 0x79906e,
    tablet: 0x6e7482,
    default: 0xa9adb7,
  });
}

export function createDomainPalette(domain: string): MaterialPalette {
  const key = String(domain || '').trim() || 'default';
  const cached = paletteCache.get(key);
  if (cached) return cached;
  const palette = (() => {
    switch (key) {
      case 'House/LivingRoom':
        return livingRoomPalette();
      case 'House/Library':
        return libraryPalette();
      case 'House/Garden':
        return gardenPalette();
      case 'House/Workshop':
        return workshopPalette();
      case 'House/Gallery':
        return galleryPalette();
      case 'House/Bathtub':
        return bathtubPalette();
      default:
        return defaultPalette();
    }
  })();
  paletteCache.set(key, palette);
  return palette;
}

function paletteSlotForNode(node: HouseNode): keyof MaterialPalette {
  switch (node.meaningClass) {
    case 'room':
      return 'room';
    case 'furniture':
      if (node.starId === 'furniture_knowledge_tree') return 'tree';
      return 'furniture';
    case 'door':
      return 'door';
    case 'book':
      return 'book';
    case 'tool_object':
      return 'tool';
    case 'display':
      return 'display';
    case 'instrument':
      return 'instrument';
    case 'branch':
    case 'leaf':
      return 'tree';
    case 'tablet':
      return 'tablet';
    default:
      return 'default';
  }
}

function owningStarId(object: THREE.Object3D): string {
  let cursor: THREE.Object3D | null = object;
  while (cursor) {
    const starId = String((cursor.userData as any)?.k3d?.star_id || '').trim();
    if (starId) return starId;
    cursor = cursor.parent;
  }
  return '';
}

export function applyHouseMaterials(loadedScene: LoadedHouseScene): void {
  loadedScene.nodesByStarId.forEach((node) => {
    const palette = createDomainPalette(node.houseRoom || node.domain);
    const slot = paletteSlotForNode(node);
    const material = palette[slot] || palette.default;
    node.object.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      if (owningStarId(child) !== node.starId) return;
      child.material = material;
    });
  });
}
