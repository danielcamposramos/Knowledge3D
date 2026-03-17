import * as THREE from 'three';

import type { ContentEntry, HouseContent } from '../contentLoader';
import type { HouseNode } from '../loadHouseScene';
import { ProjectionSurface } from './surface';

type GalaxyEntry = Pick<ContentEntry, 'star_id' | 'domain'>;

function domainKey(domain: string): string {
  const value = String(domain || '').toLowerCase();
  if (value.includes('math')) return 'mathematics';
  if (value.includes('language')) return 'language';
  if (value.includes('physics')) return 'physics';
  if (value.includes('biology')) return 'biology';
  if (value.includes('tool')) return 'tools';
  return 'default';
}

export class GalaxyPodProjector {
  readonly surface: ProjectionSurface;
  private stars: THREE.Points | null = null;
  private rotationSpeed = 0.02;

  constructor(bathtubNode: HouseNode) {
    this.surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 1.5, 0),
      bounds: new THREE.Vector3(3, 3, 3),
      mode: 'stellarium',
    });
    bathtubNode.object.add(this.surface.group);
  }

  get visible(): boolean {
    return this.surface.visible;
  }

  projectGalaxy(entries: GalaxyEntry[]): void {
    const palette: Record<string, THREE.Color> = {
      mathematics: new THREE.Color(0x4488ff),
      language: new THREE.Color(0x44ff88),
      physics: new THREE.Color(0xff8844),
      biology: new THREE.Color(0x88ff44),
      tools: new THREE.Color(0xff44ff),
      default: new THREE.Color(0xaaaaff),
    };
    const source = entries.length ? entries : [{ star_id: 'default_star', domain: 'default' }];
    const count = Math.max(entries.length, 200);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i += 1) {
      const phi = Math.acos(1 - 2 * (i + 0.5) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const radius = 1.35 + ((i % 11) / 80);
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      const entry = source[i % source.length];
      const color = palette[domainKey(entry.domain)] || palette.default;
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.stars = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 0.04,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        sizeAttenuation: true,
      }),
    );
    this.surface.setContent(this.stars);
    this.surface.show();
  }

  projectFromContent(content: HouseContent | null): void {
    if (this.stars && !this.surface.visible) {
      this.surface.show();
      return;
    }
    const entries: GalaxyEntry[] = [];
    if (content) {
      entries.push(...Object.values(content.concepts));
      for (const book of Object.values(content.books)) {
        entries.push(...book.entries);
      }
    }
    this.projectGalaxy(entries);
  }

  dismiss(): void {
    this.surface.hide();
  }

  update(delta: number): void {
    this.surface.update(delta);
    if (this.stars) {
      this.stars.rotation.y += this.rotationSpeed * delta;
      this.stars.rotation.x += this.rotationSpeed * 0.3 * delta;
    }
  }
}
